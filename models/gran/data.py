import torch
import time
import os
import pickle
import glob
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
#from utils.data_helper import *


class GRANData(object):

  def __init__(self, config, max_nodes,tag='train'):
    self.config = config
    #self.data_path = config.dataset.data_path
    #self.model_name = config.model.name
    self.max_num_nodes = max_nodes
    self.block_size = config.block_size
    self.stride = config.sample_stride

    #self.graphs = graphs
    #self.num_graphs = len(graphs)
    self.npr = np.random.RandomState(config.seed)
    #self.node_order = config.dataset.node_order
    self.num_canonical_order = config.num_canonical_order
    self.tag = tag
    self.num_fwd_pass = config.num_fwd_pass
    self.is_sample_subgraph = False
    self.num_subgraph_batch = 50
    self.is_overwrite_precompute = False
    self.device = config.device


  def _get_graph_data(self, G, perm):
    node_degree_list = [(n, d) for n, d in G.degree()]

    adj_0 = np.array(nx.to_numpy_matrix(G))

    ### Degree descent ranking
    # N.B.: largest-degree node may not be unique
    degree_sequence = sorted(
        node_degree_list, key=lambda tt: tt[1], reverse=True)
    adj_1 = np.array(
        nx.to_numpy_matrix(G, nodelist=perm))

    adj_list = [adj_1]

    # print('number of nodes = {}'.format(adj_0.shape[0]))

    return adj_list

  def __call__(self, graph, perm):
    K = self.block_size
    N = self.max_num_nodes
    S = self.stride

    # load graph
    adj_list = self._get_graph_data(graph,perm)
    num_nodes = adj_list[0].shape[0]
    num_subgraphs = int(np.floor((num_nodes - K) / S) + 1)

    if self.is_sample_subgraph:
      if self.num_subgraph_batch < num_subgraphs:
        num_subgraphs_pass = int(
            np.floor(self.num_subgraph_batch / self.num_fwd_pass))
      else:
        num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))

      end_idx = min(num_subgraphs, self.num_subgraph_batch)
    else:
      num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))
      end_idx = num_subgraphs

    ### random permute subgraph
    rand_perm_idx = self.npr.permutation(num_subgraphs).tolist()

    start_time = time.time()
    data_batch = []
    for ff in range(self.num_fwd_pass):
      ff_idx_start = num_subgraphs_pass * ff
      if ff == self.num_fwd_pass - 1:
        ff_idx_end = end_idx
      else:
        ff_idx_end = (ff + 1) * num_subgraphs_pass

      rand_idx = rand_perm_idx[ff_idx_start:ff_idx_end]

      edges = []
      node_idx_gnn = []
      node_idx_feat = []
      label = []
      subgraph_size = []
      subgraph_idx = []
      att_idx = []
      subgraph_count = 0

      for ii in range(len(adj_list)):
        # loop over different orderings
        adj_full = adj_list[ii]
        # adj_tril = np.tril(adj_full, k=-1)

        idx = -1
        for jj in range(0, num_nodes, S):
          # loop over different subgraphs
          idx += 1

          ### for each size-(jj+K) subgraph, we generate edges for the new block of K nodes
          if jj + K > num_nodes:
            break

          if idx not in rand_idx:
            continue

          ### get graph for GNN propagation
          adj_block = np.pad(
              adj_full[:jj, :jj], ((0, K), (0, K)),
              'constant',
              constant_values=1.0)  # assuming fully connected for the new block
          adj_block = np.tril(adj_block, k=-1)
          adj_block = adj_block + adj_block.transpose()
          adj_block = torch.from_numpy(adj_block).to_sparse()
          edges += [adj_block.coalesce().indices().long()]

          ### get attention index
          # exist node: 0
          # newly added node: 1, ..., K
          if jj == 0:
            att_idx += [np.arange(1, K + 1).astype(np.uint8)]
          else:
            att_idx += [
                np.concatenate([
                    np.zeros(jj).astype(np.uint8),
                    np.arange(1, K + 1).astype(np.uint8)
                ])
            ]

          ### get node feature index for GNN input
          # use inf to indicate the newly added nodes where input feature is zero
          if jj == 0:
            node_idx_feat += [np.ones(K) * np.inf]
          else:
            node_idx_feat += [
                np.concatenate([np.arange(jj) + ii * N,
                                np.ones(K) * np.inf])
            ]

          ### get node index for GNN output
          idx_row_gnn, idx_col_gnn = np.meshgrid(
              np.arange(jj, jj + K), np.arange(jj + K))
          idx_row_gnn = idx_row_gnn.reshape(-1, 1)
          idx_col_gnn = idx_col_gnn.reshape(-1, 1)
          node_idx_gnn += [
              np.concatenate([idx_row_gnn, idx_col_gnn],
                             axis=1).astype(np.int64)
          ]

          ### get predict label
          label += [
              adj_full[idx_row_gnn, idx_col_gnn].flatten().astype(np.uint8)
          ]

          subgraph_size += [jj + K]
          subgraph_idx += [
              np.ones_like(label[-1]).astype(np.int64) * subgraph_count
          ]
          subgraph_count += 1

      ### adjust index basis for the selected subgraphs
      cum_size = np.cumsum([0] + subgraph_size).astype(np.int64)
      for ii in range(len(edges)):
        edges[ii] = edges[ii] + cum_size[ii]
        node_idx_gnn[ii] = node_idx_gnn[ii] + cum_size[ii]

      ### pack tensors
      data = {}
      data['adj'] = np.tril(np.stack(adj_list, axis=0), k=-1)
      data['edges'] = torch.cat(edges, dim=1).t().long()
      data['node_idx_gnn'] = np.concatenate(node_idx_gnn)
      data['node_idx_feat'] = np.concatenate(node_idx_feat)
      data['label'] = np.concatenate(label)
      data['att_idx'] = np.concatenate(att_idx)
      data['subgraph_idx'] = np.concatenate(subgraph_idx)
      data['subgraph_count'] = subgraph_count
      data['num_nodes'] = num_nodes
      data['subgraph_size'] = subgraph_size
      data['num_count'] = sum(subgraph_size)
      data_batch += [data]

    end_time = time.time()

    return data_batch

  def collate_fn(self, batch):
    assert isinstance(batch, list)
    start_time = time.time()
    batch_size = len(batch)
    N = self.max_num_nodes
    C = self.num_canonical_order
    batch_data = []

    for ff in range(self.num_fwd_pass):
      data = {}
      batch_pass = []
      for bb in batch:
        batch_pass += [bb[ff]]

      pad_size = [self.max_num_nodes - bb['num_nodes'] for bb in batch_pass]
      subgraph_idx_base = np.array([0] +
                                   [bb['subgraph_count'] for bb in batch_pass])
      subgraph_idx_base = np.cumsum(subgraph_idx_base)

      data['subgraph_idx_base'] = torch.from_numpy(
        subgraph_idx_base).to(self.device)

      data['num_nodes_gt'] = torch.from_numpy(
          np.array([bb['num_nodes'] for bb in batch_pass])).long().view(-1).to(self.device)

      data['adj'] = torch.from_numpy(
          np.stack(
              [
                  np.pad(
                      bb['adj'], ((0, 0), (0, pad_size[ii]), (0, pad_size[ii])),
                      'constant',
                      constant_values=0.0) for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).float().to(self.device)  # B X C X N X N

      idx_base = np.array([0] + [bb['num_count'] for bb in batch_pass])
      idx_base = np.cumsum(idx_base)

      data['edges'] = torch.cat(
          [bb['edges'] + idx_base[ii] for ii, bb in enumerate(batch_pass)],
          dim=0).long().to(self.device)

      data['node_idx_gnn'] = torch.from_numpy(
          np.concatenate(
              [
                  bb['node_idx_gnn'] + idx_base[ii]
                  for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).long().to(self.device)

      data['att_idx'] = torch.from_numpy(
          np.concatenate([bb['att_idx'] for bb in batch_pass], axis=0)).long().to(self.device)

      # shift one position for padding 0-th row feature in the model
      node_idx_feat = np.concatenate(
          [
              bb['node_idx_feat'] + ii * C * N
              for ii, bb in enumerate(batch_pass)
          ],
          axis=0) + 1
      node_idx_feat[np.isinf(node_idx_feat)] = 0
      node_idx_feat = node_idx_feat.astype(np.int64)
      data['node_idx_feat'] = torch.from_numpy(node_idx_feat).long().to(self.device)

      data['label'] = torch.from_numpy(
          np.concatenate([bb['label'] for bb in batch_pass])).float().to(self.device)

      data['subgraph_idx'] = torch.from_numpy(
          np.concatenate([
              bb['subgraph_idx'] + subgraph_idx_base[ii]
              for ii, bb in enumerate(batch_pass)
          ])).long().to(self.device)

      batch_data += [data]

    end_time = time.time()
    # print('collate time = {}'.format(end_time - start_time))

    return batch_data