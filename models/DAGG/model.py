import numpy as np
import os
import torch
from torch import nn
from models.gcn.layer.mlp_readout_layer import MLPReadout, MLP
from models.DAGG.data import fact
import torch.nn.functional as F
import networkx as nx
from torch.utils.data._utils.collate import default_collate as collate
from models.DAGG.data import Graph_to_att_Matrix


# DAGG


class DAGG(nn.Module):
    def __init__(self, args, withz, decoding_model, update_model, feature_map, processor):
        super().__init__()
        self.args = args
        self.withz = withz
        self.decoding_model = decoding_model
        self.update_model= update_model
        self.feature_map = feature_map

        self.init_placeholder = nn.Parameter(torch.Tensor(self.args.n_embd))
        self.init_placeholder.data.uniform_(-1, 1)
        self.compact_cotent = MLPReadout(self.args.n_embd*2,self.args.n_embd)
        self.edge_readout=MLPReadout(self.args.n_embd*2,2)
        self.node_label_readout=MLPReadout(self.args.n_embd,1)
        self.rnninput_readout = MLP(args.feature_len, args.n_embd)
        self.softmax=torch.nn.Softmax(dim=2)
        self.processor = processor


        if self.withz==False:
            #no decoding z given, optimize this one instead.
            self.z_placeholder = nn.Parameter(torch.Tensor(self.args.n_embd))
            self.z_placeholder.data.uniform_(-1, 1)

    # probability calculation
    def forward(self, nx_g, pis):
        '''

        :param adj:tensor (sample_size,n,n)
        :param nlabel: (sample_size, n, nlabel)
        :param n: scalar
        :return: tensor(1,), log-likelihood
        '''
        data = [self.processor(nx_g, perms) for perms in pis]
        data = collate(data)
        data['n'] = int(data['n'].cpu()[0])


        content=torch.cat((self.z_placeholder, self.init_placeholder), 0)
        #expand content for all sample
        content= content.repeat(self.args.sample_size, 1)




        z = self.z_placeholder.repeat(self.args.sample_size,1)
        #compact dimension of content to n_embd


        mem=[]
        edge_c_record = torch.zeros((self.args.sample_size, fact(data['n']-1), 2),dtype=torch.float).to(self.args.device)
        s=0
        rpast=None
        upast=None
        content=self.compact_cotent(content).unsqueeze(1)
        for i in range(data['n']-1):

            attention_mask=data[i+1].to(self.args.device)

            new_node,rpast = self.decoding_model(inputs_embeds=content, past=rpast)
            mem.append(new_node)

            node_rep = torch.cat(mem,dim=1)
            #can be improved?
            edge_rep = torch.cat((node_rep,new_node.repeat(1,i+1,1)),dim=2) #(sample_size, i+1, 2*n_embd)
            pre_edge = self.edge_readout(edge_rep)  #(sample_size, i+1, 1)
            pre_edge = self.softmax(pre_edge)   #(sample_size, i+1, 2)
            e = s + i + 1



            edge_c_record[:,s:e] = pre_edge
            s = e




            rnninput = self.rnninput_readout(data['x'][:,i+1].to(self.args.device)).unsqueeze(1)
            new_node,upast = self.update_model(inputs_embeds=rnninput, past=upast, attention_mask=attention_mask)

            content = new_node#.squeeze(1)
            #content = torch.cat((z,new_node),1)



        node_rep=torch.cat(mem, dim=1) #(sample_size, n, n_embd)


        #loss_node = F.binary_cross_entropy_with_logits(edge_c_record, data['con'].to(self.args.device), reduction='none')
        loss_node = F.binary_cross_entropy(edge_c_record, data['con'].to(self.args.device),
                                                       reduction='none')
        loss_node = torch.sum(loss_node, dim=1)

        ll=loss_node      #TODO: add node label/edge label
        return ll

    def sample(self,num_nodes_pmf, N, batch_size):
        A = self._sampling(N, batch_size)
        num_nodes_pmf = torch.from_numpy(num_nodes_pmf).to(self.args.device)
        num_nodes = torch.multinomial(
            num_nodes_pmf, batch_size, replacement=True) + 1  # shape B X 1

        A_list = [
            A[ii, :num_nodes[ii], :num_nodes[ii]] for ii in range(batch_size)
        ]
        # A_list = [
        #     A[ii, :, :] for ii in range(batch_size)
        # ]
        graphs = [self.max_graph(nx.from_numpy_matrix(A.cpu().numpy())) for A in A_list]

        return graphs


    def max_graph(self, G):
        if len(G.nodes()):
            max_comp = max(nx.connected_components(G), key=len)
            G = nx.Graph(G.subgraph(max_comp))
        return G

    def _sampling(self, N , batch_size, z=None):
        '''

        :param N: max_num_nodes
        :return:adj(N,N)
        '''

        with torch.no_grad():
            #A = torch.zeros(batch_size, N, N).to(self.args.device)
            #attention_mask = torch.ones((batch_size, 1)).to(self.args.device)
            # edge_c_record = torch.zeros((self.args.sample_size, fact(N - 1), 2), dtype=torch.float).to(
            #     self.args.device)
            if z == None:
                assert self.withz == False
                content = torch.cat((self.z_placeholder, self.init_placeholder), 0)
                # expand content for all sample
                content = content.repeat(batch_size, 1)

            if z != None:
                assert self.withz == True
                content = torch.cat((z, self.init_placeholder), 1)
                # expand content for all sample
                content = content.repeat(batch_size, 1)




            mem=[]
            edge=[]

            s = 0
            rpast = None
            upast = None
            content = self.compact_cotent(content).unsqueeze(1)

            for i in range(N - 1):
                #attention_mask = data[i + 1].to(self.args.device)

                new_node, rpast = self.decoding_model(inputs_embeds=content, past=rpast)
                mem.append(new_node)

                node_rep = torch.cat(mem, dim=1)
                # can be improved?
                edge_rep = torch.cat((node_rep, new_node.repeat(1, i + 1, 1)), dim=2)  # (sample_size, i+1, 2*n_embd)
                pre_edge = self.edge_readout(edge_rep)  # (sample_size, i+1, 1)
                pre_edge = self.softmax(pre_edge)  # (sample_size, i+1, 2)
                rnnpoint, attention_mask, sample_edge = self.sample_transfer(pre_edge, batch_size)

                e = s + i + 1


                s = e

                rnninput = self.rnninput_readout(rnnpoint.to(self.args.device))#.unsqueeze(1)
                new_node, upast = self.update_model(inputs_embeds=rnninput, past=upast, attention_mask=attention_mask)

                content = new_node  # .squeeze(1)
                edge.append(sample_edge)
            #batch_size*N*N
            sample_edge = 1-np.concatenate(edge, axis=1)
            adj = self.edge_to_adj(sample_edge, batch_size, N)
            #adj = torch.tril(adj, diagonal=-1)
            adj = adj + adj.transpose(1, 2)

            return adj


    def sample_transfer(self, pre_edge, batch_size):
        '''

        :param pre_edge: (sample_size, i+1, 2)
        :param batch_size:
        :return: rnninput (batch_size, feature_len)
        '''

        n = pre_edge.size()[0]
        node_number = pre_edge.size()[1]
        node_level_input = torch.zeros(
            batch_size, 1, self.args.feature_len, device=self.args.device)
        attention_mask=torch.zeros(batch_size,node_number, device=self.args.device)
        #this is for no label case
        node_level_input[:, 0, 0] = 1 #set as default node label

        sample_edge_level_output=[]
        #batchsize*n
        for i in range(n):
            sample_edge_level_predict = torch.multinomial(
                pre_edge[i], 1).reshape(-1)
            sample_edge_level_output.append(sample_edge_level_predict)
        sample_edge_level_output=torch.cat(sample_edge_level_output,dim=0)

        sample_edge_level_output = sample_edge_level_output.reshape(n,-1)

        sample_edge_level_output = 1- sample_edge_level_output
        re_sample_edge_level_output = sample_edge_level_output.flip(dims=[1])
        for b in range(batch_size):
            for i in range(node_number):
                con = re_sample_edge_level_output[b,i]
                node_level_input[b,0,i*4+3+con] =1
                if sample_edge_level_output[b,i] == 0:
                    attention_mask[b,i] = 1


        return node_level_input, attention_mask, np.array(sample_edge_level_output.cpu())

    def edge_to_adj(self, sample_edge, batch_size, N):
        '''

        :param sample_edge:(batch_size, fact(N-1)
        :return: adj (batch_size, N,N)
        '''
        adj=np.zeros((batch_size, N,N))
        #sample_edge = 1- sample_edge

        for b in range(batch_size):
            s = 0
            for i in range(N-1):
                e=s+i+1
                adj[b, i+1, :i+1] = sample_edge[b, s:e]
                s=e

        return torch.tensor(adj, device=self.args.device)















