import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate as collate
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from models.layers import RNN, MLP_Plain, MLP_Softmax
from models.DAGG.helper import get_attributes_len_for_graph_rnn
import numpy as np
import networkx as nx
EPS = 1e-9



# DAGG


# class DAGG(nn.Module):
#     def __init__(self, args, decoding_model, update_model, feature_map, processor):
#         super().__init__()
#         self.args = args
#         self.withz = withz
#         self.decoding_model = decoding_model
#         self.update_model= update_model
#         self.feature_map = feature_map
#
#         self.init_placeholder = nn.Parameter(torch.Tensor(self.args.n_embd))
#         self.init_placeholder.data.uniform_(-1, 1)
#         self.compact_cotent = MLPReadout(self.args.n_embd*2,self.args.n_embd)
#         self.edge_readout=MLPReadout(self.args.n_embd*2,2)
#         self.node_label_readout=MLPReadout(self.args.n_embd,1)
#         self.rnninput_readout = MLP(args.feature_len, args.n_embd)
#         self.softmax=torch.nn.Softmax(dim=2)
#         self.processor = processor
#
#
#     # probability calculation
#     def forward(self, nx_g, pis):
#         '''
#
#         :param adj:tensor (sample_size,n,n)
#         :param nlabel: (sample_size, n, nlabel)
#         :param n: scalar
#         :return: tensor(1,), log-likelihood
#         '''
#         data = [self.processor(nx_g, perms) for perms in pis]
#         data = collate(data)
#         data['n'] = int(data['n'].cpu()[0])
#
#
#         content=torch.cat((self.z_placeholder, self.init_placeholder), 0)
#         #expand content for all sample
#         content= content.repeat(self.args.sample_size, 1)
#
#
#
#
#         z = self.z_placeholder.repeat(self.args.sample_size,1)
#         #compact dimension of content to n_embd
#
#
#         mem=[]
#         edge_c_record = torch.zeros((self.args.sample_size, fact(data['n']-1), 2),dtype=torch.float).to(self.args.device)
#         s=0
#         rpast=None
#         upast=None
#         content=self.compact_cotent(content).unsqueeze(1)
#         for i in range(data['n']-1):
#
#             attention_mask=data[i+1].to(self.args.device)
#
#             new_node,rpast = self.decoding_model(inputs_embeds=content, past=rpast)
#             mem.append(new_node)
#
#             node_rep = torch.cat(mem,dim=1)
#             #can be improved?
#             edge_rep = torch.cat((node_rep,new_node.repeat(1,i+1,1)),dim=2) #(sample_size, i+1, 2*n_embd)
#             pre_edge = self.edge_readout(edge_rep)  #(sample_size, i+1, 1)
#             pre_edge = self.softmax(pre_edge)   #(sample_size, i+1, 2)
#             e = s + i + 1
#
#
#
#             edge_c_record[:,s:e] = pre_edge
#             s = e
#
#
#
#
#             rnninput = self.rnninput_readout(data['x'][:,i+1].to(self.args.device)).unsqueeze(1)
#             new_node,upast = self.update_model(inputs_embeds=rnninput, past=upast, attention_mask=attention_mask)
#
#             content = new_node#.squeeze(1)
#             #content = torch.cat((z,new_node),1)
#
#
#
#         node_rep=torch.cat(mem, dim=1) #(sample_size, n, n_embd)
#
#
#         #loss_node = F.binary_cross_entropy_with_logits(edge_c_record, data['con'].to(self.args.device), reduction='none')
#         loss_node = F.binary_cross_entropy(edge_c_record, data['con'].to(self.args.device),
#                                                        reduction='none')
#         loss_node = torch.sum(loss_node, dim=1)
#
#         ll=loss_node      #TODO: add node label/edge label
#         return ll
#
#     def sample(self,num_nodes_pmf, N, batch_size):
#         A = self._sampling(N, batch_size)
#         num_nodes_pmf = torch.from_numpy(num_nodes_pmf).to(self.args.device)
#         num_nodes = torch.multinomial(
#             num_nodes_pmf, batch_size, replacement=True) + 1  # shape B X 1
#
#         A_list = [
#             A[ii, :num_nodes[ii], :num_nodes[ii]] for ii in range(batch_size)
#         ]
#         # A_list = [
#         #     A[ii, :, :] for ii in range(batch_size)
#         # ]
#         graphs = [self.max_graph(nx.from_numpy_matrix(A.cpu().numpy())) for A in A_list]
#
#         return graphs
#
#
#     def max_graph(self, G):
#         if len(G.nodes()):
#             max_comp = max(nx.connected_components(G), key=len)
#             G = nx.Graph(G.subgraph(max_comp))
#         return G
#
#     def _sampling(self, N , batch_size, z=None):
#         '''
#
#         :param N: max_num_nodes
#         :return:adj(N,N)
#         '''
#
#         with torch.no_grad():
#             #A = torch.zeros(batch_size, N, N).to(self.args.device)
#             #attention_mask = torch.ones((batch_size, 1)).to(self.args.device)
#             # edge_c_record = torch.zeros((self.args.sample_size, fact(N - 1), 2), dtype=torch.float).to(
#             #     self.args.device)
#             if z == None:
#                 assert self.withz == False
#                 content = torch.cat((self.z_placeholder, self.init_placeholder), 0)
#                 # expand content for all sample
#                 content = content.repeat(batch_size, 1)
#
#             if z != None:
#                 assert self.withz == True
#                 content = torch.cat((z, self.init_placeholder), 1)
#                 # expand content for all sample
#                 content = content.repeat(batch_size, 1)
#
#
#
#
#             mem=[]
#             edge=[]
#
#             s = 0
#             rpast = None
#             upast = None
#             content = self.compact_cotent(content).unsqueeze(1)
#
#             for i in range(N - 1):
#                 #attention_mask = data[i + 1].to(self.args.device)
#
#                 new_node, rpast = self.decoding_model(inputs_embeds=content, past=rpast)
#                 mem.append(new_node)
#
#                 node_rep = torch.cat(mem, dim=1)
#                 # can be improved?
#                 edge_rep = torch.cat((node_rep, new_node.repeat(1, i + 1, 1)), dim=2)  # (sample_size, i+1, 2*n_embd)
#                 pre_edge = self.edge_readout(edge_rep)  # (sample_size, i+1, 1)
#                 pre_edge = self.softmax(pre_edge)  # (sample_size, i+1, 2)
#                 rnnpoint, attention_mask, sample_edge = self.sample_transfer(pre_edge, batch_size)
#
#                 e = s + i + 1
#
#
#                 s = e
#
#                 rnninput = self.rnninput_readout(rnnpoint.to(self.args.device))#.unsqueeze(1)
#                 new_node, upast = self.update_model(inputs_embeds=rnninput, past=upast, attention_mask=attention_mask)
#
#                 content = new_node  # .squeeze(1)
#                 edge.append(sample_edge)
#             #batch_size*N*N
#             sample_edge = 1-np.concatenate(edge, axis=1)
#             adj = self.edge_to_adj(sample_edge, batch_size, N)
#             #adj = torch.tril(adj, diagonal=-1)
#             adj = adj + adj.transpose(1, 2)
#
#             return adj
#
#
#     def sample_transfer(self, pre_edge, batch_size):
#         '''
#
#         :param pre_edge: (sample_size, i+1, 2)
#         :param batch_size:
#         :return: rnninput (batch_size, feature_len)
#         '''
#
#         n = pre_edge.size()[0]
#         node_number = pre_edge.size()[1]
#         node_level_input = torch.zeros(
#             batch_size, 1, self.args.feature_len, device=self.args.device)
#         attention_mask=torch.zeros(batch_size,node_number, device=self.args.device)
#         #this is for no label case
#         node_level_input[:, 0, 0] = 1 #set as default node label
#
#         sample_edge_level_output=[]
#         #batchsize*n
#         for i in range(n):
#             sample_edge_level_predict = torch.multinomial(
#                 pre_edge[i], 1).reshape(-1)
#             sample_edge_level_output.append(sample_edge_level_predict)
#         sample_edge_level_output=torch.cat(sample_edge_level_output,dim=0)
#
#         sample_edge_level_output = sample_edge_level_output.reshape(n,-1)
#
#         sample_edge_level_output = 1- sample_edge_level_output
#         re_sample_edge_level_output = sample_edge_level_output.flip(dims=[1])
#         for b in range(batch_size):
#             for i in range(node_number):
#                 con = re_sample_edge_level_output[b,i]
#                 node_level_input[b,0,i*4+3+con] =1
#                 if sample_edge_level_output[b,i] == 0:
#                     attention_mask[b,i] = 1
#
#
#         return node_level_input, attention_mask, np.array(sample_edge_level_output.cpu())
#
#     def edge_to_adj(self, sample_edge, batch_size, N):
#         '''
#
#         :param sample_edge:(batch_size, fact(N-1)
#         :return: adj (batch_size, N,N)
#         '''
#         adj=np.zeros((batch_size, N,N))
#         #sample_edge = 1- sample_edge
#
#         for b in range(batch_size):
#             s = 0
#             for i in range(N-1):
#                 e=s+i+1
#                 adj[b, i+1, :i+1] = sample_edge[b, s:e]
#                 s=e
#
#         return torch.tensor(adj, device=self.args.device)

class DAGG(nn.Module):
    def __init__(self, args, node_level_transformer, edge_level_transformer, feature_map, processor):
        super().__init__()
        self.args = args
        self.node_level_transformer = node_level_transformer
        self.edge_level_transformer= edge_level_transformer
        self.feature_map = feature_map
        self.embedding_node_to_edge = MLP_Plain(
            input_size=args.hidden_size_node_level_transformer, embedding_size=args.embedding_size_node_level_transformer,
            output_size=args.hidden_size_edge_level_transformer).to(device=args.device)
        self.output_node = MLP_Softmax(
            input_size=args.hidden_size_node_level_transformer, embedding_size=args.embedding_size_node_output,
            output_size=feature_map.len_node_vec).to(device=args.device)

        len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(len(
            feature_map['node_forward']), len(feature_map['edge_forward']), args.max_prev_node, args.max_head_and_tail)
        feature_len = len_node_vec + num_nodes_to_consider * len_edge_vec
        self.node_project =  MLP_Plain(feature_len, args.embedding_size_node_level_transformer, args.embedding_size_node_level_transformer)
        self.edge_project =  MLP_Plain(len_edge_vec, args.embedding_size_edge_level_transformer, args.embedding_size_edge_level_transformer)

        self.processor = processor


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

        x_unsorted = data['x'].to(self.args.device)

        x_len_unsorted = data['len'].to(self.args.device)
        x_len_max = max(x_len_unsorted)
        x_unsorted = x_unsorted[:, 0:max(x_len_unsorted), :]

        len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
            len(self.feature_map['node_forward']), len(self.feature_map['edge_forward']),
            self.args.max_prev_node, self.args.max_head_and_tail)

        batch_size = x_unsorted.size(0)
        # sort input for packing variable length sequences
        x_len, sort_indices = torch.sort(x_len_unsorted, dim=0, descending=True)
        x = torch.index_select(x_unsorted, 0, sort_indices)




        # Teacher forcing: Feed the target as the next input
        # Start token for graph level Transformer decoder is node feature second last bit is 1
        node_level_input = torch.cat(
            (torch.zeros(batch_size, 1, x.size(2), device=self.args.device), x), dim=1)
        node_level_input[:, 0, len_node_vec - 2] = 1
        steps = x.size(1)

        x_len = x_len.cpu()
        #mask
        mask = torch.tril(torch.ones((batch_size, steps+1, steps+1)))

        # Forward propogation
        node_level_input = self.node_project(node_level_input)
        node_level_output = self.node_level_transformer(
            node_level_input, attention_mask=mask)

        # Evaluating node predictions
        x_pred_node = self.output_node(node_level_output)


        edge_mat_packed = pack_padded_sequence(
            x[:, :, len_node_vec: min(
                x_len_max - 1, num_nodes_to_consider) * len_edge_vec + len_node_vec],
            x_len, batch_first=True)

        edge_mat, edge_batch_size = edge_mat_packed.data, edge_mat_packed.batch_sizes

        idx = torch.LongTensor(
            [i for i in range(edge_mat.size(0) - 1, -1, -1)]).to(self.args.device)
        edge_mat = edge_mat.index_select(0, idx)

        edge_mat = edge_mat.reshape(edge_mat.size(0), min(
            x_len_max - 1, num_nodes_to_consider), len_edge_vec)
        edge_level_input = torch.cat(
            (torch.zeros(sum(x_len), 1, len_edge_vec, device=self.args.device), edge_mat), dim=1)
        edge_level_input[:, 0, len_edge_vec - 2] = 1

        # Compute descending list of lengths for y_edge
        x_edge_len = []
        # Histogram of y_len
        x_edge_len_bin = torch.bincount(x_len)
        for i in range(len(x_edge_len_bin) - 1, 0, -1):
            # count how many x_len is above and equal to i
            count_temp = torch.sum(x_edge_len_bin[i:]).item()

            # put count_temp of them in x_edge_len each with value min(i, num_nodes_to_consider + 1)
            x_edge_len.extend([min(i, num_nodes_to_consider + 1)] * count_temp)

        x_edge_len = torch.LongTensor(x_edge_len).to(self.args.device)

        # Get edge-level RNN hidden state from node-level RNN output at each timestamp
        # Ignore the last hidden state corresponding to END
        hidden_edge = self.embedding_node_to_edge(node_level_output[:, 0:-1, :])

        # Prepare hidden state for edge level RNN similiar to edge_mat
        # Ignoring the last graph level decoder END token output (all 0's)
        hidden_edge = pack_padded_sequence(
            hidden_edge, x_len, batch_first=True).data
        idx = torch.LongTensor(
            [i for i in range(hidden_edge.size(0) - 1, -1, -1)]).to(self.args.device)
        hidden_edge = hidden_edge.index_select(0, idx)

        # Set hidden state for edge-level RNN
        # shape of hidden tensor (num_layers, batch_size, hidden_size)
        hidden_edge = hidden_edge.view(hidden_edge.size(0), 1, hidden_edge.size(1))
        edge_level_input = self.edge_project(edge_level_input)
        edge_level_input = torch.cat([hidden_edge,edge_level_input], dim=1)



        x_pred_edge = self.edge_level_rnn(edge_level_input, attention_mask=mask)[:,1:]
        # cleaning the padding i.e setting it to zero
        x_pred_node = pack_padded_sequence(
            x_pred_node, x_len + 1, batch_first=True)
        x_pred_node, lens_pred_node = pad_packed_sequence(x_pred_node, batch_first=True)
        x_pred_edge = pack_padded_sequence(
            x_pred_edge, x_edge_len, batch_first=True)
        x_pred_edge, lens_pred_edge = pad_packed_sequence(x_pred_edge, batch_first=True)

        x_node = torch.cat(
            (x[:, :, :len_node_vec], torch.zeros(batch_size, 1, len_node_vec, device=args.device)), dim=1)
        x_node[torch.arange(batch_size), x_len, len_node_vec - 1] = 1

        x_edge = torch.cat((edge_mat, torch.zeros(
            sum(x_len), 1, len_edge_vec, device=self.args.device)), dim=1)
        x_edge[torch.arange(sum(x_len)), x_edge_len - 1, len_edge_vec - 1] = 1

        loss_node = F.binary_cross_entropy(x_pred_node, x_node, reduction='none')
        loss_edge = F.binary_cross_entropy(x_pred_edge, x_edge, reduction='none')



        loss_node = torch.sum(loss_node, dim=[1, 2])
        edge_batch_size_cum = torch.cat(
            [torch.tensor([0]).to(self.args.device), torch.cumsum(edge_batch_size, dim=0).to(self.args.device)])
        edge_indices = []
        for shift, length in enumerate(x_len):
            edge_indices.append((edge_batch_size_cum + shift)[:length])

        loss_edge = torch.cat(
            [torch.sum(loss_edge.index_select(0, indices), dim=[0, 1, 2]).view(1) for indices in edge_indices])

        loss = loss_node + loss_edge
        swapped_loss = torch.empty_like(loss)
        swapped_loss[sort_indices] = loss



        return swapped_loss

    def sample(self, eval_args):
        train_args = eval_args.train_args
        feature_map = get_model_attribute(
            'feature_map', eval_args.model_path, eval_args.device)
        train_args.device = eval_args.device

        model = create_model(train_args, feature_map)
        load_gmodel(eval_args.model_path, eval_args.device, model)

        for _, net in model.items():
            net.eval()

        max_num_node = eval_args.max_num_node
        len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
            len(feature_map['node_forward']), len(feature_map['edge_forward']),
            train_args.max_prev_node, train_args.max_head_and_tail)
        feature_len = len_node_vec + num_nodes_to_consider * len_edge_vec

        graphs = []

        for _ in range(eval_args.count // eval_args.batch_size):
            # model['node_level_rnn'].hidden = model['node_level_rnn'].init_hidden(
            #     batch_size=eval_args.batch_size)

            # [batch_size] * [num of nodes]
            x_pred_node = np.zeros(
                (eval_args.batch_size, max_num_node), dtype=np.int32)
            # [batch_size] * [num of nodes] * [num_nodes_to_consider]
            x_pred_edge = np.zeros(
                (eval_args.batch_size, max_num_node, num_nodes_to_consider), dtype=np.int32)

            node_level_input = torch.zeros(
                eval_args.batch_size, 1, feature_len, device=eval_args.device)
            # Initialize to node level start token
            node_level_input[:, 0, len_node_vec - 2] = 1
            past=None
            for i in range(max_num_node):
                # [batch_size] * [1] * [hidden_size_node_level_rnn]
                node_level_input = self.node_project(node_level_input)
                node_level_output,past = model.node_level_transformer(node_level_input, past)
                # [batch_size] * [1] * [node_feature_len]
                node_level_pred = model.output_node(node_level_output)
                # [batch_size] * [node_feature_len] for torch.multinomial
                node_level_pred = node_level_pred.reshape(
                    eval_args.batch_size, len_node_vec)
                # [batch_size]: Sampling index to set 1 in next node_level_input and x_pred_node
                # Add a small probability for each node label to avoid zeros
                node_level_pred[:, :-2] += EPS
                # Start token should not be sampled. So set it's probability to 0
                node_level_pred[:, -2] = 0
                # End token should not be sampled if i less than min_num_node
                if i < eval_args.min_num_node:
                    node_level_pred[:, -1] = 0
                sample_node_level_output = torch.multinomial(
                    node_level_pred, 1).reshape(-1)
                node_level_input = torch.zeros(
                    eval_args.batch_size, 1, feature_len, device=eval_args.device)
                node_level_input[torch.arange(
                    eval_args.batch_size), 0, sample_node_level_output] = 1

                # [batch_size] * [num of nodes]
                x_pred_node[:, i] = sample_node_level_output.cpu().data

                # [batch_size] * [1] * [hidden_size_edge_level_rnn]
                hidden_edge = model.embedding_node_to_edge(node_level_output)

                hidden_edge_rem_layers = torch.zeros(
                    train_args.num_layers -
                    1, eval_args.batch_size, hidden_edge.size(2),
                    device=eval_args.device)
                # [num_layers] * [batch_size] * [hidden_len]
                # model['edge_level_rnn'].hidden = torch.cat(
                #     (hidden_edge.permute(1, 0, 2), hidden_edge_rem_layers), dim=0)

                # [batch_size] * [1] * [edge_feature_len]
                edge_level_input = torch.zeros(
                    eval_args.batch_size, 1, len_edge_vec, device=eval_args.device)
                # Initialize to edge level start token
                edge_level_input[:, 0, len_edge_vec - 2] = 1
                for j in range(min(num_nodes_to_consider, i)):
                    # [batch_size] * [1] * [edge_feature_len]
                    edge_level_input = self.edge_project(edge_level_input)
                    edge_level_output = model.edge_level_rnn(edge_level_input)
                    # [batch_size] * [edge_feature_len] needed for torch.multinomial
                    edge_level_output = edge_level_output.reshape(
                        eval_args.batch_size, len_edge_vec)

                    # [batch_size]: Sampling index to set 1 in next edge_level input and x_pred_edge
                    # Add a small probability for no edge to avoid zeros
                    edge_level_output[:, -3] += EPS
                    # Start token and end should not be sampled. So set it's probability to 0
                    edge_level_output[:, -2:] = 0
                    sample_edge_level_output = torch.multinomial(
                        edge_level_output, 1).reshape(-1)
                    edge_level_input = torch.zeros(
                        eval_args.batch_size, 1, len_edge_vec, device=eval_args.device)
                    edge_level_input[:, 0, sample_edge_level_output] = 1

                    # Setting edge feature for next node_level_input
                    node_level_input[:, 0, len_node_vec + j * len_edge_vec: len_node_vec + (j + 1) * len_edge_vec] = \
                        edge_level_input[:, 0, :]

                    # [batch_size] * [num of nodes] * [num_nodes_to_consider]
                    x_pred_edge[:, i, j] = sample_edge_level_output.cpu().data

            # Save the batch of graphs
            for k in range(eval_args.batch_size):
                G = nx.Graph()

                for v in range(max_num_node):
                    # End node token
                    if x_pred_node[k, v] == len_node_vec - 1:
                        break
                    elif x_pred_node[k, v] < len(feature_map['node_forward']):
                        G.add_node(
                            v, label=feature_map['node_backward'][x_pred_node[k, v]])
                    else:
                        print('Error in sampling node features')
                        exit()

                for u in range(len(G.nodes())):
                    for p in range(min(num_nodes_to_consider, u)):
                        if x_pred_edge[k, u, p] < len(feature_map['edge_forward']):
                            if train_args.max_prev_node is not None:
                                v = u - p - 1
                            elif train_args.max_head_and_tail is not None:
                                if p < train_args.max_head_and_tail[1]:
                                    v = u - p - 1
                                else:
                                    v = p - train_args.max_head_and_tail[1]

                            G.add_edge(
                                u, v, label=feature_map['edge_backward'][x_pred_edge[k, u, p]])
                        elif x_pred_edge[k, u, p] == len(feature_map['edge_forward']):
                            # No edge
                            pass
                        else:
                            print('Error in sampling edge features')
                            exit()

                # Take maximum connected component
                if len(G.nodes()):
                    max_comp = max(nx.connected_components(G), key=len)
                    G = nx.Graph(G.subgraph(max_comp))

                graphs.append(G)

        return graphs















