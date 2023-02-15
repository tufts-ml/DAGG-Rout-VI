import numpy as np
import os
import torch
from torch import nn
from models.gcn.layer.mlp_readout_layer import MLPReadout, MLP
from models.graph_decoder.data import fact
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt







class att_decoder_dependency(nn.Module):
    def __init__(self, args, withz, decoding_model, update_model, feature_map):
        super().__init__()
        self.args = args
        self.withz = withz
        self.decoding_model = decoding_model
        self.update_model= update_model
        self.feature_map = feature_map

        self.init_placeholder = nn.Parameter(torch.Tensor(self.args.n_embd))
        self.init_placeholder.data.uniform_(-1, 1)
        self.compact_cotent = MLPReadout(self.args.n_embd*2,self.args.n_embd)
        self.edge_readout=MLPReadout(self.args.n_embd,2)
        self.input_readout = MLP(2, self.args.n_embd)
        self.node_label_readout=MLPReadout(self.args.n_embd,1)
        self.rnninput_readout = MLP(args.feature_len, args.n_embd)
        self.softmax=torch.nn.Softmax(dim=2)


        if self.withz==False:
            #no decoding z given, optimize this one instead.
            self.z_placeholder = nn.Parameter(torch.Tensor(self.args.n_embd))
            self.z_placeholder.data.uniform_(-1, 1)
    def forward(self, data,z=None):
        '''

        :param adj:tensor (sample_size,n,n)
        :param nlabel: (sample_size, n, nlabel)
        :param n: scalar
        :return: tensor(1,), log-likelihood
        '''
        data['n'] = int(data['n'].cpu()[0])
        if z==None:
            assert self.withz==False
            content=torch.cat((self.z_placeholder, self.init_placeholder), 0)
            #expand content for all sample
            content= content.repeat(self.args.sample_size, 1)



        if z!=None:
            assert self.withz==True
            content=torch.cat((z, self.init_placeholder), 1)
            #expand content for all sample
            content= content.repeat(self.args.sample_size, 1)

        z = self.z_placeholder.repeat(self.args.sample_size,1)
        #compact dimension of content to n_embd


        mem=[]
        edge_c_record = torch.zeros((self.args.sample_size, fact(data['n']-1), 2),dtype=torch.float).to(self.args.device)
        s=0
        rpast=None
        #upast=None
        content=self.compact_cotent(content).unsqueeze(1)
        for i in range(data['n']-1):



            new_node,rpast = self.decoding_model(inputs_embeds=content, past=rpast)
            mem.append(new_node)

            #node_rep = torch.cat(mem,dim=1)
            #can be improved?


            new_node, upast = self.update_model(inputs_embeds=new_node, past=None)
            pre_edge_record=[]
            edge_pre=self.edge_readout(new_node)
            pre_edge_record.append(edge_pre)
            for j in range(i):


                rnninput = data['x'][:,i+1, 3+j*4: 3+j*4+2].to(self.args.device).unsqueeze(1)
                rnninput = self.input_readout(rnninput)
                new_node, upast = self.update_model(inputs_embeds=rnninput, past=upast)
                edge_pre = self.edge_readout(new_node)
                pre_edge_record.append(edge_pre)

            pre_edge_record = torch.cat(pre_edge_record, dim=1)
            pre_edge_record = self.softmax(pre_edge_record)  # (sample_size, i+1, 2)
            pre_edge_record = pre_edge_record.flip(dims=[1])


            e = s + i + 1



            edge_c_record[:,s:e] = pre_edge_record
            s = e


            content = new_node#.squeeze(1)
            #content = torch.cat((z,new_node),1)



        #node_rep=torch.cat(mem, dim=1) #(sample_size, n, n_embd)


        #loss_node = F.binary_cross_entropy_with_logits(edge_c_record, data['con'].to(self.args.device), reduction='none')
        loss_node = F.binary_cross_entropy(edge_c_record, data['con'].to(self.args.device),
                                                       reduction='none')

        # debug_edge = edge_c_record.detach().cpu().numpy()
        # debug_con = data['con'].detach().cpu().numpy()
        # debug_loss = loss_node.detach().cpu().numpy()
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


        plt.show()


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
                pre_edge_record = []


                new_node, rpast = self.decoding_model(inputs_embeds=content, past=rpast)
                mem.append(new_node)

                node_rep = torch.cat(mem, dim=1)
                # can be improved?
                new_node, upast = self.update_model(inputs_embeds=new_node, past=None)

                edge_pre = self.edge_readout(new_node).squeeze(1)

                rnninput, sample_edge = self.sample_transfer(edge_pre, batch_size)
                pre_edge_record.append(sample_edge)
                for j in range(i):
                    rnninput, sample_edge = self.sample_transfer(edge_pre, batch_size)
                    rnninput = self.input_readout(rnninput)
                    new_node, upast = self.update_model(inputs_embeds=rnninput, past=upast)
                    edge_pre = self.edge_readout(new_node).squeeze(1)
                    pre_edge_record.append(sample_edge)





                content = new_node
                pre_edge_record = torch.cat(pre_edge_record, dim=1)
                pre_edge_record = pre_edge_record.flip(dims=[1])
                edge.append(pre_edge_record)

            sample_edge = torch.cat(edge, dim=1)
            #batch_size*N*N

            adj = self.edge_to_adj(sample_edge, batch_size, N)
            #adj = torch.tril(adj, diagonal=-1)
            adj = adj + adj.transpose(1, 2)

            return adj

    def sample_transfer(self, pre_edge, batch_size):


        n = pre_edge.size()[0]
        node_number = pre_edge.size()[1]
        rnn_input = torch.zeros(
            batch_size, 1, 2, device=self.args.device)

        # this is for no label case


        sample_edge_level_output = []
        pre_edge = torch.nn.Softmax(dim=1)(pre_edge)



        sample_edge_level_predict = torch.multinomial(
                pre_edge, 1).reshape(-1)




        sample_edge_level_output = sample_edge_level_predict.reshape(n, -1)


        for b in range(batch_size):

            con = sample_edge_level_output[b, 0]
            rnn_input[b, 0, con] = 1


        return rnn_input, sample_edge_level_output

    def edge_to_adj(self, sample_edge, batch_size, N):
        sample_edge = sample_edge.cpu().numpy()

        adj = np.zeros((batch_size, N, N))
        # sample_edge = 1- sample_edge

        for b in range(batch_size):
            s = 0
            for i in range(N - 1):
                e = s + i + 1
                adj[b, i + 1, :i + 1] = sample_edge[b, s:e]
                s = e

        return torch.tensor(adj, device=self.args.device)



