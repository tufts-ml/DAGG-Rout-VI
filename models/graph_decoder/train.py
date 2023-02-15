import numpy as np
import os
import torch
from torch import nn
from models.gcn.layer.mlp_readout_layer import MLPReadout, MLP
from models.graph_decoder.data import fact
import torch.nn.functional as F
import networkx as nx




class att_decoder_rnn_fast(nn.Module):
    def __init__(self, args, withz, edge_model, update_model, feature_map):
        super().__init__()
        self.args = args
        self.withz = withz
        self.edge_model = edge_model
        self.update_model= update_model
        self.feature_map = feature_map

        self.init_placeholder = nn.Parameter(torch.Tensor(self.args.n_embd))
        self.init_placeholder.data.uniform_(-1, 1)

        self.init_edge = nn.Parameter(torch.Tensor(self.args.sample_size, 1, self.args.n_embd))
        self.init_edge.data.uniform_(-1, 1)

        self.init_node = nn.Parameter(torch.Tensor(self.args.sample_size, 1, self.args.n_embd))
        self.init_edge.data.uniform_(-1, 1)




        self.compact_cotent = MLPReadout(self.args.n_embd*2,self.args.n_embd)
        self.edge_readout=MLPReadout(self.args.n_embd,2)
        self.node_label_readout=MLPReadout(self.args.n_embd,1)
        self.edge_re_readout = MLP(2, args.n_embd)
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
        t=0
        rpast=None
        upast=None
        content=self.compact_cotent(content).unsqueeze(1)
        new_node, upast = self.update_model(inputs_embeds=content, past=upast)



        for i in range(data['n']-1):



            if i ==0:
                new_node, upast = self.update_model(inputs_embeds=self.init_node, past=upast)

                e = s + i + 1

                edge_level_input = new_node


            else:
                rnninput = self.rnninput_readout(data['x'][:, i].to(self.args.device)).unsqueeze(1)
                new_node, upast = self.update_model(inputs_embeds=rnninput, past=upast)  # , attention_mask=attention_mask)

                e = s + i + 1

                true_con = data['con'][:, s:e-1]
                edge_level_input = self.edge_re_readout(true_con.to(self.args.device))  # (sample_size,i+1,embd)
                edge_level_input = torch.cat((new_node, edge_level_input), 1)






            #edge_level_output, upast = self.edge_model(inputs_embeds=edge_level_input, past=rpast)  #previous bug
            edge_level_output, _ = self.edge_model(inputs_embeds=edge_level_input, past=rpast)
            pre_edge = self.edge_readout(edge_level_output) #(sample_size,i+1,2)





            pre_edge = self.softmax(pre_edge)  # (sample_size, i+1, 2)


            edge_c_record[:, s:e] = pre_edge
            s = e
            rpast = None



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

class Rout(nn.Module):
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
        self.edge_readout=MLPReadout(self.args.n_embd*2,2)
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

        # debug_edge = edge_c_record.detach().cpu().numpy()
        # debug_con = data['con'].detach().cpu().numpy()
        # debug_loss = loss_node.detach().cpu().numpy()
        loss_node = torch.sum(loss_node, dim=1)

        ll=loss_node      #TODO: add node label/edge label
        return ll














