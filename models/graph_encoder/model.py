import torch
from torch import nn
from torch.nn import functional as F
from models.gcn.net.appnp_net_node import APPNET
from models.gcn.net.gat_net_node import GATNet
from models.gcn.net.gcn_net_node import GCNNet
from torch.autograd import Variable



class g_encoder(nn.Module):
    def __init__(self, args, feature_map):
        super(g_encoder,self).__init__()
        len_node_vec = len(feature_map['node_forward'])
        if args.gcn_type == 'gcn':
            self.gcn = GCNNet(args, len_node_vec, out_dim=args.hidden_size_node_level_rnn).to(args.device)
        elif args.gcn_type == 'gat':
            self.gcn = GATNet(args, len_node_vec, out_dim=args.hidden_size_node_level_rnn).to(args.device)
        elif args.gcn_type == 'appnp':
            self.gcn = APPNET(args, len_node_vec, out_dim=args.hidden_size_node_level_rnn).to(args.device)

        self.graph_embedding = nn.LSTM(input_size=args.hidden_size_node_level_rnn, hidden_size=args.hidden_size_node_level_rnn, num_layers=2)
        self.mean_linear = nn.Linear(in_features=args.hidden_size_node_level_rnn, out_features=args.hidden_size_node_level_rnn)
        self.logvar_linear = nn.Linear(in_features=args.hidden_size_node_level_rnn, out_features=args.hidden_size_node_level_rnn)

    def forward(self, g, h, pi=None):
        """

        :param g: dgl_graph
        :param h: tensor: (n,hidden_dimension)
        :param pi: [[pi_1], ...[pi_sample_size]]
        :return: z_mean: tensor：（sample_size, z_dim)
        z_logvar: tensor: (sample_size, z_dim)
        """
        bs, ns, feat = h.shape
        h = h.view(-1, h.size(2))
        h = self.gcn(g, h).view(bs, ns, -1).permute([1,0,2])
        # h0 = torch.zeros(2, h.size(1), h.size(2))
        # c0 = torch.zeros(2, h.size(1), h.size(2))
        if pi:
            # TODO: permute the node order-xiaohui
            pass

        output, (hn, cn) = self.graph_embedding(h)#, (h0, c0),)
        z_emb = F.relu(output[-1, :])
        z_mean = self.mean_linear(z_emb)
        z_logvar = self.logvar_linear(z_emb)
        return z_mean, z_logvar

    @staticmethod
    def sample_normal(z_mean, z_logvar):
        sd = torch.exp(z_logvar * 0.5)
        e = Variable(torch.randn(sd.size()))  # Sample from standard normal
        z = e.mul(sd).add_(z_mean)
        return z


class property_encoder(nn.Module):
    def __init__(self, args, feature_map):
        super(property_encoder, self).__init__()
        self.n_hidden = args.n_ctx
        self.main = nn.ModuleList([nn.Linear(2, self.n_hidden//2),
                                   nn.ReLU(),
                                   nn.Linear(self.n_hidden//2, self.n_hidden),
                                   nn.ReLU])
    def forward(self, z):
        return self.main(z)