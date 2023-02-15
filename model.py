# from models.dgmg.model import create_model as create_model_dgmg
from models.graph_rnn.model import create_model as create_model_graph_rnn
from models.gran.model import create_model as create_model_gran
from models.seq_attention.attention_model import AttentionModel, generation
from models.gcn.net.appnp_net_node import APPNET
from models.gcn.net.gat_net_node import GATNet
from models.gcn.net.gcn_net_node import GCNNet
from models.graph_encoder.model import g_encoder
from models.graph_decoder.att_decoder import AttentionDecoder
from models.graph_decoder.train import Rout
from models.graph_decoder.train_dependecy import att_decoder_dependency

def create_models(args, feature_map):

    print('Producing model...')

    # generative model 
    if args.note == 'GraphRNN':
        gmodel = create_model_graph_rnn(args, feature_map)

    elif args.note == 'DAGG':
        # decoding_model= AttentionDecoder(args)
        # update_model = AttentionDecoder(args)
        # gmodel=att_decoder_dependency(args, args.withz, decoding_model, update_model, feature_map)
        # gmodel={'gmodel':gmodel.to(args.device)}

        edge_model = AttentionDecoder(args)
        update_model = AttentionDecoder(args)
        gmodel =  Rout(args, args.withz, edge_model, update_model, feature_map)
        gmodel = {'gmodel': gmodel.to(args.device)}



    len_node_vec = len(feature_map['node_forward'])

    # q distribution q(pi | G)
    if args.gcn_type == 'gcn':
        gcn = GCNNet(args, len_node_vec, out_dim=32).to(args.device)
    elif args.gcn_type == 'gat':
        gcn = GATNet(args, len_node_vec, out_dim=32).to(args.device)
    elif args.gcn_type == 'appnp':
        gcn = APPNET(args, len_node_vec, out_dim=32).to(args.device)

    pmodel = AttentionModel(embedding_dim = args.gcn_out_dim,
                 hidden_dim=32,
                 state=generation,
                 args=args,
                 featuremap=feature_map,
                 #model,
                 gcn=gcn,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None).to(args.device)
    ppmodel={'pmodel':pmodel}

    return gmodel, ppmodel




