from models.seq_attention.attention_model import AttentionModel, generation
from models.gcn.net.appnp_net_node import APPNET
from models.gcn.net.gat_net_node import GATNet
from models.gcn.net.gcn_net_node import GCNNet
from models.DAGG.att_decoder import AttentionDecoder
from models.DAGG.model import DAGG
from models.DAGG.data import Graph_to_Adj_Matrix
from models.DAGG.attention import AttentionDecoder




def create_generative_model(args, feature_map):
    """
    Initialize a generative model.  
    """
    

    print('Producing model...')



    if args.note == 'DAGG':

        processor = Graph_to_Adj_Matrix(args, feature_map)
        args.feature_len = processor.feature_len

        dagg =  DAGG(args, feature_map, processor)


    return dagg



def create_inference_model(args, feature_map):
    """
    Initialize an inference model.  
    """

    len_node_vec = len(feature_map['node_forward'])

    # q distribution q(pi | G)
    if args.gcn_type == 'gcn':
        gcn = GCNNet(args, 5, out_dim=32).to(args.device)
    elif args.gcn_type == 'gat':
        gcn = GATNet(args, 5, out_dim=32).to(args.device)
    elif args.gcn_type == 'appnp':
        gcn = APPNET(args, 5, out_dim=32).to(args.device)

    Rout = AttentionModel(embedding_dim = args.gcn_out_dim,
                 hidden_dim=32,
                 state=generation,
                 args=args,
                 featuremap=feature_map,
                 #model,
                 gcn_type=args.gcn_type,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None).to(args.device)


    return Rout




