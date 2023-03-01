from models.seq_attention.attention_model import AttentionModel, generation
from models.gcn.net.appnp_net_node import APPNET
from models.gcn.net.gat_net_node import GATNet
from models.gcn.net.gcn_net_node import GCNNet
from models.DAGG.att_decoder import AttentionDecoder
from models.DAGG.model import DAGG




# TODO: separate the function into two models: the generative model, and the inference component  
#def create_generative_model(args, feature_map):
def create_generative_model(args, feature_map):
    """
    Initialize a generative model.  
    """
    

    print('Producing model...')



    if args.note == 'DAGG':
        # decoding_model= AttentionDecoder(args)
        # update_model = AttentionDecoder(args)
        # gmodel=att_decoder_dependency(args, args.withz, decoding_model, update_model, feature_map)
        # gmodel={'gmodel':gmodel.to(args.device)}

        edge_model = AttentionDecoder(args)
        update_model = AttentionDecoder(args)

        # TODO: check these names
        gmodel =  DAGG(args, args.withz, edge_model, update_model, feature_map)


    return gmodel



def create_inference_model(args, feature_map):
    """
    Initialize an inference model.  
    """

    len_node_vec = len(feature_map['node_forward'])

    # q distribution q(pi | G)
    if args.gcn_type == 'gcn':
        gcn = GCNNet(args, len_node_vec, out_dim=32).to(args.device)
    elif args.gcn_type == 'gat':
        gcn = GATNet(args, len_node_vec, out_dim=32).to(args.device)
    elif args.gcn_type == 'appnp':
        gcn = APPNET(args, len_node_vec, out_dim=32).to(args.device)

    Rout = AttentionModel(embedding_dim = args.gcn_out_dim,
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


    return Rout




