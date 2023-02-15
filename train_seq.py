import torch
import dgl
import time
import os
import pandas as pd
from collections import defaultdict
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
from utils import save_model, load_model, get_model_attribute, get_last_checkpoint
from models.graph_rnn.train import evaluate_loss as eval_loss_graph_rnn
from models.gran.model import evaluate_loss as eval_loss_gran
from torch.utils.data._utils.collate import default_collate as collate
from models.graph_rnn.data import Graph_to_Adj_Matrix
from models.graph_decoder.data import Graph_to_att_Matrix



def preprocess_copy(graphs, sample_size, device):
    unbatch_dG = [graph['dG'].to(device) for graph in graphs for _ in range(sample_size)]
    batch_dG = dgl.batch(unbatch_dG)
    batch_dGX = batch_dG.ndata['feat'].long()
    batch_dGX=batch_dGX.view(sample_size, -1, batch_dGX.size()[1])
    nx_g = graphs[0]['G'] #only support batch_size=1

    return batch_dG, batch_dGX, nx_g


def train_epoch(
        epoch, args, pmodel, gmodel, dataloader_train,optimizer, scheduler, log_history, feature_map, processor):
    # Set training mode for modules
    model=pmodel['pmodel']
    model.train()
    for _, net in gmodel.items():
        net.train()

    batch_count = len(dataloader_train)
    total_loss = 0.0

    for batch_id, graphs in enumerate(dataloader_train):

        st = time.time()
        dg, embedding, nx_g = preprocess_copy(graphs, args.sample_size, args.device)

        elbo = train_batch(epoch, args, model, gmodel, optimizer,dg, nx_g, embedding, feature_map, processor)
        total_loss = total_loss + elbo

        spent = time.time() - st
        if batch_id % args.print_interval == 0:
            print('epoch {} batch {}: elbo is {}, time spent is {}.'.format(epoch, batch_id, elbo,spent), flush=True)

        log_history['batch_elbo'].append(elbo)

        log_history['batch_time'].append(spent)

        for _, sched in scheduler.items():
            sched.step()

        # if args.log_tensorboard:
        #     summary_writer.add_scalar('{} {} Loss/train batch'.format(
        #         args.note, args.graph_type), loss, batch_id + batch_count * epoch)

    return total_loss / batch_count


def train_batch(epoch, args, pmodel, gmodel,optimizer, dg, nx_g, embedding, feature_map, processor):

    # Evaluate model, get costs and log probabilities
    pi_log_likelihood, pis = pmodel(embedding, dg, nx_g, return_pi=True)



    if args.note == 'GraphRNN':
        # put the original GraphRNN here

        data = [processor(nx_g, perms) for perms in pis]
        data = collate(data)

        # log p(G, pi | z)
        log_joint = -eval_loss_graph_rnn(args, gmodel, data, None, feature_map)
        fake_nll_q = -torch.mean(
            torch.mean((log_joint.detach() - pi_log_likelihood.detach()) * pi_log_likelihood))

    elif args.note == 'DAGG':
        data = [processor(nx_g, perms) for perms in pis]
        data = collate(data)
        log_joint = -gmodel['gmodel'](data, z=None)
        # Reinforce: [log p(G,\pi|z)  - log q(\pi|G)] * dlog q(\pi|G)
        fake_nll_q = -torch.mean(
            torch.mean((log_joint.detach() - pi_log_likelihood.detach()) * pi_log_likelihood))

    # Calculate loss
    #reinforce_loss = ((cost - bl_val) * log_likelihood).mean()






    # elbo(G, z|pi) = log p(G,\pi|z) - kl [q(z|G,\pi)||q(z)]
    nll_p = -torch.mean(log_joint)

    loss = fake_nll_q + nll_p

    # Perform backward pass and optimization step
    for _, opt in optimizer.items():
        opt.zero_grad()

    # grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    if args.clip == True:
        for _, net in gmodel.items():
            clip_grad_value_(net.parameters(), 1.0)


    for _, opt in optimizer.items():
        opt.step()


    elbo = torch.mean(log_joint.detach() - pi_log_likelihood.detach())


    return elbo.item()



def test_data(args, model, gmodel, dataloader_validate, processor,feature_map):
    for _, net in gmodel.items():
        net.eval()
    model.eval()

    batch_count = len(dataloader_validate)
    with torch.no_grad():
        total_loss = 0.0
        ll_qs = 0.0
        for _, graphs in enumerate(dataloader_validate):
            dg, embedding, nx_g = preprocess_copy(graphs, args.sample_size, args.device)
            log_likelihood, pis = model(embedding, dg, nx_g, return_pi=True)
            if args.note == 'GraphRNN':
                # data process and training for graphRNN

                data = [processor(nx_g, perms) for perms in pis]
                data = collate(data)
                cost = -eval_loss_graph_rnn(args, gmodel, data, feature_map)

            # Calculate loss
            # reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
            fake_nll_q = -torch.mean(torch.mean((cost.detach() - log_likelihood.detach()) * log_likelihood))
            nll_p = -torch.mean(cost)

            #loss = fake_nll_q + nll_p

            elbo = torch.mean(cost.detach() - log_likelihood.detach())
            total_loss = total_loss + elbo
            ll_qs = ll_qs + torch.mean(log_likelihood).item()

    return total_loss / batch_count, ll_qs / batch_count


# Main training function

def train(args, model, gmodel,feature_map, dataloader_train , processor):

    optimizer = {}
    for name, net in gmodel.items():
        # optimizer['optimizer_' + name] = optim.Adam(
        #     filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
        #     weight_decay=5e-5)
        optimizer['optimizer_'+name] = optim.Adam(net.parameters(), lr=args.lr)
    if args.enable_gcn:
        optimizer['optimizer_attention'] = optim.Adam(model['pmodel'].parameters(), lr=args.lr)


    scheduler = {}
    for name, net in gmodel.items():
        scheduler['scheduler_' + name] = MultiStepLR(
            optimizer['optimizer_' + name], milestones=args.milestones,
            gamma=args.gamma)

    if args.enable_gcn:
        scheduler['scheduler_attention'] = MultiStepLR(
            optimizer['optimizer_attention'], milestones=args.milestones,
            gamma=args.gamma)


    log_history = defaultdict(list)


    if args.log_tensorboard:
        writer = SummaryWriter(
            log_dir=args.tensorboard_path+ ' ' + args.time, flush_secs=5)
    else:
        writer = None

    epoch=0  #to be deleted when load ready
    while epoch < args.epochs:
        # train
        loss= train_epoch(
            epoch, args, model, gmodel,dataloader_train,optimizer, scheduler, log_history, feature_map, processor)

        epoch += 1

        if args.log_tensorboard:
            writer.add_scalar('{} {} Loss/train'.format(args.note, args.graph_type), loss, epoch)

        print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))
        save_model(epoch, args, gmodel, model, feature_map=feature_map)
        print('Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))



        log_history['train_elbo'].append(loss)

        #log_history['valid_elbo'].append(loss_validate)

        # save logging history
        df_iter = pd.DataFrame()
        df_epoch = pd.DataFrame()
        df_iter['batch_elbo'] = log_history['batch_elbo']
        df_iter['batch_time'] = log_history['batch_time']


        df_epoch['train_elbo'] = log_history['train_elbo']


        df_iter.to_csv(args.logging_iter_path, index=False)
        df_epoch.to_csv(args.logging_epoch_path, index=False)
