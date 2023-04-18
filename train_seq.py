import torch
import dgl
import time
import pandas as pd
from collections import defaultdict
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
from utils import save_model, load_model, get_model_attribute, get_last_checkpoint






# remove the epoch argument from the argument, and move the print clause out 
def train_epoch(args, DAGG, Rout, dataloader_train,optimizer, scheduler, log_history, feature_map,epoch):

    Rout.train()
    DAGG.train()

    batch_count = len(dataloader_train)
    total_loss = 0.0

    for batch_id, graphs in enumerate(dataloader_train):

        st = time.time()


        elbo = train_batch(args, DAGG, Rout, optimizer,graphs[0]['dG'].to(args.device), graphs[0]['G'],feature_map)
        total_loss = total_loss + elbo

        spent = time.time() - st

        #  
        if batch_id % args.print_interval == 0:
            print('epoch {} batch {}: elbo is {}, time spent is {}.'.format(epoch, batch_id, elbo,spent), flush=True)

        log_history['batch_elbo'].append(elbo)

        log_history['batch_time'].append(spent)

        for _, sched in scheduler.items():
            sched.step()

    return total_loss / batch_count




def train_batch(args, DAGG, Rout,optimizer, dg, nx_g, embedding):

    # Evaluate model, get costs and log probabilities
    pi_log_likelihood, pis = Rout(dg, args.sample_size, return_pi=True)





    log_joint = -DAGG(nx_g, pis)

    # Reinforce: [log p(G,\pi|z)  - log q(\pi|G)] * dlog q(\pi|G)
    fake_nll_q = -torch.mean(torch.mean((log_joint.detach() - pi_log_likelihood.detach()) * pi_log_likelihood))

    # elbo(G, z|pi) = log p(G,\pi|z) - kl [q(z|G,\pi)||q(z)]
    nll_p = -torch.mean(log_joint)

    loss = fake_nll_q + nll_p

    # Perform backward pass and optimization step

    optimizer.zero_grad()

    # grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    if args.clip == True:
        clip_grad_value_(DAGG.parameters(), 1.0)



    optimizer.step()


    elbo = torch.mean(log_joint.detach() - pi_log_likelihood.detach())


    return elbo.item()


# TODO: suggested function name: test  
def test(args, DAGG, Rout, dataloader_validate, feature_map):

    DAGG.eval()
    Rout.eval()

    batch_count = len(dataloader_validate)
    with torch.no_grad():
        total_elbo = 0.0
        ll_qs = 0.0
        for _, graphs in enumerate(dataloader_validate):

            log_likelihood, pis = Rout(graphs['dG'][0], args.sample_size, return_pi=True)



            log_joint = -DAGG(graphs['G'][0], pis)
            elbo = -torch.mean(log_joint.detach() - log_likelihood.detach())
            total_elbo = total_elbo + elbo




    return total_elbo / batch_count, ll_qs / batch_count


# Main training function

def train(args, DAGG, Rout,feature_map, dataloader_train):

    optimizer = optim.Adam([DAGG.parameters(), Rout.parameters()], lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones,gamma=args.gamma)



    log_history = defaultdict(list)


    if args.log_tensorboard:
        writer = SummaryWriter(
            log_dir=args.tensorboard_path+ ' ' + args.time, flush_secs=5)
    else:
        writer = None



    for epoch  in range(args.epochs):
        # train
        loss= train_epoch(args, DAGG, Rout,dataloader_train,optimizer, scheduler, log_history, feature_map,epoch)

        epoch += 1

        if args.log_tensorboard:
            writer.add_scalar('{} {} Loss/train'.format(args.note, args.graph_type), loss, epoch)

        print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))
        save_model(epoch, args, DAGG, Rout, feature_map=feature_map)
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
