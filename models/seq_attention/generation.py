import torch
from typing import NamedTuple
from models.seq_attention.sequtils import mask_long2bool, mask_long_scatter
from torch.utils.data._utils.collate import default_collate as collate
from models.graph_rnn.data import Graph_to_Adj_Matrix
from models.graph_rnn.train import evaluate_loss as eval_loss_graph_rnn

class generation(NamedTuple):
    # Fixed input
    loc: torch.Tensor #node_embedding (batch, step, feature_dim)
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    note:str #model used for graph generation

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            first_a=self.first_a[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
        )

    @staticmethod
    def initialize(loc, visited_dtype=torch.uint8):
        #log:node_embedding
        #n_loc: num_node

        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        return generation(
            loc=loc,
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            note='GraphRNN' #type generation model
        )

    #compute p(G,\pi)
    def get_final_cost(self, nx_g ,pis, model, args, feature_map):
        #bathc_G: [nxG,nxG...]
        #graph generation model

        assert self.all_finished()
        # assert self.visited_.
        nll_p = torch.empty(self.loc, device=self.loc.device)
        if self.note == 'GraphRNN':
            # data process and training for graphRNN
            processor = Graph_to_Adj_Matrix (args, feature_map, random_bfs=True)


            data = [processor(nx_g, perms) for perms in pis]
            data = collate(data)
            nll_p_m = eval_loss_graph_rnn(args, model, data, feature_map)





        return nll_p_m, None

    def update(self, selected):

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step

        # Add the length
        # cur_coord = self.loc.gather(
        #     1,
        #     selected[:, None, None].expand(selected.size(0), 1, self.loc.size(-1))
        # )[:, 0, :]
        cur_coord = self.loc[self.ids, prev_a]

        #lengths jiu mei yong!
        lengths = self.lengths
        if self.cur_coord is not None:  # Don't add length for first action (selection of start node)
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update
        first_a = prev_a if self.i.item() == 0 else self.first_a

        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        #i record sample step.
        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_,
                             lengths=lengths, cur_coord=cur_coord, i=self.i + 1, note='GraphRNN')

    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= self.loc.size(-2)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited > 0  # Hacky way to return bool or uint8 depending on pytorch version

    def get_connected_mask(self, edge_index):
        def find_connected_nodes(edge_index, nodes):
            edge_index =edge_index.cpu().numpy()
            connected_nodes = set()
            for sender, receiver in zip(edge_index[0], edge_index[1]):
                if sender in nodes:
                    connected_nodes.add(receiver)
                elif receiver in nodes:
                    connected_nodes.add(sender)
            return list(connected_nodes)
        if self.first==True:
            return self.visited > 0
        else:
            mask = self.visited > 0  # Hacky way to return bool or uint8 depending on pytorch version
            add_mask = torch.ones_like(mask)
            add_mask = add_mask>0 #all True
            batch_size=self.visited.size()[0]
            for b in range(batch_size):
                #starting get mask for graph b
                current_nodes = self.visited[b,0].nonzero(as_tuple=True)[0].cpu().numpy()
                connected_nodes = find_connected_nodes(edge_index, current_nodes)
                add_mask[b,0,connected_nodes] = False #unmask connected graph
            return mask+add_mask

