r"""
This script defines the neural network (NN) modules.
"""

# Import functional tools.
import copy
# Import packages for mathematics.
import math
# Import packages for ML.
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


def create_mlp(sizes):
    """
    Creates a multi-layer perceptron (MLP).

    Arguments:
        sizes (list): integers specifying number of neurons for each layer

    Returns:
        an MLP holding layers using a sequential container
    """
    assert len(sizes) > 1
    layers = []
    for l in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[l], sizes[l + 1])]
        # layers += [nn.BatchNorm1d(sizes[l + 1])]
        layers += [nn.ReLU()] if l < len(sizes) - 2 else []
    return nn.Sequential(*layers)


class GraphConv(nn.Module):
    """Graph convolution layer"""

    def __init__(self, num_src_feats, num_dst_feats, num_edge_feats, msg_size, num_out_feats):
        super(GraphConv, self).__init__()
        # Message function takes the input features of src nodes and edge features.
        self.fc_msg = create_mlp([num_src_feats + num_edge_feats, 16, msg_size])
        # self.fc_msg = create_mlp([num_src_feats + num_edge_feats, 64, 32, msg_size])
        # Update function takes the input features of dst nodes and aggregated messages.
        self.fc_udt = create_mlp([num_dst_feats + msg_size, 16, num_out_feats])
        # self.fc_udt = create_mlp([num_dst_feats + msg_size, 64, 32, num_out_feats])

    def message(self, edges):
        # msg = torch.cat([edges.data['h'].flatten(start_dim=1), edges.src['in_feats']], dim=1)
        msg = torch.cat([edges.data['h'], edges.src['in_feats']], dim=1)
        return {'m': self.fc_msg(msg)}

    def forward(self, graph, x):
        num_nan = 0
        for item in self.fc_msg:
            if isinstance(item, nn.Linear):
                num_nan += item.weight.isnan().sum()
        for item in self.fc_udt:
            if isinstance(item, nn.Linear):
                num_nan += item.weight.isnan().sum()
        if num_nan > 0:
            print("nan is found in model parameters.")
        graph.ndata['in_feats'] = x
        graph.update_all(self.message, fn.max('m', 'r'))
        return self.fc_udt(torch.cat([graph.dstdata['in_feats'], graph.dstdata['r']], dim=1))


class HeteroGraphConv(nn.Module):
    """Heterograph convolution layer"""

    def __init__(self, num_tx_ants, num_in_feats, msg_size, num_out_feats):
        super(HeteroGraphConv, self).__init__()
        mods = {}
        for stype in num_tx_ants.keys():
            for dtype in num_tx_ants.keys():
                mods[stype + '-to-' + dtype] = GraphConv(num_in_feats[stype], num_in_feats[dtype],
                                                         2 * (num_tx_ants[stype] + num_tx_ants[dtype]),
                                                         msg_size,
                                                         num_out_feats[dtype])
        # Holds submodules in a dictionary.
        self.mods = nn.ModuleDict(mods)

    def forward(self, g, inputs):
        """Applies graph convolution on heterograph g."""
        # Each node type has individual outputs.
        outputs = {ntype: [] for ntype in g.dsttypes}
        # Graph convolution is executed for each relation.
        for stype, etype, dtype in g.canonical_etypes:
            # Create a subgraph for the current relation.
            rel_graph = g[stype, etype, dtype]
            # Skip if the relation contains no edge.
            if rel_graph.number_of_edges() == 0:
                continue
            # Skip if no input is provided for src nodes.
            if stype not in inputs:
                continue
            # Pass the subgraph and inputs to corresponding module.
            if rel_graph.is_homogeneous:
                # If the subgraph is homogeneous, inputs should be a Tensor specified by node type.
                dstdata = self.mods[stype + '-to-' + dtype](rel_graph, inputs[stype])
            else:
                # If the subgraph is heterogeneous, inputs is direct passed as a dict.
                dstdata = self.mods[stype + '-to-' + dtype](rel_graph, inputs)
            outputs[dtype].append(dstdata)
        # Aggregate the outputs from all relations for each node type.
        rsts = {}
        for ntype, alist in outputs.items():
            if len(alist) != 0:
                # rsts[ntype] = torch.mean(torch.stack(alist), 0)
                rsts[ntype], indices = torch.max(torch.stack(alist), 0)
        return rsts


class HIGNN(nn.Module):
    """Heterogeneous interference graph neural networks"""

    def __init__(self, num_tx_ants, p_max, msg_size=8, udt_size=8, num_layers=3):
        super(HIGNN, self).__init__()
        self.num_tx_ants = num_tx_ants
        self.num_hidden_layers = max(0, num_layers - 2)
        self.max_output = {k: math.sqrt(v) for k, v in p_max.items()}

        self.enc = HeteroGraphConv(**{'num_tx_ants': num_tx_ants, 'msg_size': msg_size,
                                      'num_in_feats': {k: 2 * v for k, v in num_tx_ants.items()},
                                      'num_out_feats': {k: udt_size for k, v in num_tx_ants.items()}})

        self.core = HeteroGraphConv(**{'num_tx_ants': num_tx_ants, 'msg_size': msg_size,
                                       'num_in_feats': {k: udt_size + 2 * v for k, v in num_tx_ants.items()},
                                       'num_out_feats': {k: udt_size for k, v in num_tx_ants.items()}})

        self.dec = HeteroGraphConv(**{'num_tx_ants': num_tx_ants, 'msg_size': msg_size,
                                      'num_in_feats': {k: udt_size + 2 * v for k, v in num_tx_ants.items()},
                                      'num_out_feats': {k: 2 * v for k, v in num_tx_ants.items()}})

    def forward(self, g, x):
        feats = copy.deepcopy(x)
        # Encoder
        x = self.enc(g, x)
        # Processor
        for i in range(self.num_hidden_layers):
            x = {k: torch.cat([x[k], feats[k]], dim=1) for k in x.keys()}
            x = self.core(g, x)
            x = {k: F.relu(v) for k, v in x.items()}
        # Decoder
        x = {k: torch.cat([x[k], feats[k]], dim=1) for k in x.keys()}
        x = self.dec(g, x)
        # Impose power constraints at output layer.
        x_norm = {k: torch.sqrt(v.pow(2).sum(-1, keepdim=True)) for k, v in x.items()}
        x = {k: self.max_output[k] * torch.div(v, torch.max(x_norm[k], torch.ones(x_norm[k].size()))).view(
            -1, self.num_tx_ants[k], 2) for k, v in x.items()}
        return x

