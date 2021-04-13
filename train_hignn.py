r"""
This script trains an HIGNN for resource allocation (RA) in heterogeneous wireless networks.
"""

# Import functional tools.
import copy
import pickle
from tqdm import tqdm
# Import packages for mathematics.
import numpy as np
# Import packages for ML.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
# Import user-defined modules.
import utils, nn_modules as nn_mods


def compute_statistics(samples):
    """
    Computes the mean/variance of given samples.

    Arguments:
        samples (list):

    Returns:
        stat_h_ii (dict): the mean and variance of direct links h_ii by node type
        stat_h_ij (dict): the mean and variance of interference links h_ij by relation
    """
    num_links, num_tx_ants = samples[0]['num_links'], samples[0]['num_tx_ants']
    stat_h_ii, stat_h_ij = {}, {}
    # Statistics are computed for each relation.
    # Additionally, since the distance of direct links is different from interfering links, it is treated independently.
    for rtype in num_links.keys():
        stat_h_ii[rtype] = {}
        for ttype in num_links.keys():
            stat_h_ij[(rtype, ttype)] = {}

            # Obtain the concatenated channel matrix h_rel from h[(rtype, ttype)] of all samples.
            h_rel = torch.stack(
                [torch.tensor(samples[i]['h'][(rtype, ttype)], dtype=torch.cfloat) for i in range(len(samples))])

            if rtype == ttype:
                # Create a mask to select h_ii.
                mask_diag = torch.cat([torch.eye(num_links[rtype]).unsqueeze(-1) for i in range(num_tx_ants[rtype])],
                                      dim=-1)
                # Create a mask to select h_ij.
                mask_off_diag = torch.ones(num_links[rtype], num_links[rtype], num_tx_ants[rtype]) - mask_diag
                # Compute the statistics of h_ii.
                stat_h_ii[rtype]['mean'] = torch.view_as_real(h_rel * mask_diag).sum() / (
                            len(samples) * num_links[rtype] * num_tx_ants[rtype] * 2)
                stat_h_ii[rtype]['var'] = torch.view_as_real((h_rel - stat_h_ii[rtype]['mean']) * mask_diag).pow(
                    2).sum() / (len(samples) * num_links[rtype] * num_tx_ants[rtype] * 2)
                # Compute the statistics of h_ij.
                stat_h_ij[(rtype, ttype)]['mean'] = torch.view_as_real(h_rel * mask_off_diag).sum() / (
                        len(samples) * num_links[rtype] * (num_links[ttype] - 1) * num_tx_ants[ttype] * 2)
                stat_h_ij[(rtype, ttype)]['var'] = torch.view_as_real(
                    (h_rel - stat_h_ij[(rtype, rtype)]['mean']) * mask_off_diag).pow(2).sum() / (
                                                               len(samples) * num_links[rtype] * (
                                                                   num_links[ttype] - 1) * num_tx_ants[ttype] * 2)
            else:
                h_rel = torch.view_as_real(h_rel)  # a (n_samples, n_rx, n_tx, n_tx_ants, 2) real Tensor
                stat_h_ij[(rtype, ttype)]['mean'] = h_rel.mean()
                stat_h_ij[(rtype, ttype)]['var'] = (h_rel - h_rel.mean()).pow(2).sum() / h_rel.numel()
    return stat_h_ii, stat_h_ij


def build_heterograph(sample):
    """
    Builds a heterograph describing the IFC from given sample.

    Arguments:
        sample (dict): inputs and outputs of the closed-form FP algorithm

    Returns:
        a DGLGraph storing the selected features of the sample
    """

    # Extract params from the sample.
    num_links, num_tx_ants = sample['num_links'], sample['num_tx_ants']
    p_max, var_awgn, weight = copy.deepcopy(sample['p_max']), copy.deepcopy(sample['var_awgn']), copy.deepcopy(sample['weight'])
    h = copy.deepcopy(sample['h'])
    for rtype in num_links.keys():
        for ttype in num_links.keys():
            h[(rtype, ttype)] = torch.view_as_real(torch.tensor(h[(rtype, ttype)], dtype=torch.cfloat))

    h_ii = {}  # Channel response of direct links
    for ltype in num_links.keys():
        p_max[ltype] = torch.as_tensor(p_max[ltype], dtype=torch.float).unsqueeze(-1)
        var_awgn[ltype] = torch.as_tensor(var_awgn[ltype], dtype=torch.float).unsqueeze(-1)
        weight[ltype] = torch.as_tensor(weight[ltype], dtype=torch.float).unsqueeze(-1)
        h_ii[ltype] = (h[(ltype, ltype)][np.arange(num_links[ltype]), np.arange(num_links[ltype])] - stat_h_ii[ltype]['mean']) / stat_h_ii[ltype]['var'].sqrt()

    graph_data = {}  # Source IDs and destination IDs of all edges
    h_ij = {}  # Channel response of interfering links
    for stype in num_links.keys():
        for dtype in num_links.keys():
            graph_data[(stype, '-interfered-by-', dtype)] = ([], [])  # Two lists hold the source and destination IDs of the specified edge type.
            h_ij[(stype, '-interfered-by-', dtype)] = []

    # Add an edge if the norm of channel coefficient lie above the threshold.
    threshold = 0.05
    for stype in num_links.keys():
        for dtype in num_links.keys():
            for i in range(h[(stype, dtype)].size(0)):
                for j in range(h[(stype, dtype)].size(1)):
                    if (np.linalg.norm(h[(stype, dtype)][i, j]) > threshold) & (i != j):
                        graph_data[(stype, '-interfered-by-', dtype)][0].append(i)
                        graph_data[(stype, '-interfered-by-', dtype)][1].append(j)
                        edge_feat = torch.cat([(h[(stype, dtype)][i, j].flatten() - stat_h_ij[(stype, dtype)]['mean']) / stat_h_ij[(stype, dtype)]['var'].sqrt(),
                                               (h[(dtype, stype)][j, i].flatten() - stat_h_ij[(dtype, stype)]['mean']) / stat_h_ij[(dtype, stype)]['var'].sqrt()], dim=0)
                        h_ij[(stype, '-interfered-by-', dtype)].append(edge_feat)
            h_ij[(stype, '-interfered-by-', dtype)] = torch.stack(h_ij[(stype, '-interfered-by-', dtype)])

    # Build the heterograph from edges defined in graph_data.
    g = dgl.heterograph(graph_data, num_nodes_dict=num_links)
    # Assign node attributes.
    g.ndata['p_max'], g.ndata['weight'], g.ndata['var_awgn'], g.ndata['h'] = p_max, weight, var_awgn, h_ii
    # Assign edge attributes.
    g.edata['h'] = h_ij
    return g


def proc_data(samples, requires_label):
    """
    builds heterographs from a list of samples and create a dataset.

    Arguments:
        samples (list): a list of sapmles holding info using a dict
        requires_label (bool): if optimal beamforming vectors are computed

    Returns:
        a list of tuples holding (graph, labels)
    """
    graphs, labels = [], []
    for i in tqdm(range(len(samples))):
        # Build a heterograph from the sample.
        graphs.append(build_heterograph(samples[i]))
        h_ = {}
        for k in samples[i]['h'].keys():
            h_[k] = torch.as_tensor(samples[i]['h'][k], dtype=torch.cfloat)
        # Add optimal beamforming vectors from FP if it is required.
        if requires_label:
            labels.append({'wsr_targ': samples[i]['wsr_targ'], 'h': h_})
        else:
            labels.append({'h': h_})

    # Display the statistics of h_ii and h_ij.
    bg = dgl.batch(graphs)
    for ntype in bg.ntypes:
        print("bg.nodes[{}].data['h'].mean() = {}".format(ntype, bg.nodes[ntype].data['h'].mean()))
        print("bg.nodes[{}].data['h'].var() = {}".format(ntype, bg.nodes[ntype].data['h'].var()))
    for c_etype in bg.canonical_etypes:
        print("bg.edges[{}].data['h'].mean() = {}".format(c_etype, bg.edges[c_etype].data['h'].mean()))
        print("bg.edges[{}].data['h'].var() = {}".format(c_etype, bg.edges[c_etype].data['h'].var()))
    return list(zip(graphs, labels))


def collate(data):
    """collate_fn for training set"""
    # Encapsulate a batch of graphs/labels into two lists.
    graphs, labels = map(list, zip(*data))
    # Batch a collection of graphs into one graph.
    bg = dgl.batch(graphs)
    # Build a block diagonal channel matrix per relation from all graphs in the batch.
    h = {}
    for stype, etype, dtype in bg.canonical_etypes:
        # For each relation, put channel matrices from different graphs into a list.
        h_rel = [item['h'][(stype, dtype)] for item in labels]
        # Construct the block diagonal matrix.
        h[(stype, dtype)] = utils.build_diag_block(h_rel)
    # Seal reconstructed h and concatenated wsr_targ into batched labels.
    bl = {'h': h}
    if 'wsr_targ' in labels[0].keys():
        bl['wsr_targ'] = [item['wsr_targ'] for item in labels]
    return bg, bl


def get_in_feats(g):
    """
    Build inputs from node features.
    Direct channel response consists of a (2 * num_tx_ants,) real Tensor for each node.
    """
    return {ntype: g.nodes[ntype].data['h'].flatten(start_dim=1) for ntype in g.ntypes}


def train_per_epoch():
    """Executes one epoch of training."""
    # Set the model to training mode.
    model.train()
    for i, data in enumerate(train_loader):
        # Extract data from train_loader.
        batched_graph, batched_labels = data
        # Reset optimizer.
        optimizer.zero_grad()
        # Pass batched data to model.
        in_feats = get_in_feats(batched_graph)
        outputs = model(batched_graph, in_feats)
        # Compute negative WSR as loss.
        loss = utils.weighted_sum_rate(batched_labels['h'], outputs, batched_graph.ndata['var_awgn'], batched_graph.ndata['weight']).neg()
        # Call back propagation and execute one step of optimization.
        loss.backward()
        optimizer.step()


def test():
    """Tests performance of trained model."""
    # Set the model to test mode.
    model.eval()
    wsr_model, wsr_targ = 0., 0.
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # Extract data from train_loader.
            batched_graph, batched_labels = data
            # Pass batched data to model.
            in_feats = get_in_feats(batched_graph)
            outputs = model(batched_graph, in_feats)
            # Accumulate utilities (WSR).
            wsr_model += utils.weighted_sum_rate(batched_labels['h'], outputs, batched_graph.ndata['var_awgn'], batched_graph.ndata['weight']).item()
            wsr_targ += torch.tensor(batched_labels['wsr_targ']).sum().item()

    # Record and display the average performance.
    acc.append(wsr_model / wsr_targ)
    test_epochs.append(epoch)
    if (epoch >= 1) & (wsr_model / wsr_targ == np.max(np.array(acc))):
        torch.save(model.state_dict(), model_path)
        print("epoch: {}, acc: {:.2%}, model is saved.".format(epoch, wsr_model / wsr_targ))
    else:
        print("epoch: {}, acc: {:.2%}.".format(epoch, wsr_model / wsr_targ))


if __name__ == '__main__':
    # Read training/test data and specifications.
    PATH = './datasets/d2d_12links/'  # PATH should be consistent with the one used in `gen_data.py`.
    print("Path of datasets: {}".format(PATH))
    dir_trainset = PATH + 'train.pickle'
    dir_testset = PATH + 'test.pickle'
    dir_specs = PATH + 'specs.pickle'
    model_path = 'model_hignn.pth'

    with open(dir_trainset, 'rb') as file:
        train_data = pickle.load(file)
    with open(dir_testset, 'rb') as file:
        test_data = pickle.load(file)
    with open(dir_specs, 'rb') as file:
        specs = pickle.load(file)

    train_data = train_data[:5000]  # Select the actual size of training set.
    print("Size of training set: {}".format(len(train_data)))
    print("Size of test set: {}".format(len(test_data)))
    print("Specs: {}".format(specs))

    # Process data, build datasets and create DataLoaders.
    stat_h_ii, stat_h_ij = compute_statistics(train_data)
    print("Test the statistics of processed train_data...")
    trainset = proc_data(train_data, requires_label=False)
    train_loader = DataLoader(trainset, batch_size=64, collate_fn=collate, shuffle=True)
    print("Test the statistics of processed test_data...")
    testset = proc_data(test_data, requires_label=True)
    test_loader = DataLoader(testset, batch_size=256, collate_fn=collate, shuffle=True)

    # Create an instance of HIGNN.
    model = nn_mods.HIGNN(num_tx_ants=specs['num_tx_ants'], p_max=specs['p_max'])
    # model.load_state_dict(torch.load(model_path))
    # print("model: \n{}".format(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Adam optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # Learning rate decay

    # Training loop.
    num_epochs = 200  # Number of epochs
    test_epochs, acc = [], []  # Record of test results.
    epoch = -1
    test()
    for epoch in range(num_epochs):
        # Train the model.
        train_per_epoch()
        # Test the model performance after each 5 epochs.
        if epoch < 20 or epoch % 5 == 4:
            test()
        scheduler.step()
