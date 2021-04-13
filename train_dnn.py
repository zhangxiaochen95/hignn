r"""
This script trains a deep neural network (DNN) for resource allocation (RA) in heterogeneous wireless networks.
"""

# import functional tools
from tqdm import tqdm
import copy
import pickle
# Import packages for mathematics.
import numpy as np
# Import packages for ML.
import torch
from torch.utils.data import DataLoader
# Import user-defined modules.
import utils, nn_modules as nn_mods


def compute_statistics(samples):
    """Computes the mean/variance of given samples."""
    statistics = {}
    for rtype in num_links.keys():
        for ttype in num_links.keys():
            statistics[(rtype, ttype)] = {}
            h_rel = torch.view_as_real(torch.stack(
                [torch.tensor(samples[i]['h'][(rtype, ttype)], dtype=torch.cfloat) for i in range(len(samples))]))
            statistics[(rtype, ttype)] = {'mean': h_rel.mean(dim=0), 'var': h_rel.var(dim=0)}

    for k, v in statistics.items():
        for kk, vv in v.items():
            print("statistics[{}][{}].size() = {}".format(k, kk, vv.size()))
    return statistics


def proc_data(samples, requires_label):
    """Preprocesses given samples and creates a dataset."""
    inputs, labels = [], []
    for i in range(len(samples)):
        h = copy.deepcopy(samples[i]['h'])
        h_temp = {k: torch.view_as_real(torch.as_tensor(v, dtype=torch.cfloat)) for k, v in h.items()}
        # A dictionary of 3-D Tensors are flattened to a 1-D Tensor.
        input = torch.cat([((h_temp[('siso', 'siso')] - statistics[('siso', 'siso')]['mean']) / statistics[('siso', 'siso')]['var'].sqrt()).flatten(),
                           ((h_temp[('miso', 'siso')] - statistics[('miso', 'siso')]['mean']) / statistics[('miso', 'siso')]['var'].sqrt()).flatten(),
                           ((h_temp[('siso', 'miso')] - statistics[('siso', 'miso')]['mean']) / statistics[('siso', 'miso')]['var'].sqrt()).flatten(),
                           ((h_temp[('miso', 'miso')] - statistics[('miso', 'miso')]['mean']) / statistics[('miso', 'miso')]['var'].sqrt()).flatten()], dim=0)
        inputs.append(input)
        # Add optimal beamforming vectors from FP if it is required.
        if requires_label:
            labels.append({'h': {k: torch.as_tensor(v, dtype=torch.cfloat) for k, v in h.items()},
                           'wsr_targ': samples[i]['wsr_targ']})
        else:
            labels.append({'h': {k: torch.as_tensor(v, dtype=torch.cfloat) for k, v in h.items()}})
    return list(zip(inputs, labels))


def collate(data):
    """collate_fn for training set"""
    inputs, labels = map(list, zip(*data))
    inputs = torch.stack(inputs)
    # Build a block diagonal channel matrix per relation from all graphs in the batch.
    h = {}
    c_etypes = [('siso', 'siso'), ('miso', 'siso'), ('siso', 'miso'), ('miso', 'miso')]
    for stype, dtype in c_etypes:
        # For each relation, put channel matrices from different graphs into a list.
        h_rel = [item['h'][(stype, dtype)] for item in labels]
        # Construct the block diagonal matrix.
        h[(stype, dtype)] = utils.build_diag_block(h_rel)
    # Seal reconstructed h and concatenated wsr_targ into batched labels.
    if 'wsr_targ' in labels[0].keys():
        labels = {'wsr_targ': [item['wsr_targ'] for item in labels],
                  'h': h}
    else:
        labels = {'h': h}
    return inputs, labels


def bf_recovery(outputs):
    """Recovers the beamforming vectors for each link types from outputs of DNN."""
    x = {}
    for i in range(len(output_sizes) - 1):
        x[ntypes[i]] = outputs[:, output_sizes[i]:output_sizes[i + 1]].reshape(-1, num_tx_ants[ntypes[i]] * 2)
    x_norm = {k: torch.sqrt(v.pow(2).sum(-1, keepdim=True)) for k, v in x.items()}
    x = {k: np.sqrt(p_max[k]) * torch.div(v, torch.max(x_norm[k], torch.ones(x_norm[k].size()))).view(
        -1, num_tx_ants[k], 2) for k, v in x.items()}
    return x


def train_per_epoch():
    """Executes one epoch of training."""
    # Set the model to training mode.
    model.train()
    for i, data in tqdm(enumerate(train_loader)):
        # Extract data from train_loader.
        inputs, labels = data
        # Reset optimizer.
        optimizer.zero_grad()
        # Pass batched data to model.
        outputs = model(inputs)
        x = bf_recovery(outputs)
        # Compute negative WSR as loss.
        loss = utils.weighted_sum_rate(labels['h'], x,
                                       {k: torch.ones(v.size(0), dtype=torch.float) for k, v in x.items()},
                                       {k: torch.ones(v.size(0), dtype=torch.float) for k, v in x.items()}).neg()
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
            inputs, labels = data
            # Pass batched data to model.
            outputs = model(inputs)
            x = bf_recovery(outputs)
            var_awgn = {k: torch.ones(v.size(0), dtype=torch.float) for k, v in x.items()}
            weight = {k: torch.ones(v.size(0), dtype=torch.float) for k, v in x.items()}
            # Accumulate utilities (WSR).
            wsr_model += utils.weighted_sum_rate(labels['h'], x, var_awgn, weight).item()
            wsr_targ += torch.tensor(labels['wsr_targ']).sum().item()

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
    PATH = './datasets/d2d_12links/'  # PATH should be consistent with the one used in 'gen_data.py'.
    print("Path of datasets: {}".format(PATH))
    dir_trainset = PATH + 'train.pickle'
    dir_testset = PATH + 'test.pickle'
    dir_specs = PATH + 'specs.pickle'
    model_path = 'model_dnn.pth'

    with open(dir_trainset, 'rb') as file:
        train_data = pickle.load(file)
    with open(dir_testset, 'rb') as file:
        test_data = pickle.load(file)
    with open(dir_specs, 'rb') as file:
        specs = pickle.load(file)

    num_links, num_tx_ants, p_max = specs['num_links'], specs['num_tx_ants'], specs['p_max']  # Parameters used globally
    train_data = train_data[:100000]  # Select the actual size of training set.
    print("Size of training set: {}".format(len(train_data)))
    print("Size of test set: {}".format(len(test_data)))
    print("Specs: {}".format(specs))

    # Process data, build datasets and create DataLoaders.
    statistics = compute_statistics(train_data)
    trainset = proc_data(train_data, requires_label=False)
    train_loader = DataLoader(trainset, collate_fn=collate, batch_size=256)
    testset = proc_data(test_data, requires_label=True)
    test_loader = DataLoader(testset, collate_fn=collate, batch_size=128)

    # Compute the sizes of input/output layers of DNN.
    in_size = 0
    for rtype in num_links.keys():
        for ttype in num_links.keys():
            in_size += num_links[rtype] * num_links[ttype] * num_tx_ants[ttype] * 2
    out_size = {k: num_links[k] * num_tx_ants[k] * 2 for k in num_links.keys()}
    out_size = sum(out_size.values())
    # Create an instance of DNN.
    model = nn_mods.create_mlp([in_size, 512, 512, 512, out_size])
    # model.load_state_dict(torch.load(model_path))
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)  # Adam optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # Learning rate decay

    # Compute the indices of entries for different node types, which is later used in bf_recovery.
    ntypes = ['siso', 'miso']
    output_sizes = [2 * num_tx_ants[k] * num_links[k] for k in ntypes]
    output_sizes.insert(0, 0)
    output_sizes = torch.tensor(output_sizes)
    for i in range(1, len(output_sizes)):
        output_sizes[i:] += output_sizes[i - 1]

    # Training loop.
    num_epochs = 100  # Number of epochs
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
