r"""
This script generates training/test sets.
"""

# Import functional tools.
import pickle
from tqdm import tqdm
# Import packages for mathematics.
import numpy as np
# Import user-defined modules.
import envs


def gen_samples(size, chan_type, requires_label=True):
    """
    Arguments:
        size (int): number of samples to generate
        chan_type (str): type of channel
        requires_label (Bool): if optimal beamforming vectors are computed

    Returns:
        a list of samples holding information with a dict
    """
    # Ensure the channel type is available.
    assert chan_type in ['gaussian', 'd2d']
    n_links = {'siso': 8, 'miso': 4}
    n_t = {'siso': 1, 'miso': 2}
    specs = {'num_links': n_links, 'num_tx_ants': n_t,
             'p_max': {k: 1. for k in n_links.keys()},
             'var_awgn': {k: 1. for k in n_links.keys()},
             'weight': {k: np.ones(v) for k, v in n_links.items()}}
    if chan_type == 'gaussian':
        env = envs.InterferenceChannel(**specs)
    else:
        env = envs.HeteroD2DNetwork(**specs)

    # Generate samples.
    data_list = []
    for idx in tqdm(range(size)):
        h = env.channel_response()
        sample = {'num_links': env.num_links, 'num_tx_ants': env.num_tx_ants,
                  'var_awgn': env.var_awgn, 'weight': env.weight, 'p_max': env.p_max,
                  'h': h}
        # Get labels if 'requires_label' == True.
        if requires_label:
            # Compute the optimal transmission scheme and WSR with benchmark algorithm.
            x, wsrs = envs.max_wsr_cf_fp(h=h, var_awgn=env.var_awgn, p_max=env.p_max, w=env.weight)
            sample['x_targ'], sample['wsr_targ'] = x, wsrs[-1]
        # Append the sample to the list.
        data_list.append(sample)
    return data_list, specs


if __name__ == '__main__':
    # Execute only if the script is run as main file.

    # Create and store samples.
    size_trainset = 500000  # Number of samples to be generated
    size_testset = 1000
    chan_type = 'd2d'  # Type of channel
    PATH = './datasets/d2d_12links/'

    # Generate training/test data.
    train_data, specs = gen_samples(size_trainset, chan_type, requires_label=False)
    test_data, specs = gen_samples(size_testset, chan_type, requires_label=True)

    # Save training set.
    dir_trainset = PATH + 'train.pickle'
    file = open(dir_trainset, 'wb')
    pickle.dump(train_data, file)
    file.close()
    print("Training set is stored to directory: {}".format(dir_trainset))

    # Save test set.
    dir_testset = PATH + 'test.pickle'
    file = open(dir_testset, 'wb')
    pickle.dump(test_data, file)
    file.close()
    print("Test set is stored to directory: {}".format(dir_testset))

    # Save data specifications.
    dir_specs = PATH + 'specs.pickle'
    file = open(dir_specs, 'wb')
    pickle.dump(specs, file)
    file.close()

    # Read and check data.
    with open(dir_trainset, 'rb') as file:
        train_data = pickle.load(file)
    with open(dir_testset, 'rb') as file:
        test_data = pickle.load(file)
    print("Size of training set: {}".format(len(train_data)))
    print("Size of test set: {}".format(len(test_data)))
    print("specs: {}".format(specs))