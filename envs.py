r"""
This script defines
    - classes of heterogeneous MISO wireless channels
    - benchmark algos for beamforming
    - other required components
"""

# Import functional tools.
import copy
# Import packages for visualization.
import matplotlib.pyplot as plt
# Import packages for mathematics.
import numpy as np


class InterferenceChannel:
    """Base class of general heterogeneous interference channels (IFCs)"""

    def __init__(self, num_links, num_tx_ants, p_max, var_awgn, weight, bandwidth=1.):

        # Parameters shared by all links within each type
        self.num_links = copy.deepcopy(num_links)  # Number of links for each link type
        self.num_tx_ants = copy.deepcopy(num_tx_ants)  # Number of Tx antennas for each link type

        # Parameters individually held by each link
        self.p_max = {k: v * np.ones(self.num_links[k]) for k, v in p_max.items()}  # Power constraint on each transmitter (Watt)
        self.var_awgn = {k: v * np.ones(self.num_links[k]) for k, v in var_awgn.items()}  # Noise variance at each receiver (Watt)
        self.weight = copy.deepcopy(weight)  # Weight of each link

        # Parameters shared by the entire IFC
        self.bandwidth = bandwidth  # Total bandwidth shared by all links (Hz)
        self.total_num_links = sum(num_links.values())  # Total number of links

        # Display specs of the channel.
        print("An instance of {} is created with specs:".format(self.__class__.__name__))
        print("num_links = {}".format(self.num_links))
        print("weight = \n{}".format(self.weight))
        print("bandwidth = {} (Hz)".format(self.bandwidth))
        print("var_awgn = \n{} (Watt)".format(self.var_awgn))
        print("p_max = \n{} (Watt)".format(self.p_max))

    def channel_response(self):
        """Generates zero-mean complex Gaussian variables with unit variance as channel response for all links."""
        h = {}  # A dictionary holding channel response of all relations
        # h[(rtype, ttype)] refers to a relation between transmitters of ttype and receivers of rtype,
        # of which the (i, j)-th entry gives CSI between the j-th transmitter of ttype and the i-th receiver of rtype.
        for rtype in self.num_links.keys():
            for ttype in self.num_links.keys():
                h[(rtype, ttype)] = 1 / np.sqrt(2) * (
                        np.random.randn(self.num_links[rtype], self.num_links[ttype], self.num_tx_ants[ttype])
                        + 1j * np.random.randn(self.num_links[rtype], self.num_links[ttype], self.num_tx_ants[ttype])
                )
        return h


class HeteroD2DNetwork(InterferenceChannel):
    """Heterogeneous device-to-device (D2D) network"""

    def __init__(self, num_links, num_tx_ants, p_max, var_awgn, weight, bandwidth=5e6,
                 len_area=400., dmin=2, dmax=50):
        super(HeteroD2DNetwork, self).__init__(num_links, num_tx_ants, p_max, var_awgn, weight, bandwidth)

        self.len_area = len_area  # Length of the square area
        self.dmin, self.dmax = dmin, dmax  # Minimum/maximum link distance

        self.pos_tx, self.pos_rx = {}, {}  # Positions of Tx/Rx
        self.dist = {}  # Distance between each Tx and Rx

    def layout(self):
        """Randomly determine the Tx/Rx positions."""
        for ltype in self.num_links.keys():
            # Randomly determine the positions of Tx in the square area.
            self.pos_tx[ltype] = self.len_area * np.random.rand(self.num_links[ltype], 2)
            self.pos_rx[ltype] = []

            for idx in range(self.num_links[ltype]):
                is_valid_pos = False
                while not is_valid_pos:
                    radius = self.dmin + np.random.rand() * (self.dmax - self.dmin)  # Distance of Rx to Rx
                    ang = 2 * np.pi * np.random.rand()  # Angle of Rx to Rx
                    pos = self.pos_tx[ltype][idx] + radius * np.array([np.cos(ang), np.sin(ang)])  # Position of Rx
                    # The generated position is kept only if it lies in the valid range.
                    if 0 <= pos[0] <= self.len_area and 0 <= pos[1] <= self.len_area:
                        is_valid_pos = True
                self.pos_rx[ltype].append(pos)
            self.pos_rx[ltype] = np.array(self.pos_rx[ltype])

        # Compute distance between each Rx and Tx.
        for rtype in self.num_links.keys():
            for ttype in self.num_links.keys():
                # The (i, j)-th entry is the distance between the Rx i of rtype and Tx j of ttype.
                self.dist[(rtype, ttype)] = np.empty([self.num_links[rtype], self.num_links[ttype]])
                for i in range(self.num_links[rtype]):
                    for j in range(self.num_links[ttype]):
                        self.dist[(rtype, ttype)][i, j] = np.sqrt(np.square(self.pos_rx[rtype][i] - self.pos_tx[ttype][j]).sum())
                # Ensure that nodes are not too close.
                self.dist[(rtype, ttype)] = np.clip(self.dist[(rtype, ttype)], self.dmin, np.sqrt(2) * self.len_area)

    def channel_response(self):
        h = {}  # A dictionary holding channel response of all relations
        # Reset the layout of the network.
        self.layout()
        # Generate channel response for each relation.
        for rtype in self.num_links.keys():
            for ttype in self.num_links.keys():
                shadowing = np.random.randn(*self.dist[(rtype, ttype)].shape)
                large_scale_fading = 4.4 * 10 ** 5 / ((self.dist[(rtype, ttype)] ** 1.88) * (10 ** (shadowing * 6.3 / 20)))
                small_scale_fading = 1 / np.sqrt(2) * (
                        np.random.randn(self.num_links[rtype], self.num_links[ttype], self.num_tx_ants[ttype]) +
                        1j * np.random.randn(self.num_links[rtype], self.num_links[ttype], self.num_tx_ants[ttype]))
                h[(rtype, ttype)] = np.sqrt(np.expand_dims(large_scale_fading, axis=-1)) * small_scale_fading
        return h


def link_capacity(h, x, var_awgn):
    """
    Computes the rate profile for each link in the IFC.

    Arguments:
        h (dict of ndarrays): channel response of all relations
        x (dict of ndarrays): beamforming vectors of each transmitters
        var_awgn (dict of ndarrays): noise variance at each receiver

    Returns:
        An dict of ndarrays listing the capacity of each link
    """
    # At each receiver, compute the received signal power from each transmitter.
    rx_power, interference, sinr, rate = {}, {}, {}, {}
    for rtype, ttype in h.keys():
        rx_power[(rtype, ttype)] = np.square(np.abs((h[(rtype, ttype)] * x[ttype]).sum(-1)))

    # Compute the interference level at each receiver.
    for rtype in x.keys():
        interference[rtype] = - np.diag(rx_power[(rtype, rtype)])
        for ttype in x.keys():
            interference[rtype] += rx_power[(rtype, ttype)].sum(-1)

    # Compute the signal-to-interference-plus-noise ratio (SINR) and achievable rate.
    for rtype in x.keys():
        sinr[rtype] = np.diag(rx_power[(rtype, rtype)]) / (interference[rtype] + var_awgn[rtype])
        rate[rtype] = np.log(1 + sinr[rtype])
    return rate


def weighted_sum_rate(h, x, var_awgn, w):
    """
    Computes the weighted sum rate (WSR) of the IFC.

    Arguments:
        h (dict of ndarrays): channel response of all relations
        x (dict of ndarrays): transmission scheme of each transmitters
        var_awgn (dict of ndarrays): noise variance at each receiver
        w (dict of ndarrays): weight of each link

    Returns:
        the weighted sum rate of all links in the channel
    """
    # Compute the rate for each link.
    rate = link_capacity(h, x, var_awgn)
    # Accumulate the WSR throughout link types.
    wsr = 0.  # Weighted sum rate
    for ltype in w.keys():
        wsr += np.dot(w[ltype], rate[ltype])
    return wsr


def max_wsr_cf_fp(h, var_awgn, p_max, w, max_num_iters=500):
    """
    Maximizes the WSR of IFC via closed-form fractional programming (FP), which is originally proposed in the paper

    @ARTICLE{8314727,
        author={K. {Shen} and W. {Yu}},
        journal={IEEE Transactions on Signal Processing},
        title={Fractional Programming for Communication Systems—Part I: Power Control and Beamforming},
        year={2018},
        volume={66},
        number={10},
        pages={2616-2630},
        doi={10.1109/TSP.2018.2812733}
    }

    Arguments:
        h (dict of ndarrays): channel response of all relations
        var_awgn (dict of ndarrays): noise variance at each receiver
        p_max (dict of ndarrays): power constraint on each transmitter
        w (dict of ndarrays): weight of each link
        max_num_iters (int): Maximum number of iterations

    Returns:
        x (dict of ndarrays): optimal transmission scheme for each types of links
        log_wsrs (list): WSRs achieved through iterations
    """
    # Initialize transmission scheme.
    num_links, num_tx_ants = {}, {}  # Number of links and Tx antennas
    x, gamma, y = {}, {}, {}  # Transmission scheme and auxiliary variables
    for ltype in p_max.keys():
        num_links[ltype] = h[(ltype, ltype)].shape[1]
        num_tx_ants[ltype] = h[(ltype, ltype)].shape[-1]
        x[ltype] = np.expand_dims(np.sqrt(p_max[ltype] / num_tx_ants[ltype]), -1) * np.ones(
            (num_links[ltype], num_tx_ants[ltype]), dtype=np.complex)

    epsilon = 1e-5  # Threshold to terminate loops
    # Record the WSR achieved by initial transmission scheme
    wsrs = [weighted_sum_rate(h, x, var_awgn, w)]  # Log of WSRs

    for num_iter in range(max_num_iters):
        x_pre = copy.deepcopy(x)  # Copy of solution in the last iteration
        # Compute received signal, power and interference.
        rx_signal, rx_power, interference = {}, {}, {}
        for rtype, ttype in h.keys():
            rx_signal[(rtype, ttype)] = (h[(rtype, ttype)] * x[ttype]).sum(-1)
            rx_power[(rtype, ttype)] = np.square(np.abs(rx_signal[(rtype, ttype)]))
        for rtype in x.keys():
            interference[rtype] = -np.diag(rx_power[(rtype, rtype)])
            for ttype in x.keys():
                interference[rtype] += rx_power[(rtype, ttype)].sum(-1)
        for ltype in x.keys():
            # Update gamma by (57).
            gamma[ltype] = np.diag(rx_power[(ltype, ltype)]) / (interference[ltype] + var_awgn[ltype])
            # Update y by (60).
            y[ltype] = np.sqrt(w[ltype] * (1 + gamma[ltype])) * np.diag(rx_signal[(ltype, ltype)]) / (
                    interference[ltype] + np.diag(rx_power[(ltype, ltype)]) + var_awgn[ltype])
        # Update x by (61).
        hy, hyyh, sum_hyyh = {}, {}, {}
        for rtype in x.keys():
            for ttype in x.keys():
                # Compute $h_{ji}^{H}y_{j}$ for each i, j.
                hy[(rtype, ttype)] = np.conj(h[(rtype, ttype)]) * y[rtype].reshape(-1, 1, 1)
                # Compute $h_{ji}^{H}y_{j}y_{j}^{H}h_{ji}$ for each (i, j).
                hyyh[(rtype, ttype)] = np.matmul(np.expand_dims(hy[(rtype, ttype)], -1),
                                                 np.expand_dims(np.conj(hy[(rtype, ttype)]), -2))

        # Sum $h_{ji}^{H}y_{j}y_{j}^{H}h_{ji}$ along j (idx of Rx).
        for ttype in x.keys():
            sum_hyyh[ttype] = 0
            for rtype in x.keys():
                sum_hyyh[ttype] += hyyh[(rtype, ttype)].sum(axis=0)

        for ltype in x.keys():
            for i in range(num_links[ltype]):
                eta_l, eta_r = 0., 1.
                x[ltype][i] = np.sqrt(w[ltype][i] * (1 + gamma[ltype][i])) * np.dot(
                    np.linalg.inv(eta_r * np.eye(num_tx_ants[ltype]) + sum_hyyh[ltype][i]), hy[(ltype, ltype)][i, i])

                # Scale the lower/upper bound such that the optimal eta lies in (eta_l, eta_r).
                while np.linalg.norm(x[ltype][i]) > np.sqrt(p_max[ltype][i]):
                    eta_l = eta_r
                    eta_r = 2 * eta_r
                    x[ltype][i] = np.sqrt(w[ltype][i] * (1 + gamma[ltype][i])) * np.dot(
                        np.linalg.inv(eta_l * np.eye(num_tx_ants[ltype]) + sum_hyyh[ltype][i]),
                        hy[(ltype, ltype)][i, i])

                # Use bisection search to find the optimal eta.
                while eta_r - eta_l > epsilon:
                    eta = (eta_l + eta_r) / 2
                    x[ltype][i] = np.sqrt(w[ltype][i] * (1 + gamma[ltype][i])) * np.dot(
                        np.linalg.inv(eta * np.eye(num_tx_ants[ltype]) + sum_hyyh[ltype][i]), hy[(ltype, ltype)][i, i])
                    if np.linalg.norm(x[ltype][i]) < np.sqrt(p_max[ltype][i]):
                        eta_r = eta
                    else:
                        eta_l = eta
        # Record the achieved WSR in the current iteration.
        wsrs.append(weighted_sum_rate(h, x, var_awgn, w))
        # Break the loop in advance when the solution converges.
        diff = 0.  # Difference between the current solution and previous one.
        for ltype in x.keys():
            diff += np.linalg.norm(x[ltype] - x_pre[ltype])
        if diff <= epsilon:
            break
    return x, wsrs


if __name__ == '__main__':
    # Execute only if the script is run as main file.

    # Create an instance of wireless network as env.
    env = HeteroD2DNetwork(**{'num_links': {'siso': 8, 'miso': 4}, 'num_tx_ants': {'siso': 1, 'miso': 2},
                              'p_max': {'siso': 1., 'miso': 1.}, 'var_awgn': {'siso': 1., 'miso': 1.},
                              'weight': {'siso': np.ones(8), 'miso': np.ones(4)}})
    # Compute the channel response.
    h = env.channel_response()
    # Check the range of channel coefficients.
    print("Check the range of channel coefficients:")
    for ltype in h.keys():
        print("|h[{}]| lies in [{}, {}]".format(ltype, np.min(np.abs(h[ltype])), np.max(np.abs(h[ltype]))))

    # Test the func 'weighted_sum_rate'.
    print("Test the func 'weighted_sum_rate':")
    x = {'siso': np.random.randn(env.num_links['siso'], env.num_tx_ants['siso']),
         'miso': np.random.randn(env.num_links['miso'], env.num_tx_ants['miso'])}
    r = link_capacity(h, x, env.var_awgn)
    print("r = {}".format(r))
    wsr = weighted_sum_rate(h, x, var_awgn=env.var_awgn, w=env.weight)
    print("wsr = {}".format(wsr))

    # Test the correctness of benchmark algorithms.
    _, wsrs = max_wsr_cf_fp(h, env.var_awgn, env.p_max, env.weight)

    plt.figure()
    plt.plot(wsrs, marker='.', markevery=50, label='Closed-form FP')
    plt.legend()
    plt.title("Weighted sum rates (WSR) vs. iterations")
    plt.xlabel("iteration")
    plt.ylabel("WSR (nat/sec)")
    plt.show()
