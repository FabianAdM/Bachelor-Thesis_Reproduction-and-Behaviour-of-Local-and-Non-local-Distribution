import numpy as np
import os

import neural_network as nn
import model as m
import distribution as d
import settings as s


def prt_distribution(distribution, model=None):  # Print the distribution and if available the corresponding model

    """  # If you want to see the 'Fritz' distribution which a specific parameter, then uncomment this part
    parameter = 1
    n = 0
    while d.get_para_range()[n] < parameter:
        if n + 1 >= len(d.get_para_range()):
            break
        n = n + 1
    print('Distribution: {}, parameter: {}, sum: {}'.format(d.get_name(), d.get_para_range()[n], np.sum(distribution,
                                                                                                        axis=0)))
    """
    print('Distribution: {}, normalization (sum): {}'.format(d.get_name(), round(np.sum(distribution)), 9))
    if model is not None:
        m.print_model(model)
    d.print_distribution(distribution)


def find_min_res(distribution):  # Uses several attempts with the same start configuration to find the best distribution

    print('\nStart finding the closest distribution according to the target distribution in {} subrounds'
          .format(s.NeuralNetwork().n_of_min_samples))

    result = nn.training()
    dist_old = nn.np_discrepancy(result, target=distribution, dist_mod='ec')
    res_list = np.ones(s.NeuralNetwork().n_of_min_samples)
    res_list[0] = dist_old
    print('Subround 1 of {} with a founded distance of {}\n'.format(s.NeuralNetwork().n_of_min_samples, dist_old))

    for i in range(s.NeuralNetwork().n_of_min_samples - 1):
        temp = nn.training()

        dist_new = nn.np_discrepancy(temp, target=distribution, dist_mod='ec')
        res_list[i + 1] = dist_new
        if dist_new < dist_old:
            result = temp
            dist_old = nn.np_discrepancy(result, target=distribution, dist_mod='ec')
        print('Subround {} of {} with a new founded distance of {} (min distance is {})\n'.format(i + 2,
              s.NeuralNetwork().n_of_min_samples, dist_new, dist_old))

    nn.set_find_min_distances(res_list)
    return result


def train(distribution, model=None):

    if d.get_name() == 'fritz' or d.get_name() == 'LLL' or d.get_name() == 'LLL_random_noise':
        # Used to reproduce the Fritz and LLL distribution over a range of parameters
        d.set_target(distribution)
        result = np.zeros_like(distribution)

        for i in range(len(d.get_para_range())-1, -1, -1):
            print('\nRound {} of {}, with model {} and {} distribution of param v = {}. Normalization (sum): {}'
                  .format(len(d.get_para_range()) - i, len(d.get_para_range()), m.get_name(), d.get_name(),
                          round(s.Distribution().para_range[i], 6), round(np.sum(distribution[i], axis=0)), 9))

            nn.new_neural_network(model, distribution[i, :])
            if s.NeuralNetwork().find_min_result:
                result[i, :] = find_min_res(distribution[i, :])
            else:
                result[i, :] = nn.training()

            print('Distance calculated: ', nn.np_discrepancy(result[i, :], target=distribution[i, :], dist_mod='ec'))

        return result


    elif d.get_name() == 'quantum calculated distribution':  # Here to start searching possibly used states in order to
        d.set_target(distribution)                                                  # reproduce the quantum distribution
        prt_distribution(distribution.flatten())
        return nn.find_used_states_for_distribution(distribution)

    else:
        print('Wrong distribution, pls choose a valid distribution or define it first in <<distribution>>')


def start(distribution_name, network=None):

    model, distribution = m.new_model(network, distribution_name)
    prt_distribution(distribution)
    result = train(distribution, model)
    prt_distribution(result)
    if model is not None:
        nn.plot_result(result, distribution)


if __name__ == '__main__':
    for i in ['models_plot', 'result_distance', 'result_plot_sweep', 'saved_settings']:
        if not os.path.exists(i):
            os.makedirs(i)

    #start('fritz', 'triangle')
    #start('quantum calculated distribution')
    start('LLL', 'triangle')

    print("\n********Finish. If you used the neural network, then the resulting plots are now in the folder "
          "'D:/ETH/Bachelorarbeit/Code/result_...********")


