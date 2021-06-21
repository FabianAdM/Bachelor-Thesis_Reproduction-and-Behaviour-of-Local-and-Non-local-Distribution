import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as k
from scipy.ndimage.filters import gaussian_filter1d
from itertools import product

import settings as s
import model as m
import distribution as d
nn = s.NeuralNetwork()


def np_discrepancy(predicted, target=0, dist_mod=nn.loss_mode):  # Used to value the difference between the target and the
    if dist_mod == 'l2':                                                     # predicted distribution (for numPy arrays)
        return np.sum(np.square(target - predicted), axis=-1)  # Mean squared error
    elif dist_mod == 'l1':
        return 0.5 * np.sum(np.abs(target - predicted), axis=-1)  # Mean absolut error
    elif dist_mod == 'msl':
        return np.sum(np.square(np.log(target + 1.) - np.log(predicted + 1.)))  # Mean squared logarithmic error
    elif dist_mod == 'lc':
        return np.sum(np.log(np.cosh(np.abs(target - predicted))))  # Log-Cosh error
    elif dist_mod == 'ec':
        return np.sum(np.sqrt(np.sum(np.square(predicted - target), axis=-1)))  # Euclidean distance
    elif dist_mod == 'kl':
        target = np.clip(target, k.epsilon(), 1)
        predicted = np.clip(predicted, k.epsilon(), 1)
        return np.sum(target * np.log(np.divide(target, predicted)), axis=-1)  # Kullback-Leibler divergence
    elif dist_mod == 'js':
        target = np.clip(target, k.epsilon(), 1)
        predicted = np.clip(predicted, k.epsilon(), 1)
        avg = (target + predicted) / 2
        return np.sum(target * np.log(np.divide(target, avg)), axis=-1) + \
               np.sum(predicted * np.log(np.divide(predicted, avg)), axis=-1)


def loss_discrepancy(predicted, target):  # Used to value the difference between the target and the predicted distribution
    if nn.loss_mode == 'l2':                                                                     # (for Keras tensor)
        return k.sum(k.square(target - predicted), axis=-1)  # Mean squared error
    elif nn.loss_mode == 'l1':
        return 0.5 * k.sum(k.abs(target - predicted), axis=-1)  # Mean absolut error
    elif nn.loss_mode == 'msl':
        return k.sum(k.square(k.log(target + 1.) - k.log(predicted + 1.)))  # Mean squared logarithmic error
    elif nn.loss_mode == 'lc':
        return tf.keras.losses.logcosh(target, predicted)  # Log-Cosh error
    elif nn.loss_mode == 'ec':
        return k.sum(k.sqrt(k.sum(k.square(predicted - target), axis=-1)))  # Euclidean distance
    elif nn.loss_mode == 'kl':
        target = k.clip(target, k.epsilon(), 1)
        predicted = k.clip(predicted, k.epsilon(), 1)
        return k.sum(target * k.log(target / predicted), axis=-1)  # Kullback-Leibler divergence
    elif nn.loss_mode == 'js':
        target = k.clip(target, k.epsilon(), 1)
        predicted = k.clip(predicted, k.epsilon(), 1)
        avg = (target + predicted) / 2
        return k.sum(target * k.log(target / avg), axis=-1) + k.sum(predicted * k.log(predicted / avg), axis=-1)


def loss(target, predicted):  # 'out_target' is sampled in several rows due to reach size of 'batch_size',so
    return loss_discrepancy(m.triangle_convertion(predicted), target[0, :])  # we need here just one sample


def generator_input_output():  # Generates the necessary batch with random input_batch variables and output ('target')
    while True:
        input_batch = np.divide((np.random.random((nn.batch_size, 3)) - nn.training_mean), nn.training_sigma)
        yield input_batch, nn.out_target_batchsize  # Continually send the computed batch to the 'fit_generator'


def generator_input():  # Generates the output with random input variables ('target')
    while True:
        yield np.divide((np.random.random((nn.batch_size, 3)) - nn.training_mean), nn.training_sigma)


def optimizer():  # Define an optimizer for training the network ('fit_generator') (See Keras docs for more information)
    if nn.optimizer.lower() == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=nn.lr, decay=nn.decay, momentum=nn.momentum, nesterov=True)
    elif nn.optimizer.lower() == 'RMSprop':
        return tf.keras.optimizers.RMSprop(learning_rate=nn.lr, rho=nn.rho, momentum=nn.momentum, epsilon=nn.epsilon,
                                           centered=True)
    elif nn.optimizer.lower() == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=nn.lr, epsilon=nn.epsilon)
    elif nn.optimizer == 'adadelta':
        return tf.keras.optimizers.Adadelta(learning_rate=nn.lr, rho=nn.rho, epsilon=nn.epsilon)
    elif nn.optimizer.lower() == 'Nadam':
        return tf.keras.optimizers.Nadam(learning_rate=nn.lr, beta_1=0.9, beta_2=0.999, epsilon=nn.epsilon)
    elif nn.optimizer.lower() == 'FTRL':
        return tf.keras.optimizers.Ftrl(learning_rate=nn.lr, learning_rate_power=-0.5, initial_accumulator_value=0.1)
    else:
        print("\n Optimizer {} not defined. Please choose one of the implemented optimizer or define it in "
              "<<neural_network>>.\n 'adam' is used now." .format(nn.optimizer))
        nn.optimizer = 'adam'
        return tf.keras.optimizers.Adam(learning_rate=nn.lr, epsilon=nn.epsilon)


def evaluation(model):  # Evaluates the resulted model and return it as a distribution in a numpy array
    eval_model = model.predict(generator_input(), verbose=0, steps=1, max_queue_size=10, workers=1)
    return k.eval(m.triangle_convertion(eval_model))


def training():  # Train the neural network to reach a similar distribution
    k.clear_session()  # Removes all nodes which could be left from the previous training
    temp_model = nn.model
    temp_model.compile(loss=loss, optimizer=optimizer(), metrics=[])  # Compile the model for the training

    # Fits the model on the data yielded for every batch by a python generator
    temp_model.fit(generator_input_output(), steps_per_epoch=nn.n_samples, epochs=1, verbose=1,
                   validation_data=generator_input_output(), validation_steps=nn.n_val_samples)

    loss_tested = temp_model.evaluate(generator_input_output(), steps=nn.n_test_samples)  # Testing the loss of the nn
    print("Loss (with test samples): ", loss_tested)

    return evaluation(temp_model)


def plot_result(result, distribution):  # Plot the founded distances

    if (d.get_name() == 'fritz' or d.get_name() == 'LLL') and len(d.get_para_range()) == 1: # Plot founded min distances
        print('Distance: ', np_discrepancy(result, target=distribution, dist_mod='ec'))
        plt.clf()
        fig = plt.figure(figsize=(7, 5))
        plt.plot(np.linspace(1, nn.n_of_min_samples, num=10), nn.find_min_distances, 'rh')
        plt.title('Distance between target distribution and the computed distribution')
        plt.ylim(bottom=np.amin(nn.find_min_distances) * 0.8, top=np.amax(nn.find_min_distances) * 1.1)
        plt.savefig('./result_plot_sweep/sweep_min_distances_ ' + d.get_name() + '.png')

    elif (d.get_name() == 'Fritz' or d.get_name() == 'LLL' or d.get_name() == 'LLL_random_noise') and \
            len(d.get_para_range()) > 1:  # Plot the founded distances with their corresponding parameter
        for i in range(s.Distribution().para_range.shape[0]):
            nn.distances1[i] = np_discrepancy(result[i], target=distribution[i])
            nn.distances2[i] = np_discrepancy(result[i], target=distribution[i], dist_mod='ec')

        print(nn.distances1)
        print(nn.distances2)

        plt.clf()  # Plot (euclid) distances
        fig = plt.figure(figsize=(16, 9))
        plt.plot(s.Distribution().para_range, nn.distances2, 'rh')
        plt.title('Distance between target distribution and the computed distribution')
        plt.ylim(bottom=0, top=np.amax(nn.distances2) * 1.2)
        plt.savefig('./result_plot_sweep/sweep_parameter_ ' + d.get_name() + '.png')

        if nn.smooth_curve:  # Same plot but with a smoothed curve
            for i in range(s.Distribution().para_range.shape[0]):
                nn.smoothed_euclid_distances[i] = np_discrepancy(result[i], target=distribution[i], dist_mod='ec')

            nn.smoothed_euclid_distances = gaussian_filter1d(nn.smoothed_euclid_distances, sigma=1)
            print(nn.smoothed_euclid_distances)

            plt.clf()  # Plot (euclid) distances
            fig = plt.figure(figsize=(16, 9))
            plt.plot(s.Distribution().para_range, nn.smoothed_euclid_distances, 'rh')
            plt.title('Distance between target distribution and the computed distribution')
            plt.ylim(bottom=0, top=np.amax(nn.smoothed_euclid_distances) * 1.2)
            plt.savefig('./result_plot_sweep/sweep_parameter_smoothed_ ' + d.get_name() + '.png')

            #Später löschen
            for i in range(s.Distribution().para_range.shape[0]):
                nn.smoothed_euclid_distances[i] = np_discrepancy(result[i], target=distribution[i], dist_mod='ec')

            nn.smoothed_euclid_distances = gaussian_filter1d(nn.smoothed_euclid_distances, sigma=0.1)
            print(nn.smoothed_euclid_distances)

            plt.clf()  # Plot (euclid) distances
            fig = plt.figure(figsize=(16, 9))
            plt.plot(s.Distribution().para_range, nn.smoothed_euclid_distances, 'rh')
            plt.title('Distance between target distribution and the computed distribution')
            plt.ylim(bottom=0, top=np.amax(nn.smoothed_euclid_distances) * 1.2)
            plt.savefig('./result_plot_sweep/sweep_parameter_smoothed_small_ ' + d.get_name() + '.png')


def new_neural_network(model, distribution):  # Initialise the parameters for the neural network

    nn.model = model  # Save the 'model' and 'distribution' (as 'out_target') in settings for further use
    nn.out_target = distribution
    nn.out_target_batchsize = np.array([distribution for _ in range(nn.batch_size)])

    #nn.save()


def set_find_min_distances(f_m_d):
    nn.find_min_distances = f_m_d


def single_try(distribution, state, modus):  # Try to calculate the given distribution for two given states

    connection_between_states = np.full((4, 8), -1)  # Here to know which states are correlated (entangled) or not
    for a, b, c in product('0123', repeat=3):
        a, b, c = int(a), int(b), int(c)

        target = distribution[a, b, c]
        if target != 0:
            i = 0
            while connection_between_states[c, i] != -1:
                i += 2
            connection_between_states[c, i] = a
            connection_between_states[c, i + 1] = b

    for i in range(connection_between_states.shape[0]):
        if min(connection_between_states[i]) == -1:
            connection_between_states[i, 4] = connection_between_states[i, 0]
            connection_between_states[i, 7] = connection_between_states[i, 1]
            connection_between_states[i, 6] = connection_between_states[i, 2]
            connection_between_states[i, 5] = connection_between_states[i, 3]

    d.set_connection_between_states(connection_between_states)
    print(connection_between_states)

    alice, bob, charlie = d.create_starting_point(state, modus)

    dist = np.zeros((4, 4, 4))
    for a, b, c in product('0123', repeat=3):
        a, b, c = int(a), int(b), int(c)

        target = distribution[a, b, c]
        if target == 0:
            if np.all(alice[a, 0:3] == charlie[c, 0:3]) and np.all(bob[b, 0:3] == charlie[c, 3:6]) and modus != 2:
                if d.control_connections(alice, bob, charlie):
                    print("Error, the rule 'connection_between_states' is not correctly calculated")
                else:
                    print("Error, the rule 'connection_between_states' is not correctly obeyed by the program")
            continue

        alice, bob, charlie = d.try_different_combination_of_states(alice, bob, charlie, a, b, c)
        res = d.single_quantum_computation(alice[a], bob[b])
        dist[a, b, c] = res

    return alice, bob, charlie, dist


def find_used_states_for_distribution(distribution):  # Try to find the used states (of Alice, Bob and Charlie)
    possible_matrices = ['X', 'Y', 'Z']  # List to get all possible matrices (Pauli matrices)

    for i in range(3):
        print('Start, modus: ', i)  # The modus is used to know which type of combination of matrices is used
        alice, bob, charlie, dist = single_try(distribution, possible_matrices, i)

        for a, b, c in product('0123', repeat=3):  # Control if the founded states reproduce the correct distribution
            a, b, c = int(a), int(b), int(c)
            if round(dist[a, b, c], 10) != round(distribution[a, b, c], 10):
                print('\n-----------Next try-----------\n')
                break
            if a == b == c == 3:
                print("Founded possible states. alice:\n", alice, "\nbob:\n", bob, "\ncharlie:\n", charlie)
                return dist

    print('No possible states founded to reproduce the given distribution')
    return None
