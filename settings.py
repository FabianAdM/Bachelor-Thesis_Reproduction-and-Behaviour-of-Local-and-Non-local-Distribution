import numpy as np
import pickle


class NeuralNetwork:  # All necessary parameters and settings for the file <<neural_network>>
    def __init__(self):
        self.model = None  # The 'model', 'out_target' and 'out_target_batchsize' will be defined after a model and an
        self.out_target = None  # according distribution get chosen
        self.out_target_batchsize = None  # 'out_target' should have batch_size dimension in supervised training as well

        self.optimizer = 'adam'  # Defining an optimizer and below the necessary parameters for the training
        self.lr = 1  # Learning rate of the optimizer
        self.rho = 0.95  # Additional parameter for the 'optimizer'
        self.epsilon = 1e-07  # Used for numerical stability, so just a small constant
        self.decay = 0.001  # Additional parameter for the 'optimizer'
        self.momentum = 0.25  # Used to damp any oscillations

        self.batch_size = 6000  # Set of examples used in one iteration
        self.n_samples = 10000  # How many batches to go through during training.
        self.n_val_samples = 1000  # Define the number of batches to go through in each validation step.
        self.n_test_samples = 10  # Used to test the final loss
        self.training_mean = 0.5  ##! Try out different starts
        self.training_sigma = np.sqrt(1 / 12)

        self.loss_mode = 'lc'  # Chose which kind of calculation for the distance will be used

        self.find_min_result = True  # Calculate over several samples in order to get the min result
        self.n_of_min_samples = 10
        self.find_min_distances = np.zeros(self.n_of_min_samples)  # Sum of all founded distances in the last subrounds
        self.distances1 = np.zeros_like(Distribution().para_range)  # Summarise the founded distances
        self.distances2 = np.zeros_like(Distribution().para_range)  # Summarise the founded distances (different kind)

        self.smooth_curve = True  # Smooth the curve for the distances for the 'Fritz' distribution
        self.smoothed_euclid_distances = np.zeros_like(Distribution().para_range)

    def save(self):
        with open('./saved_settings/neural_network_para', 'wb') as f:
            pickle.dump(self, f)


class Model:  # All necessary parameters and settings for the file <<model>>
    def __init__(self):
        self.name = None
        self.outputsize = None
        self.latin_depth = 1  # Neural network depth
        self.latin_width = 4  # Neural network width

        self.greek_depth = 0  # ??????????set to 0 if trivial neural networks at sources
        self.greek_width = 1
        self.activ = 'tanh'  # Activation for the layers in the neural network
        self.activ_last_layer = 'softmax'  # Activation for last layer in the neural network
        self.kernel_reg = None
        self.weight_init_scaling = 2  # ????????

    def save(self, name):
        with open('./saved_settings/' + name, 'wb') as f:
            pickle.dump(self, f)


class Distribution:  # All necessary parameters and settings for the file <<distribution>>
    def __init__(self):
        self.name = None
        self.qubits = np.zeros((4, 4, 4)).astype(str)  # The according to the distribution used binary/quaternary states

        self.para_range = np.linspace(0, 0.2, num=21)  # Define the range of parameters used for the distributions

        self.swap_eigenvector = False  # Use other eigenvector for the measurement operator output
        self.connection_between_states = None  # Summarise the correlated states
        self.target = None  # Set the distribution which the program has to reproduce it

    def save(self, name):
        with open('./saved_settings/' + name, 'wb') as f:
            pickle.dump(self, f)


def load_settings(name):
    with open('./saved_settings/' + name, 'rb') as f:
        temp = pickle.load(f)
    return temp
