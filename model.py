import os
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda

import settings as s
import distribution as d
os.environ["PATH"] += os.pathsep + 'D:/Programme/Pycharm/Graphviz/bin'
m = s.Model()


def triangle():
    if d.get_name() == 'fritz':
        m.outputsize = 4
    elif d.get_name() == 'LLL' or d.get_name() == 'LLL_random_noise':
        m.outputsize = 2
    input_tensor = Input(3, )  # Number of hidden variables, i.e. alpha, beta, gamma

    # Group input tensor according to whether alpha, beta or gamma hidden variable.
    group_alpha = Lambda(lambda x: x[:, :1], output_shape=((1,)))(input_tensor)
    group_beta = Lambda(lambda x: x[:, 1:2], output_shape=((1,)))(input_tensor)
    group_gamma = Lambda(lambda x: x[:, 2:3], output_shape=((1,)))(input_tensor)

    # Neural network at the sources, for pre-processing (e.g. for going from uniform distribution to non-uniform one)
    ## Note that in the example code greek_depth is set to 0, so this part is trivial.
    for _ in range(m.greek_depth):
        group_alpha = Dense(m.greek_width, activation=m.activ, kernel_regularizer=m.kernel_reg)(
            group_alpha)
        group_beta = Dense(m.greek_width, activation=m.activ, kernel_regularizer=m.kernel_reg)(
            group_beta)
        group_gamma = Dense(m.greek_width, activation=m.activ, kernel_regularizer=m.kernel_reg)(
            group_gamma)

    # Route hidden variables to visibile parties Alice, Bob and Charlie
    group_a = Concatenate()([group_beta, group_gamma])
    group_b = Concatenate()([group_gamma, group_alpha])
    group_c = Concatenate()([group_alpha, group_beta])

    # Neural network at the parties Alice, Bob and Charlie.
    ## Note: increasing the variance of the initialization seemed to help in some cases, especially when the number if outputs per party is 4 or more.
    kernel_init = tf.keras.initializers.VarianceScaling(scale=m.weight_init_scaling, mode='fan_in',
                                                        distribution='truncated_normal', seed=None)
    for _ in range(m.latin_depth):
        group_a = Dense(m.latin_width, activation=m.activ, kernel_regularizer=m.kernel_reg,
                        kernel_initializer=kernel_init)(group_a)
        group_b = Dense(m.latin_width, activation=m.activ, kernel_regularizer=m.kernel_reg,
                        kernel_initializer=kernel_init)(group_b)
        group_c = Dense(m.latin_width, activation=m.activ, kernel_regularizer=m.kernel_reg,
                        kernel_initializer=kernel_init)(group_c)

    # Apply final softmax layer
    group_a = Dense(m.outputsize, activation=m.activ_last_layer, kernel_regularizer=m.kernel_reg)(group_a)
    group_b = Dense(m.outputsize, activation=m.activ_last_layer, kernel_regularizer=m.kernel_reg)(group_b)
    group_c = Dense(m.outputsize, activation=m.activ_last_layer, kernel_regularizer=m.kernel_reg)(group_c)

    output_tensor = Concatenate()([group_a, group_b, group_c])

    model = Model(input_tensor, output_tensor)
    model.summary()
    return model


# assimilated
def triangle_convertion(out_predicted):  # Convert the output 'out_predicted' of the neural network to a probability vector
    """ Converts the output of the neural network to a probability vector.
    That is from a shape of (batch_size, a_outputsize + b_outputsize + c_outputsize) to a shape of (a_outputsize * b_outputsize * c_outputsize,)
    """
    a_probs = out_predicted[:, 0: m.outputsize]
    b_probs = out_predicted[:, m.outputsize: 2 * m.outputsize]
    c_probs = out_predicted[:, 2 * m.outputsize: 3 * m.outputsize]

    a_probs = K.reshape(a_probs, (-1, m.outputsize, 1, 1))
    b_probs = K.reshape(b_probs, (-1, 1, m.outputsize, 1))
    c_probs = K.reshape(c_probs, (-1, 1, 1, m.outputsize))

    probs = a_probs * b_probs * c_probs
    probs = K.mean(probs, axis=0)
    probs = K.flatten(probs)
    return probs


def new_model(model, distribution):  # Initialise the parameters for the models

    distribution = d.new_distribution(distribution)

    m.save(str(model))
    m.name = model

    if model == 'triangle':
        m.inputsize = 3
        return triangle(), distribution
    elif model is None:
        return None, distribution
    else:
        print('Model not defined, pls define it first in the file <<model.py>> or use an existing model')
        return None


def get_name():
    return m.name


def print_model(model):
    tf.keras.utils.plot_model(model, to_file='./models_plot/model_'+m.name+'.png')
