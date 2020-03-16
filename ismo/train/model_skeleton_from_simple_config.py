import tensorflow.keras
import tensorflow.keras.models
import tensorflow.keras.layers
import tensorflow.keras.regularizers
import json


def model_skeleton_from_simple_config_file(config_filename):
    with open(config_filename) as f:

        configuration = json.load(f)
        return model_skeleton_from_simple_config(configuration)


def model_skeleton_from_simple_config(configuration):
    activation = configuration['activation']

    if 'l1_regularization' in configuration.keys():
        regularization_l1 = configuration['l1_regularization']
        regularizer = tensorflow.keras.regularizers.l1(regularization_l1)
    if 'l2_regularization' in configuration.keys():
        regularization_l2 = configuration['l2_regularization']
        regularizer = tensorflow.keras.regularizers.l2(regularization_l2)

    else:
        regularizer = None

    network_topology = configuration['network_topology']

    model = tensorflow.keras.models.Sequential()

    model.add(tensorflow.keras.layers.Dense(network_topology[1],
                                 input_shape=(network_topology[0],),
                                 activation=activation,
                                 kernel_regularizer=regularizer))
    for layer in network_topology[2:-1]:
        model.add(tensorflow.keras.layers.Dense(layer, activation=configuration['activation'],
                                     kernel_regularizer=regularizer))

    model.add(tensorflow.keras.layers.Dense(network_topology[-1]))

    return model