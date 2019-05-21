import keras.optimizers


def create_optimizer(name : str, learning_rate : float):

    if name.lower() == 'sgd':
        return keras.optimizers.SGD(lr=learning_rate)
    elif name.lower() == 'adam':
        return keras.optimizers.Adam(lr=learning_rate)

    else:
        raise Exception("Unknown optimizer {}.".format(name))