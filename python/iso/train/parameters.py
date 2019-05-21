import json

class Parameters(object):
    def __init__(self, *,
                 epochs,
                 optimizer,
                 loss,
                 should_use_early_stopping,
                 early_stopping_patience,
                 learning_rate
                 ):

        self.optimizer = optimizer
        self.epochs = epochs
        self.loss = loss
        self.should_use_early_stopping = should_use_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate = learning_rate


def parameters_from_simple_config_file(configuration_filename):
    with open(configuration_filename) as f:
        config = json.load(f)

        return Parameters(epochs=config['epochs'],
                          optimizer=config['optimizer'],
                          loss=config['loss'],
                          should_use_early_stopping=config['should_use_early_stopping'],
                          early_stopping_patience=config['early_stopping_patience'],
                          learning_rate=config['learning_rate'])