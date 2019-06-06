import json
import typing


class Parameters(typing.NamedTuple):
    epochs: int
    optimizer: str
    loss: str
    should_use_early_stopping: bool
    early_stopping_patience: int
    learning_rate: float
    retrainings: int


def parameters_from_simple_config_file(configuration_filename):
    with open(configuration_filename) as f:
        config = json.load(f)



        return Parameters(epochs=config['epochs'],
                          optimizer=config['optimizer'],
                          loss=config['loss'],
                          should_use_early_stopping=config['should_use_early_stopping'],
                          early_stopping_patience=config['early_stopping_patience'],
                          learning_rate=config['learning_rate'],
                          retrainings=config['retrainings'])