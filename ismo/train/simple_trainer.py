from ismo.train import Parameters
import ismo.train.optimizers
import h5py
import keras.callbacks
import copy
import numpy as np


class SimpleTrainer(object):

    def __init__(self, *, training_parameters: Parameters,
                 model):

        self.retrainings = training_parameters.retrainings

        self.optimizer = training_parameters.optimizer
        self.learning_rate = training_parameters.learning_rate

        self.epochs = training_parameters.epochs

        self.model = model

        self.loss = training_parameters.loss

        self.callbacks = []
        if training_parameters.should_use_early_stopping:
            self.callbacks.append(
                keras.callbacks.EarlyStopping(monitor='loss', patience=training_parameters.early_stopping_patience))

    def fit(self, parameters, values):
        best_loss = None
        for retraining_number in range(self.retrainings):
            self.__reinitialize(self.model)

            hist = self.model.fit(parameters, values, batch_size=parameters.shape[0],
                                  epochs=self.epochs, verbose=0, callbacks=self.callbacks)

            loss = hist.history['loss'][-1]
            if best_loss is None or loss < best_loss:
                best_weights = copy.deepcopy(self.model.get_weights())

        self.model.set_weights(best_weights)

    def save_to_file(self, outputname):
        self.model.save(outputname)

    def __reinitialize(self, model):
        weights = model.get_weights()
        weights_new = []

        for array in weights:
            weights_new.append(np.random.uniform(0, 1, array.shape))

        model.set_weights(weights_new)

        model.compile(optimizer=ismo.train.optimizers.create_optimizer(self.optimizer,
                                                                       self.learning_rate),
                      loss=self.loss)
