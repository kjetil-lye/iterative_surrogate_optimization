from ismo.train import Parameters
import ismo.train.optimizers
import h5py
import keras.callbacks
import copy
import numpy as np
import keras.initializers

import tf.compat.v1.keras.backend as K


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

        self.writers = []

    def fit(self, parameters, values):
        best_loss = None
        for retraining_number in range(self.retrainings):
            self.__reinitialize(self.model)

            hist = self.model.fit(parameters, values, batch_size=parameters.shape[0],
                                  epochs=self.epochs, verbose=0, callbacks=self.callbacks)

            loss = hist.history['loss'][-1]
            if best_loss is None or loss < best_loss:
                best_weights = copy.deepcopy(self.model.get_weights())
                best_loss_hist = copy.deepcopy(hist)

        self.model.set_weights(best_weights)

        self.write_best_loss_history(best_loss_hist)

    def write_best_loss_history(self, loss_history):
        for writer in self.writers:
            writer(loss_history)

    def add_loss_history_writer(self, writer):
        self.writers.append(writer)

    def save_to_file(self, outputname):
        self.model.save(outputname)

    def __reinitialize(self, model):
        # See https://stackoverflow.com/a/51727616 (with some modifications, does not run out of the box)
        session = K.get_session()
        for layer in model.layers:
            weights = np.zeros_like(layer.get_weights()[0])
            biases = np.zeros_like(layer.get_weights()[1])

            if hasattr(layer, 'kernel_initializer'):
                weights = session.run(layer.kernel_initializer(weights.shape))

            if hasattr(layer, 'bias_initializer'):
                biases = session.run(layer.bias_initializer(biases.shape))


            layer.set_weights((weights, biases))

        model.compile(optimizer=ismo.train.optimizers.create_optimizer(self.optimizer,
                                                                       self.learning_rate),
                      loss=self.loss)
