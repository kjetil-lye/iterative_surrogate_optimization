from ismo.train import Parameters
import ismo.train.optimizers
import h5py
import keras.callbacks


class SimpleTrainer(object):

    def __init__(self, *, training_parameters: Parameters,
                 model):
        model.compile(optimizer=ismo.train.optimizers.create_optimizer(training_parameters.optimizer,
                                                                      training_parameters.learning_rate),
                      loss=training_parameters.loss)

        self.epochs = training_parameters.epochs

        self.model = model

        self.callbacks = []
        if training_parameters.should_use_early_stopping:
            self.callbacks.append(keras.callbacks.EarlyStopping(monitor='loss', patience=training_parameters.early_stopping_patience))

    def fit(self, parameters, values):
        self.model.fit(parameters, values, batch_size=parameters.shape[0],
                  epochs=self.epochs, verbose=0, callbacks=self.callbacks)

    def save_to_file(self, outputname):
        self.model.save(outputname)