from ismo.train import Parameters
import ismo.train.optimizers
import h5py

class SimpleTrainer(object):

    def __init__(self, *, training_parameters: Parameters,
                 model):
        model.compile(optimizer=ismo.train.optimizers.create_optimizer(training_parameters.optimizer,
                                                                      training_parameters.learning_rate),
                      loss=training_parameters.loss)

        self.epochs = training_parameters.epochs

        self.model = model

    def fit(self, parameters, values):
        self.model.fit(parameters, values, batch_size=parameters.shape[0],
                  epochs=self.epochs)

    def save_to_file(self, outputname):
        self.model.save(outputname)