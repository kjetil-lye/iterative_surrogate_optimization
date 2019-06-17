class MultiVariateTrainer:
    def __init__(self, trainers):
        self._trainers = trainers

    def fit(self, parameters, values):
        for n, trainer in enumerate(self._trainers):
            trainer.fit(parameters, values[:,n])


    @property
    def models(self):
        return [trainer.model for trainer in self._trainers]