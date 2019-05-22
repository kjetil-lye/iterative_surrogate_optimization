from.ismo.samples import MonteCarlo, Sobol

class SampleGeneratorFactory(object):

    def __init__(self):
        self.known_names = {
            'monte-carlo' : MonteCarlo,
            'sobol' : Sobol
        }

    def create_sample_generator(self, name):
        return self.known_names[name]()


def create_sample_generator(name):
    return SampleGeneratorFactory().create_sample_generator(name)
