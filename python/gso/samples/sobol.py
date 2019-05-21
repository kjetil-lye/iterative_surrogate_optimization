import sobol
import numpy as np
class Sobol(object):



    def __call__(self, number_of_samples,
                 dimension,
                 start=0):

        samples = np.zeros((number_of_samples, dimension))

        for sample in range(number_of_samples):
            samples[sample,: ] = sobol.i4_sobol(dimension, start + sample)[0]

        return samples

