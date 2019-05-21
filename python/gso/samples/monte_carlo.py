import numpy as np

class MonteCarlo(object):



    def __call__(self, number_of_samples,
                 dimension,
                 start=0):

        # Make sure it is reproducible
        np.random.seed(0)

        for _ in range(start):
            np.random.uniform(0, 1, dimension)


        return np.random.uniform(0, 1, (number_of_samples, dimension))

