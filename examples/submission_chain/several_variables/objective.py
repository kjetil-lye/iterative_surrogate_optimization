import numpy as np


class Objective(object):
    def __call__(self, x):
        return x[0]**2 + x[1]+x[2]

    def grad(self, x):
        return np.array([2*x[0], 1, 1])