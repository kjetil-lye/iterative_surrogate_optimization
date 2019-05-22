import numpy as np


class Objective(object):
    def __call__(self, x):
        return x

    def grad(self, x):
        return np.ones_like(x)