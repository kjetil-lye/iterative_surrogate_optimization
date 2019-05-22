import numpy as np


class Objective(object):
    def __init__(self, sin_penalty=10, cos_penalty=10, tan_penalty=10):
        self.sin_penalty = sin_penalty
        self.cos_penalty = cos_penalty
        self.tan_penalty = tan_penalty

    def __call__(self, x):
        return self.sin_penalty * x[0] ** 2 + self.cos_penalty * x[1] + self.tan_penalty * x[2]

    def grad(self, x):
        return np.array([2 * x[0], 1, 1])
