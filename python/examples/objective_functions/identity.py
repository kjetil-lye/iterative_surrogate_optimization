import numpy


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return x

    def grad(self, x):
        return numpy.ones_like(x)