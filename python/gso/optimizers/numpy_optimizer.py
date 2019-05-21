import scipy.optimize

class NumpyOptimizer(object):
    def __init__(self, method_name):
        self.method_name = method_name

    def __call__(self, *, F, DF, x0, bounds):
        optimization_result = scipy.optimize.minimize(F, x0, jac=DF, method=self.method_name,
                                                      bounds=bounds)

        y = F(optimization_result.x)

        return optimization_result.x, y