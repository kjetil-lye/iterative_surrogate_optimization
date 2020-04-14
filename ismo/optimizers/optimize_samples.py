import numpy as np
def make_bounds(bounds, starting_value):
    if bounds is None:
        return None

    if type(starting_value) == float or len(starting_value.shape) == 0:
        return [bounds]

    bounds = np.array(bounds)
    if len(bounds.shape) == 2 and bounds.shape[0] == starting_value.shape[0] and bounds.shape[1] == 2:
        return bounds

    new_bounds = []
    for k in range(starting_value.shape[0]):
        new_bounds.append(bounds)

    return new_bounds

def optimize_samples(*,
                     starting_values,
                     J,
                     optimizer,
                     bounds=[0,1]):

    optimized_parameters = []

    optimization_results = []
    for starting_value in starting_values:
        x, y, results = optimizer(F = J,
                         DF = lambda x: J.grad(x),
                         x0 = starting_value,
                         bounds = make_bounds(bounds, starting_value))
        optimization_results.append(results)
        optimized_parameters.append(x)

    return np.array(optimized_parameters), optimization_results






