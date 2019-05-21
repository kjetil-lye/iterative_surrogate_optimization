import iso.optimizers

def create_optimizer(name):
    return iso.optimizers.NumpyOptimizer(name)