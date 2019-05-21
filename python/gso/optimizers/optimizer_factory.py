import gso.optimizers

def create_optimizer(name):
    return gso.optimizers.NumpyOptimizer(name)