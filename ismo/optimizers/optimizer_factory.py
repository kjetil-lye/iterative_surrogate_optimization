import ismo.optimizers

def create_optimizer(name):
    return ismo.optimizers.NumpyOptimizer(name)