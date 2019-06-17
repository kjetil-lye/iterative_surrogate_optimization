import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import ismo.iterative_surrogate_model_optimization
import ismo.train.trainer_factory
import ismo.train.multivariate_trainer
import ismo.samples.sample_generator_factory
import ismo.optimizers
import matplotlib.pyplot as plt

class Objective:
    def __call__(self, x):
        return x

    def grad(self, x):
        return np.ones_like(x)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""
Runs the function sin(4*pi*x) on the input parameters
        """)

    parser.add_argument('--number_of_samples_per_iteration', type=int, nargs='+', default=[16, 4, 4, 4, 4, 4],
                        help='Number of samples per iteration')

    parser.add_argument('--generator', type=str, default='monte-carlo',
                        help='Generator')

    parser.add_argument('--simple_configuration_file', type=str, default='training_parameters.json',
                        help='Configuration of training and network')

    parser.add_argument('--optimizer', type=str, default='L-BFGS-B',
                        help='Configuration of training and network')

    args = parser.parse_args()

    generator = ismo.samples.create_sample_generator(args.generator)

    trainer = ismo.train.MultiVariateTrainer([ismo.train.create_trainer_from_simple_file(args.simple_configuration_file)])

    optimizer = ismo.optimizers.create_optimizer(args.optimizer)

    parameters, values = ismo.iterative_surrogate_model_optimization(
        number_of_samples_per_iteration=args.number_of_samples_per_iteration,
        sample_generator=generator,
        trainer=trainer,
        optimizer=optimizer,
        simulator=lambda x: np.sin(4 * np.pi * x),
        objective_function=Objective(),
        dimension=1)

    per_iteration = []
    total_number_of_samples = 0
    for number_of_samples in args.number_of_samples_per_iteration:
        total_number_of_samples += number_of_samples
        per_iteration.append(values[:total_number_of_samples])

    plt.plot(per_iteration, 'o')
    plt.xlabel('Iteration')
    plt.ylabel('Min value')
    plt.show()
