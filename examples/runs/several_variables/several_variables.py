import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import ismo.iterative_surrogate_model_optimization
import ismo.train.trainer_factory
import ismo.train.multivariate_trainer
import ismo.samples.sample_generator_factory
import ismo.optimizers
import matplotlib.pyplot as plt

def evolve(parameters):
    values = np.zeros((parameters.shape[0], 3))

    for k in range(parameters.shape[0]):
        values[k, 0] = np.sin(parameters[k, 2] + parameters[k, 0])
        values[k, 1] = np.cos(parameters[k, 19] + parameters[k, 5])
        values[k, 2] = np.tan(parameters[k, 4] + parameters[k, 8])

    return values


class Objective(object):
    def __init__(self, sin_penalty=10, cos_penalty=10, tan_penalty=10):
        self.sin_penalty = sin_penalty
        self.cos_penalty = cos_penalty
        self.tan_penalty = tan_penalty

    def __call__(self, x):
        return self.sin_penalty * x[0] ** 2 + self.cos_penalty * x[1] + self.tan_penalty * x[2]

    def grad(self, x):
        return np.array([2 * x[0], 1, 1])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""
A multiple variable function (does not have any real use, just tests that we can do multiple variables)
        """)

    parser.add_argument('--number_of_samples_per_iteration', type=int, nargs='+', default=[16, 4, 4, 4, 4, 4],
                        help='Number of samples per iteration')

    parser.add_argument('--generator', type=str, default='monte-carlo',
                        help='Generator')

    parser.add_argument('--simple_configuration_file', type=str, default='training_parameters.json',
                        help='Configuration of training and network')

    parser.add_argument('--optimizer', type=str, default='L-BFGS-B',
                        help='Configuration of training and network')

    parser.add_argument('--retries', type=int, default=1,
                        help='Number of retries (to get mean/variance). This option is studying how well it works over multiple runs.')

    parser.add_argument('--save_result', action='store_true',
                        help='Save the result to file')

    args = parser.parse_args()


    all_values_min = []

    for try_number in range(args.retries):
        print(f"try_number: {try_number}")
        generator = ismo.samples.create_sample_generator(args.generator)

        optimizer = ismo.optimizers.create_optimizer(args.optimizer)

        trainer = ismo.train.MultiVariateTrainer(
            [ismo.train.create_trainer_from_simple_file(args.simple_configuration_file) for k in range(3)])

        parameters, values = ismo.iterative_surrogate_model_optimization(
            number_of_samples_per_iteration=args.number_of_samples_per_iteration,
            sample_generator=generator,
            trainer=trainer,
            optimizer=optimizer,
            simulator=evolve,
            objective_function=Objective(),
            dimension=20,
            starting_sample=try_number*sum(args.number_of_samples_per_iteration))


        per_iteration = []
        total_number_of_samples = 0
        for number_of_samples in args.number_of_samples_per_iteration:
            total_number_of_samples += number_of_samples
            per_iteration.append(np.min(values[:total_number_of_samples]))
        all_values_min.append(per_iteration)

        if args.save_result:
            np.savetxt(f'parameters_{try_number}_{iteration}.txt', parameters)
            np.savetxt(f'values_{try_number}_{iteration}.txt', values)
    print("Done!")
    iterations = np.arange(0, len(args.number_of_samples_per_iteration))
    plt.errorbar(iterations, np.mean(all_values_min,0), yerr=np.std(all_values_min,0), fmt='o')
    plt.xlabel('Iteration')
    plt.ylabel('Min value')
    plt.show()
