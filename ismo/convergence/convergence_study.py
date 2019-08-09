import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import ismo.iterative_surrogate_model_optimization
import ismo.train.trainer_factory
import ismo.train.multivariate_trainer
import ismo.samples.sample_generator_factory
import ismo.optimizers
import matplotlib.pyplot as plt
import collections


class LossWriter:
    def __init__(self, basename):
        self.basename = basename
        self.iteration = 0

    def __call__(self, loss):
        np.save(f'{self.basename}_iteration_{self.iteration}.npy', loss.history['loss'])
        self.iteration += 1


def convergence_study(
        *,
        generator_name,
        training_parameter_filename,
        optimizer_name,
        retries,
        save_result,
        prefix,
        with_competitor,
        dimension,
        number_of_variables,
        number_of_samples_per_iteration,
        simulator_creator,
        objective,
        variable_names,
        save_plot=lambda name: plt.savefig(f'{name}.png')
    ):
    all_values_min = collections.defaultdict(list)

    samples_as_str = "_".join(map(str, number_of_samples_per_iteration))
    for try_number in range(retries):
        print(f"try_number: {try_number}")
        generator = ismo.samples.create_sample_generator(generator_name)

        optimizer = ismo.optimizers.create_optimizer(optimizer_name)

        trainers = [ismo.train.create_trainer_from_simple_file(training_parameter_filename) for _ in
                    range(number_of_variables)]

        for var_index, trainer in enumerate(trainers):
            trainer.add_loss_history_writer(LossWriter(f'{prefix}loss_var_{var_index}_try_{try_number}'))
        trainer = ismo.train.MultiVariateTrainer(trainers)

        starting_sample = try_number * sum(number_of_samples_per_iteration)
        parameters, values = ismo.iterative_surrogate_model_optimization(
            number_of_samples_per_iteration=number_of_samples_per_iteration,
            sample_generator=generator,
            trainer=trainer,
            optimizer=optimizer,
            simulator=simulator_creator(starting_sample),
            objective_function=objective,
            dimension=dimension,
            starting_sample=starting_sample
        )
        values = np.array(values)
        objective_values = [objective(values[i, :]) for i in range(values.shape[0])]

        per_iteration = collections.defaultdict(list)

        total_number_of_samples = 0
        for number_of_samples in number_of_samples_per_iteration:
            total_number_of_samples += number_of_samples
            arg_min = np.argmin(objective_values[:total_number_of_samples])
            for n, name in enumerate(variable_names):
                per_iteration[name].append(values[arg_min, n])
            per_iteration['objective'].append(objective_values[arg_min])

        for func_name, func_values in per_iteration.items():
            all_values_min[func_name].append(per_iteration[func_name])

        if save_result:
            np.savetxt(f'{prefix}parameters_{try_number}_samples_{samples_as_str}.txt', parameters)
            np.savetxt(f'{prefix}values_{try_number}_samples_{samples_as_str}.txt', values)
            np.savetxt(f'{prefix}objective_values_{try_number}_samples_{samples_as_str}.txt', objective_values)

    if with_competitor:
        competitor_min_values = collections.defaultdict(
            lambda: np.zeros((retries, len(number_of_samples_per_iteration) - 1)))
        for try_number in range(retries):

            print(f"try_number (competitor): {try_number}")

            for iteration_number, number_of_samples_post in enumerate(number_of_samples_per_iteration[1:]):
                number_of_samples = sum(number_of_samples_per_iteration[:iteration_number + 1])
                generator = ismo.samples.create_sample_generator(generator_name)

                optimizer = ismo.optimizers.create_optimizer(optimizer_name)

                trainers = [ismo.train.create_trainer_from_simple_file(training_parameter_filename) for _ in
                            range(number_of_variables)]

                for var_index, trainer in enumerate(trainers):
                    trainer.add_loss_history_writer(LossWriter(
                        f'{prefix}loss_competitor_var_{var_index}_iteration_{iteration_number}_try_{try_number}'))
                trainer = ismo.train.MultiVariateTrainer(trainers)

                starting_sample = try_number * (number_of_samples_post + number_of_samples)
                parameters, values = ismo.iterative_surrogate_model_optimization(
                    number_of_samples_per_iteration=[number_of_samples, number_of_samples_post],
                    sample_generator=generator,
                    trainer=trainer,
                    optimizer=optimizer,
                    simulator=simulator_creator(starting_sample),
                    objective_function=objective,
                    dimension=dimension,
                    starting_sample=starting_sample
                )
                values = np.array(values)
                objective_values = [objective(values[i, :]) for i in range(values.shape[0])]

                arg_min = np.argmin(objective_values)

                competitor_min_values['objective'][try_number, iteration_number] = objective_values[arg_min]

                for n, name in enumerate(['lift', 'drag', 'area']):
                    competitor_min_values[name][try_number, iteration_number] = values[arg_min, n]

                if save_result:
                    np.savetxt(
                        f'{prefix}competitor_parameters_{try_number}_it_{iteration_number}_samples_{samples_as_str}.txt',
                        parameters)
                    np.savetxt(
                        f'{prefix}competitor_values_{try_number}_it_{iteration_number}_samples_{samples_as_str}.txt',
                        values)
                    np.savetxt(
                        f'{prefix}competitor_objective_values_{try_number}_it_{iteration_number}_samples_{samples_as_str}.txt',
                        objective_values)

    print("Done!")
    iterations = np.arange(0, len(number_of_samples_per_iteration))
    for name, values in all_values_min.items():
        plt.errorbar(iterations, np.mean(values, 0),
                     yerr=np.std(values, 0).flatten(), fmt='o',
                     label='ISMO')

        if with_competitor:
            plt.errorbar(iterations[:-1], np.mean(competitor_min_values[name], 0),
                         yerr=np.std(competitor_min_values[name], 0).flatten(), fmt='*',
                         label='DNN+Opt')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Min value')
        plt.title(name)

        save_plot(f'{prefix}optimized_value_{name}_{samples_as_str}')
        plt.close('all')
