import ismo.optimizers
import ismo.objective_function
import numpy as np


def iterative_surrogate_model_optimization(*, sample_generator,
                                           number_of_samples_per_iteration,
                                           simulator,
                                           trainer,
                                           optimizer,
                                           objective_function,
                                           dimension,
                                           starting_sample=0
                                           ):
    all_parameters = []
    all_values = []
    sample_start = starting_sample
    all_parameters.extend(sample_generator(number_of_samples=number_of_samples_per_iteration[0], dimension=dimension, start=sample_start))

    all_values.extend(simulator(np.array(all_parameters)))

    sample_start += number_of_samples_per_iteration[0]

    for n, number_of_samples in enumerate(number_of_samples_per_iteration[1:]):

        trainer.fit(np.array(all_parameters), np.array(all_values))

        new_parameters = sample_generator(number_of_samples=number_of_samples, dimension=dimension, start=sample_start)
        objective_function_dnn = ismo.objective_function.DNNObjectiveFunction(trainer.models, objective_function)
        optimized_parameters, _ = ismo.optimizers.optimize_samples(starting_values=new_parameters,
                                                                J=objective_function_dnn,
                                                                optimizer=optimizer)

        all_parameters.extend(optimized_parameters)

        all_values.extend(simulator(np.array(all_parameters)[sample_start-starting_sample:, :]))
        sample_start += number_of_samples

    return all_parameters, all_values
