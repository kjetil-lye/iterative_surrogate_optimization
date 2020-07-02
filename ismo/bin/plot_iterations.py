"""
Runs all configuration for analysis
"""
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import json
import plot_info
import collections
import logging
import scipy
import scipy.stats
from ismo.ensemble import get_configuration_name, get_iteration_sizes, get_competitor_basename

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""
    Plots the result of the ensemble run
    """)

    parser.add_argument('--aux_names', type=str, default=None,
                        help='path to json file containing the auxillary files'
                             ' (eg. drag, lift and area for an airfoil). '
                             'Should have the format {"name1":"filename1", "name2":"filename2"}')

    args = parser.parse_args()

    if args.aux_names is not None:
        with open(args.aux_names) as f:
            aux_files = json.load(f)
    else:
        aux_files = {}

    with open('ensemble_setup.json') as f:
        configuration = json.load(f)

    python_script = configuration['script_name']
    compute_budget = configuration['compute_budget']
    source_folder = configuration['source_folder']

    percentiles = [68, 95, 99.7]

    for generator in [configuration['generator']]:

        for starting_size in configuration['starting_sizes']:
            for batch_size_factor in configuration['batch_size_factors']:

                starting_sample = 0
                iterations = get_iteration_sizes(starting_size, batch_size_factor, configuration['compute_budget'])

                number_of_reruns = configuration['number_of_reruns']

                min_value_per_iteration = np.zeros((len(iterations), number_of_reruns))
                value_per_iteration = np.zeros((sum(iterations), number_of_reruns))

                aux_min_values = collections.defaultdict(lambda: np.zeros((len(iterations), number_of_reruns)))

                min_shapes_per_iteration = []
                closest_to_mean_shapes_per_iteration = []

                for rerun in range(number_of_reruns):
                    output_folder = os.path.join(get_configuration_name(configuration['basename'],
                                                                        rerun, starting_size,
                                                                        batch_size_factor ** (-1)), source_folder)
                    for iteration in range(len(iterations)):

                        try:
                            output_objective = os.path.join(output_folder,
                                                            f'objective.txt')
                            start_index = sum(iterations[:iteration])
                            end_index = sum(iterations[:iteration + 1])
                            all_values = np.loadtxt(output_objective)
                            values = all_values[start_index:end_index]
                            # values = values[~np.isnan(values)]
                            min_value = np.min(values)

                            value_per_iteration[:, rerun] = all_values

                            arg_min_value = np.argmin(all_values[:end_index])
                            if iteration > 0:
                                min_value = min(min_value, np.min(min_value_per_iteration[:iteration, rerun]))

                            min_value_per_iteration[iteration, rerun] = min_value

                            for aux_name, aux_file in aux_files.items():
                                output_aux = os.path.join(output_folder, aux_file)

                                min_aux_value = np.loadtxt(output_aux)[arg_min_value]

                                aux_min_values[aux_name][iteration, rerun] = min_aux_value




                        except Exception as e:
                            print(f"Looking for file {output_objective}")
                            logging.exception(e)
                            print(f"Failing {batch_size_factor} {starting_size} {generator}")

                # Histogram evolution
                min_objective_value = np.min(value_per_iteration[:, 0])
                max_objective_value = np.max(value_per_iteration[:, 0])
                for iteration in range(len(iterations)):
                    plt.hist(value_per_iteration[sum(iterations[:iteration]):sum(iterations[:iteration+1])],
                             bins=30, range=(min_objective_value, max_objective_value))
                    plt.xlabel("Objective value")
                    plt.ylabel("Number of samples")
                    plt.title("iteration: {}, type: {}, script: {}, generator: {}, batch_size_factor: {},\nstarting_size: {}".format(
                        iteration, source_name, python_script, generator, batch_size_factor, starting_size))
                    plot_info.savePlot("evolution_hist_{script}_{source_name}_{generator}_{batch_size}_{starting_size}_{iteration}".format(
                        script=python_script.replace(".py", ""),
                        source_name="objective",
                        batch_size=iterations[1],
                        starting_size=starting_size,
                        iteration=iteration))
                    plt.close('all')


                for iteration in range(len(iterations)):
                    mean_value = np.mean(min_value_per_iteration[iteration, :])
                    end_index = sum(iterations[:iteration + 1])
                    index_closest_to_mean_value = np.unravel_index(
                        abs(mean_value - value_per_iteration[:end_index, :]).argmin(),
                        value_per_iteration.shape)

                    output_folder_closest_to_mean = os.path.join(get_configuration_name(configuration['basename'],
                                                                                        index_closest_to_mean_value[1],
                                                                                        starting_size,
                                                                                        batch_size_factor ** (-1)),
                                                                 source_folder)

                    output_parameters_filename_closest_to_mean = os.path.join(output_folder_closest_to_mean,
                                                                              f'parameters.txt')
                    output_parameters_closest_to_mean = np.loadtxt(output_parameters_filename_closest_to_mean)

                    closest_to_mean_shapes_per_iteration.append(
                        output_parameters_closest_to_mean[index_closest_to_mean_value[0]])

                    index_min_value = np.unravel_index(value_per_iteration[:end_index, :].argmin(),
                                                       value_per_iteration.shape)

                    output_folder_min_value = os.path.join(get_configuration_name(configuration['basename'],
                                                                                  index_min_value[1], starting_size,
                                                                                  batch_size_factor ** (-1)),
                                                           source_folder)

                    output_parameters_filename_min_value = os.path.join(output_folder_min_value,
                                                                        f'parameters.txt')
                    output_parameters_min_value = np.loadtxt(output_parameters_filename_min_value)

                    min_shapes_per_iteration.append(output_parameters_min_value[index_min_value[0]])

                plot_info.saveData(
                    f'min_shapes_per_iteration_{python_script}_{generator}_{iterations[1]}_{starting_size}',
                    min_shapes_per_iteration)
                plot_info.saveData(
                    f'closest_to_mean_shapes_per_iteration_{python_script}_{generator}_{iterations[1]}_{starting_size}',
                    closest_to_mean_shapes_per_iteration)

                min_value_per_iteration_competitor = np.zeros((len(iterations), number_of_reruns))
                aux_min_values_competitor = collections.defaultdict(
                    lambda: np.zeros((len(iterations), number_of_reruns)))

                for rerun in range(number_of_reruns):

                    for iteration in range(len(iterations)):
                        try:
                            all_values = []

                            number_of_samples = sum(iterations[:iteration + 1])

                            competitor_basename = get_competitor_basename(configuration['basename'])
                            output_folder = os.path.join(get_configuration_name(competitor_basename,
                                                                                rerun, number_of_samples // 2,
                                                                                1), source_folder)

                            output_objective = os.path.join(output_folder,
                                                            f'objective.txt')

                            values = np.loadtxt(output_objective)

                            assert (values.shape[0] == number_of_samples)
                            # values = values[~np.isnan(values)]
                            all_values.extend(values)

                            min_value = np.min(all_values)
                            arg_min_value = np.argmin(all_values)

                            min_value_per_iteration_competitor[iteration, rerun] = min_value

                            for aux_name, aux_file in aux_files.items():
                                output_aux = os.path.join(output_folder, aux_file)

                                min_aux_value = np.loadtxt(output_aux)[arg_min_value]

                                aux_min_values_competitor[aux_name][iteration, rerun] = min_aux_value
                        except Exception as e:
                            print(f"Looking for file {output_objective}")
                            print(str(e))
                            print(f"Failing {batch_size_factor} {starting_size} {generator}")

                sources = {"objective": [min_value_per_iteration, min_value_per_iteration_competitor]}

                for aux_name in aux_files.keys():
                    sources[aux_name] = [aux_min_values[aux_name], aux_min_values_competitor[aux_name]]

                for source_name, source in sources.items():
                    batch_size = iterations[1]
                    iteration_range = np.arange(0, len(iterations))

                    plt.errorbar(iteration_range, np.mean(source[0], 1),
                                 yerr=np.std(source[0], 1), label='ISMO',
                                 fmt='o', uplims=True, lolims=True)

                    plt.errorbar(iteration_range + 1, np.mean(source[1], 1),
                                 yerr=np.std(source[1], 1), label='DNN+Opt',
                                 fmt='*', uplims=True, lolims=True)

                    print("#" * 80)
                    print(f"starting_size = {starting_size}, batch_size_factor = {batch_size_factor}")
                    print(f"mean(ismo)={np.mean(min_value_per_iteration, 1)}\n"
                          f" var(ismo)={np.mean(min_value_per_iteration, 1)}\n"
                          f"\n"
                          f"mean(dnno)={np.mean(min_value_per_iteration_competitor, 1)}\n"
                          f" var(dnno)={np.mean(min_value_per_iteration_competitor, 1)}\n")

                    plt.xlabel("Iteration $k$")
                    if source_name == "objective":
                        plt.ylabel("$\\mathbb{E}( J(x_k^*))$")
                    else:
                        plt.ylabel(f"$\\mathrm{{{source_name}}}(x_k^*)$")

                    plt.legend()
                    plt.title("type: {}, script: {}, generator: {}, batch_size_factor: {},\nstarting_size: {}".format(
                        source_name, python_script, generator, batch_size_factor, starting_size))
                    plot_info.savePlot("{script}_{source_name}_{generator}_{batch_size}_{starting_size}".format(
                        script=python_script.replace(".py", ""),
                        source_name=source_name,
                        batch_size=iterations[1],
                        starting_size=starting_size,
                        generator=generator))
                    plt.close('all')

                    ## min/max
                    plt.fill_between(iteration_range, np.min(source[0], 1), np.max(source[0], 1),
                                     alpha=0.3, color='C0')

                    plt.plot(iteration_range, np.mean(source[0], 1), '-o',
                             label='ISMO')

                    plt.fill_between(iteration_range, np.min(source[1], 1), np.max(source[1], 1),
                                     alpha=0.3, color='C1')

                    plt.plot(iteration_range, np.mean(source[1], 1), '-o',
                             label='DNN+Opt')

                    plt.xlabel("Iteration $k$")
                    if source_name == "objective":
                        plt.ylabel("$\\mathbb{E}( J(x_k^*))$")
                    else:
                        plt.ylabel(f"$\\mathrm{{{source_name}}}(x_k^*)$")

                    plt.legend()
                    plt.title(
                        "min/max, type: {}, script: {}, generator: {}, batch_size_factor: {},\nstarting_size: {}".format(
                            source_name, python_script, generator, batch_size_factor, starting_size))
                    plot_info.savePlot("{script}_min_max_{source_name}_{generator}_{batch_size}_{starting_size}".format(
                        script=python_script.replace(".py", ""),
                        source_name=source_name,
                        batch_size=iterations[1],
                        starting_size=starting_size,
                        generator=generator))
                    plt.close('all')

                    plot_info.saveData(f"ismo_{python_script.replace('.py', '')}_{source_name}_{generator}_{batch_size}_{starting_size}",
                                       source[0])
                    plot_info.saveData(f"dnnopt_{python_script.replace('.py', '')}_{source_name}_{generator}_{batch_size}_{starting_size}",
                                       source[1])
                    ## percentiles
                    for percentile in percentiles:


                        plt.plot(iteration_range, np.percentile(source[0], percentile, axis=1), '-o',
                                 label='ISMO')


                        plt.plot(iteration_range, np.percentile(source[1], percentile, axis=1), '-o',
                                 label='DNN+Opt')

                        plt.xlabel("Iteration $k$")
                        if source_name == "objective":
                            plt.ylabel("$\\mathbb{E}( J(x_k^*))$")
                        else:
                            plt.ylabel(f"$\\mathrm{{{source_name}}}(x_k^*)$")

                        plt.legend()
                        plt.title(
                            "percentile {}%, type: {}, script: {}, generator: {}, batch_size_factor: {},\nstarting_size: {}".format(
                                percentile,
                                source_name, python_script, generator, batch_size_factor, starting_size))
                        plot_info.savePlot("{script}_percentile_{percentile}_{source_name}_{generator}_{batch_size}_{starting_size}".format(
                            percentile=percentile,
                            script=python_script.replace(".py", ""),
                            source_name=source_name,
                            batch_size=iterations[1],
                            starting_size=starting_size,
                            generator=generator))

                        plt.plot(iteration_range, np.mean(source[0], 1), '-*',
                                 label='Mean ISMO', color='C0')


                        plt.plot(iteration_range, np.mean(source[1], 1), '-*',
                                 label='Mean DNN+Opt', color='C1')

                        plt.legend()

                        plot_info.savePlot(
                            "{script}_mean_percentile_{percentile}_{source_name}_{generator}_{batch_size}_{starting_size}".format(
                                percentile=percentile,
                                script=python_script.replace(".py", ""),
                                source_name=source_name,
                                batch_size=iterations[1],
                                starting_size=starting_size,
                                generator=generator))

                        plt.close('all')

                        # Percentile confidence

                        sources_names = ['ISMO', 'DNN+Opt']
                        formats = ['o', '*']
                        for source_index in range(2):
                            upper_lower = np.zeros((2, source[source_index].shape[0]))
                            upper_lower[0, :] = np.percentile(source[source_index], 100 - percentile, axis=1)
                            upper_lower[1, :] = np.percentile(source[source_index], percentile, axis=1)




                            plt.errorbar(iteration_range, np.mean(source[source_index], 1),
                                     yerr=upper_lower, label=sources_names[source_index],
                                     fmt=formats[source_index], uplims=True, lolims=True)

                        plt.legend()
                        plt.title(
                            "percentile confidence {}%, type: {}, script: {}, generator: {}, batch_size_factor: {},\nstarting_size: {}".format(
                                percentile,
                                source_name, python_script, generator, batch_size_factor, starting_size))
                        plot_info.savePlot(
                            "{script}_confidence_percentile_{percentile}_{source_name}_{generator}_{batch_size}_{starting_size}".format(
                                percentile=percentile,
                                script=python_script.replace(".py", ""),
                                source_name=source_name,
                                batch_size=iterations[1],
                                starting_size=starting_size,
                                generator=generator))
                        plt.close('all')


                        ## T test interval, see https://kite.com/python/answers/how-to-compute-the-confidence-interval-of-a-sample-statistic-in-python
                        ## for more information. This might not be the most accurate way of doing it though.
                        for source_index in range(2):
                            confidence_intervals = np.zeros((2, source[source_index].shape[0]))

                            for iteration in range(source[source_index].shape[0]):
                                sample = source[source_index][iteration, :]
                                degrees_freedom = source[source_index].shape[1]
                                sample_mean = np.mean(sample)
                                sample_standard_error = scipy.stats.sem(sample)

                                confidence_interval = scipy.stats.t.interval(percentile/100.,
                                                                             degrees_freedom,
                                                                             sample_mean,
                                                                             sample_standard_error)

                                confidence_intervals[0, iteration] = confidence_interval[0]
                                confidence_intervals[1, iteration] = confidence_interval[1]

                            plt.errorbar(iteration_range, np.mean(source[source_index], 1),
                                        yerr=confidence_intervals, label=sources_names[source_index],
                                        fmt=formats[source_index], uplims=True, lolims=True)
                        plt.legend()
                        plt.title(
                            "student t test confidence {}%, type: {}, script: {}, generator: {}, batch_size_factor: {},\nstarting_size: {}".format(
                                percentile,
                                source_name, python_script, generator, batch_size_factor, starting_size))
                        plot_info.savePlot(
                            "{script}_confidence_student_t_{percentile}_{source_name}_{generator}_{batch_size}_{starting_size}".format(
                                percentile=percentile,
                                script=python_script.replace(".py", ""),
                                source_name=source_name,
                                batch_size=iterations[1],
                                starting_size=starting_size,
                                generator=generator))
                        plt.close('all')
