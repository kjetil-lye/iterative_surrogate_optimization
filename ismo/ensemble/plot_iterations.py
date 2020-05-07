"""
Runs all configuration for analysis
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import subprocess
import ismo.submit
import json
import plot_info
import collections
import logging
from run_ensemble import get_configuration_name, get_iteration_sizes, get_competitor_basename

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage:\n\tpython {} <name of python script> <compute budget> <other arguments passed to python script>".format(sys.argv[0]))
        print("<compute budget> should be in terms of number of total samples calculated (integer). Reruns not included.")
        exit(1)
        
    # we also plot the corresponding lift, drag and area
    aux_files = {
            "lift" : "values_0.txt",
            "drag" : "values_1.txt",
            "area" : "values_2.txt"
            }
    python_script = sys.argv[1]
    compute_budget = int(sys.argv[2])

    with open('ensemble_setup.json') as f:
        configuration = json.load(f)

    for generator in [configuration['generator']]:

        for starting_size in configuration['starting_sizes']:
            for batch_size_factor in configuration['batch_size_factors']:

                starting_sample=0
                iterations = get_iteration_sizes(starting_size, batch_size_factor, configuration['compute_budget'])
                
                number_of_reruns = configuration['number_of_reruns']
                
                min_value_per_iteration = np.zeros((len(iterations), number_of_reruns))
                value_per_iteration = np.zeros((sum(iterations), number_of_reruns))

                
                aux_min_values = collections.defaultdict(lambda: np.zeros((len(iterations), number_of_reruns)))

                min_shapes_per_iteration = []
                closest_to_mean_shapes_per_iteration = []

                for rerun in range(number_of_reruns):
                    output_folder = os.path.join(get_configuration_name(configuration['basename'],
                                                                        rerun, starting_size, batch_size_factor**(-1)), 'airfoil_chain')
                    for iteration in range(len(iterations)):

                        try:
                            output_objective = os.path.join(output_folder,
                                                         f'objective.txt')
                            start_index = sum(iterations[:iteration])
                            end_index = sum(iterations[:iteration + 1])
                            all_values = np.loadtxt(output_objective)
                            values = all_values[start_index:end_index]
                            #values = values[~np.isnan(values)]
                            min_value = np.min(values)

                            value_per_iteration[:, rerun] = all_values
                            
                            arg_min_value = np.argmin(all_values[:end_index])
                            if iteration > 0:
                                min_value = min(min_value, np.min(min_value_per_iteration[:iteration,rerun]))
    
                            min_value_per_iteration[iteration, rerun] = min_value
                            
                            for aux_name, aux_file in aux_files.items():
                                output_aux = os.path.join(output_folder, aux_file)
                                                         
                                min_aux_value = np.loadtxt(output_aux)[arg_min_value]
                                
                                aux_min_values[aux_name][iteration, rerun] = min_aux_value



                                
                        except Exception as e:
                             print(f"Looking for file {output_objective}")
                             print(str(e))
                             logging.exception(e)
                             print(f"Failing {batch_size_factor} {starting_size} {generator}")
                for iteration in range(len(iterations)):
                    mean_value = np.mean(min_value_per_iteration[iteration, :])
                    end_index = sum(iterations[:iteration + 1])
                    index_closest_to_mean_value = np.unravel_index(abs(mean_value - value_per_iteration[:end_index, :]).argmin(),
                                                                   value_per_iteration.shape)

                    output_folder_closest_to_mean = os.path.join(get_configuration_name(configuration['basename'],
                                                                        index_closest_to_mean_value[1], starting_size,
                                                                        batch_size_factor ** (-1)), 'airfoil_chain')

                    output_parameters_filename_closest_to_mean = os.path.join(output_folder_closest_to_mean,
                                                              f'parameters.txt')
                    output_parameters_closest_to_mean = np.loadtxt(output_parameters_filename_closest_to_mean)

                    closest_to_mean_shapes_per_iteration.append(output_parameters_closest_to_mean[index_closest_to_mean_value[0]])

                    index_min_value = np.unravel_index(value_per_iteration[:end_index, :].argmin(),
                                                                   value_per_iteration.shape)

                    output_folder_min_value = os.path.join(get_configuration_name(configuration['basename'],
                                                                        index_min_value[1], starting_size,
                                                                        batch_size_factor ** (-1)), 'airfoil_chain')

                    output_parameters_filename_min_value = os.path.join(output_folder_min_value,
                                                              f'parameters.txt')
                    output_parameters_min_value = np.loadtxt(output_parameters_filename_min_value)

                    min_shapes_per_iteration.append(output_parameters_min_value[index_min_value[0]])



                plot_info.saveData(f'min_shapes_per_iteration_{python_script}_{generator}_{iterations[1]}_{starting_size}',
                                   min_shapes_per_iteration)
                plot_info.saveData(
                    f'closest_to_mean_shapes_per_iteration_{python_script}_{generator}_{iterations[1]}_{starting_size}',
                    closest_to_mean_shapes_per_iteration)
                
                min_value_per_iteration_competitor = np.zeros((len(iterations), number_of_reruns))
                aux_min_values_competitor = collections.defaultdict(lambda: np.zeros((len(iterations), number_of_reruns)))

                for rerun in range(number_of_reruns):

                    for iteration in range(len(iterations)):
                        try:
                            all_values = []
                            
                            number_of_samples = sum(iterations[:iteration+1])
                            
                            competitor_basename = get_competitor_basename(configuration['basename'])
                            output_folder = os.path.join(get_configuration_name(competitor_basename,
                                                           rerun, number_of_samples//2,
                                                           1), "airfoil_chain")
                            
                            output_objective = os.path.join(output_folder,
                                                     f'objective.txt')
    
                            values = np.loadtxt(output_objective)
                            
                            assert(values.shape[0] == number_of_samples)
                            #values = values[~np.isnan(values)]
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



                sources = {"objective" : [min_value_per_iteration, min_value_per_iteration_competitor]}
                
                for aux_name in aux_files.keys():
                    sources[aux_name] = [aux_min_values[aux_name], aux_min_values_competitor[aux_name]]
                    
                for source_name, source in sources.items():
                    iteration_range = np.arange(0, len(iterations))
    
                    plt.errorbar(iteration_range, np.mean(source[0], 1),
                                 yerr=np.std(source[0], 1), label='ISMO',
                                 fmt='o', uplims=True, lolims=True)
    
    
    
                    plt.errorbar(iteration_range+1, np.mean(source[1], 1),
                                 yerr=np.std(source[1], 1), label='DNN+Opt',
                                 fmt='*', uplims=True, lolims=True)
    
                    print("#"*80)
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
                        source_name = source_name,
                        batch_size=iterations[1],
                        starting_size=starting_size,
                        generator=generator))
                    plt.close('all')


                    ## min/max
                    plt.fill_between(iteration_range, np.min(source[0], 1), np.max(source[0], 1),
                                     alpha = 0.3, color='C0')

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
                    plt.title("min/max, type: {}, script: {}, generator: {}, batch_size_factor: {},\nstarting_size: {}".format(
                        source_name, python_script, generator, batch_size_factor, starting_size))
                    plot_info.savePlot("{script}_min_max_{source_name}_{generator}_{batch_size}_{starting_size}".format(
                        script=python_script.replace(".py", ""),
                        source_name=source_name,
                        batch_size=iterations[1],
                        starting_size=starting_size,
                        generator=generator))
                    plt.close('all')


