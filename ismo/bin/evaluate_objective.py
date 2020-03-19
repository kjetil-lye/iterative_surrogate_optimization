#!/bin/env python

import ismo.train
import json
import ismo.objective_function
import os.path

if __name__ == '__main__':
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser(description="""
Evaluates the functional on the given values

""")

    parser.add_argument('--output_objective_file', type=str, required=True,
                        help='Output filename for the objective')

    parser.add_argument('--output_append', action='store_true',
                        help='Append output to end of file')

    parser.add_argument('--input_values_files', type=str, required=True, nargs='+',
                        help='Input values file (assumed to be loadable with numpy.loadtxt)')

    parser.add_argument('--objective_python_module', type=str, required=True,
                        help='Input filename for for the objective function (python module, full path)')

    parser.add_argument('--objective_python_class', type=str, required=True,
                        help='Input classname for for the objective function')

    parser.add_argument('--objective_parameter_file', type=str,
                        help='Parameter file the optimization')

    parser.add_argument('--values_start', type=int, default=0,
                        help='Starting index to read out of the parameter file, by default reads from start of file')

    parser.add_argument('--values_end', type=int, default=-1,
                        help='Ending index (exclusive) to read out of the parameter file, by default reads to end of file')

    args = parser.parse_args()

    if args.objective_parameter_file:
        with open(args.objective_parameter_file) as config_file:
            objective_configuration = json.load(config_file)
    else:
        objective_configuration = {}

    objective_function = ismo.objective_function.load_objective_function_from_python_file(args.objective_python_module,
                                                                                          args.objective_python_class,
                                                                                          objective_configuration)

    values = []
    for filename in args.input_values_files:
        if args.values_end != -1:
            values.append(np.loadtxt(filename)[args.values_start:args.values_end])
        else:
            values.append(np.loadtxt(filename)[args.values_start:])

    values = np.array(values).T

    objective_values = np.zeros(values.shape[0])

    for k in range(values.shape[0]):
        objective_values[k] = objective_function(values[k, :])

    if args.output_append:
        if os.path.exists(args.output_objective_file):
            previous_objective_values = np.loadtxt(args.output_objective_file)

            new_objective_values = np.zeros(values.shape[0] + previous_objective_values.shape[0])

            new_objective_values[:previous_objective_values.shape[0]] = previous_objective_values
            new_objective_values[previous_objective_values.shape[0]:] = objective_values

            objective_values = new_objective_values

    np.savetxt(args.output_objective_file, objective_values)
