#!/bin/env python

import ismo.train
import json
import ismo.objective_function

if __name__ == '__main__':
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser(description="""
Evaluates the functional on the given values

""")

    parser.add_argument('--output_objective_file', type=str, required=True,
                        help='Output filename for the objective')

    parser.add_argument('--input_values_files', type=str, required=True, nargs='+',
                        help='Input values file (assumed to be loadable with numpy.loadtxt)')

    parser.add_argument('--objective_python_module', type=str, required=True,
                        help='Input filename for for the objective function (python module, full path)')

    parser.add_argument('--objective_python_class', type=str, required=True,
                        help='Input classname for for the objective function')

    parser.add_argument('--objective_parameter_file', type=str,
                        help='Parameter file the optimization')

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
        values.append(np.loadtxt(filename))

    values = np.array(values).T

    objective_values = np.zeros(values.shape[0])

    for k in range(values.shape[0]):
        objective_values[k] = objective_function(values[k, :])

    np.savetxt(args.output_objective_file, objective_values)
