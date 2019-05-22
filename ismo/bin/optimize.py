#!/bin/env python

import json
import ismo.optimizers
import ismo.objective_function
import keras.models

if __name__ == '__main__':
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser(description="""
Optimizes the parameters as starting values under the given neural network and objective function

It assumes the parameter file is of the form
parameter_sample_0
parameter_sample_1

""")

    parser.add_argument('--input_model_files', type=str, required=True, nargs='+',
                        help='Input filename for models (full path, hdf5 file)')

    parser.add_argument('--objective_python_module', type=str, required=True,
                        help='Input filename for for the objective function (python module, full path)')

    parser.add_argument('--objective_python_class', type=str, required=True,
                        help='Input classname for for the objective function')

    parser.add_argument('--input_parameters_file', type=str, required=True,
                        help='Input parameters file (assumed to be loadable with numpy.loadtxt)')

    parser.add_argument('--output_parameters_file', type=str, required=True,
                        help='Output parameters file where the optimized parameters are written')

    parser.add_argument('--optimizer_name', type=str, default='L-BFGS-B',
                        help='Name of the optimizer')

    parser.add_argument('--optimization_parameter_file', type=str,
                        help='Parameter file the optimization')

    parser.add_argument('--objective_parameter_file', type=str,
                        help='Parameter file the optimization')

    args = parser.parse_args()

    models = [keras.models.load_model(filename) for filename in args.input_model_files]

    starting_values = np.loadtxt(args.input_parameters_file)

    if args.optimization_parameter_file:
        with open (args.optimization_parameter_file) as config_file:
            optimization_configuration = json.load(config_file)
    else:
        optimization_configuration = {}


    if args.objective_parameter_file:
        with open (args.objective_parameter_file) as config_file:
            objective_configuration = json.load(config_file)
    else:
        objective_configuration = {}

    objective_function = ismo.objective_function.load_objective_function_from_python_file(args.objective_python_module,
                                                                                          args.objective_python_class,
                                                                                          objective_configuration)

    objective_function_dnn = ismo.objective_function.DNNObjectiveFunction(models, objective_function)

    optimizer = ismo.optimizers.create_optimizer(args.optimizer_name)

    optimized_parameters = ismo.optimizers.optimize_samples(
        starting_values=starting_values,
        J=objective_function_dnn,
        optimizer=optimizer)

    np.savetxt(args.output_parameters_file, optimized_parameters)
