#!/bin/env python

import ismo.train
import os.path

if __name__ == '__main__':
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser(description="""
Trains a network with the given parameters and associated values.

It assumes the parameter file is of the form
parameter_sample_0
parameter_sample_1
...

and the value file should be of the form

value_sample_0
value_sample_1
...

""")

    parser.add_argument('--output_model_file', type=str, required=True,
                        help='Output filename for model (full path, hdf5 file)')

    parser.add_argument('--input_parameters_files', type=str, required=True, nargs='+',
                        help='Input parameters file (assumed to be loadable with numpy.loadtxt)')

    parser.add_argument('--input_values_files', type=str, required=True, nargs='+',
                        help='Input values file (assumed to be loadable with numpy.loadtxt)')

    parser.add_argument('--simple_configuration_file', type=str, required=True,
                        help='A JSON file describing the configuration file')

    parser.add_argument('--reuse_model', action='store_true',
                        help='Reuse the model')


    args = parser.parse_args()

    if len(args.input_parameters_files) != len(args.input_values_files):
        raise Exception("The number of value files should be the same as the number of parameters file")

    trainer = ismo.train.create_trainer_from_simple_file(args.simple_configuration_file)

    all_parameters = None
    all_values = None

    for parameter_file, value_file in zip(args.input_parameters_files, args.input_values_files):
        parameters = np.loadtxt(parameter_file)

        values = np.loadtxt(value_file)

        if all_parameters is None:
            all_parameters = parameters
            all_values = values

        else:
            if len(all_parameters.shape) == 1:
                all_parameters = np.resize(all_parameters, (all_parameters.shape[0] + parameters.shape[0]))
                all_parameters[-parameters.shape[0]:] = parameters
            else:
                all_parameters = np.resize(all_parameters,
                                           (all_parameters.shape[0] + parameters.shape[0], *parameters.shape[1:]))
                all_parameters[-parameters.shape[0]:, :] = parameters
            all_values = np.resize(all_values, (all_values.shape[0] + values.shape[0]))
            all_values[-values.shape[0]:] = values

    if args.reuse_model and os.path.exists(args.output_model_file):
        trainer.load_from_file(args.output_model_file)

    trainer.fit(all_parameters, all_values)

    trainer.save_to_file(args.output_model_file)
