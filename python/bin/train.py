#!/bin/env python

import gso.train

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

    parser.add_argument('--input_parameters_file', type=str, required=True,
                        help='Input parameters file (assumed to be loadable with numpy.loadtxt)')

    parser.add_argument('--input_values_file', type=str, required=True,
                        help='Input values file (assumed to be loadable with numpy.loadtxt)')

    parser.add_argument('--simple_configuration_file', type=str, required=True,
                        help='A JSON file describing the configuration file')

    args = parser.parse_args()

    trainer = gso.train.create_trainer_from_simple_file(args.simple_configuration_file)

    parameters = np.loadtxt(args.input_parameters_file)
    values = np.loadtxt(args.input_values_file)

    trainer.fit(parameters, values)

    trainer.save_to_file(args.output_model_file)