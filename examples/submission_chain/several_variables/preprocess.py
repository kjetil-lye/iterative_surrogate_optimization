import numpy as np

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""
Takes the parameter input file and adds the sample number
    """)

    parser.add_argument('--input_parameters_file', type=str, required=True,
                        help='Input filename for the parameters (readable by np.loadtxt)')

    parser.add_argument('--output_parameters_file', type=str, required=True,
                        help='Output filename for the parameters (will be written by np.savetxt)')

    args = parser.parse_args()

    parameters = np.loadtxt(args.input_parameters_file)

    parameters_new = np.zeros((parameters.shape[0], parameters.shape[1]+1))

    for n in range(parameters.shape[0]):
        parameters_new[n, 1:] = parameters[n, :]
        parameters_new[n, 0] = n

    np.savetxt(args.output_parameters_file, parameters_new)
