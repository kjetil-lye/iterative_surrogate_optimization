import numpy as np

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""
Runs the function sin(4*pi*x) on the input parameters
    """)

    parser.add_argument('--input_parameters_file', type=str, required=True,
                        help='Input filename for the parameters (readable by np.loadtxt)')

    parser.add_argument('--output_values_file', type=str, required=True,
                        help='Output filename for the values (will be written by np.savetxt)')

    args = parser.parse_args()

    parameters = np.loadtxt(args.input_parameters_file)

    values = np.sin(4 * np.pi * parameters)

    np.savetxt(args.output_values_file, values)
