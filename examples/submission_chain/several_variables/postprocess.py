import numpy as np

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""
Removes the parameter number from the output file, and will split the values into N output files
    """)

    parser.add_argument('--input_values_file', type=str, required=True,
                        help='Input filename for the values (readable by np.loadtxt)')

    parser.add_argument('--output_values_files', type=str, required=True, nargs="+",
                        help='Output filenames for the values (will be written by np.savetxt)')

    args = parser.parse_args()

    values = np.loadtxt(args.input_values_file)

    values_new =[np.zeros(values.shape[0]) for k in range(values.shape[1]-1)]



    for n in range(values.shape[0]):
        for k in range(values.shape[1]-1):
            values_new[k][n] = values[n, k+1]

    for k in range(values.shape[1]-1):
        np.savetxt(args.output_values_files[k], values_new[k])
