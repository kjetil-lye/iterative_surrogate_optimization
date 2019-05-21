#!/bin/env python


if __name__ == '__main__':
    import argparse
    from gso.samples import create_sample_generator
    import numpy as np

    parser = argparse.ArgumentParser(description="""
Generate samples and write them to file using numpy.savetxt. Each row repesents a single sample, and each value
represents each component of the given sample.

Example use would be:
   y = numpy.loadtxt('filename.txt')
   # y[k,i] is component i of sample k  
        """)

    parser.add_argument('--generator', type=str, default='monte-carlo',
                        help='Name of generator to use, either "monte-carlo" or "sobol"')
    parser.add_argument('--dimension', type=int, required=True,
                        help="Number of dimensions")
    parser.add_argument('--number_of_samples', type=int, required=True,
                        help='Number of samples to generate')

    parser.add_argument('--start', type=int, default=0,
                        help='The first sample (in other words, number of samples to skip first)')

    parser.add_argument('--output_file', type=str, required=True,
                        help='Output filename (full path)')

    args = parser.parse_args()



    generator = create_sample_generator(args.generator)

    samples = generator(args.number_of_samples,
                        args.dimension,
                        start = args.start)


    np.savetxt(args.output_file, samples)



