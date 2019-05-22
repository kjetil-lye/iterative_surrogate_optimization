import ismo.submit
import ismo.submit.defaults
import sys


class SeveralVariablesCommands(ismo.submit.defaults.Commands):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.preproccsed_filename_base = 'preprocessed_values_{}.txt'
        self.simulated_output_filename_base = 'simulation_output_{}.txt'

    def do_evolve(self, submitter,
                  *,
                  iteration_number: int,
                  input_parameters_file: str,
                  output_value_files: list):
        # Preprocess
        preprocess = ismo.submit.Command([self.python_command, 'preprocess.py'])
        output_preprocess = self.preproccsed_filename_base.format(iteration_number)
        preprocess = preprocess.with_long_arguments(input_parameters_file=input_parameters_file,
                                                    output_parameters_file=output_preprocess)
        submitter(preprocess, wait_time_in_hours=24)

        # Evolve
        evolve = ismo.submit.Command([self.python_command, 'evolve_several_variables.py'])
        simulated_output_filename = self.simulated_output_filename_base.format(iteration_number)
        evolve = evolve.with_long_arguments(input_parameters_file=output_preprocess,
                                            output_values_file=simulated_output_filename)
        submitter(evolve, wait_time_in_hours=24)

        # Postprocess
        postprocess = ismo.submit.Command([self.python_command, 'postprocess.py'])
        postprocess = postprocess.with_long_arguments(input_values_file=simulated_output_filename,
                                                      output_values_files=output_value_files)
        submitter(postprocess, wait_time_in_hours=24)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""
Submits all the jobs for the sine experiments
        """)

    parser.add_argument('--number_of_samples_per_iteration', type=int, nargs='+',
                        help='Number of samples per iteration')

    parser.add_argument('--chain_name', type=str, default="several",
                        help="Name of the chain to run")

    parser.add_argument('--submitter', type=str, required=True,
                        help='Submitter to be used. Either "bash" (runs without waiting) or "lsf"')

    parser.add_argument('--dry_run', action='store_true',
                        help="Don't actually run the command, only print the commands that are to be executed")

    args = parser.parse_args()

    submitter = ismo.submit.create_submitter(args.submitter, args.chain_name, dry_run=args.dry_run)

    commands = SeveralVariablesCommands(dimension=20,
                                        number_of_output_values=3,
                                        training_parameter_config_file='training_parameters.json',
                                        optimize_target_file='objective.py',
                                        optimize_target_class='Objective',
                                        python_command=sys.executable
                                        )

    chain = ismo.submit.Chain(args.number_of_samples_per_iteration, submitter,
                              commands=commands)

    chain.run()
