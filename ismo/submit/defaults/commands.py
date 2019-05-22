from ismo.submit import Command


class Commands(object):
    """
    This class is meant to be inherited from and then you can override whatever methods you want
    """

    def __init__(self,
                 *,

                 training_parameter_config_file,
                 optimize_target_file,
                 optimize_target_class,
                 number_of_output_values=1,
                 python_command='python',
                 ):
        self.parameter_basename = 'parameters_{}.txt'
        self.model_file_basename = 'model_{iteration_number}_{value_number}.h5'
        self.values_basename = 'values_{iteration_number}_{value_number}.txt'

        self.python_command = python_command
        self.training_parameter_config_file = training_parameter_config_file

        self.optimize_target_file = optimize_target_file
        self.optimize_target_class = optimize_target_class

        self.training_wait_time_in_hours = 24
        self.optimize_wait_time_in_hours = 24

        self.number_of_output_values = number_of_output_values

    def __run_python_module(self, module):
        return Command(["python", "-m", module])

    def train(self, submitter, iteration_number):
        command = self.__run_python_module("ismo.bin.train")

        for value_number in range(self.number_of_output_values):
            input_parameters_file = self.parameter_basename.format(iteration_number)
            input_values_file = self.values_basename.format(iteration_number=iteration_number,
                                                            value_number=value_number)

            output_model_file = self.model_file_basename.format(iteration_number=iteration_number,
                                                                value_number=value_number)

            command = command.with_long_arguments(number_of_samples=number_of_samples,
                                                  input_parameters_file=input_parameters_file,
                                                  input_values_file=input_values_file,
                                                  simple_configuration_file=self.trainining_parameter_config_file,
                                                  output_model_file=output_model_file
                                                  )
            submitter(command, wait_time_in_hours=self.training_wait_time_in_hours)

    def generate_samples(self, submitter, iteration_number,*,  number_of_samples):
        command = self.__run_python_module("ismo.bin.generate_samples")

        output_parameters_file = self.parameter_basename.format(iteration_number)

        command = command.with_long_arguments(number_of_samples=number_of_samples,
                                              output_parameters_file=output_parameters_file)

        submitter(command)

    def optimize(self, submitter, iteration_number):
        command = self.__run_python_module("ismo.bin.optimize")

        input_parameters_file = self.parameter_basename.format(iteration_number-1)

        output_parameters_file = self.parameter_basename.format(iteration_number)

        models = [self.model_file_basename.format(iteration_number=iteration_number-1, value_number=k)
                  for k in range(self.number_of_output_values)]

        models_as_string = " ".join(models)

        command = command.with_long_arguments(output_parameters_file=output_parameters_file,
                                              input_model_files=models_as_string,
                                              objective_python_module=self.optimize_target_file,
                                              objective_python_class=self.optimize_target_class,
                                              input_parameters_file=input_parameters_file)

        submitter(command, wait_time_in_hours=self.optimize_wait_time_in_hours)

    def evolve(self, submitter, iteration_number):
        input_parameters_file = self.parameter_basename.format(iteration_number)
        output_value_files = [self.values_basename.format(iteration_number=iteration_number-1, value_number=k)
                  for k in range(self.number_of_output_values)]

        self.do_evolve(submitter,
                       iteration_number=iteration_number,
                       input_parameters_file = input_parameters_file,
                       output_value_files=output_value_files
                       )

    def do_evolve(self, submitter,
                  *,
                  iteration_number: int,
                  input_parameters_file: str,
                  output_value_files: list):
        raise NotImplementedError('do_evolve needs to be implemented in a subclass of ismo.submit.defaults.Commands')
