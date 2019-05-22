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

                 dimension,
                 number_of_output_values=1,
                 python_command='python',
                 prefix='',
                 starting_sample=0,
                 optimization_parameter_file=None,
                 optimizer_name='L-BFGS-B',
                 objective_parameter_file=None
                 ):
        self.prefix = prefix

        self.parameter_for_optimization_basename = prefix + 'parameters_for_optimization_{}.txt'
        self.parameter_basename = prefix + 'parameters_{}.txt'
        self.model_file_basename = prefix + 'model_{iteration_number}_{value_number}.h5'
        self.values_basename = prefix + 'values_{iteration_number}_{value_number}.txt'

        self.python_command = python_command
        self.training_parameter_config_file = training_parameter_config_file

        self.optimize_target_file = optimize_target_file
        self.optimize_target_class = optimize_target_class

        self.training_wait_time_in_hours = 24
        self.optimize_wait_time_in_hours = 24

        self.number_of_output_values = number_of_output_values
        self.dimension=dimension

        self.number_of_samples_generated = starting_sample

        self.additional_optimizer_arguments = {'optimizer_name' : optimizer_name}

        if optimization_parameter_file is not None:
            self.additional_optimizer_arguments['optimization_parameter_file'] = optimization_parameter_file

        self.additional_objective_arguments = {}

        if objective_parameter_file is not None:
            self.additional_objective_arguments['objective_parameter_file'] = objective_parameter_file


    def __run_python_module(self, module):
        return Command([self.python_command, "-m", module])

    def train(self, submitter, iteration_number):
        command = self.__run_python_module("ismo.bin.train")

        for value_number in range(self.number_of_output_values):
            input_parameters_file = self.parameter_basename.format(iteration_number)
            input_values_file = self.values_basename.format(iteration_number=iteration_number,
                                                            value_number=value_number)

            output_model_file = self.model_file_basename.format(iteration_number=iteration_number,
                                                                value_number=value_number)

            command = command.with_long_arguments(
                                                  input_parameters_file=input_parameters_file,
                                                  input_values_file=input_values_file,
                                                  simple_configuration_file=self.training_parameter_config_file,
                                                  output_model_file=output_model_file
                                                  )
            submitter(command, wait_time_in_hours=self.training_wait_time_in_hours)

    def generate_samples(self, submitter, iteration_number,*,  number_of_samples):
        command = self.__run_python_module("ismo.bin.generate_samples")
        if iteration_number == 0:
            output_parameters_file = self.parameter_basename.format(iteration_number)
        else:
            output_parameters_file = self.parameter_for_optimization_basename.format(iteration_number)

        command = command.with_long_arguments(number_of_samples=number_of_samples,
                                              output_file=output_parameters_file,
                                              dimension=self.dimension,
                                              start=self.number_of_samples_generated)

        submitter(command)

        self.number_of_samples_generated += number_of_samples

    def optimize(self, submitter, iteration_number):
        command = self.__run_python_module("ismo.bin.optimize")

        input_parameters_file = self.parameter_for_optimization_basename .format(iteration_number)

        output_parameters_file = self.parameter_basename.format(iteration_number)

        models = [self.model_file_basename.format(iteration_number=iteration_number-1, value_number=k)
                  for k in range(self.number_of_output_values)]



        command = command.with_long_arguments(output_parameters_file=output_parameters_file,
                                              input_model_files=models,
                                              objective_python_module=self.optimize_target_file,
                                              objective_python_class=self.optimize_target_class,
                                              input_parameters_file=input_parameters_file,
                                              **self.additional_optimizer_arguments,
                                              **self.additional_objective_arguments)

        submitter(command, wait_time_in_hours=self.optimize_wait_time_in_hours)

    def evolve(self, submitter, iteration_number):
        input_parameters_file = self.parameter_basename.format(iteration_number)
        output_value_files = [self.values_basename.format(iteration_number=iteration_number, value_number=k)
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
