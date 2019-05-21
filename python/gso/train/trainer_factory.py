import gso.train


def create_trainer_from_simple_file(filename):
    model = gso.train.model_skeleton_from_simple_config_file(filename)

    parameters = gso.train.parameters_from_simple_config_file(filename)

    return gso.train.SimpleTrainer(training_parameters=parameters,
                                   model=model)

