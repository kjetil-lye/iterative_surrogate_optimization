import iso.train


def create_trainer_from_simple_file(filename):
    model = iso.train.model_skeleton_from_simple_config_file(filename)

    parameters = iso.train.parameters_from_simple_config_file(filename)

    return iso.train.SimpleTrainer(training_parameters=parameters,
                                   model=model)

