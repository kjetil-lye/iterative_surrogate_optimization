import ismo.train


def create_trainer_from_simple_file(filename):
    model =ismo.train.model_skeleton_from_simple_config_file(filename)

    parameters =ismo.train.parameters_from_simple_config_file(filename)

    return ismo.train.SimpleTrainer(training_parameters=parameters,
                                   model=model)

