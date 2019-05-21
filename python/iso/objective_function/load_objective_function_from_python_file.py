import importlib.util


def load_objective_function_from_python_file(filename, classname, parameters):
    # See https://stackoverflow.com/a/67692
    spec = importlib.util.spec_from_file_location("module.name", filename)
    module_object = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_object)

    instance = getattr(module_object, classname)(**parameters)

    return instance