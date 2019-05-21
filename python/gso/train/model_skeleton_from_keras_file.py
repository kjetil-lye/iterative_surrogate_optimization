import keras.models


def model_skeleton_from_keras_file(filename):
    with open (filename) as f:
        return keras.models.model_from_json(f.read())