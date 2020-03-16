import tensorflow.keras.models


def model_skeleton_from_tensorflow.keras_file(filename):
    with open (filename) as f:
        return tensorflow.keras.models.model_from_json(f.read())