import unittest
from keras.models import Model
from keras.layers import Input, Dense
import keras.backend
import numpy as np

from  ismo.gradients import TensorGradient


class TestTrainerFactory(unittest.TestCase):

    def test_no_error(self):
        # simply that it runs without error
        a = Input(shape=(32,))
        b = Dense(32)(a)
        model = Model(inputs=a, outputs=b)
        gradient = TensorGradient(model)

        ignored = gradient(np.zeros(32))

    def test_correct_one_by_one(self):
        a = Input(shape=(1,))
        b = Dense(1)(a)
        model = Model(inputs=a, outputs=b)


        model.layers[1].set_weights([42*np.ones((1,1)), np.zeros(1)])


        gradient_computer = TensorGradient(model)

        gradient_value = gradient_computer(np.ones(1))
        self.assertEqual(42, gradient_value)
