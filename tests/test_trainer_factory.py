import unittest

import ismo.train
import numpy as np


class TestTrainerFactory(unittest.TestCase):

    def test_factory(self):
        trainer =ismo.train.create_trainer_from_simple_file(
            '../../examples/config_files/training_files/network_one_input_output.json')

    def test_factory_four_dimension(self):
        trainer =ismo.train.create_trainer_from_simple_file(
            '../../examples/config_files/training_files/network_four_input_one_output.json')

        parameters = np.loadtxt('../../examples/config_files/training_files/parameters.txt')
        values = np.loadtxt('../../examples/config_files/training_files/values.txt')

        trainer.fit(parameters, values)


if __name__ == '__main__':
    unittest.main()
