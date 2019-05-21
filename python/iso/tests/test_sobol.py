import unittest
import iso.samples


class TestSobol(unittest.TestCase):

    def test_factory(self):
        sobol = iso.samples.create_sample_generator('sobol')

        self.assertEqual(type(sobol), iso.samples.Sobol)

    def test_first_samples_start(self):
        sobol = iso.samples.create_sample_generator('sobol')

        dimension = 4
        M = 8
        start = 4


        samples = sobol(M, dimension, start)

        samples_true = sobol(M+start, dimension, 0)

        for k in range(M):
            for d in range(dimension):
                self.assertEqual(samples[k,d], samples_true[k+start,d])

if __name__ == '__main__':
    unittest.main()

