import unittest
import iso.samples


class TestMonteCarlo(unittest.TestCase):

    def test_factory(self):
        monte_carlo = iso.samples.create_sample_generator('monte-carlo')

        self.assertEqual(type(monte_carlo), iso.samples.MonteCarlo)

    def test_first_samples_start(self):
        monte_carlo = iso.samples.create_sample_generator('monte-carlo')

        dimension = 4
        M = 8
        start = 4


        samples = monte_carlo(M, dimension, start)

        samples_true = monte_carlo(M+start, dimension, 0)

        for k in range(M):
            for d in range(dimension):
                self.assertEqual(samples[k,d], samples_true[k+start,d])

if __name__ == '__main__':
    unittest.main()

