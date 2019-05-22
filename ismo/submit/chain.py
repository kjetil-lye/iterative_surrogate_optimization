import ismo.submit.defaults
import ismo.submit


class Chain(object):
    def __init__(self, number_of_samples_per_iteration,
                 submitter: ismo.submit.SubmissionScript,
                 *,
                 commands: ismo.submit.defaults.Commands
                 ):

        self.number_of_samples_per_iteration = number_of_samples_per_iteration

        self.__commands = commands

        self.submitter = submitter

    def generate_samples(self, iteration_number, *, number_of_samples):
        self.__commands.generate_samples(self.submitter, iteration_number, number_of_samples=number_of_samples)

    def evolve(self, iteration_number):
        self.__commands.evolve(self.submitter, iteration_number)

    def train(self, iteration_number):
        self.__commands.train(self.submitter, iteration_number)

    def optimize(self, iteration_number):
        self.__commands.optimize(self.submitter, iteration_number)

    def run(self):
        self.generate_samples(0, number_of_samples = self.number_of_samples_per_iteration[0])

        self.evolve(0)

        for n, number_of_samples in enumerate(self.number_of_samples_per_iteration[1:]):
            self.train(n)
            self.generate_samples(n+1, number_of_samples = number_of_samples)
            self.optimize(n+1)
            self.evolve(n+1)

