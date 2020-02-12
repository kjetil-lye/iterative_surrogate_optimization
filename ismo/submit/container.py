import ismo.submit

class Container(ismo.submit.Command):
    def __init__(self, command):
        self.command = command

    def with_long_arguments(self, **kwargs):
        raise NotImplementedError()

    def tolist(self):
        raise NotImplementedError()