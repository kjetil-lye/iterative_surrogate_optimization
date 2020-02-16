from ismo.submit import Command

class Container(Command):
    def __init__(self, command):
        self.command = command

    def with_long_arguments(self, **kwargs):
        raise NotImplementedError()

    def tolist(self):
        raise NotImplementedError()