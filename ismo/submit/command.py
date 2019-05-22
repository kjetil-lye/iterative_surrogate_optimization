import copy


class Command(object):
    def __init__(self, command: list):
        self.command = command

    def with_long_arguments(self, **kwargs):
        command_list = copy.deepcopy(self.command)

        for key, item in kwargs.items():
            command_list.append('--{}'.format(item))
            command_list.append('{}'.format(item))

        return Command(command_list)

    def tolist(self):
        return self.command