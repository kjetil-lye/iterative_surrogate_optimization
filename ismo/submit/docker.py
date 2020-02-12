import ismo.submit
import os

class Docker(ismo.submit.Container):
    def __init__(self, command: ismo.submit.Command, bind_list: list = [[os.getcwd(), os.getcwd()]],
                 working_directory: str = os.getcwd()):
        super().__init__(self, command)
        self.working_directory = working_directory
        self.bind_list = bind_list

    def tolist(self):
        command_list = self.command

        docker = ['docker',
                  'run',
                  # Working directory
                  '-w',
                  self.working_directory,
                  # Binding
                  '-v',
                  ','.join(':'.join(bind) for bind in self.bind_list)
                  ]

        full_command = [docker, *self.command.tolist()]

        return full_command






