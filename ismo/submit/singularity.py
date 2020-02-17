from ismo.submit import Container, Command
import os
import sys

class Singularity(Container):
    def __init__(self, container, command: Command, bind_list: list = [[os.getcwd(), os.getcwd()]],
                 working_directory: str = os.getcwd()):
        super().__init__(command)
        self.working_directory = working_directory
        self.bind_list = bind_list
        self.container = container

    def tolist(self):
        command_list = self.command

        docker = ['singularity',
                  'exec',
                  # Working directory
                  '-W',
                  self.working_directory,
                  # Binding
                  '-B',
                  ','.join(':'.join(bind) for bind in self.bind_list),
                  self.container
                  ]

        full_command = [*docker, *self.command.tolist()]

        return full_command






