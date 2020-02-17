from ismo.submit import Container, Command, get_user_id, get_group_id
import os
import sys

class Docker(Container):
    def __init__(self, container, command: Command, bind_list: list = [[os.getcwd(), os.getcwd()]],
                 working_directory: str = os.getcwd()):
        super().__init__(command)
        self.working_directory = working_directory
        self.bind_list = bind_list
        self.container = container

    def tolist(self):
        command_list = self.command

        docker = ['docker',
                  'run',
                  # user
                  '-u',
                  f'{get_user_id()}:{get_group_id()}',
                  # Python environment
                  '-e',
                  'PYTHONPATH=' + ':'.join(sys.path),
                  # Working directory
                  '-w',
                  self.working_directory,
                  # Binding
                  '-v',
                  ','.join(':'.join(bind) for bind in self.bind_list),
                  self.container
                  ]

        full_command = [*docker, *self.command.tolist()]

        return full_command






