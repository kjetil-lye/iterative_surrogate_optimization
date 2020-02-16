from ismo.submit import SubmissionScript, Command


class ContainerDecorator(SubmissionScript):
    def __init__(self, container_class, container, submitter, *, bind_list, working_directory):
        self.container_class = container_class
        self.submitter = submitter
        self.bind_list = bind_list
        self.working_directory = working_directory
        self.container = container

    def __call__(self, command: Command, *argv, **kwargs):
        container = self.container_class(self.container, command, bind_list=self.bind_list,
                                         working_directory=self.working_directory)

        return self.submitter(container, *argv, **kwargs)

