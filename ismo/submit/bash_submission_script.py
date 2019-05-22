from ismo.submit import SubmissionScript
from ismo.submit import Command
import subprocess


class BashSubmissionScript(SubmissionScript):
    """
    Simply runs the commands given directly in bash
    """
    def __init__(self, job_chain = None,
                 command_runner = lambda submit_command: subprocess.run(submit_command, check=True)):
        self.job_chain = job_chain

        if job_chain is not None:

            self.first_time_job_chain = True

        self.command_runner = command_runner

    def __call__(self, command : Command,
                 *,
                 number_of_processes=1,
                 wait_time_in_hours=None,
                 memory_limit_in_mb=None):


        self.command_runner(command.tolist())






