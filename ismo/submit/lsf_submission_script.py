from ismo.submit import SubmissionScript
from ismo.submit import Command
import subprocess


class LsfSubmissionScript(SubmissionScript):
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

        submit_command = ["bsub", "-n", "{}".format(number_of_processes)]

        if wait_time_in_hours is not None:
            submit_command.append("-W")
            submit_command.append("{}:00".format(int(wait_time_in_hours)))

        if memory_limit_in_mb is not None:
            submit_command.extend(["-R", "'rusage[mem={}]'".format(int(memory_limit_in_mb))])

        if self.job_chain is not None:
            submit_command.extend(['-J', '{}'.format(self.job_chain)])

        if not self.first_time_job_chain:
            submit_command.extend(['-w', 'done({})'.format(self.job_chain)])

        self.first_time_job_chain = False

        submit_command.extend(command.tolist())
        self.command_runner(submit_command)






