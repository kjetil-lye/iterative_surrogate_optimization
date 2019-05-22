from ismo.submit import SubmissionScript
import subprocess
class LsfSubmissionScript(SubmissionScript):
    def __init__(self, job_chain = None):
        self.job_chain = job_chain

        if job_chain is not None:

            self.first_time_job_chain = True

    def __call__(self, command, *,
                 number_of_processes,
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
            submit_command.extend(['-w', '"done({})"'.format(self.job_chain))])

        self.first_time_job_chain = False

        submit_command.apepnd(command)
        subprocess.call(submit_command, check=True)






