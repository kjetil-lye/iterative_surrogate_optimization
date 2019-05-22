import ismo.submit
import subprocess

def create_submitter(name, job_chain, dry_run=False):
    if not dry_run:
        command_runner = lambda submit_command: subprocess.run(submit_command, check=True)
    else:
        command_runner = lambda submit_command: print(" ".join(submit_command))

    if name.lower() == 'lsf':
        return ismo.submit.LsfSubmissionScript(job_chain, command_runner=command_runner)
    elif name.lower() == 'bash':
        return ismo.submit.BashSubmissionScript(job_chain, command_runner=command_runner)
    else:
        raise Exception("Unknown submission script {}".format(name))
