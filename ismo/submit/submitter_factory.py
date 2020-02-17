import ismo.submit
from ismo.submit import get_current_repository
import subprocess
import os


def create_submitter(name, job_chain, dry_run=False, container_type=None,
                     container=None,
                     container_bind_list: list = [[get_current_repository(), get_current_repository()]],
                     container_working_directory: str = os.getcwd()):

    if not dry_run:
        command_runner = lambda submit_command: subprocess.run(submit_command, check=True)
    else:
        command_runner = lambda submit_command: print(" ".join(submit_command))

    if name.lower() == 'lsf':
        submitter = ismo.submit.LsfSubmissionScript(job_chain, command_runner=command_runner)

        if container_type == 'singularity':
            submitter.add_parameter(['-R', 'singularity'])

    elif name.lower() == 'bash':
        submitter = ismo.submit.BashSubmissionScript(job_chain, command_runner=command_runner)
    else:
        raise Exception("Unknown submission script {}".format(name))

    if container_type is not None:
        if container_type == 'docker':
            container_class = ismo.submit.Docker
            container = container.replace('docker://', '')
        elif container_type == 'singularity':
            container_class = ismo.submit.Singularity
        else:
            raise Exception(f"Unknown container {container_type}.")

        submitter = ismo.submit.ContainerDecorator(container_class, container, submitter, bind_list = container_bind_list,
                                                   working_directory = container_working_directory)

    return submitter
