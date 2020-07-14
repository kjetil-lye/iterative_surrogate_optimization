import os
import shutil
import git
import copy
import sys
import subprocess
import glob
import json
from ismo.ensemble import ChangeFolder
from ismo.submit import get_current_repository


def all_successfully_completed():
    lsf_files = glob.glob('lsf.o*')

    for lsf_filename in lsf_files:
        with open(lsf_filename) as f:
            content = f.read()

            if 'Successfully completed' not in content:
                return False

    return True


def get_configuration_name(basename, rerun, starting_size, batch_size_factor):
    return f'{basename}_rerun_{rerun}_iterations_{starting_size}_{float(batch_size_factor)}'


def run_configuration(*, script_name, source_folder, basename, rerun, iteration_sizes, repository_path, dry_run,
                      submitter_name,
                      only_missing, container, container_type,
                      sample_generator, optimizer, do_not_draw_new_samples):
    starting_size = iteration_sizes[0]
    batch_size_factor = iteration_sizes[0] / iteration_sizes[1]

    folder_name = get_configuration_name(basename, rerun, starting_size, batch_size_factor)
    if not only_missing or not os.path.exists(folder_name):
        os.mkdir(folder_name)
    with ChangeFolder(folder_name):
        if only_missing:
            should_run = not os.path.exists(source_folder)
        if not only_missing or not os.path.exists(source_folder):
            shutil.copytree(os.path.join(repository_path, source_folder), source_folder)

        with ChangeFolder(source_folder):
            starting_sample = rerun * sum(iteration_sizes)
            iteration_sizes_as_str = [str(int(x)) for x in iteration_sizes]

            # Just some sanity check
            for iteration_size in iteration_sizes:
                if int(iteration_size) < 1:
                    raise Exception(f"Iteration size is 0, all iterations sizes: {iteration_sizes}")

            command_to_run = [sys.executable,
                              script_name,
                              '--number_of_samples_per_iteration',
                              *iteration_sizes_as_str,
                              '--number_of_processes',
                              *iteration_sizes_as_str,
                              '--submitter',
                              submitter_name,
                              '--starting_sample',
                              str(starting_sample),
                              '--chain_name',
                              folder_name,
                              '--generator',
                              sample_generator,
                              '--optimizer',
                              optimizer
                              ]

            if do_not_draw_new_samples:
                command_to_run.append('--do_not_draw_new_samples')

            if container is not None:
                command_to_run.extend(['--container', container])
            if container_type is not None:
                command_to_run.extend(['--container_type', container_type])

            if dry_run:
                command_to_run.append('--dry_run')

            if only_missing:
                should_run = should_run or not all_successfully_completed()
            else:
                should_run = True

            if should_run:
                subprocess.run(command_to_run, check=True)


def get_competitor_basename(basename):
    return f'{basename}_competitor'


def get_iteration_sizes(starting_size, batch_size_factor, compute_budget):
    iteration_sizes = [starting_size]

    while sum(iteration_sizes) < compute_budget:
        iteration_sizes.append(int(batch_size_factor * starting_size))

    return iteration_sizes


def run_all_configurations(*, script_name, source_folder, basename, number_of_reruns,
                           starting_sizes, batch_size_factors, optimizer, generator, container, container_type,
                           only_missing, repository_path, dry_run, submitter, compute_budget, do_not_draw_new_samples=False):
    # This will be to store the competitors afterwards
    all_sample_sizes = []
    # Loop through configurations
    for starting_size in starting_sizes:
        for batch_size_factor in batch_size_factors:

            iteration_sizes = get_iteration_sizes(starting_size, batch_size_factor, compute_budget)

            for rerun in range(number_of_reruns):
                run_configuration(basename=basename,
                                  rerun=rerun,
                                  iteration_sizes=iteration_sizes,
                                  repository_path=repository_path,
                                  dry_run=dry_run,
                                  submitter_name=submitter,
                                  only_missing=only_missing,
                                  container_type=container_type,
                                  container=container,
                                  sample_generator=generator,
                                  optimizer=optimizer,
                                  script_name=script_name,
                                  source_folder=source_folder,
                                  do_not_draw_new_samples=do_not_draw_new_samples)

            for iteration_number, iteration_size in enumerate(iteration_sizes):
                number_of_samples = sum(iteration_sizes[:iteration_number + 1])
                all_sample_sizes.append(number_of_samples)
    # Make sure we do not have duplications
    all_sample_sizes = set(all_sample_sizes)
    # Run competitors
    for sample_size in all_sample_sizes:
        for rerun in range(number_of_reruns):
            run_configuration(basename=get_competitor_basename(basename),
                              rerun=rerun,
                              # First is the number of points we will use to
                              # train, second is the number that will be
                              # evaluated. Our whole budget is sample_size//2,
                              # so we evaluate half at the samples for training,
                              # and then use the rest to optimize/evaluate the resulting
                              # optimized points
                              iteration_sizes=[sample_size // 2, sample_size // 2],
                              repository_path=repository_path,
                              dry_run=dry_run,
                              submitter_name=submitter,
                              only_missing=only_missing,
                              container_type=container_type,
                              container=container,
                              sample_generator=generator,
                              optimizer=optimizer,
                              script_name=script_name,
                              source_folder=source_folder)
