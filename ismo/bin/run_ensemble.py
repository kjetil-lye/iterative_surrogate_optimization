from ismo.ensemble import run_all_configurations
import json
import git
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""
Runs the ensemble for M different runs (to get some statistics).
""")

    parser.add_argument('--script_name', type=str, required=True,
                        help='Name of python script to run')

    parser.add_argument('--source_folder', type=str, required=True,
                        help='Name of source folder')

    parser.add_argument('--number_of_reruns', type=int, default=10,
                        help='Total number of reruns to get the ensemble')

    parser.add_argument('--basename', type=str, default='ensemble_run',
                        help='Basename for the ensemble')

    parser.add_argument('--compute_budget', type=int, default=512,
                        help='Maximum compute budget (in terms of number of samples that can be computed from simulator)')

    parser.add_argument('--starting_sizes', type=int, nargs='+', default=[16, 32, 64],
                        help='Starting sizes to use')

    parser.add_argument('--batch_size_factors', type=float, nargs='+', default=[0.25, 0.5, 1],
                        help='Batch sizes to use as a ratio of starting_size')

    repo = git.Repo(search_parent_directories=True)

    parser.add_argument('--repository_path', type=str, default=repo.working_dir,
                        help='Absolute path of the repository')

    parser.add_argument('--dry_run', action='store_true',
                        help='Only do a dry run, no jobs are submitted or run')

    parser.add_argument('--submitter', type=str, default='lsf',
                        help='Name of submitter to use, can be lsf or bash')

    parser.add_argument('--only_missing', action='store_true',
                        help='Only run missing configurations')

    parser.add_argument('--container_type', type=str, default=None,
                        help="Container type (none, docker, singularity)")

    parser.add_argument('--container', type=str, default='docker://kjetilly/machine_learning_base:0.1.2',
                        help='Container name')

    parser.add_argument('--generator', type=str, default="monte-carlo",
                        help="Generator to use (either 'monte-carlo' or 'sobol'")

    parser.add_argument('--optimizer', type=str, default='L-BFGS-B',
                        help='Name of optimizer')


    parser.add_argument('--do_not_draw_new_samples', action='store_true',
                        help='Reuse old optimization values for next iteration')



    args = parser.parse_args()

    # Save configuration for easy read afterwards
    with open("ensemble_setup.json", 'w') as f:
        json.dump(vars(args), f, indent=4)

    run_all_configurations(**vars(args))
