import git
import os.path

def get_current_repository(default=os.getcwd()):
    try:
        repo = git.Repo(search_parent_directories=True)
        directory = os.path.dirname(repo.git_dir)
        return directory
    except:
        return default



