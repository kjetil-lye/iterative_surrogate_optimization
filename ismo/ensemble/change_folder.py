import os


class ChangeFolder:

    def __init__(self, folder):
        self.old_folder = os.getcwd()
        self.folder = folder

    def __enter__(self):
        os.chdir(self.folder)

    def __exit__(self, type, value, traceback):
        os.chdir(self.old_folder)
