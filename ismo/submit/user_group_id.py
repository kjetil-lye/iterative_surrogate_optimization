import os

def get_user_id():
    return os.getuid()

def get_group_id():
    return os.getgid()