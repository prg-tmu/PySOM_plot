import os

def path(p, is_dir=False):
    if os.path.exists(p):
        if is_dir and os.path.isdir(p):
            return p
        return p
    return Exception(p + " is not a valid path")
