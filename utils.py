import glob
import os
import shutil
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def make_dir(dir_path, delete_if_exists=False):
    try:
        os.mkdir(dir_path)
    except:
        if delete_if_exists:
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)

        pass
