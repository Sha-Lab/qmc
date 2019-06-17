import os
import sys
import argparse
#import torch.multiprocessing as mp
#mp.set_start_method('spawn')
from ipdb import slaunch_ipdb_on_exception
from utils import batch_args, with_null
from ipdb import slaunch_ipdb_on_exception
# local file
#from main import main

main_module = None # main function to import

def import_main(fn):
    directory, module_name = os.path.split(fn)
    module_name = os.path.splitext(module_name)[0]
    path = list(sys.path)
    sys.path.insert(0, directory)
    try:
        global main_module
        main_module = __import__(module_name)
    finally:
        sys.path[:] = path # restore system path

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_path', type=str)
    parser.add_argument('exp_path', type=str)
    parser.add_argument('-d', action='store_true')
    return parser.parse_args()

def run(args, config):
    finished = False
    if config.d:
        context = slaunch_ipdb_on_exception
    else:
        context = with_null
    with context():
        main_module.main(args)
        finished = True
    return finished

if __name__ == "__main__":
    config = get_config()
    import_main(config.main_path)
    batch_args(config.exp_path, run, config)

