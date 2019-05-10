import argparse
from ipdb import slaunch_ipdb_on_exception
from utils import batch_args, with_null
from ipdb import slaunch_ipdb_on_exception
from termcolor import colored
# local file
from main import main
from utils import is_git_diff

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_path', type=str)
    parser.add_argument('-d', action='store_true')
    return parser.parse_args()

def run(args, config):
    if not config.d and is_git_diff():
        print(colored('please commit your changes before running new experiments!', 'red', attrs=['bold']))
        return False
    finished = False
    if config.d:
        context = slaunch_ipdb_on_exception
    else:
        context = with_null
    with context():
        main(args)
        finished = True
    return finished

if __name__ == "__main__":
    config = get_config()
    batch_args(config.exp_path, run, config)

