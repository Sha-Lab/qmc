import argparse
from utils import batch_args, with_null
# local file
from main import main

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_path', type=str)
    parser.add_argument('-d', action='store_true')
    return parser.parse_args()

def run(args, config):
    if config.d:
        context = slaunch_ipdb_on_exception
    else:
        context = with_null
    with context:
        main(args)


if __name__ == "__main__":
    config = get_config()
    batch_args(config.exp_path, run, config)
