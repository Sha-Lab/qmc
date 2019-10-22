import os
import sys
import json
import shutil
import shlex
import socket
import argparse
import filelock
import traceback
import subprocess
from pathlib import Path
from termcolor import colored


# for parse_tag
class LazyMap(object):
       def __init__(self, **kwargs):
           self.dict = kwargs

       def __getitem__(self, key):
           return self.dict.get(key, "".join(["{", key, "}"]))

# use format to implement this!
def parse_tag(tag_str, args):
    if tag_str is None: return None
    tag_str = tag_str.replace('[', '{').replace(']', '}')
    kwargs = vars(args)
    return tag_str.format(**kwargs)

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

def is_git_diff(): # this is slow
    return bool(subprocess.check_output(['git', 'diff']))

def has_gpu():
    try:
        return str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
    except Exception as e:
        return 0

def check_git():
    if is_git_diff():
        print(colored('please commit your changes before running new experiments!', 'red', attrs=['bold']))
        return False
    return True

def check_gpu():
    if has_gpu() and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        print(colored('please set visible gpus before running experiments!', 'red', attrs=['bold']))
        return False
    return True

class with_null:
    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        return None

### client ###
class ExpConfig:
    HOST = "dash-borg.usc.edu"
    PORT = 9999

    @classmethod
    def load_config(cls, fn):
        with open(fn) as f:
            config_dict = json.load(f)
        for k, v in config_dict.items():
            setattr(cls, k, v)


def submit_request(data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ExpConfig.HOST, ExpConfig.PORT))
        sock.sendall(bytes(json.dumps(data), "utf-8"))
        reply = json.loads(str(sock.recv(1024), "utf-8"))
    return reply


def read_args_client(args_path):
    return submit_request({
        'request': 'read_args',
        'args_path': str(Path(ExpConfig.exp_path, args_path)),
    })['args']


def push_args_client(args_str, args_path):
    return submit_request({
        'request': 'push_args',
        'args_str': args_str,
        'args_path': str(Path(ExpConfig.exp_path, args_path)),
    })

### end of client ###

# for bach experiments, but combined with argparse and put this into your main.py
def read_args(args_path, timeout=30):
    args_path = Path(args_path)
    lock_dir = Path(args_path.parent, '.lock')
    if not lock_dir.exists():
        lock_dir.mkdir(parents=True)
    lock_fn = Path(lock_dir, args_path.stem)
    lock_fn.touch(exist_ok=True)
    with filelock.FileLock(str(lock_fn)).acquire(timeout=timeout):
        with open(str(args_path)) as f:
            jobs = f.read().splitlines(True) # same as convert_arg_line_to_args
        while jobs:
            job = jobs[0].strip()
            if not job or job.startswith('#'):
                jobs = jobs[1:]
            else:
                break
        if jobs:
            # skip empty line and comments
            args = shlex.split(jobs[0])
            with open(str(args_path), 'w') as f:
                f.writelines(jobs[1:])
        else:
            args = None
    return args

def push_args(args_str, args_path, timeout=30):
    args_path = Path(args_path)
    lock_dir = Path(args_path.parent, '.lock')
    if not lock_dir.exists():
        lock_dir.mkdir(parents=True)
    lock_fn = Path(lock_dir, args_path.stem)
    lock_fn.touch(exist_ok=True) # disadvantages: this will not be cleaned up
    with filelock.FileLock(str(lock_fn)).acquire(timeout=timeout):
        with open(str(args_path)) as f:
            jobs = f.read().splitlines(True)
        jobs.insert(0, args_str + '\n')
        with open(str(args_path), 'w') as f:
            f.writelines(jobs)

def is_empty_exp(exp_path):
    is_empty = True
    with open(str(exp_path)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            return False
    return is_empty

def batch_args(exp_path, exp_f, config=None):
    if config is not None and not config.d and not check_git(): return
    if not check_gpu(): return
    exp_path = Path(exp_path)
    assert exp_path.suffix in ['', '.run']
    if exp_path.suffix == '':
        run_path = exp_path.with_suffix('.run')
        assert not run_path.exists() or is_empty_exp(run_path), 'run file already exists'
        shutil.copy(str(exp_path), str(run_path))
        exp_path = run_path
    while True:
        if config.local:
            args = read_args(exp_path)
        else:
            args = read_args_client(exp_path)
        if args is None: break
        args_str = ' '.join(args)
        exp_finished = False
        try:
            print(args)
            config = argparse.Namespace() if config is None else config
            exp_finished = exp_f(args, config)
        except Exception as e:
            traceback.print_exc() # if traceback is not import, no error will be shown
        finally:
            if not exp_finished:
                if config.local:
                    push_args(args_str, exp_path)
                else:
                    push_args_client(args_str, exp_path)
                break

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_path', type=str)
    parser.add_argument('exp_path', type=str)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--config', type=str, default='.config')
    parser.add_argument('-d', action='store_true')
    return parser.parse_args()

def run(args, config):
    finished = False
    if config.d:
        from ipdb import slaunch_ipdb_on_exception
        context = slaunch_ipdb_on_exception
    else:
        context = with_null
    with context():
        main_module.main(args)
        finished = True
    return finished

# put this in __main__ of main.py to run main function
def run_one_exp(main_f):
    if ('-d' in sys.argv or check_git()) and check_gpu():
        if '-d' in sys.argv: sys.argv.remove('-d')
        from ipdb import slaunch_ipdb_on_exception
        with slaunch_ipdb_on_exception():
            main_f()

# usage: python run.py main.py exps/experiment_file
def run_exps():
    config = get_config()
    ExpConfig.load_config(config.config)
    import_main(config.main_path)
    batch_args(config.exp_path, run, config)

### parse args ###

# this support parsing template and exp_name
# will include command into args
def parse_args(parser, args=None, exp_name_attr=None):
    template_parser = argparse.ArgumentParser()
    template_parser.add_argument('--template', type=str, nargs='*')
    command = ' '.join(sys.argv[1:] if args is None else args)
    template_args, args = template_parser.parse_known_args(args)
    if template_args.template:
        read_args = []
        for template_fn in template_args.template:
            with open(str(Path('tps', template_fn))) as f:
                for line in f:
                    read_args += line.strip().split(' ')
        read_args = parser.parse_args(read_args)
        args = parser.parse_args(args, namespace=read_args)
    else:
        args = parser.parse_args(args)
    if exp_name_attr is not None:
        setattr(args, exp_name_attr, parse_tag(getattr(args, exp_name_attr), args))
    args.command = command
    return args

### end of parse args ###

if __name__ == "__main__":
    run_exps()
