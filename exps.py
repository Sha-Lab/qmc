import re
import os
import sys
import shutil
import shlex
import argparse
import filelock
import traceback
import subprocess
import random
import inspect
from itertools import product
from pathlib import Path
from termcolor import colored

### gen exps ###
# commandr
_cmd_dict = {}

# a decorator that put the decorated function into a command dictionary, the function can then be called by its function name or name specified in input
def cmd(name=None):
    name_dict = {'name': name}
    def f(g):
        name = name_dict['name']
        if name is None:
            name = g.__name__
        _cmd_dict[name] = g
        return g
    return f

# get the function name the current code is in
# actually return the caller's name, because directly getting the name will be 'get_function_name'
def get_function_name():
    return inspect.stack()[1][3]

# parse command line arguments into function input (args, kwargs)
def parse_args_as_func(argv):
    args = []
    kwargs = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith('-'):
            kwargs[argv[i].lstrip('-')] = argv[i+1]
            i += 2
        else:
            args.append(argv[i])
            i += 1
    return args, kwargs

def annotate(arg, p):
    if isinstance(p.annotation, inspect._empty):
        return arg
    return p.annotation(arg)

def cmd_frun(name, *args, **kwargs):
    f = _cmd_dict[name]
    #sig = signature(f)
    #args = [annotate(arg, p) for arg, p in zip(args, sig.parameters.values())]
    #kwargs = {k: annotate(v, sig.parameters[k]) for k, v in kwargs.items()}
    return f(*args, **kwargs)

def cmd_run(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args, kwargs = parse_args_as_func(argv)
    cmd_frun(args[0], *args[1:], **kwargs)
# end of commandr

def generate_cmd(args=None, kwargs=None):
    cmds = ''
    if args:
        cmds += ' '.join(map(process_arg, args))
    cmds += ' '
    if kwargs:
        cmds += ' '.join(['{} {}'.format(k, process_arg(v)) for k, v in kwargs.items()])
    return cmds

def dump_args(f, args=None, kwargs=None):
    f.write(generate_cmd(args, kwargs) + '\n')

def process_arg(arg):
    if isinstance(arg, (list, tuple)):
        return ' '.join([str(v) for v in arg])
    return arg

def merge_dict(*dicts):
    merged_d = {}
    for d in dicts:
        merged_d.update(d)
    return merged_d

def product_dict(*dict_lists):
    dict_lists = product(*dict_lists)
    dicts = []
    for dict_list in dict_lists:
        dicts.append(merge_dict(*dict_list))
    return dicts

# return powerset of toggles
# currently composite toggle is not supported
def process_toggles(toggles):
    total_toggles = []
    for i in range(1 << len(toggles)):
        total_toggles.append([toggles[j] for j in range(len(toggles)) if (i & (1 << j))])
    return total_toggles

# compositite variant: {'--key': {value: {variant}}}
def process_variants(variants):
    if not variants: return [{}]
    raw_variants = {k: v for k, v in variants.items() if isinstance(v, list)}
    keys, values = zip(*raw_variants.items())
    raw_variants = [dict(zip(keys, v)) for v in product(*values)]
    composite_variants = []
    for key, variant in variants.items():
        if isinstance(variant, list): continue
        composite_variant = []
        for value, sub_variant in variant.items():
            cur_variant = process_variants(sub_variant)
            for v in cur_variant:
                v[key] = value
            composite_variant += cur_variant
        composite_variants.append(composite_variant)
    composite_variants = product_dict(*composite_variants)
    total_variants = product_dict(raw_variants, composite_variants)
    return total_variants

# generate arguments by template
# exp_path: the output file path, if None, output to stdout
# args: shared positional arguemnts
# kwargs: shared key word arguements
# toggles: the flags that considered all subsets of used or not used
# variants: the key word arguments that try all product combinations
# post_option: the function that post process the toggle, variant combination
# touch: whether to overwrite or append to the exp_path
# shuffle: whether to randomly shuffle the generated commands
def generate_args(exp_path, args, kwargs, toggles, variants, post_option=None, touch=True, shuffle=False):
    toggles = process_toggles(toggles)
    variants = process_variants(variants)
    if post_option:
        options = [post_option(toggle, variant) for toggle, variant in product(toggles, variants)]
    else:
        options = product(toggles, variants)
    if shuffle:
        random.shuffle(options)
    if exp_path is None:
        for toggle, variant in options:
            dump_args(sys.stdout, args + toggle, dict(list(kwargs.items())+list(variant.items())))
    else:
        if not Path(exp_path).parent.exists():
            Path(exp_path).parent.mkdir(parents=True)
        if touch: open(exp_path, 'w').close()
        with open(exp_path, 'a+') as f:
            for toggle, variant in options:
                dump_args(f, args + toggle, dict(list(kwargs.items())+list(variant.items())))

### end of gen exps ###

### run exps ###
def parse_tag(tag_str, args):
    if tag_str is None: return None
    groups = re.split(r'(\[.*?\])', tag_str)
    tag = []
    for group in groups:
        if group.startswith('['):
            elem = getattr(args, group[1:-1])
            if isinstance(elem, list): # expand list
                elem = '_'.join(elem)
            tag.append(str(elem))
        else:
            tag.append(group)
    return ''.join(tag)

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

def is_git_diff(): # this is slow...
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
        args = read_args(exp_path)
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
                push_args(args_str, exp_path)
                break

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_path', type=str)
    parser.add_argument('exp_path', type=str)
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

def run_one_exp(main_f):
    if ('-d' in sys.argv or check_git()) and check_gpu():
        if '-d' in sys.argv: sys.argv.remove('-d')
        from ipdb import slaunch_ipdb_on_exception
        with slaunch_ipdb_on_exception():
            main_f()

def run_exps():
    config = get_config()
    import_main(config.main_path)
    batch_args(config.exp_path, run, config)

### run exps ###

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
