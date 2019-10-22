import sys
import inspect
import random
from itertools import product
from pathlib import Path

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
    if raw_variants:
        keys, values = zip(*raw_variants.items())
        raw_variants = [dict(zip(keys, v)) for v in product(*values)]
    else:
        raw_variants = [{}]
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

# usage:
# from ipdb import launch_ipdb_on_exception
# from exps import cmd, cmd_run, generate_args
# @cmd()
# def search(touch: int=1, shuffle: int=0):
#     args = [...]
#     kwargs = {
#         '--key': value,
#         ...
#     }
#     toggles = [...]
#     variants = {...}
#     def post_variant(toggle, variant):
#         ...
#         return toggle, variant
#     generate_args('exps/search', args, kwargs, variants, post_option=post_option, touch=touch, shuffle=shuffle)
#
# if __name__ == "__main__":
#     with launch_ipdb_on_exception():
#         cmd_run()
