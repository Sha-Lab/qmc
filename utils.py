import sys
import numpy as np
import random
import torch

def set_seed(t, r=None, p=None, c=None):
    if r is None:
        r = t
    if p is None:
        p = r
    torch.manual_seed(t)
    random.seed(r)
    np.random.seed(p)
    if c is not None:
      torch.cuda.manual_seed(c)

# commandr
_cmd_dict = {} 

def cmd(name=None):
    def f(g):
        nonlocal name
        if name is None:
            name = g.__name__
        _cmd_dict[name] = g
        return g
    return f

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

def cmd_frun(name, *args, **kwargs):
    return _cmd_dict[name](*args, **kwargs)

def cmd_run(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args, kwargs = parse_args_as_func(argv)
    cmd_frun(args[0], *args[1:], **kwargs)
    
