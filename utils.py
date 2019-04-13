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

def rollout(env, K, noises):
    states = []
    actions = []
    rewards = []
    done = False
    s = env.reset()
    cur_step = 0
    while not done:
        a = K.dot(s) + noises[cur_step]
        next_s, r, done, _ = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = next_s
        cur_step += 1
    return np.asarray(states), np.asarray(actions), np.asarray(rewards)
  
def mse(a, b):
    return ((a - b) ** 2).mean()
