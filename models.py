import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from utils import Config, tensor

def get_mlp(hidden_sizes, gate=None, output_gate=None):
    layers = []
    input_dim = hidden_sizes[0]
    for output_dim in hidden_sizes[1:]:
        layer = nn.Linear(input_dim, output_dim)
        # init
        #torch.nn.init.xavier_uniform_(layer.weight.data)
        torch.nn.init.orthogonal_(layer.weight.data)
        torch.nn.init.constant_(layer.bias.data, 0)
        layers.append(layer)
        if gate: layers.append(gate()) 
        input_dim = output_dim
    layers = layers[:-1]
    if output_gate is not None:
        layers.append(output_gate())
    return nn.Sequential(*layers)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state, noise):
        raise NotImplementedError

class GaussianPolicy(Policy):
    def __init__(
        self,
        state_dim,
        action_dim,
        mean_network,
        learn_std=True,
        gate_output=False,
    ):
        super().__init__()
        self.mean = mean_network
        if learn_std: self.std = nn.Parameter(torch.zeros(action_dim))
        else: self.std = torch.ones(action_dim)
        self.gate_output = gate_output
        self.learn_std = learn_std
        self.to(Config.DEVICE)

    def distribution(self, obs):
        obs = tensor(obs)
        mean = self.mean(obs)
        if self.gate_output: mean = torch.tanh(mean)
        if self.learn_std:
            dist = torch.distributions.Normal(mean, F.softplus(self.std))
        else:
            dist = torch.distributions.Normal(mean, self.std)
        return dist 

    def forward(self, obs, noise):
        #:: there is an issue with gpu of multiprocessing, unless you want to have one GPU each process, it is not worth it.
        obs = tensor(obs)
        mean = self.mean(obs)
        if self.gate_output:
            mean = torch.tanh(mean)
        if self.learn_std:
            action = mean + tensor(noise) * F.softplus(self.std)
            #action = mean
        else:
            action = mean + tensor(noise)
            #action = mean
        return action.cpu().detach().numpy()
