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
        self._mean = mean_network
        if learn_std: self._std = nn.Parameter(torch.zeros(action_dim))
        else: self._std = torch.ones(action_dim)
        self.gate_output = gate_output
        self.learn_std = learn_std
        self.to(Config.DEVICE)

    def mean(self, obs):
        mean = self._mean(obs)
        if self.gate_output:
            mean = torch.tanh(mean)
        return mean

    @property
    def std(self):
        if self.learn_std:
            return torch.max(F.softplus(self._std), 1e-6 * torch.ones_like(self._std))
        return self._std

    def distribution(self, obs):
        obs = tensor(obs)
        dist = torch.distributions.Normal(self.mean(obs), self.std)
        return dist 

    #:: there is an issue with gpu of multiprocessing, unless you want to have one GPU each process, it is not worth it.
    def forward(self, obs, noise):
        obs = tensor(obs)
        action = self.mean(obs) + tensor(noise) * self.std
        return action.cpu().detach().numpy()
