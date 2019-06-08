import torch
import torch.nn.functional as F
from torch import nn
from utils import Config, tensor

def get_mlp(hidden_sizes, gate=None, gate_output=False):
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
    if gate is not None and not gate_output: layers = layers[:-1]
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
        learn_std=False,
    ):
        super().__init__()
        self.mean = mean_network
        self.std = torch.zeros(action_dim)
        if learn_std: self.std = nn.Parameter(self.std)
        self.to(Config.DEVICE)

    def distribution(self, obs):
        obs = tensor(obs)
        mean = self.mean(obs)
        dist = torch.distributions.Normal(mean, tensor(torch.ones_like(self.std)))
        #mean = torch.tanh(self.mean(obs))
        #dist = torch.distributions.Normal(mean, F.softplus(self.std))
        #log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        return dist 

    def forward(self, obs, noise):
        obs = tensor(obs)
        #mean = torch.tanh(self.mean(obs)) # bounded action!!!
        mean = self.mean(obs)
        #action = mean + tensor(noise) * F.softplus(self.std)
        action = mean + tensor(noise)
        return action.cpu().detach().numpy()
