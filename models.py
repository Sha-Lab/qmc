import torch
import torch.nn.functional as F


class GaussianMLPPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim, 
        hidden_sizes=(32, 32), 
        min_std=1e-6,
        gate=F.tanh,
    ):
        super().__init__()
        
