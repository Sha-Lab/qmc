import os
import gym
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt
from .utils import Render, color_interpolate


CUR_DIR = os.path.dirname(__file__)


def read_map(filename):
    m = []
    with open(filename) as f:
        for row in f:
            m.append(list(row.rstrip()))
    return m


def get_grid_position(x, y):
    return int(x), int(y)


def sample_pos(m, exclude=set(), rng=np.random):
    while True:
        x = rng.uniform(len(m))
        y = rng.uniform(len(m[0]))
        x, y = get_grid_position(x, y)
        if (m[x][y] != '#') and ((x, y) not in exclude): return np.array([x, y])


colormap = {
    ' ': color_interpolate(0.0, plt.cm.Greys(0.02), plt.cm.Greys(0.2)),
    '@': color_interpolate(0.0, plt.cm.Greys(0.02), plt.cm.Greys(0.2)),
    '#': color_interpolate(0.0, plt.cm.Greys(0.12), plt.cm.Greys(0.3)), 
}


# push everything into wrapper (sample init position, sample goal, change map etc)
# for MDP, state should contains all information, which means that to simulate in parallel n rollouts, you only need one environment and n states
class PointMass(gym.Env):
    def __init__(
        self,
        map_name,
        init_pos=None,
        n_sub_steps=10,
        seed=0,
    ):
        self.map = read_map(os.path.join(CUR_DIR, 'maps', '{}.txt'.format(map_name)))
        self.row, self.col = len(self.map), len(self.map[0])
        self.seed(seed)
        if init_pos is not None:
            self.load_params({'init_pos': init_pos})
        else:
            self._init_pos = None
        self.n_sub_steps = n_sub_steps
        self.done_threshold = done_threshold

        self.observation_space = gym.spaces.Box(np.array([0.0, 0.0]), np.array([self.row, self.col]))
        self.action_space = gym.spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        self._render = None

    def load_params(self, params):
        if 'init_pos' in params:
            assert 0 < init_pos[0] < self.row and 0 < init_pos[1] < self.col
            self._init_pos = np.array(params['init_pos'])

    def seed(self, seed=None):
        self.rng, _ = seeding.np_random(seed)

    def reset(self):
        assert self._init_pos is not None
        self.pos = self._init_pos
        return self.pos

    def _is_blocked(self, pos):
        x, y = get_grid_position(*pos)
        return self.map[x][y] == '#'

    def transition(self, pos, action):
        assert not self._is_blocked(pos), 'start position in the wall'
        action = np.clip(action, self.action_space.low, self.action_space.high) # might cause problem
        dpos = 1.0 / self.n_sub_steps
        for _ in range(self.n_sub_steps):
            next_pos = pos + action * dpos
            if self._is_blocked(next_pos): break
            pos = next_pos
        return pos

    def render(self, repeat=32):    
        self.init_render(repeat)
        img = np.zeros((self.row, self.col, 3)) # if used as states, channels should be in the front
        for i in range(self.row):
            for j in range(self.col):
                img[i][j] = colormap[self.map[i][j]]
        img = img.repeat(repeat, 0).repeat(repeat, 1)
        pos_x, pos_y = int(repeat * self.pos[0]), int(repeat * self.pos[1])
        goal_x, goal_y = int(repeat * self.goal[0]), int(repeat * self.goal[1])
        img[pos_x-2:pos_x+2, pos_y-2:pos_y+2] = np.rint(255 * np.asarray(plt.cm.Reds(0.5)[:3]))
        img[goal_x-2:goal_x+2, goal_y-2:goal_y+2] = np.rint(255 * np.asarray(plt.cm.Blues(0.5)[:3]))
        self._render.render(img)

    def init_render(self, repeat):
      if self._render is None:
          self._render = Render(size=(self.col * repeat, self.row * repeat))
      return self


class ReachPointMass(PointMass):
    def __init__(
        self,
        map_name,
        goal=None,
        init_pos=None,
        n_sub_steps=10,
        done_threshold=0.8,
        seed=0,
    ):
        super().__init__(map_name, n_sub_steps=n_sub_steps, seed=seed)
        self.done_threshold = done_threshold
        if init_pos is not None:
            self.load_params({'init_pos': init_pos})
        else:
            self._init_pos = None
        if goal is not None:
            self.load_params({'goal': goal})
        else:
            self._goal = None

    def load_params(self, params):
        super().load_params(params)
        if 'goal' in params:
            self._goal = goal
        if self._init_pos is not None and self._goal is not None:
            assert get_grid_position(*self._init_pos) != get_grid_position(*self._goal)

    def step(self, action):
        self.pos = self.transition(self.pos, action)
        dist = np.linalg.norm(self.goal - self.pos)
        r = -dist
        done = dist < self.done_threshold
        return self.pos, r, done, {}


class GaussianMixtureRewardPointMass(PointMass):
    def __init__(
        self,
        map_name,
        gaussians=[], # c, mean, sigma
        init_pos=None,
        n_sub_steps=10,
        seed=0,
    ):
        super().__init__(map_name, init_pos=init_pos, n_sub_steps=n_sub_steps, seed=seed)
        self.gaussians = gaussians

    def step(self, action):
        self.pos = self.transition(self.pos, action)
        r = 0.0
        for c, mean, sigma in self.gaussians:
            r += c * np.exp(-(self.pos - mean).square().sum() / sigma)
        return self.pos, r, False, {}


class GaussianActionNoiseWrapper(gym.Wrapper):
    def __init__(self, env, scale, seed=None):
        super().__init__(self, env)
        assert scale > 0, 'scale should be strictly positive'
        self.scale = scale
        self.rng, _ = seeding.np_random(seed)

    def step(self, action):
        action += self.scale * self.rng.randn(*action.shape)
        return self.env.step(action)
