import unittest
from tqdm import tqdm

# a way to add the directory itself and avoid package problem
def add_parent_path():
    import sys 
    import os.path as osp 
    this_dir = osp.dirname(__file__)
    lib_path = osp.abspath( osp.join(this_dir, '..') )
    if lib_path not in sys.path: sys.path.insert(0, lib_path)
add_parent_path()

class TestPointMass(unittest.TestCase):
    def test_random(self):
        from gridworld.pointmass import PointMass
        import time
        env = PointMass('fourroom')
        for i in tqdm(range(500)):
            env.seed(i)
            env.reset()
            #env.render()
            for _ in range(500):
                _, _, done, _ = env.step(env.action_space.sample())
                #env.render()
                #time.sleep(0.01)
                if done: break

if __name__ == "__main__":
    unittest.main()
