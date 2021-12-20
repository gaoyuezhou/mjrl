
from mjrl.utils.gym_env import GymEnv
import numpy as np

class FrankaSpec():
    observation_dim = 11
    action_dim = 8

    def __init__(self):
        pass


class FrankaEnv(GymEnv):
    observation_dim = 11
    action_dim = 8
    start_js = [-0.145, -0.67, -0.052, -2.3, 0.145, 1.13, 0.029] + [0.08]
    spec = FrankaSpec()
    horizon = 500
    act_repeat = 1

    # observaion mask for scaling ???
    #obs_mask = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    obs_mask = np.array([1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0])


    def __init__(self):
        pass

    def reset(self):
        sampled_jointstate = self.start_js + np.random.normal(scale=0.01, size=8)
        sampled_tag = np.random.uniform(low=[-0.3, -0.22, -0.3],high=[0.3, -0.18, 0.1])
        return np.concatenate([sampled_jointstate, sampled_jointstate, sampled_tag])


