from os import environ
environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
environ['MKL_THREADING_LAYER']='GNU'
import numpy as np
import copy
import torch
import torch.nn as nn
import pickle
import mjrl.envs
import time as timer
import argparse
import os 
import json
import mjrl.samplers.core as sampler
import mjrl.utils.tensor_utils as tensor_utils
from tqdm import tqdm
from tabulate import tabulate
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.logger import DataLog
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.algos.mbrl.nn_dynamics import WorldModel
from mjrl.algos.mbrl.model_based_npg import ModelBasedNPG
from mjrl.algos.mbrl.sampling import sample_paths, evaluate_policy

from franka_demo.addon import FrankaWrapper

import numpy as np
import pickle

from mjrl.policies.gaussian_mlp import MLP

class MORel():
    def __init__(self, policy, config=None):
        self.policy = policy
        self.config = config
        print(f'Obs dim {self.policy.observation_dim} Act dim {self.policy.action_dim}')
        print(self.policy.in_shift, self.policy.in_scale)
        print(self.policy.out_shift, self.policy.out_scale)

    # For 1-batch query only! ## TODO
    def predict(self, s):
        at = self.policy.forward(s)
        if False: # NO RANDOM
            at = at + torch.randn(at.shape).to(policy.device) * torch.exp(policy.log_std)
        # clamp states and actions to avoid blowup
        return at.to('cpu').detach().numpy()

def _init_agent_from_path(path, device='cuda:0'):

    policy = pickle.load(open(path, 'rb')) # gaussian mlp
    policy.set_param_values(policy.get_param_values())
    policy.to(device)

    return MORel(policy)

JOINT_LIMIT_MIN = np.array(
    [-2.8773, -1.7428, -2.8773, -3.0018, -2.8773, 0.0025, -2.8773])
JOINT_LIMIT_MAX = np.array(
    [2.8773, 1.7428, 2.8773, -0.1398, 2.8773, 3.7325, 2.8773])
franka_sim = FrankaWrapper()
# def roaming_policy(state):
#     robostate = state[:8] # 8
#     goal = state[8:] # 3
#     clipped_cmd = np.zeros(8)
#     clipped_cmd[:7] = franka_sim.get_ik(goal, step_iter=20)
#     clipped_cmd = np.clip(clipped_cmd,
#         robostate[:8] - 0.05,
#         robostate[:8] + 0.05)
#     clipped_cmd[:7] = np.clip(clipped_cmd[:7],
#         JOINT_LIMIT_MIN,
#         JOINT_LIMIT_MAX)
#     clipped_cmd[7] = 0.08
#     return clipped_cmd
def roaming_policy(state):
    robostate = state[:8] # 8
    goal = state[8:] # 3
    clipped_cmd = np.zeros(8)
    clipped_cmd[:7] = franka_sim.get_ik(goal, step_iter=20)
    clipped_cmd = np.clip(clipped_cmd,
        robostate[:8] - 0.05,
        robostate[:8] + 0.05)
    clipped_cmd[:7] = np.clip(clipped_cmd[:7],
        JOINT_LIMIT_MIN,
        JOINT_LIMIT_MAX)
    clipped_cmd[7] = 0.08
    return clipped_cmd

parser = argparse.ArgumentParser(description='Model accelerated policy optimization.')
parser.add_argument('--dynamics', '-d', type=str, required=True, help='path to dynamics model to be evaluated')
parser.add_argument('--bc', '-bc', type=str, required=True, help='path to BC model')
parser.add_argument('--train_data', '-t', type=str, required=True, help='path to training data')
parser.add_argument('--val_data', '-v', type=str, required=False, help='path to validation data')
args = parser.parse_args()


models = pickle.load(open(args.dynamics, 'rb'))
bc_agent = _init_agent_from_path(args.bc)

train_paths = pickle.load(open(args.train_data, 'rb'))
# best_perf = -1e8
# ts = timer.time()
train_s = np.concatenate([p['observations'][:-1] for p in train_paths])
train_a = np.concatenate([p['actions'][:-1] for p in train_paths])
train_sp = np.concatenate([p['observations'][1:] for p in train_paths])
train_r = np.concatenate([p['rewards'][:-1] for p in train_paths])
train_rollout_score = np.mean([np.sum(p['rewards']) for p in train_paths])
train_num_samples = np.sum([p['rewards'].shape[0] for p in train_paths])
if not args.val_data:
    val_paths = pickle.load(open(args.train_data, 'rb'))
else:
    val_paths = pickle.load(open(args.val_data, 'rb'))
# best_perf = -1e8
# ts = timer.time()
val_s = np.concatenate([p['observations'][:-1] for p in val_paths])
val_a = np.concatenate([p['actions'][:-1] for p in val_paths])
val_sp = np.concatenate([p['observations'][1:] for p in val_paths])
val_r = np.concatenate([p['rewards'][:-1] for p in val_paths])
val_rollout_score = np.mean([np.sum(p['rewards']) for p in val_paths])
val_num_samples = np.sum([p['rewards'].shape[0] for p in val_paths])


# # GENERAL (Training) LOSS
# from mjrl.utils.logger import DataLog
logger = DataLog()
# for i, model in enumerate(models):
#     loss_general = model.compute_loss(train_s, train_a, train_sp)
#     logger.log_kv('dyn_loss_train_gen_' + str(i), loss_general)
# # print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
# #                             logger.get_current_log().items()))
# # print(tabulate(print_data))

# # GENERAL (Validation) LOSS
# for i, model in enumerate(models):
#     loss_general = model.compute_loss(val_s, val_a, val_sp)
#     logger.log_kv('dyn_loss_val_gen_' + str(i), loss_general)
# # print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
# #                             logger.get_current_log().items()))
# # print(tabulate(print_data))

# # DISAGREEMENT on Training Data
# np.set_printoptions(suppress=True, precision=3, threshold=1000)
# delta = np.zeros(train_s.shape[0])
# for idx_1, model_1 in enumerate(models):
#     pred_1 = model_1.predict(train_s, train_a)
#     # import pdb; pdb.set_trace()
#     for idx_2, model_2 in enumerate(models):
#         if idx_2 > idx_1:
#             pred_2 = model_2.predict(train_s, train_a)
#             disagreement = np.linalg.norm((pred_1-pred_2), axis=-1)
#             delta = np.maximum(delta, disagreement)
#             if np.max(disagreement) > 0.05:
#                 rn = np.argmax(disagreement)
#                 print(train_s[rn], train_a[rn], pred_1[rn], pred_2[rn], pred_1[rn] - pred_2[rn])
# print(f"Disagreement on given dataset: {sorted(delta)[::-1][:10]}")

# # One-Step DISAGREEMENT on Training Data, with action given by BC, i.e. a = BC(s)
# train_a_prime = bc_agent.predict(train_s)
# np.set_printoptions(suppress=True, precision=3, threshold=1000)
# delta = np.zeros(train_s.shape[0])
# for idx_1, model_1 in enumerate(models):
#     pred_1 = model_1.predict(train_s, train_a_prime)
#     # import pdb; pdb.set_trace()
#     for idx_2, model_2 in enumerate(models):
#         if idx_2 > idx_1:
#             pred_2 = model_2.predict(train_s, train_a_prime)
#             disagreement = np.linalg.norm((pred_1-pred_2), axis=-1)
#             delta = np.maximum(delta, disagreement)
#             if np.max(disagreement) > 0.05:
#                 rn = np.argmax(disagreement)
#                 print(train_s[rn], train_a_prime[rn], pred_1[rn], pred_2[rn], pred_1[rn] - pred_2[rn])
# print(f"One-Step Disagreement (with a=BC(s)) on given dataset: {sorted(delta)[::-1][:10]}")

# n-Step (Validation) LOSS
# models = [models[0]]
min_n = 3
max_n = 600
loss_nstep_all = [[] for i in range(4)]
loss_nstep_GT_all = [[] for i in range(4)]
n_all = [i for i in range(min_n, max_n)]

n = max_n
for i, model in enumerate(models):
    val_s_n = np.zeros([n, val_s.shape[1]])
    val_s_n[0] = val_s[0]
    val_a_n = val_a[:n]
    val_sp_n = val_sp[:n]
    for j in range(n):
        cur_s = val_s_n[j]
        # cur_a = val_a_n[j]
        cur_a = bc_agent.predict(cur_s)
        cur_sp = model.predict(cur_s, cur_a)
        if j < n - 1:
            val_s_n[j+1] = cur_sp
    
    for idx in range(min_n, max_n):
        loss_nstep = model.compute_loss(val_s_n[:idx], val_a_n[:idx], val_sp_n[:idx])
        loss_nstep_all[i].append(loss_nstep)

    # loss_nstep = model.compute_loss(val_s_n, val_a_n, val_sp_n)
    # loss_nstep_all[i].append(loss_nstep)
    # logger.log_kv('dyn_loss_val_nstep_' + str(i), loss_nstep)
# print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
#                             logger.get_current_log().items()))


# import matplotlib.pyplot as plt
# paths = val_paths
# # import pdb; pdb.set_trace()
# for p in range(len(paths)):
#     fig, axs = plt.subplots(3, 7, figsize=(21, 9))
#     count = 0
#     for i in range(3):
#         for j in range(7):
#             axs[i, j].plot(paths[p]['observations'][:, count], label="expert state[{}]".format(count))
#             axs[i, j].set_title("state[{}]".format(count))
#             count += 1
#             # axs[i, j].title("States for expert_{}".format(i))
#     fig.savefig("/home/franka/Desktop/plots/all_exp_pushing/traj_{}".format(p))

# import pdb; pdb.set_trace()

# print(tabulate(print_data))

import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 7, figsize=(41, 18))
paths = val_paths
for p in range(1):
    count = 0
    for i in range(3):
        for j in range(7):
            axs[i, j].plot(paths[p]['observations'][:, count], label="expert state[{}]".format(count))
            axs[i, j].set_title("state[{}]".format(count))
            count += 1
            # axs[i, j].title("States for expert_{}".format(i))
    # fig.savefig("/home/franka/Desktop/plots/expert_states_pushing/traj_{}".format(p))

    
# rollout on dynamics model[0]
expert_path = paths[0]
expert_states = paths[0]['observations']
for m in range(4):
    model = models[m]
    dynamics_states = np.zeros(expert_states.shape)
    dynamics_states[0] = expert_states[0]
    for j in range(expert_states.shape[0] - 1):
        cur_s = dynamics_states[j]
        cur_a = expert_path['actions'][j]
        cur_sp = model.predict(cur_s, cur_a)
        if j < n - 1:
            dynamics_states[j+1] = cur_sp

    for p in range(1):
        count = 0
        for i in range(3):
            for j in range(7):
                axs[i, j].plot(dynamics_states[:, count], label="dyn model {} state[{}]".format(m, count))
                axs[i, j].set_title("state[{}]".format(count))
                axs[i, j].legend(loc="upper left")
                if count > 17:
                    axs[i, j].set_ylim([-1, 1])
                count += 1
fig.suptitle("Expert states vs. states given by dynamics models with expert actions")
fig.savefig("/home/franka/Desktop/exp-lifting")
import pdb; pdb.set_trace()




# import pdb; pdb.set_trace()
# n-Step (Validation) LOSS using Optimal Roaming Policy
# n = 5
# for i, model in enumerate(models):
#     val_s_n_GT = np.zeros([n, val_s.shape[1]])
#     val_s_n_GT[0] = val_s[0]
#     val_a_n_GT = np.zeros([n, val_a.shape[1]])
#     val_a_n_GT[0] = val_a[0]
#     val_sp_n_GT = val_sp[:n]
#     for j in range(n):
#         cur_s = val_s_n_GT[j]
#         if j > 0: # switch to roaming action if not the first step
#             val_a_n_GT[j] = roaming_policy(cur_s)
#         cur_a = val_a_n_GT[j]
#         cur_sp = model.predict(cur_s, cur_a)
#         # print(np.sum(~(cur_a == val_a[j])),  np.sum(~(cur_sp == val_sp[j])))
#         if j < n - 1:
#             val_s_n_GT[j+1] = cur_sp

#     for idx in range(min_n, max_n):
#         loss_nstep_GT = model.compute_loss(val_s_n_GT[:idx], val_a_n_GT[:idx], val_sp_n_GT[:idx])
#         loss_nstep_GT_all[i].append(loss_nstep_GT)
#     # loss_nstep_GT = model.compute_loss(val_s_n_GT, val_a_n_GT, val_sp_n_GT)
#     # loss_nstep_GT_all[i].append(loss_nstep_GT)
#         # logger.log_kv('dyn_loss_val_nstep_GT_' + str(i), loss_nstep_GT)
#     # print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
#     #                             logger.get_current_log().items()))
#     # print(tabulate(print_data))

# import pdb; pdb.set_trace()
import matplotlib.pyplot as plt
for i in range(4):
    plt.plot(n_all, loss_nstep_all[i], label="model_{}".format(i))
plt.legend(loc="upper left")
plt.title("/home/franka/Desktop/plots/loss_{}_step".format(max_n))
plt.savefig("/home/franka/Desktop/plots/loss_{}_step.png".format(max_n))
plt.close()
# for i in range(4):
#     plt.plot(n_all, loss_nstep_GT_all[i], label="model_{}".format(i))
# plt.legend(loc="upper left")
# plt.title("loss_{}_step_GT".format(max_n))
# plt.savefig("loss_{}_step_GT.png".format(max_n))
import pdb; pdb.set_trace()