import numpy as np
import pickle
import matplotlib.pyplot as plt


def plot_action_distribution(train_paths):
    train_a = np.concatenate([p['actions'][:-1] for p in train_paths])
    fig, axs = plt.subplots(2, 4, figsize=(41, 18))
    count = 0
    for i in range(2):
        for j in range(4):
            axs[i, j].hist(train_a[:, count], color = 'blue', edgecolor = 'black', bins = int(180/5))
            axs[i, j].set_title("joint {} action".format(count))
            axs[i, j].legend(loc="upper left")
            count += 1
    fig.savefig("action_distribution")



#### dataset version 1: rescaled absolute jointstates as actions
train_paths = pickle.load(open('morel_pushing-lid-delta-abs.pkl', 'rb'))
for p in range(len(train_paths)):
    train_paths[p]['actions'] *= np.array([1, 1, 1, 1/2.6, 1, 1/2.2, 1, 1]) # scale actions so each dimension is in range [-1, 1]
with open('parsed_datasets/morel_pushing-lid-delta-abs-rescaled.pkl', 'wb') as handle:
    pickle.dump(train_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
#### verify dataset version 1
# train_paths = pickle.load(open('morel_pushing-lid-delta-abs-rescaled.pkl', 'rb'))
# plot_action_distribution(train_paths)



#### dataset version 2: add noise to each trajectory for num_replay times
train_paths = pickle.load(open('morel_pushing-lid-delta-abs.pkl', 'rb'))
num_replay = 3 # the number of times each trajectory is replayed (i.e add noise to states)
replayed_paths = []
for p in range(len(train_paths)): # only add noise to states (including goals)! delta actions are computed based on the noisy states at the next step
    for i in range(num_replay):
        cur_replay_path = train_paths[p].copy()
        cur_replay_path['observations'] += np.random.standard_normal(size=train_paths[p]['observations'].shape) * 0.0001
        replayed_paths.append(cur_replay_path)
train_paths.extend(replayed_paths)
for p in range(len(train_paths)):
    horizon, cmd_shape = train_paths[p]['actions'].shape
    train_paths[p]['actions'] = np.zeros([horizon - 1, cmd_shape])
    for i in range(horizon - 1):
        delta_state = train_paths[p]['observations'][i + 1] - train_paths[p]['observations'][i]
        delta_state = delta_state[:8]
        delta_state[:-1] = delta_state[:-1] * 20 # not scaling the gripper action
        train_paths[p]['actions'][i] = delta_state
    train_paths[p]['observations'] = train_paths[p]['observations'][:-1]
with open('parsed_datasets/morel_pushing-lid-delta-abs-deltastate.pkl', 'wb') as handle:
    pickle.dump(train_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
#### verify dataset version 2
# train_paths = pickle.load(open('morel_pushing-lid-delta-abs-deltastate.pkl', 'rb'))
# plot_action_distribution(train_paths)
