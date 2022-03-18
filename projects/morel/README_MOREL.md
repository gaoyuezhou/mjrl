# MOReL Pipeline

For the pushing lid task

## Dataset Download
Download the pushing lid dataset from [Google Drive](https://drive.google.com/drive/folders/1GPmEGEUkOtGkLHl7yaFjMiyZaPaupjky?usp=sharing)

All datasets have the same format. When loaded, each dataset is a list of dictionaries with the following keys:

    `['traj_id', 'jointstates', 'commands', 'goals', 'cam3c', 'cam4c', 'marker4', 'marker5', 'marker10', 'marker11', 'observations', 'actions', 'terminated', 'rewards']`
`observations`, `actions`, and `rewards` will be loaded and used during dynamics model training and MOReL training.
- `observations` (dim 21): jointstates (7 joints + 1 gripper) + joint velocities (of 7 joints) + lid pisition (x, y, z) + goal position (x, y, z)
- `actions` (dim 8): actions to 7 joints and the gripper

#### Datasets
- **morel_pushing-lid-delta-abs.pkl :** the original dataset. Actions are absolute joint angles. 731 trajectories in total.
- **morel_pushing-lid-delta-abs-rescaled.pkl :** generated from `morel_pushing-lid-delta-abs.pkl` with actions scaled to be all within range [-1, 1]. The vector multiplied to actions of all 8 dimensions are [1, 1, 1, 1/2.6, 1, 1/2.2, 1, 1]. Everything else is the same as the original dataset. 731 trajectories in total.
- **morel_pushing-lid-delta-abs-deltastate.pkl :** generated from `morel_pushing-lid-delta-abs.pkl` with each trajectory replayed for 3 times. Replay here means to directly add uniform noise (scale 0.001) to `observations`. Then, actions are computed by taking the difference of the first 8 dimensions of adjacent `observations`, and rescale them to be in range [-1, 1]. 731 + 731 * 3 = 2924 trajectories in total.


After downloading, put the datasets to folder `parsed_datasets` under the current `morel` directory. The script for generating `morel_pushing-lid-delta-abs-rescaled.pkl` and `morel_pushing-lid-delta-abs-deltastate.pkl` is also in folder `parsed_datasets`. 


## Change config file 
Before launching experiments, change the config file for pushing lid [here](https://github.com/gaoyuezhou/mjrl/blob/debug/projects/morel/configs/franka_pushing_lid.txt).
- `data_file` : path to the dataset for learning dynamics models, e.g `morel_pushing-lid-delta-abs.pkl`
- `bc_data`: path to the dataset for for BC training. Should be the same as `data_file`.
- `model_file`: path to dynamics models used for MOReL training. When launching `learn_model.py` to train dynamics models, this will be the name of the saved model.
- `num_iter`: when this is set to 0, `run_morel.py` will only train and save a BC model. When this is non-zero, it will train a MOReL model for this number of epochs.
- `init_policy`: if `num_iter` is non-zero so a MOReL model will be trained, this should not be `None` since we previously found that it's preferred to first train a BC policy, save it, and launch the script again with `init_policy` set to the saved BC policy. 

## Launch experiments
- Run `python learn_model.py --config configs/franka_pushing_lid.txt`. This will train and save `num_models` dynamics models under filename `model_file`.
- Set `num_iter` to 0 in the config file, and then run `python run_morel.py --config configs/franka_pushing_lid.txt --output <output_path_BC>` to train a BC model, where `<output_path_BC>` is the name of the model directory.
- Set `num_iter` to be, e.g. 300, set `init_policy` to `<output_path_BC>` (so the MOReL policy is initialized by the BC model trained in the previous step), and run `python run_morel.py --config configs/franka_pushing_lid.txt --output <output_path>` to train a MOReL model, where `<output_path>` is the name of the MOReL model directory. 

## Generate training plots
I usually use [this script](https://github.com/gaoyuezhou/mjrl/blob/debug/mjrl/utils/plot_from_logs.py) to generate plots. I modified it so the plot will be saved in the model directory.

Example script: `python plot_from_logs.py -d <output_path>`
