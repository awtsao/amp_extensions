# amp_extensions
This repo contains the code for (AMP/DeepMimic)(https://github.com/xbpeng/DeepMimic) and its extensions. My thesis work was about imitation learning for simulated characters and the framework I focused on was AMP. The original code was added to adapt it to the OpenAI Gym environment (see ```gym-deepmimic```). To test this framework, code was adapted from [MILO](https://github.com/jdchang1/milo) and a new gym environment was created to hold the dynamics model (see ```gym-simenv```). 

## Setup
Follow the [setup instructions](setup.md) before getting started. You can follow the original instructions for [DeepMimic](https://github.com/xbpeng/DeepMimic) and [MILO](https://github.com/jdchang1/milo) separately but the setup instructions should have everything necessary. 

After following the setup instructions, install the 5 local packages (```deepmimic, gym-deepmimic, gym-simenv, mjrl, milo```) using ```pip```. That is, please run the commands from the base directory:

`cd deepmimic && pip install -e .`

`cd gym-deepmimic && pip install -e .`

`cd gym-simenv && pip install -e .`

`cd milo && pip install -e .`

`cd mjrl && pip install -e .`



## Running Experiments
#### MILO with AMP
#### Downloading example datasets
MILO requires an offline and expert data. To generate these, use ```collect_data.py``` and ```collect_expert.py``` in ```milo```. Details can be found in the [milo readme](milo/README.md)

An offline and expert dataset can be downloaded from this [google drive](https://drive.google.com/drive/folders/14kz7eBiitF-ddvduElgPbI1sDtAE7t4d?usp=sharing). 
Place this inside a new directory ```data```. 

#### Running the experiment
Run the following command to run an experiment 

```python run.py --dynamic_dense_connect --dynamic_save_models --dynamic_transform --dynamic_id 0 --milo_id 0 --seed 100 --dynamic_num_model 4 --num_cpu 4 --dynamic_hidden_size 512 512 512 512 --dynamic_lr 0.0001 --dynamic_eps 0.0001 --dynamic_batch_size 128 --dynamic_epochs 500 --milo_train --actor_model_hidden 32 32 --critic_model_hidden 128 128 --samples_per_step 40000 --kl_dist 0.01 --gamma 0.995 --cg_iter 25 --cg_damping 1e-5 --gae_lambda 0.97 --vf_iters 2 --vf_lr 1e-4 --vf_reg_coef 1e-4 --lambda_b 0.0025 --pg_iter 1 --bw_quantile 0.1 --n_iter 300 --cg_iter 25 --bc_epochs 0  --experiment_name example```

This will train a dynamics ensemble of 4 models and then run the imitation learning portion of MILO. For more information on the possible parameters, see [```arguments.py```](milo/milo/arguments.py)

### Runnning Behavior Cloning
Running ```python run_BC.py``` will train a policy using behavior cloning in the ```mjrl``` package on the offline dataset in ```data```.

## Visualizing a policy in AMP
There are several methods of visualizing a policy. If using a policy trained by the AMP framework directly, then running 

```python DeepMimic.py --arg_file <arg_file>``` 

with the appropriate file to load in the policy will visualize the policy at 60 Hz. If using a different policy, there are two different options. One is to use the ```render``` function of ```gym_deepmimic```. The other is to use ```visualize.py```. Before using `visualize.py`, make sure to change ```load_policy``` and ```step_policy``` functions to fit your policy. Running ```python visualize.py``` should visualize the policy at 30Hz. The ```r``` key resets the character, ```t``` toggles animation, and the space bar is used for stepping one frame forward in time.

## Setting up the argument file 
Many different argument files can be found in [```deepmimic/deepmimic/args```](deepmimic/deepmimic/args). These argument files dictate what scene is used, whether DeepMimic or AMP is used, the parameters of the character and world, whether we train or load a policy, etc. 
#### Changing the paths
These files are originally meant to be used from the DeepMimic directory. If the entry point is not in DeepMimic directory, the paths in the argument files need to be changed. For example, [```run_amp_humanoid3d_spinkick_args.txt```](run_amp_humanoid3d_spinkick_args.txt) is used by ```run.py``` so the paths in the argument file are relative to the location of ```run.py```. 

#### Adding time limit to run files
The original ```run``` arg files originally don't have limits on the time so any looping motions will go on until the character falls. To set a horizon, add the lines 
- ```time_end_lim_min <seconds>```
- ```time_end_lim_max <seconds>```


The seconds should be the same here. These two lines are copied over from the ```train``` arg files. 


#### DeepMimic path 
One thing that is required when using the argument files outside of the DeepMimic directory is the addition of the ```--deepmimic_path``` argument. 

#### Loading a model
To load in a different model (that was trained by AMP), change the path to the model. 

## Documentation
Besides comments in the code, we have added extra documentation in the ```README.md``` files in [`milo`](milo/README.md), [`mjrl`](mjrl/README.md), [`gym-simenv`](gym-simenv/README.md), [`gym-deepmimic`](gym-deepmimic/README.md), and [`DeepMimic`](deepmimic/README.md)
that discusses code changes and use cases for functions. 

## Possible Issues and Solutions
This section will be updated as new problems/solutions arise. 
 - If running on linux and you run into an error about too many processes or files open, it may be beneficial to increase the resource limit by using `ulimit`. For example, `ulimit -n 4096`. 
