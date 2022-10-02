# AMP/DeepMimic
This pacakge contains the code for the AMP/DeepMimic framework. Please see the original code and documentation [here](https://github.com/xbpeng/DeepMimic). That contains information on how to use the interface when visualizing policies, information about the mocap data, as well as fixes for potential bugs and errors.


## Setup
Make sure to follow the setup instructions [here][(../setup.md)] and then install the DeepMimic.pacakge by running 

`pip install -e .`


## How to Use
This section is taken directly from the [`README.md`](https://github.com/xbpeng/DeepMimic/blob/master/README.md) in the original DeepMimic repository. 

The following commands should be run inside the [`DeepMimic`](DeepMimic) child directory. 

Once the python wrapper has been built, training is done entirely in python using Tensorflow.
`DeepMimic.py` runs the visualizer used to view the simulation. Training is done with `mpi_run.py`, 
which uses MPI to parallelize training across multiple processes.

`DeepMimic.py` is run by specifying an argument file that provides the configurations for a scene.
For example,
```
python DeepMimic.py --arg_file args/run_humanoid3d_spinkick_args.txt
```

will run a pre-trained policy for a spinkick. Similarly,
```
python DeepMimic.py --arg_file args/play_motion_humanoid3d_args.txt
```

will load and play a mocap clip. To run a pre-trained policy for a simulated dog, use this command
```
python DeepMimic.py --arg_file args/run_dog3d_pace_args.txt
```

To train a policy, use `mpi_run.py` by specifying an argument file and the number of worker processes.
For example,
```
python mpi_run.py --arg_file args/train_humanoid3d_spinkick_args.txt --num_workers 16
```

will train a policy to perform a spinkick using 16 workers. **If using OpenMPI, this may not work as `mpi_run.py` uses the subprocess module which may not be compatiable with mpi when using OpenMPI. Instead of running the above command, try running the following**:
```
mpiexec -n 16 python DeepMimic_Optimizer.py --arg_file args/train_amp_humanoid3d_spinkick_args.txt
```

As training progresses, it will regularly
print out statistics and log them to `output/` along with a `.ckpt` of the latest policy.
It typically takes about 60 millions samples to train one policy, which can take a day
when training with 16 workers. 16 workers is likely the max number of workers that the
framework can support, and it can get overwhelmed if too many workers are used.

A number of argument files are already provided in `args/` for the different skills. 
`train_[something]_args.txt` files are setup for `mpi_run.py` to train a policy, and 
`run_[something]_args.txt` files are setup for `DeepMimic.py` to run one of the pretrained policies.
To run your own policies, take one of the `run_[something]_args.txt` files and specify
the policy you want to run with `--model_file`. Make sure that the reference motion `--motion_file`
corresponds to the motion that your policy was trained for, otherwise the policy will not run properly.

Similarly, to train a policy using amp, run with the corresponding argument files:
```
python mpi_run.py --arg_file args/train_amp_target_humanoid3d_locomotion_args.txt --num_workers 16
```

Pretrained AMP models can be evaluated using:
```
python DeepMimic.py --arg_file args/run_amp_target_humanoid3d_locomotion_args.txt
```

## Debugging DeepMimicCore in Visual Studio(Windows Only)
In order to use the debugger with `DeepMimicCore.sln`, first make sure everything is set up according to [`setup.md`](../setup.md). Before debugging, either copy the [`args`](deepmimic/args) and [`data`](deepmimic/data) folder to DeepMimicCore or copy an argument file and modify the paths like in the example argument file [here](deepmimic/DeepMimicCore/run_amp_humanoid3d_spinkick_args.txt). With the argument file setup, edit the `DeepMimicCore` properties in Visual Studio (right click on DeepMimicCore in the solution explorer -> Properties). Make sure the configuration is set to `Debug` and all the `C/C++` and `Linker` settings are setup correctly. 

Go to `Configuration Properties -> Debugging` and make sure `Debugger to launch` is set to `Local Windows Debugger`. Set `Command Arguments` to the following: `--arg_file <path to arg file here>`. Apply the settings.

To use the debugger properly, make sure to set breakpoints in the code. It is recommended to set `enable_draw` in `SetUpDeepMimicCore` to False and to also comment out `InitDraw`, `SetUpDraw`, and `DrawMainLoop` in the `main` function.

In the `main` function, `SetupDeepMimicCore` handles the setup of the world, characters, etc. Add any extra code here to step through different parts of the code. For example, the following code resets the world and character, updates a few times, and checks termination.

```
int main(int argc, char** argv)
{
	FormatArgs(argc, argv, gArgs);

	SetupDeepMimicCore();
	gCore->Reset(0)
	gCore->Update(0.1)
	gCore->CheckTerminate()

	return EXIT_SUCCESS;
}
```

To actually begin debugging, set the configuration in the taskbar to `Debug` and `x64` and click on `Local Windows Debugger`. 

## Debugging DeepMimic in Visual Studio
It is also possible to debug the RL code for DeepMimic. Open [`DeepMimic.sln`](deepmimic/DeepMimic.sln). First, setup the script arguments by going to the Solution Explorer and opening up the properties for the `DeepMimic` project where a new tab should open. Select `Debug`. Make sure `Launch mode` is set to `Standard Python launcher` and `Script Arguments` is set to `--arg_file <path to arg file here>`. 

From here, there are multiple ways to debug. If you set the configuration to `Debug` and press the arrow to `Start`, it will execute `DeepMimic.py`. Make sure you have breakpoints set, otherwise the code will not stop. Other ways to debug are to rightclick on `DeepMimic.py` or `DeepMimic_Optimizer.py` and select `Start with Debugging`.

If the debugger doesn't work or throws an error, you should check to make sure Visual Studio is configured correctly with Python. You may have to make a new environment in Visual Studio for this. 


## Resetting the character with noise
`DeepMimicCore` was modified a bit to add noise to the character when resetting. Originally, the character is reset to a random time in the motion and matches the reference motions exactly. The functions `AddNoise`, `AddNoisePoseVel`, and `RandomRotatePoseVel` in [`KinCharacter.cpp`](deepmimic/DeepMimicCore/anim/KinCharacter.cpp) add this noise. See the specifications for those functions for more details about the parameters.
- `AddNoisePoseVel` samples a random noisy vector and adds it to the internal pose, `mPose`, and velocity, `mVel`, of the character. `mPose` and `mVel` store information about the position and velocity of the root as well as the rotation and angular velocity of joints. Note that for spherical joints, the rotations and angular velocities are stored as quaternions. 
- `RandomRotatePoseVel` adds noise by sampling a random rotation (1D for revolute and 3D for spherical joints) to add to each joint. 
- `AddNoise` calls these two prior functions to add noise to the state. 


The functions `reset_index` and `reset_time` in [`deepmimic_env.py`](deepmimic/env/deepmimic_env.py) reset the characters to a specific index frame in the mocap data or time in the motion, respectively, while adding noise. The specifications for the two functions give more details into the parameters and uses. 

## Getting discriminator features
The original AMP framework uses the function `record_amp_obs_agent` in `deepmimic_env.py` to get the features for the discriminator. Looking into `DeepMimicCore` as to how it is implemented, the function actually returns the features for the previous state and current state concatenated together in a "zig-zag" fashion. That is. the vector contains the features for the pose of the current state, the pose of the previous state, the velocity of the current state, and finally the velocity of the previous state. 

For future use cases, the function `record_amp_obs_agent_current` was added which simply returns the features (for both the pose and velocity) of the current state. 


## DeepMimicCore FAQ
The easiest way to understand how DeepMimicCore is to utilize the debugger or step through the code to understand the flow of the code. This section will serve as a very-high level directory/guide for DeepMimicCore. This will be added to as times goes on. 

### Kinematic Character
The kinematic character contains information about the reference motion. The relevant file for the kinematic character is `KinCharacter.cpp`. The controller for the kinematic character can be found in `KinController.cpp` and contains the reference motion itself. The way the simulated character is reset is by first loading the relevant frame from the reference motion onto the kinematic character and then syncing the simulated character with the kinematic character. To add noise, noise is added to the internal state and velocity of the kinematic character (see `AddNoise` in `KinController.cpp`). 

Information about the joints, links, etc., are retrieved using `KinTree.cpp`. 

### Simulated Chatacter
The simulated character is represented with `SimCharacter.cpp`. Like with the kinematic character, this class contains functions about its position, rotation, velocity, etc. It also contains information about the simulated links and simulated joints (see `mBodyParts` and `mJoints` in `SimCharacter.cpp`). The links and joints are represented by `SimBodyLink.cpp` and `SimJoint.cpp`. These links and joints can be differnet shapes (see `SimCapsule.cpp, SimSphere.cpp, SimObj.cpp`). 

To understand the controller, it is recommended to look at the 4 controller files `Controller.cpp, DeepMimicCharController.cpp, CtController,cpp, CtPDController.cpp`. The hierarcy of inheritance is as listed with `Controller.cpp` being the parent. 

When recording the state (i.e., for use in RL), the functions used are `RecordState` in `DeepMimicCharController.cpp`. These call `BuildStatePose` and `BuildStateVel` which use the implementation in `CtController.cpp`. These are also implemented in `DeepMimicCharController.cpp` but `CtController.cpp` is used as it is the child. 

#### Recording the state
As mentioned above, the state consists information about the pose and velocity of the character. If the scene is of type DeepMimic, then the phase information is also involved. From top to bottom, the information in the state vector consists of the phase, pose, and finally the velocity information. 
#### Pose
For the pose, `BuildStatePose` first builds the `origin_trans` which is the `world-to-origin` transform that transforms world coordinates to origin coordinates. The origin coordinate system is centered at the point right under the root of the character on th xz plane with x-axis aligned along the character's heading. This transform can essentially be used to compute coordinates as seen from the characters "point-of-view". Note that only the x- and z- coordinates will change when transforming the world coordinates to origin coordinates so the y-coordinates stay the same. 

From here, `BuildStatePose` iterates through each link in the character (see `humanoid3d.txt` to see which ID corresponds to which joint/link/body part) and computes its relative position compared to the root joint in the origin coordinate system as well as its rotation in the origin coordinate system. There are two steps for computing this relative position. First, the world position fo the root is multiplied by the `world-to-origin` transform to get the coordinates of the link in the origin coordinate system. Finally, the coordinates for the root joint (in the origin coordinate) system are subtracted from the link's coordinates to get the relative coordinates. Note that the x- and z- origin coordiantes for the root joint position are 0 since the origin coordinate system is cenetered at the root.

The rotation internaally is usually represented by 4D quaternions. However, when recording the state, the rotation is represented by two vectors: the norm and tangent vectors. The third axis of rotation can be determined using the cross product between the norm and tangent vector so these two vectors are sufficient for specifying a rotation. 
#### Velocity
The velocity is recorded in a similar fashion. The state contains information about the linear and angular velocity of each link in the origin coordinate system. Both are represented with 3D vectors. 



### Scenes
These scene files take care of loading the world, characters, etc., and provide many utility functions. For example, `SceneImitateAmp.cpp` consists of functions for computing the DTW cost, recording features for the agent and expert for the discriminator, etc. 

