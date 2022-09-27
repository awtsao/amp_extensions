# Setup
# Preliminary code changes
Before building DeepMimic, make sure that the issues discussed in two tickets https://github.com/xbpeng/DeepMimic/issues/21 and https://github.com/xbpeng/DeepMimic/issues/17 are resolved. That is, make the following code changes if they haven't been made already:
- In `DeepMimicCore.cpp`, comment `glutPostRedisplay` in the `Reshape` function.
- Add the following block of code to the beginning of  `MathUtil.h`
```
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
```
## Linux
#### Dependencies
``sudo apt install libgl1-mesa-dev libx11-dev libxrandr-dev libxi-dev``

``sudo apt install mesa-utils``

``sudo apt install clang``

``sudo apt install cmake``
``sudo apt install libopenmpi-dev``

The other dependencies (i.e. bullet, eigen, freeglut, glew, swig) discussed in the original [DeepMimic](https://github.com/xbpeng/DeepMimic) will be installed through conda instead of building manually. 
#### Creating the Conda Environment
It is recommended to create a new conda environment and installing the necessary packages by following the commands below
```
conda create -n <name> python=3.7
conda install -c conda-forge bullet=2.88 eigen=3.3.7 freeglut=3.0.0 glew=2.1.0 gdown tqdm gym matplotlib tabulate swig tensorflow=1.13.1 
conda install -c pytorch pytorch
conda install -c anaconda scipy
pip install mpi4py PyOpenGL PyOpenGL_accelerate tensorboard==2.10.0
```
If conda get stucks at solving environment or times out, try installing the packages from the ```conda-forge``` channel one at a time.

`Tensorboard v2.10.0` is technically not compatible with `Tensorflow v1.13.1`. The only reason it is installed is the MILO code uses `Tensorboard` and `PyTorch` needs a newer version than the `Tensorboard` installed with `Tensorflow v1.13.1`. If you want to use `Tensorboard` with `Tensorflow`, you may have to downgrade. Additionally, having `Tensorboard v2.10.0` may cause conflicts in the environment so only use it when necessary. 
#### Building DeepMimicCore 
DeepMimicCore will be built using ```clang```
1. Modify the `Makefile` in `DeepMimicCore/` by specifying the following,
	- `EIGEN_DIR`: Eigen include directory
	- `BULLET_INC_DIR`: Bullet source directory
	- `PYTHON_INC`: python include directory
	- `PYTHON_LIB`: python lib directory
	- `GL_INC`: GL include directory
	- `GL_LIB`: GL lib directory

Since all these were installed using conda, the only thing that should be edited is the path to the conda environment. 

2. Build wrapper,
	```
	make python
	```
This should generate `DeepMimicCore.py` in `DeepMimicCore/`. If there are errors about missing libraries, try searching them using the ```ld``` command such as ```ld -lGL --verbose```. It is possible that a symbolic link needs to be created to rename the libraries to something that the compiler will recognize. 

## Windows
#### MPI Dependency
Follow the instructions from Microsoft https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi

##### Eigen
Download Eigen 3.3.7 from its homepage http://www.eigen.tuxfamily.org/index.php?title=Main_Page. If the version is not listed on the home page, download Eigen 3.3.7 from https://gitlab.com/libeigen/eigen/-/releases and unzip the folder into ```deps```. We don't need to build Eigen as we only need the source code. 
#### FreeGlut
Download the Windows binaries for FreeGlut 3.0.0 from https://www.transmissionzero.co.uk/software/freeglut-devel/ (freeglut 3.0.0 MSVC package) and unzip it to ```deps``. Add the path to the the x64 ```freeglut.dll``` (this should be located in ```bin/x64```) to the ```Path``` environment variable. 
#### Glew
Download the Window binaries for GLEW http://glew.sourceforge.net/ to ```deps```. Add the path to the ```glew32.dll``` (this should be located in ```bin/Release/x64```) to the ```Path``` environment variable. 
#### Bullet
Download Bullet 2.88 from https://github.com/bulletphysics/bullet3/releases. Bullet will be built with Visual Studio like DeepMimicCore. The instructions for building bullet will be included in the section for [Bullet](#building-bullet)


### Windows Conda Environment
Follow the following commands. 
```
conda create -n <name> python=3.7
conda install -c conda-forge gdown tqdm gym matplotlib tabulate tensorflow=1.13.1 
conda install -c pytorch pytorch
conda install -c anaconda scipy
pip install mpi4py PyOpenGL_accelerate tensorboard==2.10.0
```
Do not install PyOpenGL directly from pip as the version downloaded (for python 3.7) is missing the freeglut DLL. Instead, download the pacakge wheel `PyOpenGL-3.1.6-cp37-cp37m-win_amd64.whl` from the [Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl). Once this is downloaded, install the wheel:

`pip install <path to PyOpenGL binary here>`

`Tensorboard v2.10.0` is technically not compatible with `Tensorflow v1.13.1`. The only reason it is installed is the MILO code uses `Tensorboard` and `PyTorch` needs a newer version than the `Tensorboard` installed with `Tensorflow v1.13.1`. If you want to use `Tensorboard` with `Tensorflow`, you may have to downgrade. Additionally, having `Tensorboard v2.10.0` may cause conflicts in the environment so only use it when necessary. 
### Building Bullet
Bullet can be built in ```Release``` or ```Debug``` mode. The ```Debug``` version allows us to use the debugger for ```DeepMimicCore```. The method for buildling both are similar. 
#### Using premake

1. In ```build_visual_studio_vr_pybullet_double.bat```, remove the –double flag to build bullet with single precision. Set ```myvar``` to the path for Python. Execute the bat file to build the necessary files using premake. 
2. Open ```build3/vs2010/0_Bullet3Solution.sln``` in Visual Studio. If prompted to retarget to a new version of Visual studio, accept. This solution contains many different projects which you will be building. 
3. All the projects should be configured automatically except potentially ```pybullet```. To configure the ```pybullet``` project, right click on the project and entire ```Properties```. First, select the correct ```Configuration``` mode (either ```Release``` or ```Debug```). Under ```C/C++->General->Additional Include Directories```, add the path to the ```include``` library (should contain ```Python.h```) for the python being used. In ```Linker->General->Additional Lib Directories```, make sure the path to the python ```libs``` folder is correct (this folder should contain ```python.lib```). 
4. Build the solution using the options in the taskbar ```Build->Build Solution```. 
5. The output files will be in ```bin```. However, the postfix ```vs2010_x64_release``` needs to be removed from each ```lib``` file. For example, ```Bullet3Dynamics_vs2010_x64_release.lib -> Bullet3Dynamics.lib```. If built in ```Debug``` mode, the prefix ```vs2010_x64_debug``` needs to be removed. The following code block may be helpful
```
import glob, os
for file in glob.glob("*.lib"):
    prefix = file[:file.find("_vs2010_x64_release")]
    os.rename(file, prefix+".lib")
```

6. The output directory for building in ```Debug``` and ```Release``` mode is the same ```bin``` folder so if you want to have both lib versions, it is recommended transfer either the ```Release``` or ```Debug``` files into a separate folder. 


#### Using Cmake
An alternative method is using ```CMake```. 
1. The first step is to install [CMake](https://cmake.org). 
2. Create an empty ```build_cmake``` folder inside the main directory. In that folder, run the following command for the release build:
```cmake -DBUILD_PYBULLET=ON -DBUILD_PYBULLET_NUMPY=ON -DUSE_DOUBLE_PRECISION=OFF -DBT_USE_EGL=ON -DCMAKE_BUILD_TYPE=Release ..```
For the debug build, run:
```cmake -DBUILD_PYBULLET=ON -DBUILD_PYBULLET_NUMPY=ON -DUSE_DOUBLE_PRECISION=OFF -DBT_USE_EGL=ON -DCMAKE_BUILD_TYPE=Debug ..```
3. Open ```BULLET_PHYSICS.sln``` inside ```build_cmake``` in Visual Studio. 
4. Follow step 4 in the steps for [using premake](#using-premake)
5. The ```.lib``` files should be inside ```build_cmake/lib```. Like in step 4 of [using premake](#using-premake), the postfix ```_release``` or ```_debug``` needs to be removed from the name. 

### DeepMimicCore in Windows
Visual studio will be used to build DeepMimicCore. One benefit of being on Windows and using Visual studio is that you can build DeepMimicCore in ```Debug``` mode. This allows you to use the debugger and step through the C++ code line by line. There are two configurations for building DeepMimicCore: ```Release_Swig``` and ```Debug```. ```Release_Swig``` creates the Python wrapper while ```Debug``` is for debugging purposes. The same steps can be used for building both. There are slight differences which will be noted in the steps below. 

1. The first step is to setup ```Swig``` on Windows so Visual Studio can recognize it.  To setup ```Swig```, download the files at: http://www.swig.org/. Add a new enviornment variable ```SWIG_DIR``` which links to the ```Swig``` directory. 
2. Open the ```DeepMimicCore.sln``` solution in Visual Studio. There should be 1 project, ```DeepMimicCore```, in the solution explorer. Right click on the project to access its properties. 
3. Select the configuration ```Release_Swig``` or ```Debug```. 
4. In ```C/C++->General->Additional Include Directory```, add the following:
      - Path to `Python include` directory. 
      - Path to ```FreeGLUT include``` directory. If the files were setup following the exact steps in [Windows Dependencies](#windows-dependencies), then the path should be ```$(SolutionDir)deps\freeglut-3.0.0\include. 
      - Path to `Bullet src` (i.e., $(SolutionDir)deps\bullet3-2.8.8\src)
      - Path to ```Eigen 3.3.7``` (i.e., $(SolutionDir)deps\eigen-3.3.7)
      - Path to ```GLEW include``` (i.e., $(SolutionDir)deps\glew-2.1.0\include)

5. In ```Link->General->Additional Library Directories```, add:
    - Path to ```Python libs``` directory. 
    - Path to ```FreeGLUT .lib``` (i.e., $(SolutionDir)deps\freeglut-3.0.0\lib\x64)
    - Path to ```Bullet .lib``` files (i.e., $(SolutionDir)deps\bullet3-2.88\bin). **Make sure to link the path to the bullet release .lib files if building ```Release_Swig``` or the bullet debug .lib files if building ```Debug```**.
6. Make sure that ```Linker->Input->Additional Dependencies``` has the following:
    - ```opengl32.lib```
    - ```glu32.lib```
    - ```glew32.lib```
    - ```BulletDynamics.lib```
    - ```BulletCollision.lib```
    - ```LinearMath.lib```
    - ```freeglut.lib```
7. Follow one of the following two options:
    - Add the paths to ```freeglut.dll``` and ```glew32.dll``` inside ```FreeGLUT``` and ```GLEW``` (make sure to choose the x64 configuration) to the ```Path``` system environment variable. 
    -  Copy (not **cut**)```freeglut.dll``` and ```glew32.dll``` to the ```DeepMimicCore``` directory. 
9. Set the configuration to ```Release_Swig``` or ```Debug```. Additionally, select the ```x64``` configuration. If in ```Release_Swig```, build the solution and this should generate ```DeepMimicCore.py``` inside ```DeepMimicCore```. If the configuration is ```Debug```, see the DeepMimic [`README.md`](deepmimic/README.md)
