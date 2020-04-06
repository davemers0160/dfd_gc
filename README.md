# Depth From Defocus Using the Graph Cuts Algorithm
Depth from Defocus using Graph Cuts Repositiory

## Dependencies

The code in this repository has the following dependecies:

1. [CMake 2.8.12+](https://cmake.org/download/)
2. [OpenCV v4+](https://opencv.org/releases/)
3. [davemers0160 common code repository](https://github.com/davemers0160/Common)
4. [davemers0160 dfd_common code repository](https://github.com/davemers0160/dfd_common)

Follow the instruction for each of the dependencies according to your operating system.

## Build

The project uses CMake as the pmrimary mechanism to build the executables.  There are some modifications that may have to be made to the CMakeLists.txt file in order to get the project to build successfully.

The first thing that must be done is to create an environment variable called "PLATFORM".  The CMakeLists.txt file uses this variable to determine where to look for the other required repositories and/or libraries.  These will be machine specific.

To create an environment variable in Windows (drop the -m if you do not have elevated privileges):
```
setx -m PLATFORM MY_PC
```

In Linux (usually placed in .profile or .bashrc):
```
export PLATFORM=MY_PC
```

In the CMakeLists.txt file make sure to add a check for the platform you've added and point to the right locations for the repositories/libraries.

### Windows

Execute the following commands in a Windows command window:

```
mkdir results
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" -T host=x64 ..
cmake --build . --config Release
```

Or you can use the cmake-gui and set the "source code" location to the location of the CmakeLists.txt file and the set the "build" location to the build folder. 

### Linux

Execute the following commands in a terminal window:

```
mkdir results
mkdir build
cd build
cmake ..
cmake --build . --config Release -- -j4
```

Or you can use the cmake-gui and set the "source code" location to the location of the CmakeLists.txt file and the set the "build" location to the build folder. Then open a terminal window and navigate to the build folder and execute the follokwing command:

```
cmake --build . --config Release -- -j4
```

The -- -j4 tells the make to use 4 cores to build the code.  This number can be set to as many cores as you have on your PC.

## Running

To run the code the best option is to supply a file that contains the image pairs and ground truth data. There are some samples images provided in the sample_images folder and a sample input text file is supplied in the inputs folder.

The following parameters can be supplied to the executable:

* -f input text file that contains a comma seprated list of images and ground truth data; Example: -f ../inputs/dfd_mb_input.txt
* -o directory where the results will be saved; Example: -o ../results/
* -s list of sigma values in the form (min:step:max) Example: -s 0.32:0.01:2.88
* -n index into the list supplied in option 'f' that causes the code to only run that image pair; Example: -n 10


To run this code from the command line in Windows using the sample input text file type the following:

```
dfd_gc_ex -f ../inputs/dfd_mb_sm_all.txt -o ../results/
```

## References

When using this work please cite the follokwing:

[3-D SCENE RECONSTRUCTION FOR PASSIVE RANGING USING DEPTH FROM DEFOCUS AND DEEP LEARNING](https://hammer.figshare.com/articles/3-D_SCENE_RECONSTRUCTION_FOR_PASSIVE_RANGING_USING_DEPTH_FROM_DEFOCUS_AND_DEEP_LEARNING/8938376/1)

DataCite
```
Emerson, David Ross (2019): 3-D SCENE RECONSTRUCTION FOR PASSIVE RANGING USING DEPTH FROM DEFOCUS AND DEEP LEARNING. figshare. Thesis. https://doi.org/10.25394/PGS.8938376.v1
```

BiBtex
```
@article{Emerson2019,
author = "David Ross Emerson",
title = "{3-D SCENE RECONSTRUCTION FOR PASSIVE RANGING USING DEPTH FROM DEFOCUS AND DEEP LEARNING}",
year = "2019",
month = "10",
url = "https://hammer.figshare.com/articles/3-D_SCENE_RECONSTRUCTION_FOR_PASSIVE_RANGING_USING_DEPTH_FROM_DEFOCUS_AND_DEEP_LEARNING/8938376",
doi = "10.25394/PGS.8938376.v1"
}
```

