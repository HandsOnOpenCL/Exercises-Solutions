Source Code for the Exercises and Solutions
==========================================

These examples, and the slides, and released under the "attribution CC BY" creative commons license.

Introduction
------------

Please checkout the repository using git with the following command:

`git clone git://github.com/HandsOnOpenCL/Exercises-Solutions.git`

Found any issues or have some comments? Please submit a bug report in the
[Issue tab](https://github.com/HandsOnOpenCL/Exercises-Solutions/issues).

Pre-requisites
--------------

* OpenCL 1.1 (or greater)
* Python 2.7 (or greater)
* C99 compiler (we use gcc) with OpenMP support (used for timing the runs [optional])
* C++11 compiler (we use g++ or clang, also tested with Intel's icc)

Need help setting up OpenCL?
Check out the first section in the lecture slides for information
about setting up OpenCL on Linux for AMD (CPU, GPU, APU),
Intel CPUs and NVIDIA GPUs.

Building
--------

We assume here that your current working directory is the location of the source code;
e.g. `/path/to/Exercises-Solutions/Solutions/Exercise04/C`

**Python**

Just run `python source.py` to run the code.

**C**

You must first run `make` to build the binary.
We assume that your environment is set up to find the OpenCL library; if you have trouble
try `export CPATH=/path/to/OpenCL/include` and `export LD_LIBRARY_PATH=/path/to/OpenCL/lib`.

You can also run `make` in the Examples/ and Solutions/ high-level directory;
this calls all the sub-directory make files so all the examples can be built in one command.
This also builds all the C++ examples.

Define the variable `DEVICE` in the Makefiles to be one of the OpenCL device types to vary the device type the C applications use.
This can be done easily in the two global Makefiles found in the Exercises and Solutions directories.
To use a GPU, for example, change the line `DEVICE = CL_DEVICE_TYPE_DEFAULT` to `DEVICE=CL_DEVICE_TYPE_GPU`.

Note: you can also edit each of the source files to use a specific device type, but we would recommend using the global Makefile method above.

Define the variable `CC` to change the C compiler used.
By default, this is set to gcc for all platforms.

**C++**

You must first run `make` to build the binary.
We assume that your environment is set up to find the OpenCL library.

You can also run `make` in the Examples/ and Solutions/ high-level directory;
this calls all the sub-directory make files so all the examples can be built in one command.
This also builds all the C examples.

Define the variable `DEVICE` in the Makefiles to be one of the OpenCL device types to vary the device type the C++ applications use.
This can be done easily in the two global Makefiles found in the Exercises and Solutions directories.
To use a GPU, for example, change the line `DEVICE = CL_DEVICE_TYPE_DEFAULT` to `DEVICE=CL_DEVICE_TYPE_GPU`.

Note: you can also edit each of the source files to use a specific device type, but we would recommend using the global Makefile method above.

Define the variable `CPPC` to change the C compiler used.
By default, this is set to g++ on Linux, and clang++ on OS X.

Directory structure
-------------------

The Exercises directory contains all the code
needed to be handed out at the start of the
tutorial for the exercises to be completed.

The Solutions directory contains sample code
providing an example implementation which
solves the exercises in the lecture notes.

Each directory contains a subdirectory for
each of the exercises. Each of these contains
an implementation of the exercise or solution
in C, C++ and Python.
