###################################
###################################

MLC is the Machine Learned Collisions code.

We use this code to build data driven collision models that can be dropped into a Boltzmann solver.
The included source code is mainly there to interface with TensorFlow and handle scaling/transformations.

Installing is fairly simple if you've used cmake before.
See the scripts/do-configure-mlc file to see the general cmake command.

When building against the MLC library, use the simple command:

find_package(mlc REQUIRED)
target_link_libraries(<target> PUBLIC mlc)

which will include everything for TensorFlow and MLC (tested on Linux only).

Note you will need to set mlc_ROOT to wherever you installed the MLC library (the lib directory).

MLC was written by Sean T. Miller as part of Sandia LDRD 218322

MLC is SCR 2831.0

###################################
###################################
