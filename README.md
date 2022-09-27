###################################
###################################

MLC is the Machine Learned Collisions code.
We use this code to build data driven collision models that can be dropped into a boltzmann solver.
The included source code is mainly there to interface with TensorFlow and handle scaling/transformations.

Installing is fairly simple if you've used cmake before.
See the scripts/do-configure-mlc file to see the general cmake command.
The only real challenge is building the TensorFlow C API. 
See https://www.tensorflow.org/install/lang_c for details about the installation process (or just download the binaries).
Note that if you need to install tensorflow using Bazel, you will probably run into issues with firewalls (Proxy and SSL interception).
I've always had to drop off the VPN to get Bazel to work.

When building against the MLC library, use the simple command:

find_package(mlc REQUIRED)
target_link_libraries(<target> PUBLIC mlc)

which will include everything for tensorflow and mlc (tested on linux only).

Note you will need to set mlc_ROOT to wherever you installed the mlc library (the lib directory).

MLC is SCR 2831.0
###################################
###################################
