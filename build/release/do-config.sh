rm CMakeCache.txt
cmake \
-G "Ninja" \
-D TENSORFLOW_ROOT=/usr/local \
-D CMAKE_INSTALL_PREFIX=/usr/local \
../..
