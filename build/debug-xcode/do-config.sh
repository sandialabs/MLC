cmake \
-G "Xcode" \
-D TENSORFLOW_ROOT=/usr/local \
-D CMAKE_CXX_FLAGS_DEBUG="-g -O0" \
-D CMAKE_INSTALL_PREFIX=/usr/local/mlc \
-D RANGE_CHECK=ON \
../..
