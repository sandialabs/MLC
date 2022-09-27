cmake_minimum_required(VERSION 3.16.0)
project(mlc VERSION 1.0 LANGUAGES C CXX)

# This is the Machine Learned Collisions (MLC) code. It is used to infer a TensorFlow model designed to represent collisional transport for a Boltzmann application.

# ---------------------------------------
# Set input options

set(RANGE_CHECK OFF CACHE BOOL "Enable Range Checking")
set(TENSORFLOW_ROOT "" CACHE PATH "Root directory for tensorflow (must include 'include' and 'lib' for c API)")

if("${TENSORFLOW_ROOT}" STREQUAL "")
  message(FATAL_ERROR "Must set '-D TENSORFLOW_ROOT' to the c-API root directory. See 'https://www.tensorflow.org/install/lang_c' for install process.")
else()
  message(STATUS "Tensorflow root directory set to: ${TENSORFLOW_ROOT}")
endif()

if(RANGE_CHECK)
  message(STATUS "Enable: RANGE_CHECK")
  add_definitions(-DMLC_RANGE_CHECK)
endif(RANGE_CHECK)

# ---------------------------------------
# Get headers and source files
set(HEADERS "")
set(SOURCES "")

add_subdirectory(src)

# ---------------------------------------
# Create library
 
set(LIBRARY_NAME mlc)

add_library(${LIBRARY_NAME})

# ---------------------------------------
# Setup environment
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_11)

# Make sure we have access to cmake utilities
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
target_include_directories(${LIBRARY_NAME} 
                           INTERFACE 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                           $<INSTALL_INTERFACE:include>
)

# This is a crude method for including tensorflow, but TF currently lacks a stable cmake interface
target_include_directories(${LIBRARY_NAME} PUBLIC ${TENSORFLOW_ROOT}/include)
target_link_directories(${LIBRARY_NAME} PUBLIC ${TENSORFLOW_ROOT}/lib)

target_sources(${LIBRARY_NAME} PRIVATE ${SOURCES})
target_link_libraries(${LIBRARY_NAME} PUBLIC tensorflow)
set_property(TARGET ${LIBRARY_NAME} PROPERTY CXX_STANDARD 14)

# ---------------------------------------
# Create tests

enable_testing()
add_subdirectory(tests)

# ---------------------------------------
# Setup install

set_target_properties(${LIBRARY_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

install(TARGETS ${LIBRARY_NAME}
        EXPORT ${LIBRARY_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include
)
install(EXPORT ${LIBRARY_NAME} DESTINATION lib)
install(FILES mlc-config.cmake DESTINATION lib/cmake/mlc)
