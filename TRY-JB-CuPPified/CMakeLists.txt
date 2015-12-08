PROJECT(opensteer)

cmake_minimum_required(VERSION 2.4)

# set CuPP path
SET (CuPP_PATH cupp)

# build CuPP
SUBDIRS(${CuPP_PATH})

# setup the CUDA path
SET (CUDA_INSTALL_PREFIX /usr/local/cuda/)

# load the cuda macros
INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCuda.cmake)

# set the CUDA build type to device
SET (CUDA_BUILD_TYPE Device)

# set the include path
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include/)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/)
INCLUDE_DIRECTORIES(${CuPP_PATH}/include)     # CuPP
INCLUDE_DIRECTORIES(/usr/local/cuda/include/) # CUDA

# set the CUDA include path
CUDA_INCLUDE_DIRECTORIES(${CuPP_PATH}/include/)
CUDA_INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include/)
CUDA_INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/)

# set the link directories
LINK_DIRECTORIES(/usr/local/cuda/lib)    # CUDA

SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib/)
LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/lib/) # CuPP

# get all the source files
# this is dirty, if you add a new file to one of the directories, you need to rerun cmake(!) by hand(!!)
AUX_SOURCE_DIRECTORY(src             SRC)
AUX_SOURCE_DIRECTORY(src/Graphics    SRC_GRAPHICS)
AUX_SOURCE_DIRECTORY(plugins         SRC_PLUGINS)
AUX_SOURCE_DIRECTORY(kapaga          SRC_KAPAGA)
AUX_SOURCE_DIRECTORY(kapaga/posix    SRC_KAPAGA_POSIX)

# set the compiler macros
ADD_DEFINITIONS(-DOPENSTEER)
ADD_DEFINITIONS(-DUSEOpenGL)

# if you don't want the full compiler output, remove the following line
#SET(CMAKE_VERBOSE_MAKEFILE ON)

# add definitions, compiler switches, etc.
# no -pedantic, the CUDA includes throw to many warnings...

#ADD_DEFINITIONS(-Wall -W -O2 -g)
#ADD_DEFINITIONS(-fkeep-inline-functions -Wall -W -O2 -g) # profiling

# define the opensteer executable
ADD_EXECUTABLE(opensteer ${SRC} ${SRC_GRAPHICS} ${SRC_PLUGINS} ${SRC_KAPAGA} ${SRC_KAPAGA_POSIX})

# Link cuda code in a library to something else.
CUDA_ADD_LIBRARY(kernel_boidsCupp_find_neighbours plugins/cuda/BoidsCupp_find_neighbours_kernel.cu )
CUDA_ADD_LIBRARY(kernel_boidsCupp_find_neighbours_simulate plugins/cuda/BoidsCupp_find_neighbours_simulate_kernel.cu )
CUDA_ADD_LIBRARY(kernel_boidsCupp_find_neighbours_simulate_frequency plugins/cuda/BoidsCupp_find_neighbours_simulate_frequency_kernel.cu )
CUDA_ADD_LIBRARY(kernel_boidsCupp_find_neighbours_simulate_frequency_grid plugins/cuda/BoidsCupp_find_neighbours_simulate_frequency_grid_kernel.cu )
#CUDA_ADD_LIBRARY(kernel_boidsCupp_find_neighbours_simulate_frequency_gpu_grid plugins/cuda/BoidsCupp_find_neighbours_simulate_frequency_gpu_grid_kernel.cu )

# dyn grid kernels
CUDA_ADD_LIBRARY(kernel_boidsCupp_find_neighbours_simulate_dyn_grid plugins/cuda/BoidsCupp_find_neighbours_simulate_dyn_grid_kernel.cu )
CUDA_ADD_LIBRARY(kernel_boidsCupp_find_neighbours_simulate_dyn_grid_clever plugins/cuda/BoidsCupp_find_neighbours_simulate_dyn_grid_clever_kernel.cu )
CUDA_ADD_LIBRARY(kernel_boidsCupp_find_neighbours_simulate_dyn_grid_clever_tf plugins/cuda/BoidsCupp_find_neighbours_simulate_dyn_grid_clever_tf_kernel.cu )
CUDA_ADD_LIBRARY(kernel_boidsCupp_find_neighbours_simulate_dyn_grid_clever_v2 plugins/cuda/BoidsCupp_find_neighbours_simulate_dyn_grid_clever_v2_kernel.cu )


# update kernel
CUDA_ADD_LIBRARY(kernel_boidsCupp_update plugins/cuda/BoidsCupp_update_kernel.cu )

# PREPROCESS grid
#CUDA_ADD_LIBRARY(kernel_grid_preprocess plugins/cuda/grid_preprocess.cu )

# PREPROCESS grid V1+
#CUDA_ADD_LIBRARY(kernel_v1_count plugins/cuda/grid_v1_preprocess/count.cu )
#CUDA_ADD_LIBRARY(kernel_v1_prescan plugins/cuda/grid_v1_preprocess/prescan.cu )
#CUDA_ADD_LIBRARY(kernel_v1_fill plugins/cuda/grid_v1_preprocess/fill.cu )

# which libraries needed to be linked with opensteer
TARGET_LINK_LIBRARIES(opensteer glut GLU GL pthread cupp cuda cudart
kernel_boidsCupp_find_neighbours
kernel_boidsCupp_find_neighbours_simulate
kernel_boidsCupp_update
kernel_boidsCupp_find_neighbours_simulate_frequency
kernel_boidsCupp_find_neighbours_simulate_frequency_grid
kernel_grid_preprocess
#kernel_boidsCupp_find_neighbours_simulate_frequency_gpu_grid
#kernel_v1_count kernel_v1_prescan kernel_v1_fill
kernel_boidsCupp_find_neighbours_simulate_dyn_grid
kernel_boidsCupp_find_neighbours_simulate_dyn_grid_clever
kernel_boidsCupp_find_neighbours_simulate_dyn_grid_clever_tf
kernel_boidsCupp_find_neighbours_simulate_dyn_grid_clever_v2
)
