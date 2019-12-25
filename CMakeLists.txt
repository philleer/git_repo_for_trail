#===============================================================================
# CMAKE VERSION
#===============================================================================
cmake_minimum_required(VERSION 3.5)


#===============================================================================
# PROJECT CONFIG
#===============================================================================
project(GIT_REPO)


#===============================================================================
# CMAKE CXX_STANDARD SET, CMAKE FLAGS AND CUDA SET
#===============================================================================
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE RELEASE)
set(CUDA_VERSION 8.0)
set(CUDA_NVCC_FLAGS -O3)


#===============================================================================
# FIND CORRESPONDING PACKAGES NEEDED
#===============================================================================
find_package(OpenCV 3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Ceres REQUIRED)
find_package(CUDA ${CUDA_VERSION} REQUIRED)

if (OPENMP_FOUND)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

if (CUDA_FOUND)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED")
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif(CUDA_FOUND)


#===============================================================================
# CORRESPONDING HEADER FILES INCLUDED IN THE PROJECT
#===============================================================================
set(EIGEN_DIR "/usr/include/eigen3")
set(CUDA_TARGET_INCLUDE "/usr/local/cuda-8.0/target/x86_64-linux/include")


#===============================================================================
# INCLUDE DIRECTORIES CONFIGURATION
#===============================================================================
include_directories(
		src
		${OpenCV_INCLUDE_DIRS}
		${OpenMP_INCLUDE_DIRS}
		${CERES_INCLUDE_DIRS}
		${EIGEN_DIR}
		${CUDA_TARGET_INCLUDE}
	)


#===============================================================================
# SUB-DIRECTORY SET
#===============================================================================
add_subdirectory(src bin)
include_directories("./src")

aux_source_directory(./src DIR_SRCS)


#===============================================================================
# SOURCE FILES IN THE UPPEST FOLDER (such as main.cc CacheFunction.h
#				     CacheFunction.cc)
#===============================================================================
aux_source_directory(./src/ DIR_SRC)


#===============================================================================
# EXECUTABLE PROGRAM SETTING
#===============================================================================
add_executable(main ${DIR_SRC})


#===============================================================================
# TARGET LINK LIBRARIES NEEDED
#===============================================================================
target_link_libraries(main
		src
		${OpenCV_LIBS}
		${OpenMP_LIBRARIES}
		${CERES_LIBRARIES}
		${CUDA_LIBRARIES}
	)

