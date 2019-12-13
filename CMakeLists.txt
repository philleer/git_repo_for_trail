cmake_minimum_required(VERSION 3.5)

project(GIT_REPO)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE RELEASE)
set(CUDA_VERSION 8.0)
set(CUDA_NVCC_FLAGS -O3)

find_package(OpenCV REQUIRED)
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

set(EIGEN_DIR "/usr/include/eigen3")
set(CUDA_TARGET_INCLUDE "/usr/local/cuda-8.0/target/x86_64-linux/include")

include_directories(
	src
	${OpenCV_INCLUDE_DIRS}
	${OpenMP_INCLUDE_DIRS}
	${CERES_INCLUDE_DIRS}
	${EIGEN_DIR}
	${CUDA_TARGET_INCLUDE}
	)

add_subdirectory(src bin)

aux_source_directory(./src/ DIR_SRC)

add_executable(main ${DIR_SRC})

target_link_libraries(main
		src
		${OpenCV_LIBS}
		${OpenMP_LIBRARIES}
		${CERES_LIBRARIES}
		${CUDA_LIBRARIES}
	)
