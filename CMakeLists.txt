cmake_minimum_required(VERSION 3.5)

project(GIT_REPO)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE RELEASE)

find_package(g2o REQUIRED)
find_package(OpenCV 3 REQUIRED)

set(Eigen_DIR "/usr/include/eigen3")

include_directories(src)
include_directories(${Eigen_DIR})
include_directories(${g2o_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(src bin)

aux_source_directory(./src/ DIR_SRCS)

add_executable(main ${DIR_SRCS})

target_link_libraries(main
		src
		${g2o_LIBRARIES}
		${OpenCV_LIBS}
	)