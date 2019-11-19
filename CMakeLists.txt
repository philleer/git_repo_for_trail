cmake_minimum_required(VERSION 3.5)

project(GIT_REPO)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE RELEASE)

set(Eigen_DIR "/usr/include/eigen3")
include_directories(${Eigen_DIR})

find_package(g2o REQUIRED)
include_directories(${g2o_INCLUDE_DIRS})

add_subdirectory(src bin)
include_directories("./src")

aux_source_directory(./src DIR_SRCS)

add_executable(main ${DIR_SRCS})

target_link_libraries(main
		src
		${g2o_LIBRARIES}
		)

