cmake_minimum_required(VERSION 3.5)

project(GIT_REPO)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE RELEASE)

find_package(g2o REQUIRED)
find_package(OpenCV 3 REQUIRED)
set(Eigen_DIR "/usr/include/eigen3")

set(PRO_INCLUDE_DIRS
	${Eigen_DIR}
	${g2o_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	)

include_directories(
	src
	${PRO_INCLUDE_DIRS}
	)

add_subdirectory(src bin)

# 一种做法是先链接到各级目录，再由下面的目录去链接底层库
include_directories(lib)
add_subdirectory(lib)

aux_source_directory(./src DIR_SRCS)
add_executable(git_repo ${DIR_SRCS})

target_link_libraries(git_repo
		src
		lib
		${g2o_LIBRARIES}
		${OpenCV_LIBS}
		)

# 另一种做法是将所有的c文件链接到当前项目中
# include_directories(lib/franch)
# add_library(VLSIFT STATIC lib/franch/vlsift.c)
# add_library(FEATURE STATIC lib/franch/feature.c)
# 
# aux_source_directory(./src DIR_SRCS)
# add_executable(git_repo ${DIR_SRCS})
# 
# target_link_libraries(git_repo
# 		src
# 		VLSIFT
# 		FEATURE
# 		${g2o_LIBRARIES}
# 		${OpenCV_LIBS}
# 		)