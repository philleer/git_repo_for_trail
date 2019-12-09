cmake_minimum_required(VERSION 3.5)

project(GIT_REPO)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE RELEASE)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

include_directories(
	src
	${OpenCV_INCLUDE_DIRS}
	${OpenMP_INCLUDE_DIRS}
	)

add_subdirectory(src bin)

aux_source_directory(./src/ DIR_SRC)

add_executable(main ${DIR_SRC})

target_link_libraries(main src ${OpenCV_LIBS} ${OpenMP_LIBRARIES})
