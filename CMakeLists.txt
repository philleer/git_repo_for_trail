cmake_minimum_required(VERSION 3.5)

project(GIT_REPO)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE RELEASE)

include_directories(src)
add_subdirectory(src bin)

aux_source_directory(./src/ DIR_SRC)

add_executable(main
		#		./src/main.cc
		#		./src/CacheFunction.h
		#		./src/CacheFunction.cc
		${DIR_SRC}
		)

target_link_libraries(main src)
