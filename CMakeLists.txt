cmake_minimum_required(VERSION 3.5)

project(ISLAND)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE RELEASE)

aux_source_directory(./src/ DIR_SRC)

add_executable(main ${DIR_SRC})

target_link_libraries(main)
