#===============================================================================
# CMAKE VERSION
#===============================================================================
cmake_minimum_required(VERSION 3.5)


#===============================================================================
# PROJECT CONFIG
#===============================================================================
project(GIT_REPO LANGUAGES CXX)


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
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Ceres REQUIRED)
find_package(CUDA ${CUDA_VERSION} REQUIRED)
find_package(
		Boost 1.58.0 COMPONENTS
		regex
		thread
		system
		filesystem
		unit_test_framework
		REQUIRED
	)

if (OPENMP_FOUND)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

if (CUDA_FOUND)
		set(
			CMAKE_CXX_FLAGS
			"${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED"
			)
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
		${Boost_INCLUDE_DIRS}
	)


#===============================================================================
# SUB-DIRECTORY SET
#===============================================================================
add_subdirectory(src bin)


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
		${Boost_LIBRARIES}
	)


#===============================================================================
# CacheFunction TEST UNIT
#===============================================================================
add_executable( cache_test
		src/cache_test.cc
		src/CacheFunction.cc
		src/CacheFunction.h
	)
target_link_libraries( cache_test 
	PRIVATE
		src
		Boost::unit_test_framework
	)

target_compile_definitions( cache_test
	PRIVATE
		BOOST_TEST_DYN_LINK
	)

enable_testing()
add_test( NAME SignTest COMMAND $<TARGET_FILE:cache_test> )


#===============================================================================
# THREAD TEST UNIT
#===============================================================================
add_executable(thread_test
		src/util/threading.cc
		src/util/threading.h
		src/util/threading_test.cc
	)
target_link_libraries( thread_test
	PRIVATE
		src
		Boost::unit_test_framework
	)

target_compile_definitions( thread_test
	PRIVATE
		BOOST_TEST_DYN_LINK
	)

# enable_testing()
add_test( NAME TestThreadWait COMMAND $<TARGET_FILE:thread_test> )
add_test( NAME TestThreadPause COMMAND $<TARGET_FILE:thread_test> )
add_test( NAME TestThreadStop COMMAND $<TARGET_FILE:thread_test> )
add_test( NAME TestThreadPauseStop COMMAND $<TARGET_FILE:thread_test> )
add_test( NAME TestThreadRestart COMMAND $<TARGET_FILE:thread_test> )
# add_test( NAME TestThreadValidSetup COMMAND $<TARGET_FILE:thread_test> )
# add_test( NAME TestThreadInvalidSetup COMMAND $<TARGET_FILE:thread_test> )
add_test( NAME TestCallback COMMAND $<TARGET_FILE:thread_test> )
add_test( NAME TestDefaultCallback COMMAND $<TARGET_FILE:thread_test> )
# add_test( NAME TestThreadTimer COMMAND $<TARGET_FILE:thread_test> )
# add_test( NAME TestThreadPoolNoArgNoReturn COMMAND $<TARGET_FILE:thread_test> )
# add_test( NAME TestThreadPoolArgNoReturn COMMAND $<TARGET_FILE:thread_test> )
# add_test( NAME TestThreadPoolNoArgReturn COMMAND $<TARGET_FILE:thread_test> )
# add_test( NAME TestThreadPoolArgReturn COMMAND $<TARGET_FILE:thread_test> )
add_test( NAME TestGetEffectiveNumThreads COMMAND $<TARGET_FILE:thread_test> )