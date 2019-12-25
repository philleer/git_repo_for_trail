
set(MVS_SRCS
		depth_map.h depth_map.cc
		image.h image.cc
		mat.h
		model.h model.cc
		normal_map.h normal_map.cc
		patch_match.h patch_match.cc
	)
set(MVS_CUDA_SRCS
		gpu_mat.h
		patch_match_cuda.h patch_match_cuda.cu
	)

# set(UTIL_THREAD
# 	../util/util_sys.h ../util/util_sys.cc
# 	../util/threading.h ../util/threading.cc
# 	)

add_library( mvs STATIC ${MVS_SRCS}
		# ${UTIL_THREAD}
		# ../util/cache.h
		# ../util/util_sys.h
		/usr/include/eigen3
	)

cuda_add_library( mvscuda STATIC ${MVS_CUDA_SRCS})

target_link_libraries( mvs
		util
		${OpenCV_LIBS}
	)