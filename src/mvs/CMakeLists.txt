
set(MVS_SRCS
	depth_map.h depth_map.cc
	mat.h
	normal_map.h normal_map.cc
	patch_match.h patch_match.cc
	image.h image.cc
	)
set(MVS_CUDA_SRCS
	patch_match_cuda.h patch_match_cuda.cu
	)

add_library( mvs STATIC ${MVS_SRCS} )

cuda_add_library( mvscuda STATIC ${MVS_CUDA_SRCS} )

target_link_libraries( mvs ${OpenCV_LIBS} )
