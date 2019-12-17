aux_source_directory(. DIR_SRC)

include_directories( mvs ${CUDA_TARGET_INCLUDE})

add_subdirectory( mvs )
add_subdirectory( util )
# add_subdirectory( base )
# add_subdirectory( math )
# add_subdirectory( reserve )

add_library(src STATIC ${DIR_SRC})

target_link_libraries(
		src
		mvs
		mvscuda
		util
		# base
		# math 
		# reserve
		${OpenCV_LIBS}
		${CERES_LIBRARIES}
		${CUDA_LIBRARIES}
 	)
