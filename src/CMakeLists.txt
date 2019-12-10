
set(CMAKE_CXX_STANDARD 11)
 
aux_source_directory(. DIR_SRC)

include_directories(mvs)
add_subdirectory( mvs )

# add_subdirectory( base )
# add_subdirectory( math )
# add_subdirectory( reserve )

add_library(src STATIC ${DIR_SRC})

target_link_libraries(
		src
		mvs
		# base math reserve
		${OpenCV_LIBS}
 		)
