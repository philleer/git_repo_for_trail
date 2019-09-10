aux_source_directory(. DIR_SRC)
 
# add_subdirectory( base )
# add_subdirectory( math )
# add_subdirectory( reserve )
 
add_library(src STATIC ${DIR_SRC})
 
target_link_libraries(
		src
		# base math reserve
 	)
