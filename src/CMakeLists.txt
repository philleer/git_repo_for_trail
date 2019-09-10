
set(CMAKE_CXX_STANDARD 11)
 
aux_source_directory(. DIR_SRC)
 
# add_subdirectory(
# 		base
# 		math
# 		reserve
# 		)
# 
add_library(src STATIC ${DIR_SRC})
# 
target_link_libraries(
		src
# 		base
# 		math
# 		reserve
 		)
