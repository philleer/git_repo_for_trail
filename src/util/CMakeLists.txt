
aux_source_directory(. DIR_UTIL)

add_library(util STATIC ${DIR_UTIL})

target_link_libraries(
		util
	)
