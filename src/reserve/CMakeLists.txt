
aux_source_directory(. DIR_RESERVE)

add_library(reserve STATIC ${DIR_RESERVE})

target_link_libraries(reserve)
