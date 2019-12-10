
aux_source_directory(. DIR_MVS)

add_library(mvs STATIC ${DIR_MVS})

target_link_libraries( mvs )
