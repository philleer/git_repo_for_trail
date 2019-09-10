
aux_source_directory(. DIR_MATH)

add_library(math STATIC ${DIR_MATH})

target_link_libraries(math)
