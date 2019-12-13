
# aux_source_directory(. DIR_BASE)

# link_libraries(base STATIC ${DIR_BASE})



file(GLOB_RECURSE CURRENT_HEADERS *.h)
file(GLOB CURRENT_SOURCES *.cc *.cu)

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

cuda_add_library(base STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

target_link_libraries( base ${CUDA_LIBRARIES} )
