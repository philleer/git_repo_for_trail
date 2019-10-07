include_directories(franch)
add_subdirectory(franch)

# 可以使用set将文件夹下的各源文件包含到制定变量中
# set(LIB_SRCS franch/vlsift.c franch/feature.c)
# 也可以直接使用CMake的指令包含源文件，更加方便实用
aux_source_directory(./franch/ LIB_SRCS)

# 这里的lib要与上一级的CMakeLists.txt中的库名称保持一致
add_library(
	lib
	STATIC
	${LIB_SRCS}
	)

target_link_libraries(lib)