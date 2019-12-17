
aux_source_directory(. DIR_UTIL)

add_library(util STATIC ${DIR_UTIL})

target_link_libraries(
		util
	)

#===============================================================================
# TEST UNIT
#===============================================================================
# add_executable(thread_test threading.cc)
# target_link_libraries(thread_test
# 	PRIVATE
# 		util
# 		Boost::unit_test_framework
# 	)

# target_compile_definitions(thread_test
# 	PRIVATE
# 		BOOST_TEST_DYN_LINK
# 	)

# enable_testing()

# add_test(
# 	NAME TestThreadWait
# 	COMMAND $<TARGET_FILE:thread_test>
# 	)