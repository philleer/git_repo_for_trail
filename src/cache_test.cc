#include "CacheFunction.h"
#include <vector>

#define BOOST_TEST_MODULE SignTest
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(SignTest) {
	std::vector<int> integers = {1, 2, 3, 4, 5};
	auto result = github::Option().sum_integers(integers);
	BOOST_REQUIRE(result == 15);
}