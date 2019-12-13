#include "CacheFunction.h"
#include <sstream>

void github::Option::Run() {
	auto start = std::chrono::system_clock::now();

	/*
	 * source code here to test
	 */
	github::Solution().myqueen();

	
	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration<double>(end - start);
	std::cout << "Cost time " << duration.count() << " s" << std::endl;
}
