#include "CacheFunction.h"

void github::Option::Run()
{
	auto start = std::chrono::system_clock::now();
	int len = config.size();
	if(len > 0) {
		for (int i = 0; i < len; i++) {
			std::cout << config[i] << std::endl;
		}
	} else {
		std::cerr << "Nothing has been given for processing!" << std::endl;
	}
	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration<double>(end - start);
	std::cout << "Cost time " << duration.count() << " s" << std::endl;
}
