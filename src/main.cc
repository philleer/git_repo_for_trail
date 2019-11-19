#include <iostream>
#include <vector>
#include "CacheFunction.h"

int main(int argc, char const*argv[]) {
	github::Option obj;
	if (argc < 2) {
		std::cout << "hello world!" << std::endl;
	} else {
		for (int i = 0; i < argc; i++) {
			obj.config.push_back(argv[i]);
		}
		obj.Run();
	}

	return 0;
}

