#include <iostream>
#include <vector>
#include "CacheFunction.h"

extern "C" {
#include "../lib/franch/vlsift.h"	
#include "../lib/franch/feature.h"
}

int main(int argc, char **argv)
{
	github::Option obj;
	if (argc < 2) {
		std::cout << "hello world!" << std::endl;
	} else {
		fun_git(argc, argv[0]);
		double num = findFeatures(2, 3);
		printf("Num of features: %f\n", num);
		for (int i = 0; i < argc; i++) {
			obj.config.push_back(argv[i]);
		}
		obj.Run();
	}

	return 0;
}
