#include <iostream>
#include <vector>
#include <string>
#include "CacheFunction.h"

int main(int argc, char *argv[]) {
	github::Option obj;
	if (argc < 2) {
		std::cout << "Hello world error!" << std::endl;
	} else {
		obj.Run();
	}
	
	return 0;
}
