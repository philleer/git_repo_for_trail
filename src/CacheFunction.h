#ifndef CACHEFUNCTION_H
#define CACHEFUNCTION_H
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstddef>
#include <utility>
#include <stack>
#include <queue>
#include <cmath>

namespace github { // Begin of namespace github

class Option {
	public:
		Option() {}
		~Option() {}
		void Run();

	public:
		std::vector<std::string> config;
};

} // End of namespace github
#endif /* ifndef CACHEFUNCTION_H */
