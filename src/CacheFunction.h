#ifndef CACHEFUNCTION_H
#define CACHEFUNCTION_H
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace github { // Begin of namespace github

class Option
{
	public:
		Option() {}
		~Option() {}
		void Run();

	public:
		std::vector<std::string> config;
};

} // End of namespace github
#endif /* ifndef CACHEFUNCTION_H */
