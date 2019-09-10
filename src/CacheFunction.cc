#include "CacheFunction.h"
#include <sstream>

void github::Option::Run() {
	auto start = std::chrono::system_clock::now();
	int len = config.size();
	if(len > 0) {
		for (int i = 0; i < len; i++) {
			std::cout << config[i] << std::endl;
		}
		std::string base_path = "/home/phil/Downloads";
		char path[100];
		sprintf(path, "%s/%s", base_path.c_str(), "der_hass-20140923/IMG_0137.JPG");
		cv::Mat img = cv::imread(path, 0);
		int width = img.cols;
		int height = img.rows;
		std::cout << "height * width " << height << " " << width << std::endl;
	} else {
		std::cerr << "Nothing has been given for processing!" << std::endl;
	}

	/*
	 * source code here to test
	 */
	github::Solution().myqueen();

	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration<double>(end - start);
	std::cout << "Cost time " << duration.count() << " s" << std::endl;
}

