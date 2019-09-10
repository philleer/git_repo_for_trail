#include "CacheFunction.h"
#include <sstream>

void github::Option::Run() {
	auto start = std::chrono::system_clock::now();

	/*
	 * source code here to test
	 */
	github::Solution().myqueen();

	std::string base_path = "/home/phil/Downloads/dino";
	char img_path[100];
	sprintf(img_path, "%s/%s.png", base_path.c_str(), "dino0100");
	cv::Mat img = cv::imread(img_path, 0);
	// cv::imshow("dino_img", img);
	// cv::waitKey(0);

	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration<double>(end - start);
	std::cout << "Cost time " << duration.count() << " s" << std::endl;
}
