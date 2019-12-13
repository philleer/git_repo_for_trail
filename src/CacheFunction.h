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

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace github { // Begin of namespace github

class Solution {
public:
	void nQueen(int k, int ld, int rd) {
		if (k == max) {
			++count;
			solutions.push_back(result);
			return;
		}

		int pos = max & ~(k | ld | rd);
		int index = count1Bit(k);
		while (pos) {
			int p = pos & (~pos+1);
			pos -= p;
			result[index] = (p==1 ? 0 : 1+(int)log2(p>>1));
			nQueen(k | p, (ld | p) << 1, (rd | p) >> 1);
		}
	}

	int count1Bit(int n) {
		int countOneBit = 0;
		while (n) {
			++countOneBit;
			n &= (n-1);
		}
		return countOneBit;
	}

	void myqueen() {
		int n = 4;
		result.resize(n, -1);
		max = (max << n) - 1;
		nQueen(0, 0, 0);
		visualize(solutions);
	}

	void visualize(std::vector<std::vector<int>> &solutions) {
		for (int r = 0; r < solutions.size(); ++r) {
			int l = solutions[r].size();
			for (int c = 0; c < l; ++c)
				std::cout << solutions[r][c] << " ";
			std::cout<< std::endl;

			for (int c = 0; c < l; ++c) {
				for (int i = 0; i < l; ++i)
					std::cout << (i==solutions[r][c] ? "Q " : "* ");
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		std::cout << "Total solutions: " << solutions.size() << std::endl;
	}

public:
	std::vector<std::vector<int>> solutions;
	std::vector<int> result;
	int count = 0;
	int max = 1;
};

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
