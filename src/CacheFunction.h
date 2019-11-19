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

// 79. Word Search
class Solution {
public:
	/**************************************************************************
	 * Description: Given a 2D board and a word, find if the word exists in the
	 *		grid.  The word can be constructed  from  letters  of  sequentially
	 *		adjacent cell, where "adjacent" are those horizontally / vertically
	 *		neighboring. The same letter cell may not be used more than once.
	 * Example:
	 * 		board = [['A','B','C','E'], ['S','F','C','S'], ['A','D','E','E']]
	 * 		Given word = "ABCCED", return true
	 *		Given word = "SEE", return true
	 *		Given word = "ABCB", return false
	 **************************************************************************
	 */
	bool exist(std::vector<std::vector<char>> &board, std::string word) {
		if (board.empty() || word.empty()) return false;
		size_t rows = board.size();
		size_t cols = board[0].size();
		if (word.length() > rows*cols) return false;
		visited.resize(rows);
		for (size_t r = 0; r < rows; ++r) visited[r].resize(cols, 0);

		size_t index = 0;
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {		
				if (board[i][j] == word[index]) {

					// 下面的算法通过了83/87个测试案例
					// 当前卡在vec3和word3的组合上，问题出在visited数组
					++index;
					std::stack<std::pair<size_t, size_t>> s;
					s.push(std::make_pair(i, j));
					visited[i][j] = 1;

					std::pair<size_t, size_t> second_last;
					size_t flag = 0;
					while (!s.empty()) {
						std::pair<size_t, size_t> tmp = s.top();
						size_t x= tmp.first;
						size_t y = tmp.second;

						if (x < rows-1 && visited[x+1][y] == 0 && board[x+1][y] == word[index]) {
							std::cout << x+1 << ", " << y << "; " << index << std::endl;
							std::cout << word[index] << std::endl;
							++index;
							s.push(std::make_pair(x+1, y));
							visited[x+1][y] = 1;
						} else if (y < cols-1 && visited[x][y+1] == 0 && board[x][y+1] == word[index]) {
							std::cout << x << ", " << y+1 << "; " << index << std::endl;
							std::cout << word[index] << std::endl;
							++index;
							s.push(std::make_pair(x, y+1));
							visited[x][y+1] = 1;
						} else if (x > 0 && visited[x-1][y] == 0 && board[x-1][y] == word[index]) {
							std::cout << x-1 << ", " << y << "; " << index << std::endl;
							std::cout << word[index] << std::endl;
							++index;
							s.push(std::make_pair(x-1, y));
							visited[x-1][y] = 1;
						} else if (y > 0 && visited[x][y-1] == 0 && board[x][y-1] == word[index]) {
							std::cout << x << ", " << y-1 << "; " << index << std::endl;
							std::cout << word[index] << std::endl;
							++index;
							s.push(std::make_pair(x, y-1));
							visited[x][y-1] = 1;
						} else {
							if (s.size() == word.length()) return true;
							s.pop();
							--index;
							if (flag == 0) second_last = std::make_pair(x, y);
							++flag;
							if (flag == 2) {
								visited[second_last.first][second_last.second] = 0;
								--flag;
								second_last = std::make_pair(x, y);
							}
						}

						if (s.size() == word.length()) return true;
					}
					for (size_t r = 0; r < rows; ++r) {
						visited[r].clear();
						visited[r].resize(cols, 0);
					}
				}
			}
		}

		return false;
	}

	void update(std::vector<std::vector<int>> &is_occupied, int row, int col) {
		int n = is_occupied.size();
		for (int r = row+1; r < n; ++r) {
			if (0 == is_occupied[r][col]) is_occupied[r][col] = 1;
			if (col+r-row < n && 0 == is_occupied[r][col+r-row])
				is_occupied[r][col+r-row] = 1;
			if (col+row >= r && 0 == is_occupied[r][col+row-r])
				is_occupied[r][col+row-r] = 1;
		}
	}

	void visualize(std::vector<std::vector<int>> &solutions) {
		int n = solutions.size();
		for (int r = 0; r < n; ++r) {
			int l = solutions[r].size();
			for (int c = 0; c < l; ++c) std::cout << solutions[r][c] << " ";
			std::cout<< std::endl;

			for (int c = 0; c < l; ++c) {
				int pos = solutions[r][c];
				for (int i = 0; i < l; ++i) {
					if (i == pos) std::cout << "Q ";
					else std::cout << "* ";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}

		std::cout << "Total solutions: " << solutions.size() << std::endl;
	}

	// 递归加回溯解法二
	// ========================================================================
	// 存在问题：只能找到第一个解，如何继续向下寻找？
	bool getSolution(int n, int row, std::vector<std::pair<int,int>> &p) {
		if (n <= 3)
			return false;

		if (row >= n) {
			solvers.push_back(p);
			return true;
		}

		for (int c = 0; c < n; ++c) {
			bool isValid = true;
			p[row] = std::make_pair(row, c);
			for (int pos = 0; pos < row; ++pos) {
				if (p[pos].second == c ||
					p[pos].first-p[pos].second == row-c ||
					p[pos].first+p[pos].second == row+c) {
					isValid = false;
				}
			}

			if (isValid)
				if (getSolution(n, row+1, p))
					return true;
		}
		return false;
	}

	void nQueen2() {
		int n = 5;
		std::vector<std::pair<int, int>> p(n);

		if (getSolution(n, 0, p)) {
			std::cout << solvers.size() << std::endl;
			for (int i = 0; i < solvers.size(); ++i) {
				for (int j = 0; j < n; j++)
					std::cout << solvers[i][j].second << " ";
				std::cout << std::endl;
			}
		}
	}

	// ========================================================================
	void nQueen(int k, int ld, int rd) {
		if (k == upperlim) {
			++count;
			return;
		}

		int pos = upperlim & ~(k | ld | rd);
		while (pos) {
			int p = pos & (~pos+1);
			pos -= p;
			nQueen(k | p, (ld | p) << 1, (rd | p) >> 1);
		}
	}

	void myqueen() {
		int n = 8;
		result.resize(n, -1);
		upperlim = (upperlim << n) - 1;
		nQueen(0, 0, 0);
		visualize(solutions);
	}

public:
	std::vector<std::vector<char>> visited;
	std::vector<std::vector<int>> is_occupied;
	std::vector<std::vector<int>> solutions;
	std::vector<std::vector<std::pair<int,int>>> solvers;
	int count = 0;
	int upperlim = 1;
	std::vector<int> result;
};

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
