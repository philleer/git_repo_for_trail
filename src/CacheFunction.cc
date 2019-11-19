#include "CacheFunction.h"

void github::Option::Run()
{
	auto start = std::chrono::system_clock::now();

	github::Solution solver;
	// std::vector<std::vector<char>> vec1 = {
	// 		{'A', 'B', 'C', 'E'},
	// 		{'S', 'F', 'C', 'S'},
	// 		{'A', 'D', 'E', 'E'}
	// };
	// std::string word1 = "ABCCED";
	// std::vector<std::vector<char>> vec2 = {
	// 	{'C', 'A', 'A'},
	// 	{'A', 'A', 'A'},
	// 	{'B', 'C', 'D'}
	// };
	// std::string word2 = "AAB";

	// std::vector<std::vector<char>> vec3 = {
	// 	{'A', 'B', 'C', 'E'},
	// 	{'S', 'F', 'E', 'S'},
	// 	{'A', 'D', 'E', 'E'}
	// };
	// std::string word3 = "ABCESEEEFS";

	// std::cout << "The given grid:" << std::endl;
	// for (size_t i = 0; i < vec1.size(); ++i) {
	// 	for (size_t j = 0; j < vec1[0].size(); ++j) {
	// 		std::cout << vec1[i][j];
	// 	}
	// 	std::cout << std::endl;
	// }
	// std::cout << "And the given word: " << word1 << std::endl;
	// std::cout << "Does the word exist in the grid? " << std::endl;
	// if (solver.exist(vec3, word3))
	//    std::cout << "True"	<< std::endl;
	// else
	// 	std::cout << "False" << std::endl;

	// solver.fourQueen();
	// solver.nQueen2(6);
	solver.myqueen();

	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration<double>(end - start);
	std::cout << "Cost time " << duration.count() << " s" << std::endl;
}
