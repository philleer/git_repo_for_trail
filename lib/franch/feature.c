#include "feature.h"

double findFeatures(int x, int y) {
	double num = 0;

	int i;
	for (i = 0; i < 3; i++) {
		num += sqrt(x * x + y * y);
	}
	
	return num;
}