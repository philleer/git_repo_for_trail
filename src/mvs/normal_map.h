#ifndef SRC_NORMAL_MAP_H
#define SRC_NORMAL_MAP_H

#include "mvs/mat.h"
#include <cstddef>

class NormalMap : public Mat<float> {
public:
	NormalMap();
	NormalMap(const size_t width, const size_t height);
	explicit NormalMap(const Mat<float> &mat);

	void Rescale(const float factor);
	void Downsize(const size_t max_width, const size_t max_height);
};

#endif