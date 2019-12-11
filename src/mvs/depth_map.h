#ifndef SRC_DEPTH_MAP_H
#define SRC_DEPTH_MAP_H

#include "mvs/mat.h"

class DepthMap : public Mat<float> {
public:
	DepthMap();
	DepthMap(const size_t width, const size_t height,
			 const float depth_min, const float depth_max);
	DepthMap(const Mat<float> &mat, const float	depth_min,
			 const float depth_max);

	inline float getDepthMin() const;
	inline float getDepthMax() const;
private:
	float depth_min_ = -1.0f;
	float depth_max_ = -1.0f;
};

#endif
