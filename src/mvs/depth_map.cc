#include "depth_map.h"

float DepthMap::getDepthMin() const { return depth_min_; }
float DepthMap::getDepthMax() const { return depth_max_; }

void DepthMap::Rescale(const float factor) {
	if (width_ * height_ == 0) return ;

	const size_t new_width = std::round(width_ * factor);
	const size_t new_height = std::round(height_ * factor);
	std::vector<float> new_data(new_width * new_height);

	// DownsampleImage(data_.data(), height_, width_, new_height, new_width,
	// 				new_data.data());

	data_ = new_data;
	width_ = new_width;
	height_ = new_height;

	// 在 C++11 中，vector增加了新的成员shrink_to_fit()来使容器的容量
	// 收缩到尽量等于元素的个数，这并不保证容器的容量与元素个数确切相等。
	// std::vector<int>( vec ).swap( vec );
	data_.shrink_to_fit();
}
