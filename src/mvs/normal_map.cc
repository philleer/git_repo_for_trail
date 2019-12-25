#include "mvs/normal_map.h"
#include <iostream>
#include <algorithm>
#include <Eigen/Core>

NormalMap::NormalMap(): Mat<float>(0, 0, 3) {}

NormalMap::NormalMap(const size_t width, const size_t height):
	Mat<float>(width, height, 3) {}

NormalMap::NormalMap(const Mat<float> &mat):
	Mat<float>(mat.getWidth(), mat.getHeight(), mat.getDepth())
{
	if (mat.getDepth() != 3) {
		std::cerr << "channel of depth not equals to 3!\n";
	}
	data_ = mat.getData();
}

void NormalMap::Rescale(const float factor) {
	if (0 == width_ * height_) return ;

	const size_t new_width = std::round(width_ * factor);
	const size_t new_height = std::round(height_ * factor);
	std::vector<float> new_data(new_width * new_height * 3);

	for (size_t d = 0; d < 3; ++d) {
		const size_t offset = d * width_ * height_;
		const size_t new_offset = d * new_width * new_height;
		// DownsampleImage(data_.data()+offset, height_, width_,
		// 	new_height, new_width, new_data.data()+new_offset);
	}

	data_ = new_data;
	width_ = new_width;
	height_ = new_height;

	data_.shrink_to_fit();

	// Re-normalize the normal vectors
	for (size_t r = 0; r < height_; ++r) {
		for (size_t c = 0; c < width_; ++c) {
			Eigen::Vector3f normal(get(r, c, 0), get(r, c, 1), get(r, c, 2));
			const float squared_norm = normal.squaredNorm();
			if (squared_norm > 0) {
				normal /= std::sqrt(squared_norm);
			}
			set(r, c, 0, normal(0));
			set(r, c, 1, normal(1));
			set(r, c, 2, normal(2));
		}
	}
}

void NormalMap::Downsize(const size_t max_width, const size_t max_height) {
	if (height_ <= max_height && width_ <= max_width) return;

	const float factor_x = static_cast<float>(max_width) / width_;
	const float factor_y = static_cast<float>(max_height) / height_;
	Rescale(std::min(factor_x, factor_y));
}