#include "mvs/image.h"
#include <iostream>
#include <limits>
#include <cstdlib>
#include <algorithm>

Image::Image() {}

Image::Image(const std::string &path, const size_t width, const size_t height,
			 const float *K, const float *R, const float *T):
	path_(path), width_(width), height_(height)
{
	memcpy(K_, K, 9 * sizeof(float));
	memcpy(R_, R, 9 * sizeof(float));
	memcpy(T_, T, 9 * sizeof(float));
}

cv::Mat Image::getImage() const { return src_img; }

void DownsampleImage(const float* data, const int rows, const int cols,
                     const int new_rows, const int new_cols,
                     float* downsampled) {
	if (data == nullptr) {
		std::cerr << "data empty!\n";
		exit(EXIT_FAILURE);
	}
	if (downsampled == nullptr) {
		std::cerr << "downsampled array empty!\n";
		exit(EXIT_FAILURE);
	}
	if (new_rows > rows || new_cols > cols) {
		std::cerr << "new dimension is not proper!\n";
		exit(EXIT_FAILURE);
	}
	if (rows <= 0 || cols <= 0 || new_rows <= 0 || new_cols <= 0) {
		std::cerr << "dimension size invalid!\n";
		exit(EXIT_FAILURE);
	}

	const float scale_c = static_cast<float>(cols) / static_cast<float>(new_cols);
	const float scale_r = static_cast<float>(rows) / static_cast<float>(new_rows);

	const float kSigmaScale = 0.5f;
	const float sigma_c = std::max(std::numeric_limits<float>::epsilon(),
	                             kSigmaScale * (scale_c - 1));
	const float sigma_r = std::max(std::numeric_limits<float>::epsilon(),
	                             kSigmaScale * (scale_r - 1));

	std::vector<float> smoothed(rows * cols);
	// SmoothImage(data, rows, cols, sigma_r, sigma_c, smoothed.data());
	// vl_imsmooth_f(smoothed, cols, data, cols, rows, cols, sigma_c, sigma_r)

	for (int r = 0; r < new_rows; ++r) {
		const float r_i = (r + 0.5f) * scale_r - 0.5f;
		const int r_i_min = std::floor(r_i);
		const int r_i_max = r_i_min + 1;
		const float d_r_min = r_i - r_i_min;
		const float d_r_max = r_i_max - r_i;

		for (int c = 0; c < new_cols; ++c) {
			const float c_i = (c + 0.5f) * scale_c - 0.5f;
			const int c_i_min = std::floor(c_i);
			const int c_i_max = c_i_min + 1;
			const float d_c_min = c_i - c_i_min;
			const float d_c_max = c_i_max - c_i;

			// Interpolation in column direction.
			const float value1 = d_c_max * smoothed[r_i_min * cols + c_i_min] +
								 d_c_min * smoothed[r_i_min * cols + c_i_max];
			const float value2 = d_c_max * smoothed[r_i_max * cols + c_i_min] +
								 d_c_min * smoothed[r_i_max * cols + c_i_max];

			// Interpolation in row direction.
			downsampled[r * new_cols + c] = d_r_max * value1 + d_r_min * value2;
		}
	}

}