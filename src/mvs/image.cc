#include "mvs/image.h"

Image::Image() {}

Image::Image(const std::string &path, const size_t width, const size_t height,
			 const float *K, const float *R, const float *T):
	path_(path), width_(width), height_(height)
{
	memcpy(K_, K, 9 * sizeof(float));
	memcpy(R_, R, 9 * sizeof(float));
	memcpy(T_, T, 9 * sizeof(float));
}
<<<<<<< HEAD

=======
 
>>>>>>> 0d52f80de8c82b397eb80a2ce02c7f1990a1102f
cv::Mat Image::getImage() const { return src_img; }
