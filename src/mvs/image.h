#ifndef SRC_IMAGE_H
#define SRC_IMAGE_H

#include <opencv2/imgproc/imgproc.hpp>
#include <cstddef>
#include <string>

class Image {
public:
	Image();
	Image(const std::string &path, const size_t width, const size_t height,
		  const float *K, const float *R, const float *T);
	cv::Mat getImage() const;

	inline size_t getWidth() const;
	inline size_t getHeight() const;
	inline const cv::Mat &getBitmap() const;

	inline const std::string& GetPath() const;
	inline const float* GetK() const;
	inline const float* GetR() const;
	inline const float* GetT() const;
	inline const float* GetP() const;
	inline const float* GetInvP() const;

private:
	cv::Mat src_img;
	std::string path_;
	size_t width_;
	size_t height_;
	float K_[9];
	float R_[9];
	float T_[3];
	float P_[12];
	float inv_P_[12];
};

size_t Image::getWidth() const { return height_; }
size_t Image::getHeight() const { return width_; }

const cv::Mat &Image::getBitmap() const { return src_img; }

const std::string& Image::GetPath() const { return path_; }

const float* Image::GetK() const { return K_; }
const float* Image::GetR() const { return R_; }
const float* Image::GetT() const { return T_; }
const float* Image::GetP() const { return P_; }
const float* Image::GetInvP() const { return inv_P_; }

void DownsampleImage(const float* data, const int rows, const int cols,
                     const int new_rows, const int new_cols,
                     float* downsampled);

#endif