#ifndef SRC_MAT_H
#define SRC_MAT_H

#include <vector>
#include <cstddef>

// Definition of mat class
template<typename T>
class Mat {
public:
	Mat();
	Mat(const size_t width, const size_t height, const size_t depth);
	size_t getWidth() const;
	size_t getHeight() const;
	size_t getDepth() const;

	const std::vector<T>& getData() const;

protected:
	size_t width_ = 0;
	size_t height_ = 0;
	size_t depth_ = 0;
	std::vector<T> data_;
};

template<typename T>
Mat<T>::Mat(): Mat(0, 0, 0) {}

template<typename T>
Mat<T>::Mat(const size_t width, const size_t height, const size_t depth):
	width_(width), height_(height), depth_(depth) {
	data_.resize(width_ * height_ * depth_, 0);
}

template<typename T>
size_t Mat<T>::getWidth() const {
	return width_;
}

template<typename T>
size_t Mat<T>::getHeight() const {
	return height_;
}

template<typename T>
size_t Mat<T>::getDepth() const {
	return depth_;
}

template<typename T>
const std::vector<T>& Mat<T>::getData() const {
	return data_;
}

#endif