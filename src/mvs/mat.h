#ifndef SRC_MAT_H
#define SRC_MAT_H

#include <iostream>
#include <fstream>	 // std::fstream
#include <string>
#include <vector>
#include <cstddef>	 // size_t
#include <cstdlib>	 // EXIT_FAILURE
#include <algorithm> // std::fill

// Definition of mat class
template<typename T>
class Mat {
public:
	Mat();
	Mat(const size_t width, const size_t height, const size_t depth);
	size_t getWidth() const;
	size_t getHeight() const;
	size_t getDepth() const;
	size_t getNumBytes() const;

	T get(const size_t row, const size_t col, const size_t slice = 0) const;
	void getSlice(const size_t row, const size_t col, T* values) const;

	T* getPtr();
	const T* getPtr() const;
	const std::vector<T>& getData() const;

	void set(const size_t row, const size_t col, const T value);
	void set(const size_t row, const size_t col, const size_t slice, const T value);

	void Fill(const T value);
	void Read(const std::string& path);
	void Write(const std::string& path) const;

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
size_t Mat<T>::getWidth() const { return width_; }

template<typename T>
size_t Mat<T>::getHeight() const { return height_; }

template<typename T>
size_t Mat<T>::getDepth() const { return depth_; }

template<typename T>
size_t Mat<T>::getNumBytes() const { return data_.size() * sizeof(T); }

template<typename T>
T Mat<T>::get(const	size_t row, const size_t col, const size_t slice) const {
	return data_.at(slice * width_ * height_ + row * width_ + col);
}

template<typename T>
void Mat<T>::getSlice(const size_t row, const size_t col, T* values) const {
	for (size_t slice = 0; slice < depth_; ++slice) {
		values[slice] = get(row, col, slice);
	}
}

template<typename T>
T* Mat<T>::getPtr() { return data_.data(); }

template<typename T>
const T* Mat<T>::getPtr() const { return data_.data(); }

template<typename T>
const std::vector<T>& Mat<T>::getData() const { return data_; }

template<typename T>
void Mat<T>::set(const size_t row, const size_t col, const T value) {
	set(row, col, 0, value);
}

template<typename T>
void Mat<T>::set(const size_t row, const size_t col, const size_t slice, const T value) {
	data_.at(slice * width_ * height_ + row * width_ + col) = value;
}

template<typename T>
void Mat<T>::Fill(const T value) {
	std::fill(data_.begin(), data_.end(), value);
}

template<typename T>
void Mat<T>::Read(const std::string& path) {
	std::fstream text_file(path, std::ios::in | std::ios::binary);
	if (!text_file.is_open()) {
		std::cerr << path << " load failed!\n";
		exit(EXIT_FAILURE);
	}
	char unused_char;
	text_file >> width_ >> unused_char
		>> height_ >> unused_char
		>> depth_ >> unused_char;
	std::streampos pos = text_file.tellg();
	text_file.close();

	if (width_ <= 0 || height_ <= 0 || depth_ <= 0) {
		std::cerr << "mat param invalid!\n";
		exit(EXIT_FAILURE);
	}
	data_.resize(width_ * height_ * depth_);

	std::fstream binary_file(path, std::ios::in | std::ios::binary);
	if (!binary_file.is_open()) {
		std::cerr << path << " loaded failed!\n";
		exit(EXIT_FAILURE);
	}
	binary_file.seekg(pos);
	for (size_t i = 0; i < data_.size(); ++i) {
		binary_file.read(reinterpret_cast<char*>(&data_[i]), sizeof(T));
	}

	binary_file.close();
}

template<typename T>
void Mat<T>::Write(const std::string& path) const {
	std::fstream text_file(path, std::ios::out);
	if (!text_file.is_open()) {
		std::cerr << path << " loaded failed!\n";
		exit(EXIT_FAILURE);
	}
	text_file << width_ << "&" << height_ << "&" << depth_ << "&";
	text_file.close();
	std::fstream binary_file(path, std::ios::out | std::ios::binary | std::ios::app);
	if (!binary_file.is_open()) {
		std::cerr << path << " loaded failed!\n";
		exit(EXIT_FAILURE);
	}
	for (size_t i = 0; i < data_.size(); ++i) {
		binary_file.write(reinterpret_cast<char*>(&data_[i]), sizeof(T));
	}
	binary_file.close();
}

#endif
