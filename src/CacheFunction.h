#ifndef CACHEFUNCTION_H
#define CACHEFUNCTION_H
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <cstddef>
#include <utility>
#include <stack>
#include <queue>
#include <cmath>
#include <sstream>

namespace github { // Begin of namespace github

// template<typename T>
// class Matrix2D {
// public:
// 	Matrix2D();
// 	Matrix2D(size_t rows, size_t cols);

// 	T& operator()(size_t row, size_t col);
// 	const T& operator()(size_t row, size_t col) const;

// public:
// 	size_t rows;
// 	size_t cols;

// private:
// 	std::vector<std::vector<T>> data;
// };

// template<typename T>
// Matrix2D<T>::Matrix2D() {}

// template<typename T>
// Matrix2D<T>::Matrix2D(size_t rows, size_t cols):
// 	rows(rows), cols(cols), data(rows, std::vector<T>(cols)) {}

// template<typename T>
// inline T& Matrix2D<T>::operator()(size_t row, size_t col) { return data[row][col]; }

// template<typename T>
// inline const T& Matrix2D<T>::operator()(size_t row, size_t col) const {
// 	return data[row][col];
// }

// class Plane{
// public:
// 	Plane() {}
// 	Plane(cv::Vec3f point, cv::Vec3f normal);
	
// 	float operator[] (int idx) const;
// 	cv::Vec3f operator()();

// 	cv::Vec3f getPoint();
// 	cv::Vec3f getNormal();
// 	cv::Vec3f getCoeff();
// 	Plane viewTransform(int x, int y, int sign, int &qx, int &qy);

// private:
// 	cv::Vec3f point;
// 	cv::Vec3f normal;
// 	cv::Vec3f coeff;
// };

// bool check_image(const cv::Mat &img, std::string name="Image");
// bool check_dimensions(const cv::Mat &img1, const cv::Mat &img2);
// void compute_gray_gradient(const cv::Mat &img, cv::Mat &grad);

// class PatchMatch {
// public:
// 	PatchMatch(float alpha, float gamma, float tc, float tg);
// 	PatchMatch(const PatchMatch &pm) = delete;

// 	PatchMatch &operator=(const	PatchMatch &pm) = delete;

// 	void set(const cv::Mat &img1, const cv::Mat &img2);
// 	void process(int iterations, bool reverse = false);
// 	void postProcess();

// 	cv::Mat getLeftDisparityMap() const;
// 	cv::Mat getRightDisparityMap() const;

// 	float alpha;
// 	float gamma;
// 	float tc;
// 	float tg;
	
// private:
// 	void precompute_pixels_weights(const cv::Mat &img, cv::Mat &weights,
// 				       int window_size);
// 	void initialize_random_planes(Matrix2D<Plane> &planes, float max_d);
// 	void evaluate_planes_cost(int idx);
// 	float plane_match_cost(const Plane &p, int cx, int cy, int ws, int idx);
// 	float dissimilarity(const cv::Vec3f &pp, const cv::Vec3f &qq,
// 			    const cv::Vec2f &pg, const cv::Vec2f &qg);
// 	void process_pixel(int x, int y, int cpv, int iter);
// 	void planes_to_disparity(const Matrix2D<Plane> &planes, cv::Mat &disp);

// 	void spatial_propagation(int x, int y, int cpv, int iter);
// 	void view_propagation(int x, int y, int cpv);
// 	void plane_refinement(int x, int y, int cpv, float max_delta_z,
// 			      float max_delta_n, float end_dz);
// 	void fill_invalid_pixels(int y, int x, Matrix2D<Plane> &planes,
// 				 const cv::Mat &validity);
// 	void weighted_median_filter(int cx, int cy, cv::Mat &disp,
// 				    const cv::Mat &weights,
// 				    const cv::Mat &validity,
// 				    int ws, bool use_invalid);

// 	int rows;
// 	int cols;
// 	cv::Mat views[2];
// 	cv::Mat grads[2];
// 	cv::Mat disps[2];
// 	cv::Mat weights[2];
// 	Matrix2D<Plane> planes[2];
// 	cv::Mat costs[2];
// };

class Option {
public:
	Option() {}
	~Option() {}
	void Run();

public:
	std::vector<std::string> config;
};

} // End of namespace github

#endif /* ifndef CACHEFUNCTION_H */

