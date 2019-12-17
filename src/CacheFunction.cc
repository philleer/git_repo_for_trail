#include "CacheFunction.h"
#include <sstream>	// std::stringstream
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdlib>	// std::rand, RAND_MAX
#include <random>	// std::mt19937
#include <algorithm>

#include <cuda_runtime.h>
#include "util/threading.h"

#define WINDOW_SIZE 35
#define MAX_DISPARITY 60
#define PLANE_PENALTY 120

template<typename T>
class Matrix2D {
public:
	Matrix2D();
	Matrix2D(size_t rows, size_t cols);

	T& operator()(size_t row, size_t col);
	const T& operator()(size_t row, size_t col) const;

public:
	size_t rows;
	size_t cols;

private:
	std::vector<std::vector<T>> data;
};

template<typename T>
Matrix2D<T>::Matrix2D() {}

template<typename T>
Matrix2D<T>::Matrix2D(size_t rows, size_t cols):
	rows(rows), cols(cols), data(rows, std::vector<T>(cols)) {}

template<typename T>
inline T& Matrix2D<T>::operator()(size_t row, size_t col) { return data[row][col]; }

template<typename T>
inline const T& Matrix2D<T>::operator()(size_t row, size_t col) const {
	return data[row][col];
}

class Plane{
public:
	Plane() {}
	Plane(cv::Vec3f point, cv::Vec3f normal);
	
	float operator[] (int idx) const;
	cv::Vec3f operator()();

	cv::Vec3f getPoint();
	cv::Vec3f getNormal();
	cv::Vec3f getCoeff();
	Plane viewTransform(int x, int y, int sign, int &qx, int &qy);

private:
	cv::Vec3f point;
	cv::Vec3f normal;
	cv::Vec3f coeff;
};

// ax + by +cz = 0
// z = - a/c * x - b/c * y - d/c
// a = nx, b = ny, c = nz, d = -(nx * p[0] + ny * p[1] + nz * p[2])
Plane::Plane(cv::Vec3f point, cv::Vec3f normal): point(point), normal(normal) {
	coeff[0] = -normal[0] / normal[2];
	coeff[1] = -normal[1] / normal[2];
	coeff[2] = (normal[0]*point[0]+normal[1]*point[1]+normal[2]*point[2])/normal[2];
}

inline float Plane::operator[](int idx) const { return coeff[idx]; }
inline cv::Vec3f Plane::operator()() { return coeff; }
inline cv::Vec3f Plane::getPoint() { return point; }
inline cv::Vec3f Plane::getNormal() { return normal; }
inline	cv::Vec3f Plane::getCoeff() { return coeff; }

Plane Plane::viewTransform(int x, int y, int sign, int &qx, int &qy) {
	float z = coeff[0]*x + coeff[1]*y + coeff[2];
	qx = x + sign*z;
	qy = y;

	cv::Vec3f p(qx, qy, z);
	return Plane(p, this->normal);
}

bool check_image(const cv::Mat &img, std::string name="Image") {
	if (!img.data) {
		std::cerr << name << " data not loaded!\n";
		return false;
	}
	return true;
}

bool check_dimensions(const cv::Mat &img1, const cv::Mat &img2) {
	if (img1.cols != img2.cols || img1.rows != img2.rows) {
		std::cerr << "Dimension of the two images do not corresponds!\n";
		return false;
	}
	return true;
}

void compute_gray_gradient(const cv::Mat &img, cv::Mat &grad) {
	int scale = 1, delta = 0;
	cv::Mat gray, x_grad, y_grad;
	// rgb图像转换成灰度图
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

	// 使用Sobel算子计算一阶梯度
	cv::Sobel(gray, x_grad, CV_32F, 1, 0, 3);
	cv::Sobel(gray, y_grad, CV_32F, 0, 1, 3);
	x_grad /= 8.0f;
	y_grad /= 8.0f;

	#pragma omp parallel for
	for (int y = 0; y < img.rows; ++y) {
		for (int x = 0; x < img.cols; ++x) {
			grad.at<cv::Vec2f>(y, x)[0] = x_grad.at<float>(y, x);
			grad.at<cv::Vec2f>(y, x)[1] = y_grad.at<float>(y, x);
		}
	}
}

class PatchMatch {
public:
	PatchMatch(float alpha, float gamma, float tc, float tg);
	PatchMatch(const PatchMatch &pm) = delete;

	PatchMatch &operator=(const	PatchMatch &pm) = delete;

	void set(const cv::Mat &img1, const cv::Mat &img2);
	void process(int iterations, bool reverse = false);
	void postProcess();

	cv::Mat getLeftDisparityMap() const;
	cv::Mat getRightDisparityMap() const;

	float alpha;
	float gamma;
	float tc;
	float tg;
	
private:
	void precompute_pixels_weights(const cv::Mat &img, cv::Mat &weights,
				       int window_size);
	void initialize_random_planes(Matrix2D<Plane> &planes, float max_d);
	void evaluate_planes_cost(int idx);
	float plane_match_cost(const Plane &p, int cx, int cy, int ws, int idx);
	float dissimilarity(const cv::Vec3f &pp, const cv::Vec3f &qq,
			    const cv::Vec2f &pg, const cv::Vec2f &qg);
	void process_pixel(int x, int y, int cpv, int iter);
	void planes_to_disparity(const Matrix2D<Plane> &planes, cv::Mat &disp);

	void spatial_propagation(int x, int y, int cpv, int iter);
	void view_propagation(int x, int y, int cpv);
	void plane_refinement(int x, int y, int cpv, float max_delta_z,
			      float max_delta_n, float end_dz);
	void fill_invalid_pixels(int y, int x, Matrix2D<Plane> &planes,
				 const cv::Mat &validity);
	void weighted_median_filter(int cx, int cy, cv::Mat &disp,
				    const cv::Mat &weights,
				    const cv::Mat &validity,
				    int ws, bool use_invalid);

	int rows;
	int cols;
	cv::Mat views[2];
	cv::Mat grads[2];
	cv::Mat disps[2];
	cv::Mat weights[2];
	Matrix2D<Plane> planes[2];
	cv::Mat costs[2];
};

PatchMatch::PatchMatch(float alpha, float gamma, float tc, float tg) :
	alpha(alpha), gamma(gamma), tc(tc), tg(tg) {}

cv::Mat PatchMatch::getLeftDisparityMap() const { return this->disps[0]; }
cv::Mat PatchMatch::getRightDisparityMap() const { return this->disps[1]; }

void PatchMatch::precompute_pixels_weights(const cv::Mat &img, cv::Mat &weights, int ws)
{
	int ws_half = ws / 2;

	#pragma omp parallel for
	for (int cx = 0; cx < this->cols; ++cx) {
		for (int cy = 0; cy < this->rows; ++cy) {
			for (int x = cx-ws_half; x <= cx+ws_half; ++x) {
				for (int y = cy-ws_half; y <= cy+ws_half; ++y) {
					if (x>=0 && x<this->cols && y>=0 && y<this->rows) {
						cv::Vec<int, 4> pos = cv::Vec<int, 4>(cy, cx,
								y-cy+ws_half, x-cx+ws_half);
						cv::Vec3f p = static_cast<cv::Vec3f>(
								img.at<cv::Vec3b>(cy, cx));
						cv::Vec3f q = static_cast<cv::Vec3f>(
								img.at<cv::Vec3b>(y, x));
						weights.at<float>(pos) = std::exp(-cv::norm(p-q)/this->gamma);
					}
				}
			}
		}
	}
}

void PatchMatch::initialize_random_planes(Matrix2D<Plane> &planes, float max_d) {
	cv::RNG random_generator;
	const int RAND_HALF = RAND_MAX / 2;
	#pragma omp	parallel for
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			float z = random_generator.uniform(0.0f, max_d);
			cv::Vec3f point(x, y, z);

			float nx = (float)(std::rand()-RAND_HALF)/RAND_HALF;
			float ny = (float)(std::rand()-RAND_HALF)/RAND_HALF;
			float nz = (float)(std::rand()-RAND_HALF)/RAND_HALF;
			cv::Vec3f normal(nx, ny, nz);
			cv::normalize(normal, normal);
			planes(y, x) = Plane(point, normal);
		}
	}
}

void PatchMatch::evaluate_planes_cost(int idx) {
	#pragma omp parallel for
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			this->costs[idx].at<float>(y, x) = plane_match_cost(planes[idx](y, x), 
									    x, y, WINDOW_SIZE,
									    idx);
		}
	}
}

float PatchMatch::plane_match_cost(const Plane &p, int cx, int cy, int ws, int idx)
{
	int sign = -1+2*idx;
	float cost = 0;
	int half = ws/2;
	const cv::Mat &f1 = views[idx];
	const cv::Mat &f2 = views[1-idx];
	const cv::Mat &g1 = grads[idx];
	const cv::Mat &g2 = grads[1-idx];
	const cv::Mat &w1 = weights[idx];

	for (int x = cx-half; x <= cx+half; ++x) {
		for (int y = cy-half; y <= cy+half; ++y) {
			if (!(x>=0 && x < cols && y>=0 && y < rows)) continue;

			// computing disparity
			float dsp = p[0]*x + p[1]*y + p[2];
			if (dsp < 0 || dsp > MAX_DISPARITY) {
				cost += PLANE_PENALTY;
			} else {
				float match = x+sign*dsp;
				int x_match = (int)match;
				float wm = 1 - (match - x_match);
				if (x_match > cols-2) x_match = cols-2;
				if (x_match < 0) x_match = 0;

				// wx*x + (1-wx)*y
				cv::Vec3b mcolo = wm*f2.at<cv::Vec3b>(y, x_match) +
						  (1-wm)*f2.at<cv::Vec3b>(y, x_match+1);
				cv::Vec2f mgrad = wm*g2.at<cv::Vec2f>(y, x_match) + (1-wm)*g2.at<cv::Vec2f>(y, x_match+1);
				float w = w1.at<float>(cv::Vec<int, 4>{cy, cx, y-cy+half, x-cx+half});
				cv::Vec3f pp, qq;
				for (int i = 0; i < 3; ++i) {
					pp[i] = static_cast<float>(f1.at<cv::Vec3b>(y, x)[i]);
					qq[i] = static_cast<float>(mcolo[i]);
				}
				cost += w*dissimilarity(pp, qq, g1.at<cv::Vec2f>(y, x), mgrad);
			}
		}
	}
	return cost;
}

float PatchMatch::dissimilarity(const cv::Vec3f &pp, const cv::Vec3f &qq,
				const cv::Vec2f &pg, const cv::Vec2f &qg)
{
	float cost_c = cv::norm(pp-qq, cv::NORM_L1);
	float cost_g = cv::norm(pg-qg, cv::NORM_L1);
	cost_c = std::min(cost_c, this->tc);
	cost_g = std::min(cost_g, this->tg);
	return (1-this->alpha)*cost_c + this->alpha*cost_g;
}

void PatchMatch::process(int iterations, bool reverse) {
	for (int iter = 0+reverse; iter < iterations+reverse; ++iter) {
		bool iter_type = (iter%2 == 0);
		std::cerr << "Iteration " << iter-reverse+1 << "/"
				<< iterations-reverse << "\r";

		for (int view_id = 0; view_id < 2; ++view_id) {
			if (iter ^ 1) {
				for (int y = 0; y < rows; ++y) {
					for (int x = 0; x < cols; ++x) {
						process_pixel(x, y, view_id, iter);
					}
				}
			} else {
				for (int y = rows-1; y >= 0; --y) {
					for (int x = cols-1; x >= 0; --x) {
						process_pixel(x, y, view_id, iter);
					}
				}
			}
		}
		std::cerr << std::endl;

		this->planes_to_disparity(this->planes[0], this->disps[0]);
		this->planes_to_disparity(this->planes[1], this->disps[1]);
	}
}

void PatchMatch::process_pixel(int x, int y, int idx, int iter) {
	spatial_propagation(x, y, idx, iter);
	plane_refinement(x, y, idx, MAX_DISPARITY/2, 1.0f, 0.1f);
	view_propagation(x, y, idx);
}

void PatchMatch::spatial_propagation(int x, int y, int idx, int iter) {
	std::vector<std::pair<int, int>> offsets;
	if (iter^1) {
		offsets.push_back(std::make_pair(1, 0));
		offsets.push_back(std::make_pair(0, 1));
	} else {
		offsets.push_back(std::make_pair(-1, 0));
		offsets.push_back(std::make_pair(0, -1));
	}

	
	Plane &old_plane = planes[idx](y, x);
	float &old_cost = costs[idx].at<float>(y, x);
	for (auto it = offsets.begin(); it != offsets.end(); ++it) {
		std::pair<int, int> off_tmp = *it;
		int ny = y + off_tmp.first;
		int nx = x + off_tmp.second;
		if (!(nx >= 0 && nx < cols && ny >= 0 && ny < rows)) continue;

		Plane p_neigh = planes[idx](ny, nx);
		float new_cost = plane_match_cost(p_neigh, x, y, WINDOW_SIZE, idx);
		if (old_cost > new_cost) {
			old_plane = p_neigh;
			old_cost = new_cost;
		}
	}
}

void PatchMatch::plane_refinement(int x, int y, int idx, float max_delta_z,
				  float max_delta_n, float end_dz)
{
	float max_dz = max_delta_z;
	float max_dn = max_delta_n;
	Plane &old_plane = planes[idx](y, x);
	float &old_cost = costs[idx].at<float>(y, x);

	while (max_dz >= end_dz) {
		std::random_device rd;
		std::mt19937 gen(rd());

		std::uniform_real_distribution<> rand_z(-max_dz, max_dz);
		std::uniform_real_distribution<> rand_n(-max_dn, max_dn);
		float z = old_plane[0]*x + old_plane[1]*y + old_plane[2];
		float delta_z = rand_z(gen);
		cv::Vec3f new_point(x, y, z+delta_z);
		cv::Vec3f n = old_plane.getNormal();

		cv::Vec3f delta_n(rand_n(gen), rand_n(gen), rand_n(gen));
		cv::Vec3f new_normal = n+delta_n;
		cv::normalize(new_normal, new_normal);

		Plane new_plane(new_point, new_normal);
		float new_cost = plane_match_cost(new_plane, x, y, WINDOW_SIZE, idx);
		if (new_cost < old_cost) {
			old_plane = new_plane;
			old_cost = new_cost;
		}

		max_dz /= 2.0f;
		max_dn /= 2.0f;
	}
}

void PatchMatch::planes_to_disparity(const Matrix2D<Plane> &planes, cv::Mat &disp)
{
	#pragma omp parallel for
	for (int x = 0; x < cols; ++x) {
		for (int y = 0; y < rows; ++y) {
			disp.at<float>(y, x) = planes(y, x)[0]*x +
					       planes(y, x)[1]*y +
					       planes(y, x)[2];
		}
	}
}

void PatchMatch::view_propagation(int x, int y, int idx) {
	Plane view_plane = this->planes[idx](y, x);

	int mx, my;
	Plane new_plane = view_plane.viewTransform(x, y, 2*idx-1, mx, my);
	if (!(mx >= 0 && mx < cols && my >= 0 && my < rows)) return;

	float &old_cost = costs[1-idx].at<float>(my, mx);
	float new_cost = plane_match_cost(new_plane, mx, my, WINDOW_SIZE, 1-idx);
	if (new_cost < old_cost) {
		planes[1-idx](my, mx) = new_plane;
		old_cost = new_cost;
	}
}

void PatchMatch::set(const cv::Mat &img1, const cv::Mat &img2) {
	this->rows = img1.rows;
	this->cols = img1.cols;
	this->views[0] = img1;
	this->views[1] = img2;

	int wmat_size[] = {rows, cols, WINDOW_SIZE, WINDOW_SIZE};
	this->weights[0] = cv::Mat(4, wmat_size, CV_32F);
	this->weights[1] = cv::Mat(4, wmat_size, CV_32F);
	precompute_pixels_weights(img1, weights[0], WINDOW_SIZE);
	precompute_pixels_weights(img2, weights[1], WINDOW_SIZE);

	this->grads[0] = cv::Mat(rows, cols, CV_32FC2);
	this->grads[1] = cv::Mat(rows, cols, CV_32FC2);
	compute_gray_gradient(img1, this->grads[0]);
	compute_gray_gradient(img2, this->grads[1]);

	this->planes[0] = Matrix2D<Plane>(rows, cols);
	this->planes[1] = Matrix2D<Plane>(rows, cols);
	this->initialize_random_planes(this->planes[0], MAX_DISPARITY);
	this->initialize_random_planes(this->planes[1], MAX_DISPARITY);

	this->costs[0] = cv::Mat(rows, cols, CV_32F);
	this->costs[1] = cv::Mat(rows, cols, CV_32F);
	this->evaluate_planes_cost(0);
	this->evaluate_planes_cost(1);

	this->disps[0] = cv::Mat(rows, cols, CV_32F);
	this->disps[1] = cv::Mat(rows, cols, CV_32F);
}

void PatchMatch::postProcess() {
	std::cerr << "Post processing...\n";
	cv::Mat lft_validity(rows, cols, CV_8UC1);
	cv::Mat rgt_validity(rows, cols, CV_8UC1);

	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			int x_rgt_match = std::max(0.0f,
						   std::min((float)cols,
							    x-disps[0].at<float>(y, x))
						  );
			int x_lft_match = std::max(0.0f,
						   std::min((float)rows,
							    x+disps[1].at<float>(y, x))
						  );
			lft_validity.at<unsigned char>(y, x) = (
					std::abs(disps[0].at<float>(y, x) -
					disps[1].at<float>(y, x_rgt_match)) <= 1 ? 1 : 0);
			rgt_validity.at<unsigned char>(y, x) = (
					std::abs(disps[1].at<float>(y, x) -
					disps[0].at<float>(y, x_lft_match)) <= 1 ? 1 : 0);
		}
	}

	#pragma omp parallel for
	for (int y=0; y < rows; ++y) {
		for (int x=0; x < cols; ++x) {
			if (!lft_validity.at<unsigned char>(y, x)) {
				fill_invalid_pixels(y, x, planes[0], lft_validity);
			}

			if (!rgt_validity.at<unsigned char>(y, x)) {
				fill_invalid_pixels(y, x, planes[1], rgt_validity);
			}
		}
	}

	this->planes_to_disparity(this->planes[0], this->disps[0]);
	this->planes_to_disparity(this->planes[1], this->disps[1]);

	for (int x=0; x <cols; ++x) {
		for (int y=0; y < rows; ++y) {
			weighted_median_filter(x, y, disps[0], weights[0], lft_validity,
					       WINDOW_SIZE, false);
			weighted_median_filter(x, y, disps[1], weights[1], rgt_validity,
					       WINDOW_SIZE, false);
		}
	}
}

void PatchMatch::fill_invalid_pixels(int y, int x, Matrix2D<Plane> &planes,
	const cv::Mat &validity)
{
	int x_lft = x-1;
	int x_rgt = x+1;
	while (!validity.at<unsigned char>(y, x_lft) && x_lft >= 0) --x_lft;
	while (!validity.at<unsigned char>(y, x_rgt) && x_rgt >= 0) ++x_rgt;

	int best_plane_x = x;
	if (x_lft >= 0 && x_rgt < cols) {
		float disp_l = planes(y, x_lft)[0]*x + planes(y, x_lft)[1]*y +
					   planes(y, x_lft)[2];
		float disp_r = planes(y, x_rgt)[0]*x + planes(y, x_rgt)[1]*y +
					   planes(y, x_rgt)[2];
		best_plane_x = (disp_l < disp_r ? x_lft : x_rgt);
	} else if (x_lft >= 0) {
		best_plane_x = x_lft;
	} else if (x_rgt < cols) {
		best_plane_x = x_rgt;
	}

	planes(y, x) = planes(y, best_plane_x);
}

void PatchMatch::weighted_median_filter(int cx, int cy, cv::Mat &disp, 
					const cv::Mat &weights,
					const cv::Mat &validity,
					int ws, bool use_invalid)
{
	int half = ws / 2;
	float w_tot = 0;
	float w = 0;
	std::vector<std::pair<float,float>> disps_w;
	for (int x=cx-half; x<=cx+half; ++x) {
		for (int y=cy-half; y<=cy+half; ++y) {
			if (x>=0 && x<cols && y>=0 && y<rows &&
				(use_invalid || validity.at<unsigned char>(y, x))) {
				cv::Vec<int, 4> w_ids({cy, cx, y-cy+half, x-cx+half});
				w_tot += weights.at<float>(w_ids);
				disps_w.push_back(std::make_pair(weights.at<float>(w_ids),
								 disp.at<float>(y, x)));
			}
		}
	}

	std::sort(disps_w.begin(), disps_w.end());
	float med_w = w_tot / 2.0f;

	for (auto dw=disps_w.begin(); dw < disps_w.end(); ++dw) {
		w += dw->first;
		if (w >= med_w) {
			if (dw == disps_w.begin())
				disp.at<float>(cy, cx) = dw->second;
			else
				disp.at<float>(cy, cx) = ((dw-1)->second + dw->second) / 2.0f;
		}
	}
}

int github::Option::sum_integers(const std::vector<int> &integers) {
	auto sum = 0;
	for (auto i : integers) {
		sum += i;
	}

	return sum;
}

void github::Option::Run() {
	// Begin to count time
	auto start = std::chrono::system_clock::now();

	/*
	 * source code here to test
	 */
	printf("=========================================\n");
	printf("             PatchMatch Test             \n");
	printf("=========================================\n");
	std::string base_path = "/root/mvs_project/dataset/";
	std::string img1_path = base_path + "fountain_dense/urd/0001.png";
	std::string img2_path = base_path + "fountain_dense/urd/0002.png";
	// std::string img1_path = base_path + "cones/im2.png";
	// std::string img2_path = base_path + "cones/im6.png";
	// std::string img1_path = base_path + "dino/dino0001.png";
	// std::string img2_path = base_path + "dino/dino0002.png";
	
	// std::string img1_path = base_path + "image_13a/2a_005.jpg";
	// std::string img2_path = base_path + "image_13a/2a_010.jpg";
	// Load image and check the validity
	cv::Mat img1 = cv::imread(img1_path, cv::IMREAD_COLOR);
	cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_COLOR);
	cv::resize(img1, img1, cv::Size(img1.cols/4, img1.rows/4), cv::INTER_LINEAR);
	cv::resize(img2, img2, cv::Size(img2.cols/4, img2.rows/4), cv::INTER_LINEAR);
	if (!check_image(img1, img1_path) || !check_image(img2, img2_path)) exit(1);
	if (!check_dimensions(img1, img2)) exit(1);

	// Processing
	// const float alpha = 0.9f;
	// const float gamma = 10.0f;
	// const float tc = 10.f;
	// const float tg = 2.0f;
	// PatchMatch patchmatch(alpha, gamma, tc, tg);
	// patchmatch.set(img1, img2);
	// patchmatch.process(5);
	// patchmatch.postProcess();

	// cv::Mat disp1 = patchmatch.getLeftDisparityMap();
	// cv::Mat disp2 = patchmatch.getRightDisparityMap();
	// 数组的数值被平移或缩放到一个指定的范围，线性归一化
	// cv::normalize(disp1, disp1, 0, 255, cv::NORM_MINMAX);
	// cv::normalize(disp2, disp2, 0, 255, cv::NORM_MINMAX);
	// cv::imwrite("left_disparity.png", disp1);
	// cv::imwrite("right_disparity.png", disp2);

	void* data = nullptr;
	auto err = cudaMalloc(&data, 256);
	printf("%s\n", cudaGetErrorString(err));

	// cache test
	std::vector<int> integers{1, 2, 3, 4, 5, 6, 7, 8, 9};
	auto sum = sum_integers(integers);
	std::cout << sum << std::endl;

	// Time count finished here and then print to the screen
	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration<double>(end - start);
	std::cout << "Cost time " << duration.count() << " s" << std::endl;
}
