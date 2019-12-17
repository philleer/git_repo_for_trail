#ifndef SRC_PATCH_MATCH_H
#define SRC_PATCH_MATCH_H

#include <iostream>	// std::cout, std::endl, std::cerr
#include <vector>	// std::vector
#include <cstddef>	// size_t
#include <memory>	// std::unique_ptr
#include <set>		// std::set
#include <mutex>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include "mvs/mat.h"
#include "mvs/image.h"
#include "mvs/depth_map.h"
#include "mvs/normal_map.h"
// #include "util/threading.h"

// Maximum possible window radius for the photometric consistency cost. This
// value is equal to THREADS_PER_BLOCK in patch_match_cuda.cu and the limit
// arises from the shared memory implementation.
const static size_t kMaxPatchMatchWindowRadius = 32;

<<<<<<< HEAD
=======
// class Image {
// public:
// 	Image() {}
// 	cv::Mat getImage() const;
// 	size_t getWidth() const;
// 	size_t getHeight() const;
// 	inline const cv::Mat &getBitmap() const;
// 	inline const float* GetK() const;

// private:
// 	cv::Mat src_img;
// 	std::string path_;
// 	size_t width_;
// 	size_t height_;
// 	float K_[9];
// 	float R_[9];
// 	float T_[3];
// 	float P_[12];
// 	float inv_P_[12];
// };

// cv::Mat Image::getImage() const { return src_img; }
// size_t Image::getHeight() const { return width_; }
// size_t Image::getWidth() const { return height_; }
// const cv::Mat &Image::getBitmap() const { return src_img; }

// const float *Image::GetK() const { return K_; }

>>>>>>> 42917ec... add the necessary headers implementations
class ConsistencyGraph {
public:
	ConsistencyGraph();
	ConsistencyGraph(const size_t width, const size_t height,
					 const std::vector<int> &data);

	void getImageIdxs(const int row, const int col, int *num_images,
					  const int **image_idxs) const;

private:
	void InitializeMap(const size_t width, const size_t height);
	const static int kNoConsistentImageIds;
	std::vector<int> data_;
	Eigen::MatrixXi map_;
};

namespace gpu {
class PatchMatchCuda;
}

class Workspace;

struct PatchMatchOptions {
	int max_image_size = -1;	// Maximum image size in either dimension.	
	std::string gpu_index = "-1";	// Index of the GPU used for patch match.

	// Depth range in which to randomly sample depth hypotheses.
	double depth_min = -1.0f;
	double depth_max = -1.0f;

	// Half window size to compute NCC photo-consistency cost.
	int window_radius = 5;

	// Number of pixels to skip when computing NCC. For a value of 1, every
	// pixel is used to compute the NCC. For larger values, only every n-th row
	// and column is used and the computation speed thereby increases roughly by
	// a factor of window_step^2. Note that not all combinations of window sizes
	// and steps produce nice results, especially if the step is greather than 2.
	int window_step = 1;

	// Parameters for bilaterally weighted NCC.
	double sigma_spatial = -1;
	double sigma_color = 0.2f;

	// Number of random samples to draw in Monte Carlo sampling.
	int num_samples = 15;
	
	double ncc_sigma = 0.6f; // Spread of the NCC likelihood function.
	double min_triangulation_angle = 1.0f; // Minimum triang-angle in degrees.

	// Spread of the incident angle likelihood function.
	double incident_angle_sigma = 0.9f;

	// Number of coordinate descent iterations. Each iteration consists
	// of four sweeps from left to right, top to bottom, and vice versa.
	int num_iterations = 5;

	// Whether to add a regularized geometric consistency term to the cost
	// function. If true, the `depth_maps` and `normal_maps` must not be null.
	bool geom_consistency = true;

	// The relative weight of the geometric consistency term w.r.t. to
	// the photo-consistency term.
	double geom_consistency_regularizer = 0.3f;

	// Maximum geometric consistency cost in terms of the forward-backward
	// reprojection error in pixels.
	double geom_consistency_max_cost = 3.0f;

	bool filter = true;	// Whether to enable filtering.

	// Minimum NCC coefficient for pixel to be photo-consistent.
	double filter_min_ncc = 0.1f;

	// Minimum triangulation angle to be stable.
	double filter_min_triangulation_angle = 3.0f;

	// Minimum number of source images have to be consistent
	// for pixel not to be filtered.
	int filter_min_num_consistent = 2;

	// Maximum forward-backward reprojection error for pixel
	// to be geometrically consistent.
	double filter_geom_consistency_max_cost = 1.0f;

	// Cache size in gigabytes for patch match, which keeps the bitmaps, depth
	// maps, and normal maps of this number of images in memory. A higher value
	// leads to less disk access and faster computation, while a lower value
	// leads to reduced memory usage. Note that a single image can consume a lot
	// of memory, if the consistency graph is dense.
	double cache_size = 32.0;
	
	bool write_consistency_graph = false; // Whether to write the consistency graph.

	void Print() const;
	bool Check() const {
		if (depth_min != -1.0f || depth_max != -1.0f) {
			if (depth_max < depth_min) {
				std::cerr << "depth configuration error min > max!\n";
			}
			if (depth_min < 0.0f) {
				std::cerr << "depth min set error min < 0 !\n";
			}
			return false;
		}

		if (window_radius > static_cast<int>(kMaxPatchMatchWindowRadius) ||
			window_radius <= 0)
		{
			std::cerr << "window radius set unproperly!\n";
			return false;
		}

		if (sigma_color <= 1e-8f) {
			std::cerr << "sigma_color set unproperly!\n";
			return false;
		}

		if (window_step <= 0 || window_step > 2) {
			std::cerr << "window_step set unproperly!\n";
			return false;
		}

		if (num_samples <= 0) {
			std::cerr << "num_samples set unproperly!\n";
			return false;
		}

		if (ncc_sigma < 1e-8f) {
			std::cerr << "ncc_sigma set unproperly!\n";
			return false;
		}

		if (min_triangulation_angle < -1e-8f || min_triangulation_angle >= 180.f) {
			std::cerr << "min_triangulation_angle set unproperly!\n";
			return false;
		}
		
		if (incident_angle_sigma < 1e-8f) {
			std::cerr << "incident_angle_sigma set unproperly!\n";
			return false;
		}

		if (num_iterations <= 0) {
			std::cerr << "num_iterations set unproperly!\n";
			return false;
		}

		if (geom_consistency_regularizer < -1e-8f) {
			std::cerr << "geom_consistency_regularizer set unproperly!\n";
			return false;
		}

		if (geom_consistency_max_cost < -1e-8f) {
			std::cerr << "geom_consistency_max_cost set unproperly!\n";
			return false;
		}
		
		if (filter_min_ncc > 1.0f || filter_min_ncc < -1.0f) {
			std::cerr << "filter_min_ncc set unproperly!\n";
			return false;
		}

		if (filter_min_triangulation_angle < -1e-8f ||
			filter_min_triangulation_angle > 180.0f)
		{
			std::cerr << "filter_min_triangulation_angle set unproperly!\n";
			return false;
		}

		if (filter_min_num_consistent < 0) {
			std::cerr << "filter_min_num_consistent set unproperly!\n";
			return false;
		}

		if (filter_geom_consistency_max_cost < 1e-8f) {
			std::cerr << "filter_geom_consistency_max_cost set unproperly!\n";
			return false;
		}

		if (cache_size <= 1e-8f) {
			std::cerr << "cache_size set unproperly!\n";
			return false;
		}

		return true;
	}
};

// This is a wrapper class around the actual PatchMatchCuda implementation. This
// class is necessary to hide Cuda code from any boost or Eigen code, since
// NVCC/MSVC cannot compile complex C++ code.
class PatchMatch {
public:
	struct Problem {
		int ref_image_idx = -1;	// Index of the reference image.
		std::vector<int> src_image_idxs;	// Indices of the source images.

		// Input images for the photometric consistency term.
		std::vector<Image>* images = nullptr;
		// Input depth maps for the geometric consistency term.
		std::vector<DepthMap>* depth_maps = nullptr;
		// Input normal maps for the geometric consistency term.
		std::vector<NormalMap>* normal_maps = nullptr;

		void Print() const;	// Print the configuration to stdout.
	};

	PatchMatch(const PatchMatchOptions& options, const Problem& problem);
	~PatchMatch();
	
	void Check() const;	// Check the options and the problem for validity.
	void Run();

	// Get the computed values after running the algorithm.
	DepthMap getDepthMap() const;
	NormalMap getNormalMap() const;
	ConsistencyGraph getConsistencyGraph() const;
	Mat<float> getSelProbMap() const;

private:
	size_t char_in_string (const std::string &s, char c) const {
		if (s.empty()) return 0;
		size_t count = 0;
		for (auto ch : s) if (ch == c) ++count;
		return count;
	}

private:
	const PatchMatchOptions options_;
	const Problem problem_;
	std::unique_ptr<gpu::PatchMatchCuda> patch_match_cuda_;
};


// __CUDACC__ is defined on both device and host.
// __CUDA_ARCH__ is defined on the device only
// __NVCC__ Defined when compiling C/C++/CUDA source files.
// __CUDACC__ Defined when compiling CUDA source files.

// architecture identification macro __CUDA_ARCH__
// The architecture identification macro __CUDA_ARCH__ is assigned
// a three-digit value string xy0 (ending in a literal 0) during each
// nvcc compilation stage 1 that compiles for compute_xy.
// This macro can be used in the implementation of GPU functions
// for determining the virtual architecture for which it is currently
// being compiled. The host code (the non-GPU code) must not depend on it.

#ifndef __CUDACC__

class PatchMatchController /*: public Thread*/ {
public:
	PatchMatchController(const PatchMatchOptions& options,
						 const std::string& workspace_path,
						 const std::string& workspace_format,
						 const std::string& pmvs_option_name);

private:
	void Run();
	void ReadWorkspace();
	void ReadProblems();
	void ReadGpuIndices();
	void ProcessProblem(const PatchMatchOptions& options,
						const size_t problem_idx);

	const PatchMatchOptions options_;
	const std::string workspace_path_;
	const std::string workspace_format_;
	const std::string pmvs_option_name_;

	// std::unique_ptr<ThreadPool> thread_pool_;
	std::mutex workspace_mutex_;
// 	std::unique_ptr<Workspace> workspace_;
	std::vector<PatchMatch::Problem> problems_;
	std::vector<int> gpu_indices_;
	std::vector<std::pair<float, float>> depth_ranges_;

private:
	size_t char_in_string (const std::string &s, char c) const {
		if (s.empty()) return 0;
		size_t count = 0;
		for (auto ch : s) if (ch == c) ++count;
		return count;
	}
};

#endif

#endif