#ifndef SRC_PATCH_MATCH_H
#define SRC_PATCH_MATCH_H

#include <iostream>	// std::cout, std::endl, std::cerr
#include <vector>	// std::vector
#include <cstddef>	// size_t
#include <algorithm>// transform
#include <memory>	// std::unique_ptr
#include <set>		// std::set
#include <mutex>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include "mvs/mat.h"
#include "mvs/image.h"
#include "mvs/depth_map.h"
#include "mvs/normal_map.h"
#include "mvs/model.h"
#include "util/cache.h"
#include "util/util_sys.h"

#ifndef __CUDACC__
#include "util/threading.h"
#endif

// Maximum possible window radius for the photometric consistency cost. This
// value is equal to THREADS_PER_BLOCK in patch_match_cuda.cu and the limit
// arises from the shared memory implementation.
const static size_t kMaxPatchMatchWindowRadius = 32;

// List of geometrically consistent images, in the following format:
//
//    r_1, c_1, N_1, i_11, i_12, ..., i_1N_1,
//    r_2, c_2, N_2, i_21, i_22, ..., i_2N_2, ...
//
// where r, c are the row and column image coordinates of the pixel,
// N is the number of consistent images, followed by the N image indices.
// Note that only pixels are listed which are not filtered and that the
// consistency graph is only filled if filtering is enabled.
class ConsistencyGraph {
public:
	ConsistencyGraph();
	ConsistencyGraph(const size_t width, const size_t height,
					 const std::vector<int> &data);
	
	size_t getNumBytes() const;

	void getImageIdxs(const int row, const int col, int *num_images,
					  const int **image_idxs) const;

	void Read(const std::string& path);
	void Write(const std::string& path) const;

private:
	void InitializeMap(const size_t width, const size_t height);
	const static int kNoConsistentImageIds;
	std::vector<int> data_;
	Eigen::MatrixXi map_;
};

ConsistencyGraph::ConsistencyGraph() {}

ConsistencyGraph::ConsistencyGraph(const size_t width, const size_t height,
	const std::vector<int> &data) : data_(data) {
	InitializeMap(width, height);
}

size_t ConsistencyGraph::getNumBytes() const {
	return (data_.size() + map_.size())*sizeof(int);
}

void ConsistencyGraph::getImageIdxs(const int row, const int col, int* num_images,
	const int **image_idxs) const
{
	const int index = map_(row, col);
	if (index == kNoConsistentImageIds) {
		*num_images = 0;
		*image_idxs = nullptr;
	} else {
		*num_images = data_.at(index);
		*image_idxs = &data_.at(index+1);
	}
}

void ConsistencyGraph::Read(const std::string &path) {
	std::fstream text_file(path, std::ios::in | std::ios::binary);
	if (!text_file.is_open()) {
		std::cerr << path << " load failed!" << std::endl;
		exit(EXIT_FAILURE);
	}

	size_t width = 0, height = 0, depth = 0;
	char unused_char;
	text_file >> width >> unused_char >> height >> unused_char
			>> depth >> unused_char;
	const std::streampos pos = text_file.tellg();
	text_file.close();

	if (width <= 0 || height <= 0 || depth <= 0) {
		std::cerr << "dimension of the file error!" << std::endl;
		exit(EXIT_FAILURE);
	}

	std::fstream binary_file(path, std::ios::in | std::ios::binary);
	if (!binary_file.is_open()) {
		std::cerr << path << " load failed!" << std::endl;
		exit(EXIT_FAILURE);
	}

	binary_file.seekg(0, std::ios::end);
	const size_t num_bytes = binary_file.tellg() - pos;
	data_.resize(num_bytes / sizeof(int));

	binary_file.seekg(pos);
	for (size_t i = 0; i < data_.size(); ++i) {
		binary_file.read(reinterpret_cast<char*>(&data_[i]), sizeof(int));
	}

	binary_file.close();
	InitializeMap(width, height);
}

void ConsistencyGraph::Write(const std::string& path) const {
	std::fstream text_file(path, std::ios::out);
	if (!text_file.is_open()) {
		std::cerr << path << " load failed!" << std::endl;
		exit(EXIT_FAILURE);
	}
	text_file << map_.cols() << "&" << map_.rows() << "&" << 1 << "&";
	text_file.close();

	std::fstream binary_file(path,
	                       std::ios::out | std::ios::binary | std::ios::app);
	if (!binary_file.is_open()) {
		std::cerr << path << " load failed!" << std::endl;
		exit(EXIT_FAILURE);
	}
	for (size_t i = 0; i < data_.size(); ++i) {
		binary_file.write(reinterpret_cast<const char*>(&data_[i]), sizeof(int));
	}
	binary_file.close();
}

void ConsistencyGraph::InitializeMap(const size_t width, const size_t height) {
	map_.resize(height, width);
	map_.setConstant(kNoConsistentImageIds);
	for (size_t i = 0; i < data_.size();) {
		const int num_images = data_.at(i+2);
		if (num_images > 0) {
			const int col = data_.at(i);
			const int row = data_.at(i+1);
			map_(row, col) = i + 2;
		}
		i += (3+num_images);
	}
}

class PatchMatchCuda;

class Workspace {
public:
	struct Options {
		// The maximum cache size in gigabytes.
		double cache_size = 32.0;

		// Maximum image size in either dimension.
		int max_image_size = -1;

		// Whether to read image as RGB or gray scale.
		bool image_as_rgb = true;

		// Location and type of workspace.
		std::string workspace_path;
		std::string workspace_format;
		std::string input_type;
		std::string stereo_folder = "stereo";
	};
	Workspace(const Options& options);

	void ClearCache();

	const Options& GetOptions() const;

	const Model& GetModel() const;
	const cv::Mat& GetSrcmap(const int image_idx);
	// const Bitmap& GetBitmap(const int image_idx);
	const DepthMap& GetDepthMap(const int image_idx);
	const NormalMap& GetNormalMap(const int image_idx);

	// Get paths to bitmap, depth map, normal map and consistency graph.
	// std::string GetBitmapPath(const int image_idx) const;
	std::string GetSrcmapPath(const int image_idx) const;
	std::string GetDepthMapPath(const int image_idx) const;
	std::string GetNormalMapPath(const int image_idx) const;

	// Return whether bitmap, depth map, normal map, and consistency graph exist.
	// bool HasBitmap(const int image_idx) const;
	bool HasDepthMap(const int image_idx) const;
	bool HasNormalMap(const int image_idx) const;

private:
	std::string GetFileName(const int image_idx) const;

	class CachedImage {
	public:
		CachedImage();
		CachedImage(CachedImage&& other);
		CachedImage& operator=(CachedImage&& other);
		size_t NumBytes() const;
		size_t num_bytes = 0;
		// std::unique_ptr<Bitmap> bitmap;
		std::unique_ptr<cv::Mat> srcmap;
		std::unique_ptr<DepthMap> depth_map;
		std::unique_ptr<NormalMap> normal_map;

	private:
		CachedImage(CachedImage const& obj) = delete;
		void operator=(CachedImage const& obj) = delete;
	};

	Options options_;
	Model model_;
	MemoryConstrainedLRUCache<int, CachedImage> cache_;
	std::string depth_map_path_;
	std::string normal_map_path_;
};

Workspace::CachedImage::CachedImage() {}

Workspace::CachedImage::CachedImage(CachedImage&& other) {
	num_bytes = other.num_bytes;
	// bitmap = std::move(other.bitmap);
	srcmap = std::move(other.srcmap);
	depth_map = std::move(other.depth_map);
	normal_map = std::move(other.normal_map);
}

Workspace::CachedImage& Workspace::CachedImage::operator=(CachedImage&& other) {
	if (this != &other) {
		num_bytes = other.num_bytes;
		// bitmap = std::move(other.bitmap);
		srcmap = std::move(other.srcmap);
		depth_map = std::move(other.depth_map);
		normal_map = std::move(other.normal_map);
	}
	return *this;
}

size_t Workspace::CachedImage::NumBytes() const { return num_bytes; }

Workspace::Workspace(const Options& options) : options_(options),
      cache_(1024*1024*1024*static_cast<size_t>(options_.cache_size),
             [](const int) { return CachedImage(); })
{
	std::string &input = options_.input_type;
	std::transform(input.begin(), input.end(), input.begin(), ::tolower);
	model_.Read(options_.workspace_path, options_.workspace_format);
	if (options_.max_image_size > 0) {
		for (auto& image : model_.images) {
			// image.Downsize(options_.max_image_size, options_.max_image_size);
		}
	}

	depth_map_path_ = options_.workspace_path + "/" +
					  options_.stereo_folder + "/depth_maps/";
	if (depth_map_path_.back() != '/') {
		depth_map_path_ += "/";
	}
	normal_map_path_ = options_.workspace_path + "/" +
					   options_.stereo_folder + "/normal_maps/";
	if (normal_map_path_.back() != '/') {
		normal_map_path_ += "/";
	}
}

void Workspace::ClearCache() { cache_.Clear(); }

const Workspace::Options& Workspace::GetOptions() const { return options_; }

const Model& Workspace::GetModel() const { return model_; }

const cv::Mat& Workspace::GetSrcmap(const int image_idx) {
	cv::Mat tmpimg = cv::Mat(cv::Size(1024, 1024), CV_8UC3);
	return tmpimg;
}

// const Bitmap& Workspace::GetBitmap(const int image_idx) {
//   auto& cached_image = cache_.GetMutable(image_idx);
//   if (!cached_image.bitmap) {
//     cached_image.bitmap.reset(new Bitmap());
//     cached_image.bitmap->Read(GetBitmapPath(image_idx), options_.image_as_rgb);
//     if (options_.max_image_size > 0) {
//       cached_image.bitmap->Rescale(model_.images.at(image_idx).GetWidth(),
//                                    model_.images.at(image_idx).GetHeight());
//     }
//     cached_image.num_bytes += cached_image.bitmap->NumBytes();
//     cache_.UpdateNumBytes(image_idx);
//   }
//   return *cached_image.bitmap;
// }

const DepthMap& Workspace::GetDepthMap(const int image_idx) {
	auto& cached_image = cache_.GetMutable(image_idx);
	if (!cached_image.depth_map) {
		cached_image.depth_map.reset(new DepthMap());
		cached_image.depth_map->Read(GetDepthMapPath(image_idx));
		if (options_.max_image_size > 0) {
			cached_image.depth_map->Downsize(model_.images.at(image_idx).getWidth(),
											 model_.images.at(image_idx).getHeight());
    	}
		cached_image.num_bytes += cached_image.depth_map->getNumBytes();
		cache_.UpdateNumBytes(image_idx);
	}
	return *cached_image.depth_map;
}

const NormalMap& Workspace::GetNormalMap(const int image_idx) {
	auto& cached_image = cache_.GetMutable(image_idx);
	if (!cached_image.normal_map) {
		cached_image.normal_map.reset(new NormalMap());
		cached_image.normal_map->Read(GetNormalMapPath(image_idx));
		if (options_.max_image_size > 0) {
			cached_image.normal_map->Downsize(
				model_.images.at(image_idx).getWidth(),
				model_.images.at(image_idx).getHeight());
		}
		cached_image.num_bytes += cached_image.normal_map->getNumBytes();
		cache_.UpdateNumBytes(image_idx);
	}
	return *cached_image.normal_map;
}

// std::string Workspace::GetBitmapPath(const int image_idx) const {
// 	return model_.images.at(image_idx).GetPath();
// }

std::string Workspace::GetSrcmapPath(const int image_idx) const {
	return model_.images.at(image_idx).GetPath();
}

std::string Workspace::GetDepthMapPath(const int image_idx) const {
	return depth_map_path_ + GetFileName(image_idx);
}

std::string Workspace::GetNormalMapPath(const int image_idx) const {
	return normal_map_path_ + GetFileName(image_idx);
}

// bool Workspace::HasBitmap(const int image_idx) const {
//   return ExistsFile(GetBitmapPath(image_idx));
// }

bool Workspace::HasDepthMap(const int image_idx) const {
	if (is_file_exist(GetDepthMapPath(image_idx).c_str()) == 0) {
		return true;
	}
	return false;
}

bool Workspace::HasNormalMap(const int image_idx) const {
	if (is_file_exist(GetNormalMapPath(image_idx).c_str()) == 0) {
		return true;
	}
	return false;
}

std::string Workspace::GetFileName(const int image_idx) const {
	const auto& image_name = model_.GetImageName(image_idx);
	char stmp[100];
	sprintf(stmp, "%s.%s.bin", image_name.c_str(), options_.input_type.c_str());
	return std::string(stmp);
}

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
	std::unique_ptr<PatchMatchCuda> patch_match_cuda_;
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

class PatchMatchController : public Thread {
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

	std::unique_ptr<ThreadPool> thread_pool_;
	std::mutex workspace_mutex_;
	std::unique_ptr<Workspace> workspace_;
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