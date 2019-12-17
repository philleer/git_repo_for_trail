#include "mvs/patch_match.h"

// #include "mvs/patch_match_cuda.h"
#include <sstream>
#include <cstdlib>	// exit, EXIT_SUCCESS, EXIT_FAILURE
#include <cstdio>

void PatchMatchOptions::Print() const {
	std::cout << "\nPatchMatchOptions\n";
	std::cout << std::string(15, '-') << std::endl;
	std::cout << "max_image_size: " << max_image_size << std::endl;
	std::cout << "gpu_index: " << gpu_index << std::endl;
	std::cout << "depth_min: " << depth_min
			<< ", depth_max: " << depth_max << std::endl;
	std::cout << "window_radius: " << window_radius << std::endl;
	std::cout << "window_step: " << window_step << std::endl;
	std::cout << "sigma_spatial: " << sigma_spatial << std::endl;
	std::cout << "sigma_color: " << sigma_color << std::endl;
	std::cout << "num_samples: " << num_samples << std::endl;
	std::cout << "ncc_sigma: " << ncc_sigma << std::endl;
	std::cout << "min_triangulation_angle: " << min_triangulation_angle << std::endl;
	std::cout << "incident_angle_sigma: " << incident_angle_sigma << std::endl;
	std::cout << "num_iterations: " << num_iterations << std::endl;
	std::cout << "geom_consistency: " << geom_consistency << std::endl;
	std::cout << "geom_consistency_regularizer: " << geom_consistency_regularizer << std::endl;
	std::cout << "filter: " << filter << std::endl;
	std::cout << "filter_min_ncc: " << filter_min_ncc << std::endl;
	std::cout << "filter_min_num_consistent: " << filter_min_num_consistent << std::endl;
	std::cout << "filter_min_triangulation_angle: " << filter_min_triangulation_angle << std::endl;
	std::cout << "filter_geom_consistency_max_cost: " << filter_geom_consistency_max_cost << std::endl;
	std::cout << "write_consistency_graph: " << write_consistency_graph << std::endl;
}

PatchMatch::PatchMatch(const PatchMatchOptions& options, const Problem& problem):
	options_(options), problem_(problem) {}

PatchMatch::~PatchMatch() {}

void PatchMatch::Problem::Print() const {
	printf("PatchMatch::Problem\n");
	printf("option : %d\n", ref_image_idx);
	printf("src_image_idxs: ");
	if (!src_image_idxs.empty()) {
		for (size_t i = 0; i < src_image_idxs.size()-1; ++i) {
			printf("%d ", src_image_idxs[i]);
		}
		printf("%d\n", src_image_idxs.back());
	} else {
		printf("\n");
	}
}

void PatchMatch::Check() const {
	if (!options_.Check()) {
		std::cerr << "PatchMatchOptions failed!\n";
		exit(EXIT_FAILURE);
	}

	if (options_.gpu_index.empty()) {
		std::cerr << "gpu_index list empty!\n";
		exit(EXIT_FAILURE);
	}
	std::stringstream ss;
	ss.str(options_.gpu_index);
	size_t index_nums = this->char_in_string(options_.gpu_index, ',');
	std::vector<int> gpu_indices(index_nums+1, 0);
	for (int i = 0; i < index_nums; ++i) {
		ss >> gpu_indices[i];
	}
	if (gpu_indices.size() != 1) {
		std::cerr << "gpu index number greater than 1.\n";
		exit(EXIT_FAILURE);
	}
	if (gpu_indices[0] < -1) {
		std::cerr << "gpu index set unproperly!\n";
		exit(EXIT_FAILURE);
	}
	if (problem_.images == nullptr) {
		std::cerr << "no images given!\n";
		exit(EXIT_FAILURE);
	}

	if (options_.geom_consistency) {
		if (problem_.depth_maps == nullptr) {
			std::cerr << "depth_maps empty!\n";
			exit(EXIT_FAILURE);
		}
		if (problem_.normal_maps == nullptr) {
			std::cerr << "normal_maps empty!\n";
			exit(EXIT_FAILURE);
		}

		if (problem_.depth_maps->size() != problem_.images->size()) {
			std::cerr << "number of depth_maps not equals to images!\n";
			exit(EXIT_FAILURE);
		}
		if (problem_.normal_maps->size() != problem_.images->size()) {
			std::cerr << "number of normal_maps not equals to images!\n";
			exit(EXIT_FAILURE);
		}
	}

	if (problem_.src_image_idxs.size() <= 0) {
		std::cerr << "numbers of src_image_idxs less than 0!\n";
		exit(EXIT_FAILURE);
	}

	std::set<int> unique_image_idxs(problem_.src_image_idxs.begin(),
									problem_.src_image_idxs.end());
	unique_image_idxs.insert(problem_.ref_image_idx);
	if (problem_.src_image_idxs.size()+1 != unique_image_idxs.size()) {
		std::cerr << "src images are not all unique!\n";
		exit(EXIT_FAILURE);
	}

	for (const int image_idx : unique_image_idxs) {
		if (image_idx < 0) {
			std::cerr << "image_idx less than 0! "
			<< image_idx << std::endl;
			exit(EXIT_FAILURE);
		}
		if (image_idx >= problem_.images->size()) {
			std::cerr << "image_idx should less than images->size! "
				<< image_idx << std::endl;
			exit(EXIT_FAILURE);
		}

		const Image& image = problem_.images->at(image_idx);
		if (image.getWidth() <= 0 || image.getHeight() <= 0) {
			std::cerr << "image size not as required! "
				<< image_idx << std::endl;
			exit(EXIT_FAILURE);
		}

		if (image.getBitmap().channels() != 1) {
			std::cerr << "image is not in grayscale! "
				<< image_idx << std::endl;
			exit(EXIT_FAILURE);
		}
		if (image.getWidth() != image.getBitmap().cols ||
			image.getHeight() != image.getBitmap().rows) {
			std::cerr << "image size is not as required! "
				<< image_idx << std::endl;
			exit(EXIT_FAILURE);
		}

		// Make sure, the calibration matrix only contains fx, fy, cx, cy.
		if (std::abs(image.GetK()[1]-0.0f) > 1e-6f ||
			std::abs(image.GetK()[3]-0.0f) > 1e-6f ||
			std::abs(image.GetK()[6]-0.0f) > 1e-6f ||
			std::abs(image.GetK()[7]-0.0f) > 1e-6f ||
			std::abs(image.GetK()[8]-1.0f) > 1e-6f)
		{
			std::cerr << "The calibration matrix not formatted! "
				<< image_idx << std::endl;
			exit(EXIT_FAILURE);
		}

		if (options_.geom_consistency) {
			if (image_idx >= problem_.depth_maps->size()) {
				std::cerr << "image_idx cannot >= depth_maps.size()! "
					<< image_idx << std::endl;
				exit(EXIT_FAILURE);
			}

			const DepthMap& depth_map = problem_.depth_maps->at(image_idx);
			if (image.getWidth() != depth_map.getWidth()) {
				std::cerr << "width of image and depth map not equal! "
					<< image_idx << std::endl;
				exit(EXIT_FAILURE);
			}

			if (image.getHeight() != depth_map.getHeight()) {
				std::cerr << "height of image and depth map not equal! "
					<< image_idx << std::endl;
				exit(EXIT_FAILURE);
			}
		}
	}

	// 1. operator[]不做边界检查，越界了也会返回一个引用，当然这个引用是错误的，
	// 如何不小心调用了这个引用对象的方法，会直接导致应用退出
	// 2. at会做边界检查，如果越界，会抛出异常，应用可以try catch这个异常，之后还能继续运行
	if (options_.geom_consistency) {
		const Image &ref_image = problem_.images->at(problem_.ref_image_idx);
		const NormalMap &ref_normal_map = problem_.normal_maps->at(problem_.ref_image_idx);
		if (ref_image.getWidth() != ref_normal_map.getWidth()) {
			std::cerr << "width of ref image and normal map not equal!\n";
			exit(EXIT_FAILURE);
		}
		if (ref_image.getHeight() != ref_normal_map.getHeight()) {
			std::cerr << "height of ref image and normal map not equal!\n";
			exit(EXIT_FAILURE);
		}
	}
}

void PatchMatch::Run() {
	std::cout << "\nPatchMatch::Run\n";
	std::cout << std::string(15, '-') << std::endl;
	Check();

	patch_match_cuda_.reset(new gpu::PatchMatchCuda(options_, problem_));
	patch_match_cuda_->Run();
}

DepthMap PatchMatch::getDepthMap() const {
	return patch_match_cuda_->getDepthMap();
}

NormalMap PatchMatch::getNormalMap() const {
	return patch_match_cuda_->getNormalMap(); 
}

Mat<float> PatchMatch::getSelProbMap() const {
	return patch_match_cuda_->getSelProbMap();
}

ConsistencyGraph PatchMatch::getConsistencyGraph() const {
	const auto &ref_image = problem_.images->at(problem_.ref_image_idx);
	return ConsistencyGraph(ref_image.getWidth(), ref_image.getHeight(),
							patch_match_cuda_->getConsistentImageIdxs());
}

PatchMatchController::PatchMatchController(const PatchMatchOptions& options,
	const std::string& workspace_path,
	const std::string& workspace_format,
	const std::string& pmvs_option_name):
	options_(options),
	workspace_path_(workspace_path),
	workspace_format_(workspace_format),
	pmvs_option_name_(pmvs_option_name)
{
	// std::vector<int> gpu_indices = ;
	std::stringstream ss;
	ss.str(options_.gpu_index);
	size_t index_nums = this->char_in_string(options_.gpu_index, ',');
	std::vector<int> gpu_indices(index_nums+1, 0);
	for (int i = 0; i < index_nums; ++i) {
		ss >> gpu_indices[i];
	}
}

void PatchMatchController::Run() {
	ReadWorkspace();
	ReadProblems();
	ReadGpuIndices();

	if (options_.geom_consistency) {
		auto photometric_options = options_;
		photometric_options.geom_consistency = false;
		photometric_options.filter = false;

		for (size_t problem_idx = 0; problem_idx < problem_idx.size(); ++problem_idx) {
			//
		}
	}
}

void PatchMatchController::ReadWorkspace() {
	std::cout << "Reading workspace..." << std::endl;
}

void PatchMatchController::ReadProblems() {
	std::cout << "Reading configuration..." << std::endl;
	problems_.clear();
}

void PatchMatchController::ReadGpuIndices() {

}

void PatchMatchController::ProcessProblem(const PatchMatchOptions& options,
										  const size_t problem_idx)
{
	if (IsStopped()) return;

	auto patch_match_options = options;
	if (patch_match_options.depth_min < 0 || patch_match_options.depth_max < 0) {
		patch_match_options.depth_min = depth_ranges_.at(
											problem.ref_image_idx).first;
		patch_match_options.depth_max = depth_ranges_.at(
											problem.ref_image_idx).second;
		if (patch_match_options.depth_min <= 0 || patch_match_options.depth_max <= 0) {
			printf(" - You must manually set the minimum and maximum depth, since no \
	       			sparse model is provided in the workspace.\n");
		}	    
	}

	PatchMatch patch_match(patch_match_options, problem);
	patch_match.Run();
	printf("\nWriting %s output for %s\n", output_type.c_str(), image_name.c_str());
	if (options.write_consistency_graph) {
		// patch_match.getConsistencyGraph().Write(consistency_graph_path);
	}
}