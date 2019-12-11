#include "mvs/patch_match_cuda.h"

PatchMatchCuda::PatchMatchCuda(const PatchMatchOptions& options,
							   const PatchMatch::Problem& problem):
	options_(options), problem_(problem), ref_width_(0), ref_height_(0),
	rotation_in_half_pi_(0)
{
	InitRefImage();
	InitSourceImages();
	InitTransforms();
	// /* TODO */
	InitWorkspaceMemory();
}

PatchMatchCuda::~PatchMatchCuda() {
	for (size_t i = 0; i < 4; ++i) {
		poses_device_[i].reset();
	}
}

void PatchMatchCuda::Run() {
	//
}

DepthMap PatchMatchCuda::getDepthMap() const {
	return DepthMap(depth_map_->CopyToMat(), options_.depth_min,
					options_.depth_max);
}

NormalMap PatchMatchCuda::getNormalMap() const {
	return NormalMap(normal_map_->CopyToMat());
}

Mat<float> PatchMatchCuda::getSelProbMap() const {
	return prev_sel_prob_map_->CopyToMat();
}

std::vector<int> PatchMatchCuda::getConsistencyImageIdxs() const {
	const Mat<uint8_t> mask = consistency_mask_->CopyToMat();
	std::vector<int> consistent_image_idxs;
	std::vector<int> pixel_consistent_image_idxs;
	pixel_consistent_image_idxs.reserve(mask.getDepth());

	for (size_t r = 0; r < mask.getHeight(); ++r) {
		for (size_t c = 0; c < mask.getWidth(); ++c) {
			pixel_consistent_image_idxs.clear();
			for (size_t d = 0; d < mask.getDepth(); ++d) {
				if (mask.Get(r, c, d)) {
					pixel_consistent_image_idxs.push_back(problem_.src_image_idxs[d]);
				}
			}
			if (pixel_consistent_image_idxs.size() > 0) {
				consistent_image_idxs.push_back(c);
				consistent_image_idxs.push_back(r);
				consistent_image_idxs.push_back(pixel_consistent_image_idxs.size());
				consistent_image_idxs.insert(consistent_image_idxs.end(),
											 pixel_consistent_image_idxs.begin(),
											 pixel_consistent_image_idxs.end());
			}
		}
	}
	return consistent_image_idxs;
}
