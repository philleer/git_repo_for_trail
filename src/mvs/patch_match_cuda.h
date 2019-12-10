#ifndef SRC_PATCH_MATCH_CUDA_H
#define SRC_PATCH_MATCH_CUDA_H

#include <iostream>
#include <vector>
#include <memory>

#include "mvs/image.h"
#include "mvs/mat.h"
#include "mvs/depth_map.h"
#include "mvs/normal_map.h"
#include "mvs/patch_match.h"

class PatchMatchCuda {
public:
	PatchMatchCuda(const PatchMatchOptions& options,
				   const PatchMatch::Problem& problem);

	~PatchMatchCuda();

	void Run();

	DepthMap getDepthMap() const;
	NormalMap getNormalMap() const;
	Mat<float> getSelProbMap() const;
	std::vector<int> getConsistentImageIdxs() const;

private:
	template<int kWindowSize, int kWindowStep>
	void RunWithWindowSizeAndStep();

	void ComputeCudaConfig();

	void InitRefImage();
	void InitSourceImages();
	void InitTransforms();
	void InitWorkspaceMemory();

	const PatchMatchOptions options_;
	const PatchMatch::Problem problem_;
	size_t ref_width_;
	size_t ref_height_;
	int rotation_in_half_pi_;

	std::unique_ptr<CudaArrayWrapper<float>> pose_device_[4];
}

#endif