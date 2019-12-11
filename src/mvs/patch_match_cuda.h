#ifndef SRC_PATCH_MATCH_CUDA_H
#define SRC_PATCH_MATCH_CUDA_H

#include <iostream>
#include <vector>
#include "mvs/image.h"
#include "mvs/depth_map.h"
#include "mvs/normal_map.h"
#include "mvs/patch_match.h"

class PatchMatchCuda {
public:
	PatchMatchCuda(const PatchMatchOptions &options,
				   const PatchMatch::Problem &problem);
	~PatchMatchCuda();

	void Run();

	DepthMap getDepthMap() const;
	NormalMap getNormalMap() const;
	Mat<float> getSelProbMap() const;
	std::vector<int> getConsistencyImageIdxs() const;

private:
	template <int kWindowSize, int kWindowStep>
	void RunWithWindowSizeAndStep();

	void ComputeCudaConfig();

	void InitRefImage();
	void InitSourceImages();
	void InitTransforms();
	void InitWorkspaceMemory();

	// Rotate reference image by 90 degrees in counter-clockwise direction.
	void Rotate();

	const PatchMatchOptions options_;
	const PatchMatch::Problem problem_;

	size_t ref_width_;
	size_t ref_height_;

	int rotation_in_half_pi_;

	std::unique_ptr<CudaArrayWrapper<float>> poses_devices[4];

	std::unique_ptr<GpuMat<float>> global_workspace_;
};

#endif