#ifndef SRC_MVS_MODEL_H
#define SRC_MVS_MODEL_H

#include <iostream>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include "mvs/depth_map.h"
#include "mvs/image.h"
#include "mvs/normal_map.h"

template <typename T>
T Percentile(const std::vector<T>& elems, const double p) {
    if (elems.empty()) {
        std::cerr << "elems empty!" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (p < 0 || p > 100) {
        std::cerr << "Invalid percentile!" << std::endl;
        exit(EXIT_FAILURE);
    }

    const int idx = static_cast<int>(std::round(p / 100 * (elems.size() - 1)));
    const size_t percentile_idx = std::max(0, std::min(static_cast<int>(elems.size()-1), idx));

    std::vector<T> ordered_elems = elems;
    std::nth_element(ordered_elems.begin(),
                     ordered_elems.begin()+percentile_idx, ordered_elems.end());

    return ordered_elems.at(percentile_idx);
}

// Simple sparse model class.
struct Model {
    struct Point {
        float x = 0;
        float y = 0;
        float z = 0;
        std::vector<int> track;
    };

    // Read the model from different data formats.
    void Read(const std::string& path, const std::string& format);
    void ReadFromCOLMAP(const std::string& path);
    void ReadFromPMVS(const std::string& path);

    // Get the image index for the given image name.
    int GetImageIdx(const std::string& name) const;
    std::string GetImageName(const int image_idx) const;

    // For each image, determine the maximally overlapping images, sorted based on
    // the number of shared points subject to a minimum robust average
    // triangulation angle of the points.
    std::vector<std::vector<int>> GetMaxOverlappingImages(
        const size_t num_images, const double min_triangulation_angle) const;

    // Get the overlapping images defined in the vis.dat file.
    const std::vector<std::vector<int>>& GetMaxOverlappingImagesFromPMVS() const;

    // Compute the robust minimum and maximum depths from the sparse point cloud.
    std::vector<std::pair<float, float>> ComputeDepthRanges() const;

    // Compute the number of shared points between all overlapping images.
    std::vector<std::map<int, int>> ComputeSharedPoints() const;

    // Compute the median triangulation angles between all overlapping images.
    std::vector<std::map<int, float>> ComputeTriangulationAngles(
        const float percentile = 50) const;

    // Note that in case the data is read from a COLMAP reconstruction, the index
    // of an image or point does not correspond to its original identifier in the
    // reconstruction, but it corresponds to the position in the
    // images.bin/points3D.bin files. This is mainly done for more efficient
    // access to the data, which is required during the stereo fusion stage.
    std::vector<Image> images;
    std::vector<Point> points;

private:
    bool ReadFromBundlerPMVS(const std::string& path);
    bool ReadFromRawPMVS(const std::string& path);

    std::vector<std::string> image_names_;
    std::unordered_map<std::string, int> image_name_to_idx_;
    std::vector<std::vector<int>> pmvs_vis_dat_;
};

#endif  // SRC_MVS_MODEL_H
