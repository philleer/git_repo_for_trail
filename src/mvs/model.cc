#include "mvs/model.h"

// #include "base/camera_models.h"
// #include "base/pose.h"
// #include "base/projection.h"
// #include "base/reconstruction.h"
// #include "base/triangulation.h"

#include <algorithm>
#include <stdexcept>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "util/util_sys.h"

void Model::Read(const std::string& path, const std::string& format) {
    std::string format_lower_case = format;
    std::transform(format_lower_case.begin(), format_lower_case.end(),
                   format_lower_case.begin(), ::tolower);
    if (format_lower_case == "colmap") {
        ReadFromCOLMAP(path);
    } else if (format_lower_case == "pmvs") {
        ReadFromPMVS(path);
    } else {
        std::cerr << "Invalid input format" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Model::ReadFromCOLMAP(const std::string& path) {
    // Reconstruction reconstruction;
    // reconstruction.Read(JoinPaths(path, "sparse"));

    // images.reserve(reconstruction.NumRegImages());
    // std::unordered_map<image_t, size_t> image_id_to_idx;
    // for (size_t i = 0; i < reconstruction.NumRegImages(); ++i) {
    //     const auto image_id = reconstruction.RegImageIds()[i];
    //     const auto& image = reconstruction.Image(image_id);
    //     const auto& camera = reconstruction.Camera(image.CameraId());

    //     CHECK_EQ(camera.ModelId(), PinholeCameraModel::model_id);

    //     const std::string image_path = JoinPaths(path, "images", image.Name());
    //     const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K =
    //         camera.CalibrationMatrix().cast<float>();
    //     const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R =
    //         QuaternionToRotationMatrix(image.Qvec()).cast<float>();
    //     const Eigen::Vector3f T = image.Tvec().cast<float>();

    //     images.emplace_back(image_path, camera.Width(), camera.Height(), K.data(),
    //                         R.data(), T.data());
    //     image_id_to_idx.emplace(image_id, i);
    //     image_names_.push_back(image.Name());
    //     image_name_to_idx_.emplace(image.Name(), i);
    // }

    // points.reserve(reconstruction.NumPoints3D());
    // for (const auto& point3D : reconstruction.Points3D()) {
    //     Point point;
    //     point.x = point3D.second.X();
    //     point.y = point3D.second.Y();
    //     point.z = point3D.second.Z();
    //     point.track.reserve(point3D.second.Track().Length());
    //     for (const auto& track_el : point3D.second.Track().Elements()) {
    //       point.track.push_back(image_id_to_idx.at(track_el.image_id));
    //     }
    //     points.push_back(point);
    // }
}

void Model::ReadFromPMVS(const std::string& path) {
    if (ReadFromBundlerPMVS(path)) return;
    if (ReadFromRawPMVS(path)) return;    
    std::cerr << "Invalid PMVS format" << std::endl;
    exit(EXIT_FAILURE);
}

int Model::GetImageIdx(const std::string& name) const {
    if (image_name_to_idx_.count(name) <= 0) {
        std::cerr << "Image with name `" << name
                << "` does not exist" << std::endl;
        exit(EXIT_FAILURE);
    }

    return image_name_to_idx_.at(name);
}

std::string Model::GetImageName(const int image_idx) const {
    if (image_idx < 0 || image_idx >= image_names_.size()) {
        std::cerr << "image index invalid!" << std::endl;
        exit(EXIT_FAILURE);
    }
    return image_names_.at(image_idx);
}

std::vector<std::vector<int>> Model::GetMaxOverlappingImages(
    const size_t num_images,
    const double min_triangulation_angle) const
{
    std::vector<std::vector<int>> overlapping_images(images.size());
    // const float min_triangulation_angle_rad = DegToRad(min_triangulation_angle);
    const auto shared_num_points = ComputeSharedPoints();

    const float kTriangulationAnglePercentile = 75;
    // const auto triangulation_angles =
    //   ComputeTriangulationAngles(kTriangulationAnglePercentile);

    for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
        const auto& shared_images = shared_num_points.at(image_idx);
        // const auto& overlapping_triangulation_angles =
        //     triangulation_angles.at(image_idx);

        std::vector<std::pair<int, int>> ordered_images;
        ordered_images.reserve(shared_images.size());
        for (const auto& image : shared_images) {
            // if (overlapping_triangulation_angles.at(image.first) >=
            //   min_triangulation_angle_rad) {
            //     ordered_images.emplace_back(image.first, image.second);
            // }
        }

        const size_t eff_num_images = std::min(ordered_images.size(), num_images);
        if (eff_num_images < shared_images.size()) {
            std::partial_sort(ordered_images.begin(),
                              ordered_images.begin() + eff_num_images,
                              ordered_images.end(),
                              [](const std::pair<int, int> image1,
                                 const std::pair<int, int> image2) {
                                 return image1.second > image2.second;
                              });
        } else {
            std::sort(ordered_images.begin(), ordered_images.end(),
                      [](const std::pair<int, int> image1,
                         const std::pair<int, int> image2) {
                         return image1.second > image2.second;
                     });
        }

        overlapping_images[image_idx].reserve(eff_num_images);
        for (size_t i = 0; i < eff_num_images; ++i) {
            overlapping_images[image_idx].push_back(ordered_images[i].first);
        }
    }

    return overlapping_images;
}

const std::vector<std::vector<int>>& Model::GetMaxOverlappingImagesFromPMVS()
    const
{
    return pmvs_vis_dat_;
}

std::vector<std::pair<float, float>> Model::ComputeDepthRanges() const {
    std::vector<std::vector<float>> depths(images.size());
    for (const auto& point : points) {
        const Eigen::Vector3f X(point.x, point.y, point.z);
        for (const auto& image_idx : point.track) {
            const auto& image = images.at(image_idx);
            const float depth =
                Eigen::Map<const Eigen::Vector3f>(&image.GetR()[6]).dot(X) +
                image.GetT()[2];
            if (depth > 0) {
                depths[image_idx].push_back(depth);
            }
        }
    }

    std::vector<std::pair<float, float>> depth_ranges(depths.size());
    for (size_t image_idx = 0; image_idx < depth_ranges.size(); ++image_idx) {
        auto& depth_range = depth_ranges[image_idx];
        auto& image_depths = depths[image_idx];

        if (image_depths.empty()) {
            depth_range.first = -1.0f;
            depth_range.second = -1.0f;
            continue;
        }

        std::sort(image_depths.begin(), image_depths.end());
        const float kMinPercentile = 0.01f;
        const float kMaxPercentile = 0.99f;
        depth_range.first = image_depths[image_depths.size() * kMinPercentile];
        depth_range.second = image_depths[image_depths.size() * kMaxPercentile];

        const float kStretchRatio = 0.25f;
        depth_range.first *= (1.0f - kStretchRatio);
        depth_range.second *= (1.0f + kStretchRatio);
    }

    return depth_ranges;
}

std::vector<std::map<int, int>> Model::ComputeSharedPoints() const {
    std::vector<std::map<int, int>> shared_points(images.size());
    for (const auto& point : points) {
        for (size_t i = 0; i < point.track.size(); ++i) {
            const int image_idx1 = point.track[i];
            for (size_t j = 0; j < i; ++j) {
                const int image_idx2 = point.track[j];
                if (image_idx1 != image_idx2) {
                    shared_points.at(image_idx1)[image_idx2] += 1;
                    shared_points.at(image_idx2)[image_idx1] += 1;
                }
            }
        }
    }
    return shared_points;
}

std::vector<std::map<int, float>> Model::ComputeTriangulationAngles(
    const float percentile) const
{
    // std::vector<Eigen::Vector3d> proj_centers(images.size());
    // for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
    //     const auto& image = images[image_idx];
    //     Eigen::Vector3f C;
    //     ComputeProjectionCenter(image.GetR(), image.GetT(), C.data());
    //     proj_centers[image_idx] = C.cast<double>();
    // }

    std::vector<std::map<int, std::vector<float>>> all_triangulation_angles(images.size());
    for (const auto& point : points) {
        for (size_t i = 0; i < point.track.size(); ++i) {
            const int image_idx1 = point.track[i];
            for (size_t j = 0; j < i; ++j) {
                const int image_idx2 = point.track[j];
                if (image_idx1 != image_idx2) {
                    // const float angle = CalculateTriangulationAngle(
                    //     proj_centers.at(image_idx1), proj_centers.at(image_idx2),
                    //     Eigen::Vector3d(point.x, point.y, point.z));
                    const float angle = 0.0f;
                    all_triangulation_angles.at(image_idx1)[image_idx2].push_back(angle);
                    all_triangulation_angles.at(image_idx2)[image_idx1].push_back(angle);
                }
            }
        }
    }

    std::vector<std::map<int, float>> triangulation_angles(images.size());
    for (size_t image_idx = 0; image_idx < all_triangulation_angles.size(); ++image_idx) {
        const auto& overlapping_images = all_triangulation_angles[image_idx];
        for (const auto& image : overlapping_images) {
            triangulation_angles[image_idx].emplace(
                image.first, Percentile(image.second, percentile));
        }
    }

    return triangulation_angles;
}

bool Model::ReadFromBundlerPMVS(const std::string& path) {
    const std::string bundle_file_path = path+"/bundle.rd.out";
    if (is_dir_exist(bundle_file_path.c_str()) == -1) return false;

    std::ifstream file(bundle_file_path);
    if (!file.is_open()) {
        std::cerr << bundle_file_path << " open failed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Header line.
    std::string header;
    int num_images, num_points;
    std::getline(file, header);
    file >> num_images >> num_points;

    images.reserve(num_images);
    for (int image_idx = 0; image_idx < num_images; ++image_idx) {
        char imgpath[100];
        sprintf(imgpath, "%08d.jpg", image_idx);
        const std::string image_name = std::string(imgpath);
        const std::string image_path = path+ "/visualize/" + image_name;

        float K[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        file >> K[0];
        K[4] = K[0];

        cv::Mat srcmap = cv::imread(image_path);
        if (!srcmap.data) {
            std::cerr << "image read failed!" << std::endl;
            exit(EXIT_FAILURE);
        }
        K[2] = srcmap.cols / 2.0f;
        K[5] = srcmap.rows / 2.0f;
        // Bitmap bitmap;
        // CHECK(bitmap.Read(image_path));
        // K[2] = bitmap.Width() / 2.0f;
        // K[5] = bitmap.Height() / 2.0f;

        float k1, k2;
        file >> k1 >> k2;
        if (k1 != 0.0f || k2 != 0.0f) {
            std::cerr << "k1 or k2 invalid!" << std::endl;
            exit(EXIT_FAILURE);
        }

        float R[9];
        for (size_t i = 0; i < 9; ++i) {
            file >> R[i];
        }
        for (size_t i = 3; i < 9; ++i) {
            R[i] = -R[i];
        }

        float T[3];
        file >> T[0] >> T[1] >> T[2];
        T[1] = -T[1];
        T[2] = -T[2];

        // images.emplace_back(image_path, bitmap.Width(), bitmap.Height(), K, R, T);
        images.emplace_back(image_path, srcmap.cols, srcmap.rows, K, R, T);
        image_names_.push_back(image_name);
        image_name_to_idx_.emplace(image_name, image_idx);
    }

    points.resize(num_points);
    for (int point_id = 0; point_id < num_points; ++point_id) {
        auto& point = points[point_id];
        file >> point.x >> point.y >> point.z;

        int color[3];
        file >> color[0] >> color[1] >> color[2];

        int track_len;
        file >> track_len;
        point.track.resize(track_len);

        for (int i = 0; i < track_len; ++i) {
            int feature_idx;
            float imx, imy;
            file >> point.track[i] >> feature_idx >> imx >> imy;
            if (point.track[i] >= images.size()) {
                std::cerr << "index invalid!" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    return true;
}

bool Model::ReadFromRawPMVS(const std::string& path) {
    const std::string vis_dat_path = path+"/vis.dat";
    if (is_dir_exist(vis_dat_path.c_str()) == -1) return false;

    for (int image_idx = 0;; ++image_idx) {
        char imgpath[100];
        sprintf(imgpath, "%08d.jpg", image_idx);
        const std::string image_name = std::string(imgpath);
        const std::string image_path = path+ "/visualize/" + image_name;

        if (is_file_exist(image_path.c_str()) == -1) break;

        cv::Mat srcmap = cv::imread(image_path);
        if (!srcmap.data) {
            std::cerr << "image load failed!" << std::endl;
            exit(EXIT_FAILURE);
        }
        // Bitmap bitmap;
        // CHECK(bitmap.Read(image_path));

        sprintf(imgpath, "%08d.txt", image_idx);
        const std::string proj_matrix_path = path + "/txt/" + std::string(imgpath);

        std::ifstream proj_matrix_file(proj_matrix_path);
        if (!proj_matrix_file.is_open()) {
            std::cerr << proj_matrix_path << " proj matrix file open failed!" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string contour;
        proj_matrix_file >> contour;
        if (contour != "CONTOUR") {
            std::cerr << "contour invalid!" << std::endl;
            exit(EXIT_FAILURE);
        }

        Eigen::Matrix<double, 3, 4> P;
        for (int i = 0; i < 3; ++i) {
            proj_matrix_file >> P(i, 0) >> P(i, 1) >> P(i, 2) >> P(i, 3);
        }

        Eigen::Matrix3d K;
        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        // DecomposeProjectionMatrix(P, &K, &R, &T);

        // The COLMAP patch match algorithm requires that there is no skew.
        K(0, 1) = 0.0f;
        K(1, 0) = 0.0f;
        K(2, 0) = 0.0f;
        K(2, 1) = 0.0f;
        K(2, 2) = 1.0f;

        const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K_float = K.cast<float>();
        const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_float = R.cast<float>();
        const Eigen::Vector3f T_float = T.cast<float>();

        images.emplace_back(image_path, srcmap.cols, srcmap.rows,
                            K_float.data(), R_float.data(), T_float.data());
        // images.emplace_back(image_path, bitmap.Width(), bitmap.Height(),
        //                     K_float.data(), R_float.data(), T_float.data());
        image_names_.push_back(image_name);
        image_name_to_idx_.emplace(image_name, image_idx);
    }

    std::ifstream vis_dat_file(vis_dat_path);
    if (!vis_dat_file.is_open()) {
        std::cerr << vis_dat_path << " path is invalid!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string visdata;
    int num_images;
    vis_dat_file >> visdata;
    if (visdata != "VISDATA") {
        std::cerr << visdata << " invalid!" << std::endl;
        exit(EXIT_FAILURE);
    }
    vis_dat_file >> num_images;
    if (num_images < 0 || num_images >= images.size()) {
        std::cerr << num_images << " is invalid!" << std::endl;
        exit(EXIT_FAILURE);
    }

    pmvs_vis_dat_.resize(num_images);
    for (int i = 0; i < num_images; ++i) {
        int image_idx;
        vis_dat_file >> image_idx;
        if (image_idx < 0 || image_idx >= num_images) {
            std::cerr << image_idx << " is invalid!" << std::endl;
            exit(EXIT_FAILURE);
        }

        int num_visible_images;
        vis_dat_file >> num_visible_images;

        auto& visible_image_idxs = pmvs_vis_dat_[image_idx];
        visible_image_idxs.reserve(num_visible_images);

        for (int j = 0; j < num_visible_images; ++j) {
            int visible_image_idx;
            vis_dat_file >> visible_image_idx;
            if (visible_image_idx < 0 || visible_image_idx >= num_images) {
                std::cerr << visible_image_idx << " is invalid!" << std::endl;
                exit(EXIT_FAILURE);
            }
            if (visible_image_idx != image_idx) {
                visible_image_idxs.push_back(visible_image_idx);
            }
        }
    }

    return true;
}
