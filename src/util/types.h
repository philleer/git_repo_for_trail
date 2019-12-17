#ifndef SRC_UTIL_TYPES_H
#define SRC_UTIL_TYPES_H

#include <Eigen/Core>

typedef unsigned int uint32;

namespace Eigen {
	
typedef Eigen::Matrix<float, 3, 4> Matrix3x4f;
typedef Eigen::Matrix<double, 3, 4> Matrix3x4d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

}

#endif // SRC_UTIL_TYPES_H
