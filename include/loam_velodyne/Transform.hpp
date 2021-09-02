
/* Transform.hpp */

#ifndef LOAM_TRANSFORM_HPP
#define LOAM_TRANSFORM_HPP

#include <cmath>
#include <Eigen/Core>

#include "loam_velodyne/Vector3.h"

namespace loam {

/* Create a 3D rotation matrix that rotates a 3D vector around
 * fixed z, x, and y axes, i.e., Ry(ry) Rx(rx) Rz(rz) */
inline Eigen::Matrix3f rotationMatrixZXY(
    const float rx, const float ry, const float rz)
{
    const float cosX = std::cos(rx);
    const float sinX = std::sin(rx);
    const float cosY = std::cos(ry);
    const float sinY = std::sin(ry);
    const float cosZ = std::cos(rz);
    const float sinZ = std::sin(rz);

    Eigen::Matrix3f rotationMat;
    rotationMat(0, 0) =  cosY * cosZ + sinY * sinX * sinZ;
    rotationMat(0, 1) = -cosY * sinZ + sinY * sinX * cosZ;
    rotationMat(0, 2) =  sinY * cosX;
    rotationMat(1, 0) =  cosX * sinZ;
    rotationMat(1, 1) =  cosX * cosZ;
    rotationMat(1, 2) = -sinX;
    rotationMat(2, 0) = -sinY * cosZ + cosY * sinX * sinZ;
    rotationMat(2, 1) =  sinY * sinZ + cosY * sinX * cosZ;
    rotationMat(2, 2) =  cosY * cosX;

    return rotationMat;
}

/* Create a 3D rotation matrix that rotates a 3D vector around
 * fixed y, x, and z axes, i.e., Rz(rz) Rx(rx) Ry(ry) */
inline Eigen::Matrix3f rotationMatrixYXZ(
    const float rx, const float ry, const float rz)
{
    const float cosX = std::cos(rx);
    const float sinX = std::sin(rx);
    const float cosY = std::cos(ry);
    const float sinY = std::sin(ry);
    const float cosZ = std::cos(rz);
    const float sinZ = std::sin(rz);

    Eigen::Matrix3f rotationMat;
    rotationMat(0, 0) =  cosY * cosZ - sinY * sinX * sinZ;
    rotationMat(0, 1) = -cosX * sinZ;
    rotationMat(0, 2) =  sinY * cosZ + cosY * sinX * sinZ;
    rotationMat(1, 0) =  cosY * sinZ + sinY * sinX * cosZ;
    rotationMat(1, 1) =  cosX * cosZ;
    rotationMat(1, 2) =  sinY * sinZ - cosY * sinX * cosZ;
    rotationMat(2, 0) = -sinY * cosX;
    rotationMat(2, 1) =  sinX;
    rotationMat(2, 2) =  cosY * cosX;

    return rotationMat;
}

/* Get Euler angles from a 3D rotation matrix that rotates a 3D vector
 * around fixed z, x, and y axes, i.e., get Euler angles rx, ry, and rz
 * from a rotation matrix Ry(ry) Rx(rx) Rz(rz) */
inline void eulerAnglesFromRotationZXY(
    const Eigen::Matrix3f& rotationMat, float& rx, float& ry, float& rz)
{
    rx = -std::asin(rotationMat(1, 2));
    const float cosX = std::cos(rx);

    ry = std::atan2(rotationMat(0, 2) / cosX, rotationMat(2, 2) / cosX);
    rz = std::atan2(rotationMat(1, 0) / cosX, rotationMat(1, 1) / cosX);
}

} // namespace loam

#endif // LOAM_TRANSFORM_HPP
