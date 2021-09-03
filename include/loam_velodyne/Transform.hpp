
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

/* Create a transpose of a 3D rotation matrix that rotates a 3D vector around
 * fixed y, x, and z axes, which is written as (Rz(rz) Rx(rx) Ry(ry))^T or
 * Ry(-ry) Rx(-rx) Rz(-rz) */
inline Eigen::Matrix3f rotationMatrixYXZT(
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
    rotationMat(0, 1) =  cosY * sinZ + sinY * sinX * cosZ;
    rotationMat(0, 2) = -sinY * cosX;
    rotationMat(1, 0) = -cosX * sinZ;
    rotationMat(1, 1) =  cosX * cosZ;
    rotationMat(1, 2) =  sinX;
    rotationMat(2, 0) =  sinY * cosZ + cosY * sinX * sinZ;
    rotationMat(2, 1) =  sinY * sinZ - cosY * sinX * cosZ;
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

/* Compute a partial derivative of a rotation matrix R = Ry(ry) Rx(rx) Rz(rz)
 * with respect to rx (the rotation angle around X axis), where the rotation
 * matrix rotates a 3D vector around fixed z, x, and y axes */
inline Eigen::Matrix3f partialXFromRotationZXY(
    const float rx, const float ry, const float rz)
{
    const float cosX = std::cos(rx);
    const float sinX = std::sin(rx);
    const float cosY = std::cos(ry);
    const float sinY = std::sin(ry);
    const float cosZ = std::cos(rz);
    const float sinZ = std::sin(rz);

    Eigen::Matrix3f partialXMat;
    partialXMat(0, 0) =  sinY * cosX * sinZ;
    partialXMat(0, 1) =  sinY * cosX * cosZ;
    partialXMat(0, 2) = -sinY * sinX;
    partialXMat(1, 0) = -sinX * sinZ;
    partialXMat(1, 1) = -sinX * cosZ;
    partialXMat(1, 2) = -cosX;
    partialXMat(2, 0) =  cosY * cosX * sinZ;
    partialXMat(2, 1) =  cosY * cosX * cosZ;
    partialXMat(2, 2) = -cosY * sinX;

    return partialXMat;
}

/* Compute a partial derivative of a rotation matrix R = Ry(ry) Rx(rx) Rz(rz)
 * with respect to ry (the rotation angle around Y axis), where the rotation
 * matrix rotates a 3D vector around fixed z, x, and y axes */
inline Eigen::Matrix3f partialYFromRotationZXY(
    const float rx, const float ry, const float rz)
{
    const float cosX = std::cos(rx);
    const float sinX = std::sin(rx);
    const float cosY = std::cos(ry);
    const float sinY = std::sin(ry);
    const float cosZ = std::cos(rz);
    const float sinZ = std::sin(rz);

    Eigen::Matrix3f partialYMat;
    partialYMat(0, 0) = -sinY * cosZ + cosY * sinX * sinZ;
    partialYMat(0, 1) =  sinY * sinZ + cosY * sinX * cosZ;
    partialYMat(0, 2) =  cosY * cosX;
    partialYMat(1, 0) =  0.0f;
    partialYMat(1, 1) =  0.0f;
    partialYMat(1, 2) =  0.0f;
    partialYMat(2, 0) = -cosY * cosZ - sinY * sinX * sinZ;
    partialYMat(2, 1) =  cosY * sinZ - sinY * sinX * cosZ;
    partialYMat(2, 2) = -sinY * cosX;

    return partialYMat;
}

/* Compute a partial derivative of a rotation matrix R = Ry(ry) Rx(rx) Rz(rz)
 * with respect to rz (the rotation angle around Z axis), where the rotation
 * matrix rotates a 3D vector around fixed z, x, and y axes */
inline Eigen::Matrix3f partialZFromRotationZXY(
    const float rx, const float ry, const float rz)
{
    const float cosX = std::cos(rx);
    const float sinX = std::sin(rx);
    const float cosY = std::cos(ry);
    const float sinY = std::sin(ry);
    const float cosZ = std::cos(rz);
    const float sinZ = std::sin(rz);

    Eigen::Matrix3f partialZMat;
    partialZMat(0, 0) = -cosY * sinZ + sinY * sinX * cosZ;
    partialZMat(0, 1) = -cosY * cosZ - sinY * sinX * sinZ;
    partialZMat(0, 2) =  0.0f;
    partialZMat(1, 0) =  cosX * cosZ;
    partialZMat(1, 1) = -cosX * sinZ;
    partialZMat(1, 2) =  0.0f;
    partialZMat(2, 0) =  sinY * sinZ + cosY * sinX * cosZ;
    partialZMat(2, 1) =  sinY * cosZ - cosY * sinX * sinZ;
    partialZMat(2, 2) =  0.0f;

    return partialZMat;
}

/* Compute a partial derivative of a rotation matrix written as
 * R = Ry(-ry) Rx(-rx) Rz(-rz) = (Rz(rz) Rx(rx) Ry(ry))^T
 * with respect to rx (the rotation angle around X axis) */
inline Eigen::Matrix3f partialXFromRotationYXZT(
    const float rx, const float ry, const float rz)
{
    const float cosX = std::cos(rx);
    const float sinX = std::sin(rx);
    const float cosY = std::cos(ry);
    const float sinY = std::sin(ry);
    const float cosZ = std::cos(rz);
    const float sinZ = std::sin(rz);

    Eigen::Matrix3f partialXMat;
    partialXMat(0, 0) = -sinY * cosX * sinZ;
    partialXMat(0, 1) =  sinY * cosX * cosZ;
    partialXMat(0, 2) =  sinY * sinX;
    partialXMat(1, 0) =  sinX * sinZ;
    partialXMat(1, 1) = -sinX * cosZ;
    partialXMat(1, 2) =  cosX;
    partialXMat(2, 0) =  cosY * cosX * sinZ;
    partialXMat(2, 1) = -cosY * cosX * cosZ;
    partialXMat(2, 2) = -cosY * sinX;

    return partialXMat;
}

/* Compute a partial derivative of a rotation matrix written as
 * R = Ry(-ry) Rx(-rx) Rz(-rz) = (Rz(rz) Rx(rx) Ry(ry))^T
 * with respect to ry (the rotation angle around Y axis) */
inline Eigen::Matrix3f partialYFromRotationYXZT(
    const float rx, const float ry, const float rz)
{
    const float cosX = std::cos(rx);
    const float sinX = std::sin(rx);
    const float cosY = std::cos(ry);
    const float sinY = std::sin(ry);
    const float cosZ = std::cos(rz);
    const float sinZ = std::sin(rz);

    Eigen::Matrix3f partialYMat;
    partialYMat(0, 0) = -sinY * cosZ - cosY * sinX * sinZ;
    partialYMat(0, 1) = -sinY * sinZ + cosY * sinX * cosZ;
    partialYMat(0, 2) = -cosY * cosX;
    partialYMat(1, 0) =  0.0f;
    partialYMat(1, 1) =  0.0f;
    partialYMat(1, 2) =  0.0f;
    partialYMat(2, 0) =  cosY * cosZ - sinY * sinX * sinZ;
    partialYMat(2, 1) =  cosY * sinZ + sinY * sinX * cosZ;
    partialYMat(2, 2) = -sinY * cosX;

    return partialYMat;
}

/* Compute a partial derivative of a rotation matrix written as
 * R = Ry(-ry) Rx(-rx) Rz(-rz) = (Rz(rz) Rx(rx) Ry(ry))^T
 * with respect to rz (the rotation angle around Z axis) */
inline Eigen::Matrix3f partialZFromRotationYXZT(
    const float rx, const float ry, const float rz)
{
    const float cosX = std::cos(rx);
    const float sinX = std::sin(rx);
    const float cosY = std::cos(ry);
    const float sinY = std::sin(ry);
    const float cosZ = std::cos(rz);
    const float sinZ = std::sin(rz);

    Eigen::Matrix3f partialZMat;
    partialZMat(0, 0) = -cosY * sinZ - sinY * sinX * cosZ;
    partialZMat(0, 1) =  cosY * cosZ - sinY * sinX * sinZ;
    partialZMat(0, 2) =  0.0f;
    partialZMat(1, 0) = -cosX * cosZ;
    partialZMat(1, 1) = -cosX * sinZ;
    partialZMat(1, 2) =  0.0f;
    partialZMat(2, 0) = -sinY * sinZ + cosY * sinX * cosZ;
    partialZMat(2, 1) =  sinY * cosZ + cosY * sinX * sinZ;
    partialZMat(2, 2) =  0.0f;

    return partialZMat;
}

} // namespace loam

#endif // LOAM_TRANSFORM_HPP
