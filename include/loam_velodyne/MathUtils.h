
// MathUtils.h

#ifndef LOAM_MATH_UTILS_H
#define LOAM_MATH_UTILS_H

#include "loam_velodyne/Angle.h"
#include "loam_velodyne/Vector3.h"

#include <cmath>

namespace loam {

/** \brief Convert the given radian angle to degrees.
 *
 * @param radians The radian angle to convert.
 * @return The angle in degrees.
 */
inline double rad2deg(double radians)
{
    return radians * 180.0 / M_PI;
}

/** \brief Convert the given radian angle to degrees.
 *
 * @param radians The radian angle to convert.
 * @return The angle in degrees.
 */
inline float rad2deg(float radians)
{
    return static_cast<float>(radians * 180.0 / M_PI);
}

/** \brief Convert the given degree angle to radian.
 *
 * @param degrees The degree angle to convert.
 * @return The radian angle.
 */
inline double deg2rad(double degrees)
{
    return degrees * M_PI / 180.0;
}

/** \brief Convert the given degree angle to radian.
 *
 * @param degrees The degree angle to convert.
 * @return The radian angle.
 */
inline float deg2rad(float degrees)
{
    return static_cast<float>(degrees * M_PI / 180.0);
}

/** \brief Calculate the squared difference of the given two points.
 *
 * @param a The first point.
 * @param b The second point.
 * @return The squared difference between point a and b.
 */
template <typename PointT>
inline float calcSquaredDiff(const PointT& a, const PointT& b)
{
    const float diffX = a.x - b.x;
    const float diffY = a.y - b.y;
    const float diffZ = a.z - b.z;

    return diffX * diffX + diffY * diffY + diffZ * diffZ;
}

/** \brief Calculate the squared difference of the given two points.
 *
 * @param a The first point.
 * @param b The second point.
 * @param wb The weighting factor for the SECOND point.
 * @return The squared difference between point a and b.
 */
template <typename PointT>
inline float calcSquaredDiff(const PointT& a, const PointT& b, const float wb)
{
    const float diffX = a.x - b.x * wb;
    const float diffY = a.y - b.y * wb;
    const float diffZ = a.z - b.z * wb;

    return diffX * diffX + diffY * diffY + diffZ * diffZ;
}

/** \brief Calculate the absolute distance of the point to the origin.
 *
 * @param p The point.
 * @return The distance to the point.
 */
template <typename PointT>
inline float calcPointDistance(const PointT& p)
{
    return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

/** \brief Calculate the squared distance of the point to the origin.
 *
 * @param p The point.
 * @return The squared distance to the point.
 */
template <typename PointT>
inline float calcSquaredPointDistance(const PointT& p)
{
    return p.x * p.x + p.y * p.y + p.z * p.z;
}

/** \brief Rotate the given vector by the specified angle around the x-axis.
 *
 * @param v The vector to rotate
 * @param ang The rotation angle
 */
inline void rotX(Vector3& v, const Angle& ang)
{
    const float y = v.y();
    v.y() = ang.cos() * y - ang.sin() * v.z();
    v.z() = ang.sin() * y + ang.cos() * v.z();
}

/** \brief Rotate the given point by the specified angle around the x-axis.
 *
 * @param p The point to rotate
 * @param ang The rotation angle
 */
template <typename PointT>
inline void rotX(PointT& p, const Angle& ang)
{
    float y = p.y;
    p.y = ang.cos() * y - ang.sin() * p.z;
    p.z = ang.sin() * y + ang.cos() * p.z;
}

/** \brief Rotate the given vector by the specified angle around the y-axis.
 *
 * @param v The vector to rotate
 * @param ang The rotation angle
 */
inline void rotY(Vector3& v, const Angle& ang)
{
    float x = v.x();
    v.x() =  ang.cos() * x + ang.sin() * v.z();
    v.z() = -ang.sin() * x + ang.cos() * v.z();
}

/** \brief Rotate the given point by the specified angle around the y-axis.
 *
 * @param p The point to rotate
 * @param ang The rotation angle
 */
template <typename PointT>
inline void rotY(PointT& p, const Angle& ang)
{
    float x = p.x;
    p.x =  ang.cos() * x + ang.sin() * p.z;
    p.z = -ang.sin() * x + ang.cos() * p.z;
}

/** \brief Rotate the given vector by the specified angle around the z-axis.
 *
 * @param v The vector to rotate
 * @param ang The rotation angle
 */
inline void rotZ(Vector3& v, const Angle& ang)
{
    float x = v.x();
    v.x() = ang.cos() * x - ang.sin() * v.y();
    v.y() = ang.sin() * x + ang.cos() * v.y();
}

/** \brief Rotate the given point by the specified angle around the z-axis.
 *
 * @param p The point to rotate
 * @param ang The rotation angle
 */
template <typename PointT>
inline void rotZ(PointT& p, const Angle& ang)
{
    float x = p.x;
    p.x = ang.cos() * x - ang.sin() * p.y;
    p.y = ang.sin() * x + ang.cos() * p.y;
}

/** \brief Rotate the given vector by the specified angles
 * around the z, x, and y axes.
 *
 * @param v The vector to rotate
 * @param angZ The rotation angle around the z-axis
 * @param angX The rotation angle around the x-axis
 * @param angY The rotation angle around the y-axis
 */
inline void rotateZXY(
    Vector3& v, const Angle& angZ, const Angle& angX, const Angle& angY)
{
    rotZ(v, angZ);
    rotX(v, angX);
    rotY(v, angY);
}

/** \brief Rotate the given point by the specified angles
 * around the z, x, and y axes.
 *
 * @param p The point to rotate
 * @param angZ The rotation angle around the z-axis
 * @param angX The rotation angle around the x-axis
 * @param angY The rotation angle around the y-axis
 */
template <typename PointT>
inline void rotateZXY(
    PointT& p, const Angle& angZ, const Angle& angX, const Angle& angY)
{
    rotZ(p, angZ);
    rotX(p, angX);
    rotY(p, angY);
}

/** \brief Rotate the given vector by the specified angles around
 * the y, x, and z axes.
 *
 * @param v The vector to rotate
 * @param angY The rotation angle around the y-axis
 * @param angX The rotation angle around the x-axis
 * @param angZ The rotation angle around the z-axis
 */
inline void rotateYXZ(
    Vector3& v, const Angle& angY, const Angle& angX, const Angle& angZ)
{
    rotY(v, angY);
    rotX(v, angX);
    rotZ(v, angZ);
}

/** \brief Rotate the given point by the specified angles around
 * the y, x, and z axes.
 *
 * @param p The point to rotate
 * @param angY The rotation angle around the y-axis
 * @param angX The rotation angle around the x-axis
 * @param angZ The rotation angle around the z-axis
 */
template <typename PointT>
inline void rotateYXZ(
    PointT& p, const Angle& angY, const Angle& angX, const Angle& angZ)
{
    rotY(p, angY);
    rotX(p, angX);
    rotZ(p, angZ);
}

} // namespace loam

#endif // LOAM_MATH_UTILS_H
