
// Vector3.h

#ifndef LOAM_VECTOR3_H
#define LOAM_VECTOR3_H

#include <pcl/point_types.h>

namespace loam {

/** \brief Vector4f decorator for convenient handling.
 *
 */
class Vector3 : public Eigen::Vector4f
{
public:
    Vector3() :
        Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f) { }

    Vector3(float x, float y, float z) :
        Eigen::Vector4f(x, y, z, 0.0f) { }

    template<typename OtherDerived>
    Vector3(const Eigen::MatrixBase<OtherDerived>& other) :
        Eigen::Vector4f(other) { }

    Vector3(const pcl::PointXYZI& p) :
        Eigen::Vector4f(p.x, p.y, p.z, 0.0f) { }

    template<typename OtherDerived>
    Vector3& operator=(const Eigen::MatrixBase<OtherDerived>& rhs)
    {
        this->Eigen::Vector4f::operator=(rhs);
        return *this;
    }

    inline Vector3& operator=(const pcl::PointXYZ& rhs)
    {
        this->x() = rhs.x;
        this->y() = rhs.y;
        this->z() = rhs.z;
        return *this;
    }

    inline Vector3& operator=(const pcl::PointXYZI& rhs)
    {
        this->x() = rhs.x;
        this->y() = rhs.y;
        this->z() = rhs.z;
        return *this;
    }

    inline float x() const { return (*this)(0); }
    inline float y() const { return (*this)(1); }
    inline float z() const { return (*this)(2); }

    inline float& x() { return (*this)(0); }
    inline float& y() { return (*this)(1); }
    inline float& z() { return (*this)(2); }

    inline operator pcl::PointXYZI() const
    {
        pcl::PointXYZI dst;
        dst.x = this->x();
        dst.y = this->y();
        dst.z = this->z();
        dst.intensity = 0.0f;
        return dst;
    }
};

} // namespace loam

#endif // LOAM_VECTOR3_H
