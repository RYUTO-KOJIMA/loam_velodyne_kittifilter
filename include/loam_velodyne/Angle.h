
// Angle.h

#ifndef LOAM_ANGLE_H
#define LOAM_ANGLE_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>

namespace loam {

/** \brief Class for holding an angle.
 *
 * This class provides buffered access to sine and cosine values
 * to the represented angular value.
 */
class Angle
{
public:
    Angle() : _radian(0.0),
              _cos(1.0),
              _sin(0.0) { }

    Angle(float radValue) : _radian(radValue),
                            _cos(std::cos(radValue)),
                            _sin(std::sin(radValue)) { }

    Angle(const Angle& other) : _radian(other._radian),
                                _cos(other._cos),
                                _sin(other._sin) { }

    inline void operator=(const Angle& rhs)
    {
        this->_radian = rhs._radian;
        this->_cos = rhs._cos;
        this->_sin = rhs._sin;
    }

    inline void operator+=(const float& radValue)
    { *this = static_cast<Angle>(this->_radian + radValue); }
    inline void operator+=(const Angle& other)
    { *this = static_cast<Angle>(this->_radian + other._radian); }

    inline void operator-=(const float& radValue)
    { *this = static_cast<Angle>(this->_radian - radValue); }
    inline void operator-=(const Angle& other)
    { *this = static_cast<Angle>(this->_radian - other._radian); }

    inline Angle operator-() const
    {
        Angle out;
        out._radian = -this->_radian;
        out._cos = this->_cos;
        out._sin = -this->_sin;
        return out;
    }

    inline float rad() const { return this->_radian; }
    inline float deg() const { return this->_radian * 180.0f / M_PI; }
    inline float cos() const { return this->_cos; }
    inline float sin() const { return this->_sin; }

private:
    // Angle value in radians
    float _radian;
    // Cosine of the angle in radians
    float _cos;
    // Sine of the angle in radians
    float _sin;
};

} // namespace loam

#endif // LOAM_ANGLE_H
