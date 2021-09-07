
// Twist.h

#ifndef LOAM_TWIST_H
#define LOAM_TWIST_H

#include "loam_velodyne/Angle.h"
#include "loam_velodyne/Vector3.h"

namespace loam {

/** \brief Twist composed by three angles and a three-dimensional position.
 *
 */
class Twist
{
public:
    /* Default constructor */
    Twist() = default;
    /* Constructor */
    Twist(const float tx, const float ty, const float tz,
          const float rx, const float ry, const float rz) :
        rot_x(rx), rot_y(ry), rot_z(rz), pos(tx, ty, tz) { }

    Angle   rot_x;
    Angle   rot_y;
    Angle   rot_z;
    Vector3 pos;
};

} // namespace loam

#endif // LOAM_TWIST_H
