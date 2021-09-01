
// Twist.h

#ifndef LOAM_TWIST_H
#define LOAM_TWIST_H

#include "Angle.h"
#include "Vector3.h"

namespace loam {

/** \brief Twist composed by three angles and a three-dimensional position.
 *
 */
class Twist
{
public:
    Twist() = default;

    Angle   rot_x;
    Angle   rot_y;
    Angle   rot_z;
    Vector3 pos;
};

} // namespace loam

#endif // LOAM_TWIST_H
