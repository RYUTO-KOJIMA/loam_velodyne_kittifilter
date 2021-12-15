# coding: utf-8
# odometry_util.py

import json
import pykitti
import numpy as np

def load_odometry(file_name: str, topic: str):
    # Open the odometry file
    f = open(file_name, "r", encoding="utf-8")
    odometry_json = json.load(f)
    # Close the odometry file
    f.close()

    poses_json = odometry_json[topic]
    poses = []

    for pose_json in poses_json["Results"]:
        values = list(map(float, pose_json.split()))
        _, trans_x, trans_y, trans_z, rot_x, rot_y, rot_z = values

        # Create a 3x1 translation vector
        t = np.array([-trans_x, -trans_y, trans_z])
        # Create a 3x3 rotation matrix
        Rx = pykitti.utils.rotx(-rot_x)
        Ry = pykitti.utils.roty(-rot_y)
        Rz = pykitti.utils.rotz(rot_z)
        R = Ry @ Rx @ Rz
        # Create a 4x4 transformation matrix
        trans = pykitti.utils.transform_from_rot_trans(R, t)

        # Append the transformation
        poses.append(trans)

    return poses
