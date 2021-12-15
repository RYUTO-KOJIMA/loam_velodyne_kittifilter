#!/usr/bin/env python3
# coding: utf-8
# evaluate_kitti_odometry.py

import argparse
import os
import subprocess
import sys

from typing import List, Tuple

import pykitti
import pykitti.utils
import numpy as np
import matplotlib.pyplot as plt

from odometry_util import load_odometry

lengths = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]

class Errors(object):
    def __init__(self, first_frame: int, error_rot: float, error_trans: float,
                 length: float, speed: float) -> None:
        self.first_frame = first_frame
        self.error_rot = error_rot
        self.error_trans = error_trans
        self.length = length
        self.speed = speed

def trajectory_distances(odometry: List[np.ndarray]) -> List[np.float64]:
    # Refer to trajectoryDistances() in evaluate_odometry.cpp
    dists = [0.0]

    for prev_pose, pose in zip(odometry[0:], odometry[1:]):
        dx = pose[0,3] - prev_pose[0,3]
        dy = pose[1,3] - prev_pose[1,3]
        dz = pose[2,3] - prev_pose[2,3]
        dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        dists.append(dists[-1] + dist)

    return dists

def last_frame_from_segment_length(dists: List[float], first_frame: int,
                                   length: float) -> int:
    # Refer to lastFrameFromSegmentLength() in evaluate_odometry.cpp
    for i in range(first_frame, len(dists)):
        if dists[i] > dists[first_frame] + length:
            return i
    return -1

def rotation_error(pose_error: np.ndarray) -> np.float64:
    # Refer to rotationError() in evaluate_odometry.cpp
    # Compute the rotation angle around the axis
    r00 = pose_error[0,0]
    r11 = pose_error[1,1]
    r22 = pose_error[2,2]
    d = 0.5 * (r00 + r11 + r22 - 1.0)
    return np.arccos(np.clip(d, -1.0, 1.0))

def translation_error(pose_error: np.ndarray) -> np.float64:
    # Refer to translationError() in evaluate_odometry.cpp
    dx = pose_error[0,3]
    dy = pose_error[1,3]
    dz = pose_error[2,3]
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

def compute_sequence_errors(poses_gt: List[np.ndarray],
                            poses_result: List[np.ndarray]) -> List[Errors]:
    # Refer to calcSequenceErrors() in evaluate_odometry.cpp
    errors: List[Errors] = []
    # LiDAR scan is acquired at 10Hz and `step_size` represents 1Hz
    step_size = 10
    dists_gt = trajectory_distances(poses_gt)

    for first_frame in range(0, len(poses_gt), step_size):
        for length in lengths:
            last_frame = last_frame_from_segment_length(
                dists_gt, first_frame, length)

            if last_frame == -1:
                continue

            # Compute rotational and translational errors
            pose_delta_gt = np.linalg.inv(poses_gt[first_frame]) @ \
                            poses_gt[last_frame]
            pose_delta_result = np.linalg.inv(poses_result[first_frame]) @ \
                                poses_result[last_frame]
            pose_error = np.linalg.inv(pose_delta_result) @ pose_delta_gt
            error_rot = rotation_error(pose_error)
            error_trans = translation_error(pose_error)

            # Convert the number of frames to seconds
            num_frames = last_frame - first_frame + 1
            # Compute average speed
            speed = length / (0.1 * num_frames)

            # Collect error information
            errors.append(Errors(first_frame, error_rot / length,
                                 error_trans / length, length, speed))

    return errors

def save_sequence_errors(errors: List[Errors], file_name: str) -> None:
    # Refer to saveSequenceErrors() in evaluate_odometry.cpp
    with open(file_name, "w", encoding="utf-8") as f:
        for error in errors:
            f.write("{} {:.7f} {:.7f} {:.7f} {:.7f}\n".format(
                    error.first_frame, error.error_rot, error.error_trans,
                    error.length, error.speed))

def compute_roi(poses_gt: List[np.ndarray], poses_result: List[np.ndarray]) \
    -> Tuple[int, int, int, int]:
    # Refer to computeRoi() in evaluate_odometry.cpp
    x_min = sys.float_info.max
    x_max = sys.float_info.min
    z_min = sys.float_info.max
    z_max = sys.float_info.min

    for pose_gt in poses_gt:
        x, z = pose_gt[0,3], pose_gt[2,3]
        x_min = x if x < x_min else x_min
        x_max = x if x > x_max else x_max
        z_min = z if z < z_min else z_min
        z_max = z if z > z_max else z_max

    for pose_result in poses_result:
        x, z = pose_result[0,3], pose_result[2,3]
        x_min = x if x < x_min else x_min
        x_max = x if x > x_max else x_max
        z_min = z if z < z_min else z_min
        z_max = z if z > z_max else z_max

    dx = 1.1 * (x_max - x_min)
    dz = 1.1 * (z_max - z_min)
    mx = 0.5 * (x_max + x_min)
    mz = 0.5 * (z_max + z_min)
    r = 0.5 * max(dx, dz)

    return (int(mx - r), int(mx + r), int(mz - r), int(mz + r))

def save_path(poses_gt: List[np.ndarray], poses_result: List[np.ndarray],
              file_name: str) -> None:
    # Refer to savePathPlot() in evaluate_odometry.cpp
    step_size = 3

    with open(file_name, "w", encoding="utf-8") as f:
        for i in range(0, len(poses_gt), step_size):
            f.write("{} {} {} {}\n".format(
                    poses_gt[i][0,3], poses_gt[i][2,3],
                    poses_result[i][0,3], poses_result[i][2,3]))

def plot_path(out_file_dir: str, out_file_name: str, path_file_name: str,
              roi: Tuple[int, int, int, int]) -> None:
    # Refer to plotPathPlot() in evaluate_odometry.cpp
    # Create a Gnuplot file name
    gnuplot_file_name = os.path.join(out_file_dir, out_file_name + ".gp")

    # Create images in PNG and EPS format
    png_file_name = os.path.join(out_file_dir, out_file_name + ".png")
    eps_file_name = os.path.join(out_file_dir, out_file_name + ".eps")

    for i in range(2):
        with open(gnuplot_file_name, "w", encoding="utf-8") as f:
            if i == 0:
                f.write("set term png size 900,900\n")
                f.write("set output \"{}\"\n".format(png_file_name))
            else:
                f.write("set term postscript eps enhanced color\n")
                f.write("set output \"{}\"\n".format(eps_file_name))

            f.write("set size ratio -1\n")
            f.write("set xrange [{}:{}]\n".format(roi[0], roi[1]))
            f.write("set yrange [{}:{}]\n".format(roi[2], roi[3]))
            f.write("set xlabel \"x [m]\"\n")
            f.write("set ylabel \"z [m]\"\n")
            f.write("plot \"{}\" using 1:2 lc rgb \"#FF0000\" "
                    "title 'Ground Truth' w lines,".format(path_file_name))
            f.write("\"{}\" using 3:4 lc rgb \"#0000FF\" "
                    "title 'Odometry' w lines,".format(path_file_name))
            f.write("\"< head -1 {}\" using 1:2 lc rgb \"#000000\" "
                    "pt 4 ps 1 lw 2 "
                    "title 'Sequence Start' w points\n".format(path_file_name))

        # Run Gnuplot with the file above
        subprocess.run(["gnuplot", gnuplot_file_name])

    # Create a cropped image in PDF format
    pdf_file_name = os.path.join(out_file_dir, out_file_name + "-large.pdf")
    subprocess.run(["ps2pdf", eps_file_name, pdf_file_name])
    pdf_cropped_file_name = os.path.join(out_file_dir, out_file_name + ".pdf")
    subprocess.run(["pdfcrop", pdf_file_name, pdf_cropped_file_name])
    os.remove(pdf_file_name)

def save_error(out_file_dir: str, out_file_name: str,
               seq_errors: List[Errors]) -> None:
    # Refer to saveErrorPlots() in evaluate_odometry.cpp
    trans_length_name = os.path.join(out_file_dir, out_file_name + "-tl.txt")
    rot_length_name = os.path.join(out_file_dir, out_file_name + "-rl.txt")
    trans_speed_name = os.path.join(out_file_dir, out_file_name + "-ts.txt")
    rot_speed_name = os.path.join(out_file_dir, out_file_name + "-rs.txt")

    tl_file = open(trans_length_name, "w", encoding="utf-8")
    rl_file = open(rot_length_name, "w", encoding="utf-8")
    ts_file = open(trans_speed_name, "w", encoding="utf-8")
    rs_file = open(rot_speed_name, "w", encoding="utf-8")

    for length in lengths:
        error_trans, error_rot, num = 0.0, 0.0, 0
        for error in seq_errors:
            if np.fabs(error.length - length) < 1.0:
                error_trans += error.error_trans
                error_rot += error.error_rot
                num += 1

        if num > 2:
            tl_file.write("{:.7f} {:.7f}\n".format(length, error_trans / num))
            rl_file.write("{:.7f} {:.7f}\n".format(length, error_rot / num))

    for speed in range(2, 25, 2):
        error_trans, error_rot, num = 0.0, 0.0, 0.0
        for error in seq_errors:
            if np.fabs(error.speed - speed) < 2.0:
                error_trans += error.error_trans
                error_rot += error.error_rot
                num += 1

        if num > 2:
            ts_file.write("{:.7f} {:.7f}\n".format(speed, error_trans / num))
            rs_file.write("{:.7f} {:.7f}\n".format(speed, error_rot / num))

    tl_file.close()
    rl_file.close()
    ts_file.close()
    rs_file.close()

def plot_error(out_file_dir: str, out_file_name: str) -> None:
    # Refer to plotErrorPlots() in evaluate_odometry.cpp
    suffix = ["tl", "rl", "ts", "rs"]

    for i in range(4):
        base_file_name = "{}-{}".format(out_file_name, suffix[i])
        data_file_name = os.path.join(out_file_dir, base_file_name + ".txt")

        # Create a Gnuplot file name
        gnuplot_file_name = os.path.join(out_file_dir, base_file_name + ".gp")
        # Create images in PNG and EPS format
        png_file_name = os.path.join(out_file_dir, base_file_name + ".png")
        eps_file_name = os.path.join(out_file_dir, base_file_name + ".eps")

        for j in range(2):
            with open(gnuplot_file_name, "w", encoding="utf-8") as f:
                if j == 0:
                    f.write("set term png size 500,250 font \"Helvetica\" 11\n")
                    f.write("set output \"{}\"\n".format(png_file_name))
                else:
                    f.write("set term postscript eps enhanced color\n")
                    f.write("set output \"{}\"\n".format(eps_file_name))

                f.write("set size ratio 0.5\n")
                f.write("set yrange [0:*]\n")

                if i <= 1:
                    f.write("set xlabel \"Path Length [m]\"\n")
                else:
                    f.write("set xlabel \"Speed [km/h]\"\n")

                if i % 2 == 0:
                    f.write("set ylabel \"Translation Error [%]\"\n")
                else:
                    f.write("set ylabel \"Rotation Error [deg/m]\"\n")

                f.write("plot \"{}\" using ".format(data_file_name))

                if i == 0:
                    f.write("1:($2*100) title 'Translation Error' ")
                elif i == 1:
                    f.write("1:($2*57.3) title 'Rotation Error' ")
                elif i == 2:
                    f.write("($1*3.6):($2*100) title 'Translation Error' ")
                elif i == 3:
                    f.write("($1*3.6):($2*57.3) title 'Rotation Error' ")

                f.write("lc rgb \"#0000FF\" pt 4 w linespoints\n")

            # Run Gnuplot with the file above
            subprocess.run(["gnuplot", gnuplot_file_name])

        # Create a cropped image in PDF format
        pdf_name = os.path.join(out_file_dir, base_file_name + "-large.pdf")
        subprocess.run(["ps2pdf", eps_file_name, pdf_name])
        pdf_cropped_name = os.path.join(out_file_dir, base_file_name + ".pdf")
        subprocess.run(["pdfcrop", pdf_name, pdf_cropped_name])
        os.remove(pdf_name)

def save_stats(seq_errors: List[Errors], file_name: str) -> None:
    # Refer to saveStats() in evaluate_odometry.cpp
    error_trans = sum([e.error_trans for e in seq_errors])
    error_rot = sum([e.error_rot for e in seq_errors])

    with open(file_name, "w", encoding="utf-8") as f:
        f.write("{:.7f} {:.7f}\n".format(error_trans / len(seq_errors),
                error_rot / len(seq_errors)))

def main():
    # Setup command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--sequence", required=True, type=str)
    parser.add_argument("--odometry", required=True, type=str)
    parser.add_argument("--topic", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--prefix", required=True, type=str)

    # Parse command-line arguments
    args = parser.parse_args()

    args.dataset_dir = os.path.expanduser(args.dataset_dir)
    args.out_dir = os.path.expanduser(args.out_dir)

    if args.topic not in ("Odometry", "Mapping", "Integrated"):
        print("Topic name should be `Odometry`, `Mapping`, or `Integrated`")
        sys.exit(1)

    # Load the odometry data
    dataset = pykitti.odometry(args.dataset_dir, args.sequence)

    # Load the odometry results from the loam_velodyne package
    odometry = load_odometry(args.odometry, args.topic)
    # Transform the odometry poses to the rectified camera coordinate
    # odometry = [dataset.calib.T_cam0_velo.dot(x) for x in odometry]

    if len(dataset.poses) != len(odometry):
        print("Number of ground-truth poses and output poses are different: "
              "({} and {})".format(len(dataset.poses), len(odometry)))
        sys.exit(1)

    odometry_errors = compute_sequence_errors(dataset.poses, odometry)
    sequence_file_name = args.prefix + "-sequence-errors.txt"
    sequence_file_name = os.path.join(args.out_dir, sequence_file_name)
    save_sequence_errors(odometry_errors, sequence_file_name)

    path_file_name = os.path.join(args.out_dir, args.prefix + "-path.txt")
    save_path(dataset.poses, odometry, path_file_name)
    roi = compute_roi(dataset.poses, odometry)
    plot_path(args.out_dir, args.prefix, path_file_name, roi)

    save_error(args.out_dir, args.prefix + "-error", odometry_errors)
    plot_error(args.out_dir, args.prefix + "-error")

    stats_file_name = os.path.join(args.out_dir, args.prefix + "-stats.txt")
    save_stats(odometry_errors, stats_file_name)

if __name__ == "__main__":
    main()
