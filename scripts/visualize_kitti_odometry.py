#!/usr/bin/env python3
# coding: utf-8
# visualize_kitti_odometry.py

# Command-line tool to visualize the Kitti odometry results obtained from the
# loam_velodyne package

import argparse
import sys

import pykitti
import pykitti.utils

import numpy as np
import matplotlib.pyplot as plt

from odometry_util import load_odometry

def main():
    # Setup command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--sequence", required=True, type=str)
    parser.add_argument("--odometry", required=True, type=str)
    parser.add_argument("--topic", required=True, type=str)
    parser.add_argument("--xmin", default=None, type=float)
    parser.add_argument("--ymin", default=None, type=float)
    parser.add_argument("--xmax", default=None, type=float)
    parser.add_argument("--ymax", default=None, type=float)
    parser.add_argument("--legend_ncol", default=1, type=int)
    parser.add_argument("--legend_bbox", default="1.05,1.0", type=str)
    parser.add_argument("--legend_loc", default="upper left", type=str)

    # Parse command-line arguments
    args = parser.parse_args()
    args.legend_bbox = tuple(map(float, args.legend_bbox.split(",")))

    if args.topic not in ("Odometry", "Mapping", "Integrated"):
        print("Topic name should be `Odometry`, `Mapping`, or `Integrated`")
        sys.exit(1)

    # Load the odometry data
    dataset = pykitti.odometry(args.dataset_dir, args.sequence)

    # Load the odometry results from the loam_velodyne package
    odometry = load_odometry(args.odometry, args.topic)
    # Transform the odometry poses to the rectified camera coordinate
    # odometry = [dataset.calib.T_cam0_velo.dot(x) for x in odometry]

    # Setup figure
    fig = plt.figure()
    ax = fig.add_subplot()

    # Ground-truth poses are available for sequences 0 to 10
    if int(args.sequence) < 11:
        # Get the ground-truth poses
        ground_truth = np.array([(x[0,3], x[2,3]) for x in dataset.poses])
        # Plot the ground-truth poses
        ax.plot(ground_truth[:,0], ground_truth[:,1], label="Ground-truth")
    else:
        print("Ground-truth poses are not available for sequence {}"
              .format(args.sequence))

    # Plot the odometry results
    poses = np.array([(x[0,3], x[2,3]) for x in odometry])
    ax.plot(poses[:,0], poses[:,1], label="LOAM ({})".format(args.topic))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

    ax_x_min, ax_x_max = ax.get_xlim()
    ax_y_min, ax_y_max = ax.get_ylim()
    x_min = args.xmin if args.xmin is not None else ax_x_min
    x_max = args.xmax if args.xmax is not None else ax_x_max
    y_min = args.ymin if args.ymin is not None else ax_y_min
    y_max = args.ymax if args.ymax is not None else ax_y_max
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.legend(ncol=args.legend_ncol, bbox_to_anchor=args.legend_bbox,
              loc=args.legend_loc, borderaxespad=0.0, edgecolor="black",
              fancybox=False, framealpha=1.0)

    plt.show()

if __name__ == "__main__":
    main()
