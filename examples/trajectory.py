#!/usr/bin/env python3

from visual_kinematics.RobotSerial import *
from visual_kinematics.RobotTrajectory import *
import numpy as np
from math import pi


def main():
    np.set_printoptions(precision=3, suppress=True)

    dh_params = np.array([[0.163, 0., 0.5 * pi, 0.],
                          [0., 0.632, pi, 0.5 * pi],
                          [0., 0.6005, pi, 0.],
                          [0.2013, 0., -0.5 * pi, -0.5 * pi],
                          [0.1025, 0., 0.5 * pi, 0.],
                          [0.094, 0., 0., 0.]])

    robot = RobotSerial(dh_params)

    # =====================================
    # trajectory
    # =====================================

    frames = [Frame.from_euler_3(np.array([0.5 * pi, 0., pi]), np.array([[0.28127], [0.], [1.13182]])),
              Frame.from_euler_3(np.array([0.25 * pi, 0., 0.75 * pi]), np.array([[0.48127], [0.], [1.13182]])),
              Frame.from_euler_3(np.array([0.5 * pi, 0., pi]), np.array([[0.48127], [0.], [0.63182]]))]
    time_points = np.array([0., 6., 10.])
    trajectory = RobotTrajectory(robot, frames, time_points)
    trajectory.show(motion="p2p")


if __name__ == "__main__":
    main()
