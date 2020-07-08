#!/usr/bin/env python3

from visual_kinematics import Robot, Frame
import numpy as np
from math import pi


def main():
    np.set_printoptions(precision=3, suppress=True)

    dh_params = np.array([[0.34, 0., 0., -pi / 2],
                          [0., 0., 0., pi / 2],
                          [0.4, 0., 0., -pi / 2],
                          [0., 0., 0., pi / 2],
                          [0.4, 0., 0., -pi / 2],
                          [0., 0., 0., pi / 2],
                          [0.126, 0., 0., 0.]])

    robot = Robot(dh_params)

    # =====================================
    # inverse
    # =====================================

    trajectory = []
    trajectory.append(Frame.from_euler_3(np.array([0.5 * pi, 0., pi]), np.array([[0.28127], [0.], [0.63182]])))
    trajectory.append(Frame.from_euler_3(np.array([0.25 * pi, 0., 0.75 * pi]), np.array([[0.48127], [0.], [0.63182]])))
    trajectory.append(Frame.from_euler_3(np.array([0.5 * pi, 0., pi]), np.array([[0.48127], [0.], [0.23182]])))

    robot.show_trajectory(trajectory, motion="p2p")


if __name__ == "__main__":
    main()
