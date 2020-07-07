#!/usr/bin/env python3

from visual_kinematics.Robot import *
from math import pi


def main():
    np.set_printoptions(precision=3, suppress=True)

    dh_params = np.array([[0.163, 0., 0., 0.5 * pi],
                          [0., 0.5 * pi, 0.632, pi],
                          [0., 0., 0.6005, pi],
                          [0.2013, -0.5 * pi, 0., -0.5 * pi],
                          [0.1025, 0., 0., 0.5 * pi],
                          [0.094, 0., 0., 0.]])

    robot = Robot(dh_params)

    # =====================================
    # trajectory
    # =====================================

    trajectory = []
    trajectory.append(Frame.from_euler_3(np.array([0.5 * pi, 0., pi]), np.array([[0.28127], [0.], [1.13182]])))
    trajectory.append(Frame.from_euler_3(np.array([0.25 * pi, 0., 0.75 * pi]), np.array([[0.48127], [0.], [1.13182]])))
    trajectory.append(Frame.from_euler_3(np.array([0.5 * pi, 0., pi]), np.array([[0.48127], [0.], [0.63182]])))

    robot.show_trajectory(trajectory, motion="p2p")


if __name__ == "__main__":
    main()
