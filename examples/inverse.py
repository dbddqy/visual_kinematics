#!/usr/bin/env python3

from visual_kinematics import Robot, Frame
import numpy as np
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
    # inverse
    # =====================================

    xyz = np.array([[0.28127], [0.], [1.13182]])
    abc = np.array([0.5 * pi, 0., pi])
    end = Frame.from_euler_3(abc, xyz)
    robot.inverse(end)

    print("axis values: ")
    print(robot.axis_values)

    robot.show()


if __name__ == "__main__":
    main()
