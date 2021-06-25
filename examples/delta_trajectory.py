#!/usr/bin/env python3

from visual_kinematics.RobotDelta import *
from visual_kinematics.RobotTrajectory import *
import numpy as np


def main():
    np.set_printoptions(precision=3, suppress=True)

    robot = RobotDelta(np.array([0.16, 0.06, 0.30, 0.50]))  # r1 r2 l1 l2

    # =====================================
    # trajectory
    # =====================================

    frames = [Frame.from_euler_3(np.array([0., 0., 0.]), np.array([[0.], [0.], [-0.6]])),
              Frame.from_euler_3(np.array([0., 0., 0.]), np.array([[0.0], [0.], [-0.45]])),
              Frame.from_euler_3(np.array([0., 0., 0.]), np.array([[-0.2], [-0.2], [-0.45]])),
              Frame.from_euler_3(np.array([0., 0., 0.]), np.array([[-0.2], [-0.2], [-0.6]]))]

    trajectory = RobotTrajectory(robot, frames)
    trajectory.show(motion="p2p")


if __name__ == "__main__":
    main()
