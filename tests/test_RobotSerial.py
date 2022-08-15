from copy import copy
from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from visual_kinematics.RobotSerial import RobotSerial
from visual_kinematics.Tool import Tool


class TestRobotSerial(TestCase):
    # robot dh parameters (ur5, in mm)
    dh_params = np.array(
        [
            [89.159, 0.0, np.pi / 2, 0.0],
            [0.0, -425.0, 0.0, 0.0],
            [0.0, -392.25, 0.0, 0.0],
            [109.15, 0.0, np.pi / 2, 0.0],
            [94.65, 0.0, -np.pi / 2, 0.0],
            [82.3, 0.0, 0.0, 0.0],
        ]
    )
    # tool - rotate from [0, 0, 1] to [1, 1, 1], translate 1 on y, 2 on z
    sqrt_3_inv = 1 / np.sqrt(3)
    t_4_4 = np.array(
        [
            [0.5 + 0.5 * sqrt_3_inv, -0.5 + 0.5 * sqrt_3_inv, sqrt_3_inv, 0],
            [-0.5 + 0.5 * sqrt_3_inv, 0.5 + 0.5 * sqrt_3_inv, sqrt_3_inv, 50],
            [-sqrt_3_inv, -sqrt_3_inv, sqrt_3_inv, 100],
            [0, 0, 0, 1],
        ]
    )

    def test_inverse_numerical(self):
        joint_angles_exp = np.array([1, -2, 2, 1, 1, 1])

        with self.subTest("without tool"):
            robot_no_tool = RobotSerial(
                dh_params=self.dh_params,
                dh_type="normal",
                ws_lim=None,
            )
            end_frame = robot_no_tool.forward(joint_angles_exp)
            joint_angles = robot_no_tool.inverse(end_frame)
            [
                self.assertAlmostEqual(joint_angle, joint_angle_exp)
                for joint_angle, joint_angle_exp in zip(joint_angles, joint_angles_exp)
            ]

        with self.subTest("with tool"):
            robot_with_tool = RobotSerial(
                dh_params=self.dh_params,
                dh_type="normal",
                tool=Tool(self.t_4_4),
                ws_lim=None,
            )
            end_frame = robot_with_tool.forward(joint_angles_exp)
            joint_angles = robot_with_tool.inverse(end_frame)
            [
                self.assertAlmostEqual(joint_angle, joint_angle_exp)
                for joint_angle, joint_angle_exp in zip(joint_angles, joint_angles_exp)
            ]
