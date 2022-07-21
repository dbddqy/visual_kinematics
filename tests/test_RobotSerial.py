from copy import copy
from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

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

    def test_draw(self):
        joint_angles = np.array([1, -2, 2, 1, 1, 1])

        with self.subTest("without tool"):
            # convert to m so it fits in the default plot window
            dh_in_m = copy(self.dh_params)
            dh_in_m[:, :2] = 0.001 * dh_in_m[:, :2]
            robot_no_tool = RobotSerial(
                dh_params=dh_in_m,
                dh_type="normal",
                ws_lim=None,
            )
            robot_no_tool.forward(joint_angles)
            robot_no_tool.draw()

        with self.subTest("with tool, use auto zoom"):
            robot_with_tool = RobotSerial(
                dh_params=self.dh_params,
                dh_type="normal",
                tool=Tool(self.t_4_4),
                plot_xlim=None,
                plot_ylim=None,
                plot_zlim=None,
                ws_lim=None,
            )
            robot_with_tool.forward(joint_angles)
            robot_with_tool.draw()

        with self.subTest("plot size specified (distorted), remove cylinders and orientation markers"):
            robot_specify_plot_size = RobotSerial(
                dh_params=self.dh_params,
                dh_type="normal",
                plot_xlim=(-200, 150),
                plot_ylim=(-1000, 1000),
                plot_zlim=(0, 500),
                ws_lim=None,
            )
            robot_specify_plot_size.forward(joint_angles)
            robot_specify_plot_size.draw(
                cylinder_relative_size=0, orientation_relative_size=0
            )

        plt.show()

    def test_draw_with_slider(self):
        robot = RobotSerial(
            dh_params=self.dh_params,
            dh_type="normal",
            tool=Tool(self.t_4_4),
            plot_xlim=None,
            plot_ylim=None,
            plot_zlim=None,
            ws_lim=None,
        )
        numberOfPositions = 100
        joint_angles = np.array(
            [
                np.arange(1, 2, 1 / numberOfPositions),
                np.arange(-2, 0, 2 / numberOfPositions),
                np.arange(2, 1, -1 / numberOfPositions),
                np.arange(1, 2, 1 / numberOfPositions),
                np.arange(1, 0, -1 / numberOfPositions),
                np.arange(1, 3, 2 / numberOfPositions),
            ]
        ).T
        # draw start position
        position = joint_angles[0]
        robot.forward(position)
        robot.draw()

        # create slider with update function
        axSlider: plt.Axes = plt.axes([0.15, 0.06, 0.75, 0.05])
        self._slider = Slider(axSlider, "progress", 0.0, 1.0, valinit=0)

        def update(_):
            position_index = int(self._slider.val * (numberOfPositions - 1))
            position = joint_angles[position_index]
            robot.forward(position)
            robot.draw()

        self._slider.on_changed(update)

        plt.show()

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
