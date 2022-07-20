from unittest import TestCase

import numpy as np

from visual_kinematics.RobotSerial import RobotSerial
from visual_kinematics.Tool import Tool


class TestRobotSerial(TestCase):
    # robot dh parameters
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
    dh_params[:, 0] = 1e-3 * dh_params[:, 0]
    dh_params[:, 1] = 1e-3 * dh_params[:, 1]
    # tool
    # rotate from [0, 0, 1] to [1, 1, 1], translate 1 on y, 2 on z
    sqrt_3_inv = 1 / np.sqrt(3)
    t_4_4 = np.array([
        [0.5 + 0.5 * sqrt_3_inv, -0.5 + 0.5 * sqrt_3_inv, sqrt_3_inv, 0],
        [-0.5 + 0.5 * sqrt_3_inv, 0.5 + 0.5 * sqrt_3_inv, sqrt_3_inv, 0.050],
        [-sqrt_3_inv, -sqrt_3_inv, sqrt_3_inv, 0.100],
        [0, 0, 0, 1]
    ])

    plot_limit = 1.1 * np.sum(np.absolute(dh_params[:, 1]))
    robot_with_tool = RobotSerial(
        dh_params=dh_params,
        dh_type="normal",
        tool=Tool(t_4_4),
        # plot_xlim=[-plot_limit, plot_limit],
        # plot_ylim=[-plot_limit, plot_limit],
        # plot_zlim=[0, plot_limit],
        ws_lim=None
    )
    robot_no_tool = RobotSerial(
        dh_params=dh_params,
        dh_type="normal",
        # plot_xlim=[-plot_limit, plot_limit],
        # plot_ylim=[-plot_limit, plot_limit],
        # plot_zlim=[0, plot_limit],
        ws_lim=None
    )

    def test_draw(self):
        joint_angles = np.array([1, 1, 1, 1, 1, 1])
        self.robot_with_tool.forward(joint_angles)
        self.robot_with_tool.draw()

        self.robot_no_tool.forward(joint_angles)
        self.robot_no_tool.show()
