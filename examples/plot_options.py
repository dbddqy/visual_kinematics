from copy import copy

import numpy as np
from matplotlib import pyplot as plt

from visual_kinematics.RobotSerial import RobotSerial
from visual_kinematics.Tool import Tool


def test_draw():
    # robot dh parameters (ur5, in m)
    dh_params = np.array(
        [
            [0.089159, 0.0, np.pi / 2, 0.0],
            [0.0, -0.4250, 0.0, 0.0],
            [0.0, -0.39225, 0.0, 0.0],
            [0.10915, 0.0, np.pi / 2, 0.0],
            [0.09465, 0.0, -np.pi / 2, 0.0],
            [0.0823, 0.0, 0.0, 0.0],
        ]
    )
    # tool - rotate from [0, 0, 1] to [1, 1, 1], translate 0.05m in y, 0.1m in z
    sqrt_3_inv = 1 / np.sqrt(3)
    t_4_4 = np.array(
        [
            [0.5 + 0.5 * sqrt_3_inv, -0.5 + 0.5 * sqrt_3_inv, sqrt_3_inv, 0],
            [-0.5 + 0.5 * sqrt_3_inv, 0.5 + 0.5 * sqrt_3_inv, sqrt_3_inv, 0.050],
            [-sqrt_3_inv, -sqrt_3_inv, sqrt_3_inv, 0.100],
            [0, 0, 0, 1],
        ]
    )
    tool = Tool(t_4_4)

    joint_angles = np.array([1, -2, 2, 1, 1, 1])

    draw_without_tool(joint_angles, dh_params)
    draw_with_tool_auto_zoom(joint_angles, dh_params, tool)
    draw_distorted_without_markers(joint_angles, dh_params)

    plt.show()


def draw_without_tool(joint_angles, dh_params):
    robot_no_tool = RobotSerial(
        dh_params=dh_params,
        dh_type="normal",
        ws_lim=None,
    )
    robot_no_tool.forward(joint_angles)
    robot_no_tool.draw()


def draw_distorted_without_markers(joint_angles, dh_params):
    robot_specify_plot_size = RobotSerial(
        dh_params=dh_params,
        dh_type="normal",
        plot_xlim=(-0.2, 0.1),
        plot_ylim=(-1, 1),
        plot_zlim=(0, 0.5),
        ws_lim=None,
    )
    robot_specify_plot_size.forward(joint_angles)
    robot_specify_plot_size.draw(
        cylinder_relative_size=0, orientation_relative_size=0
    )


def draw_with_tool_auto_zoom(joint_angles, dh_params, tool):
    # convert to mm so it does not fit in the default plot window
    dh_in_mm = copy(dh_params)
    dh_in_mm[:, :2] = 1000 * dh_in_mm[:, :2]
    robot_with_tool = RobotSerial(
        dh_params=dh_in_mm,
        dh_type="normal",
        tool=tool,
        plot_xlim=None,
        plot_ylim=None,
        plot_zlim=None,
        ws_lim=None,
    )
    robot_with_tool.forward(joint_angles)
    robot_with_tool.draw()


if __name__ == "__main__":
    test_draw()
