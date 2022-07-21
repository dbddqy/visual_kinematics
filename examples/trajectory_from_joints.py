import numpy as np
from matplotlib import pyplot as plt

from visual_kinematics.RobotSerial import RobotSerial
from visual_kinematics.RobotTrajectory import RobotTrajectory
from visual_kinematics.Tool import Tool


def test_draw_with_slider():
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

    robot = RobotSerial(
        dh_params=dh_params,
        dh_type="normal",
        tool=Tool(t_4_4),
        ws_lim=None,
    )
    numberOfPositions = 100
    joint_angles = np.array(
        [
            np.arange(-1, -3, -2 / numberOfPositions),
            np.arange(-2, 0, 2 / numberOfPositions),
            np.arange(2, -2, -4 / numberOfPositions),
            np.arange(-2, -1, 1 / numberOfPositions),
            np.arange(3, 0, -3 / numberOfPositions),
            np.arange(-3, 3, 6 / numberOfPositions),
        ]
    ).T

    slider = RobotTrajectory.draw_from_joint_positions(robot, joint_angles)

    plt.show()


if __name__ == "__main__":
    test_draw_with_slider()
