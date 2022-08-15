from unittest import TestCase

import numpy as np
from scipy.spatial.transform import Rotation

from visual_kinematics import Frame
from visual_kinematics.Tool import Tool


class TestTool(TestCase):
    def test_init(self):
        # rotate from [0, 0, 1] to [1, 1, 1], translate 1 on y, 2 on z
        sqrt_3_inv = 1 / np.sqrt(3)
        t_4_4 = np.array(
            [
                [0.5 + 0.5 * sqrt_3_inv, -0.5 + 0.5 * sqrt_3_inv, sqrt_3_inv, 0],
                [-0.5 + 0.5 * sqrt_3_inv, 0.5 + 0.5 * sqrt_3_inv, sqrt_3_inv, 1],
                [-sqrt_3_inv, -sqrt_3_inv, sqrt_3_inv, 2],
                [0, 0, 0, 1],
            ]
        )
        tool = Tool(t_4_4=t_4_4, name="test_tool")

        with self.subTest("assert that tool is derived from frame"):
            self.assertIsInstance(tool, Frame)

        with self.subTest("assert that tool rotation is right"):
            tool_rotation = Rotation.from_matrix(tool.r_3_3)
            tool_rot_vec = tool_rotation.as_rotvec()
            self.assertEqual(-tool_rot_vec[0], tool_rot_vec[1])
            self.assertEqual(tool_rot_vec[2], 0)
            self.assertAlmostEqual(np.linalg.norm(tool_rot_vec), 0.9553166197423132)
