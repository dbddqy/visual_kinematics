import numpy as np
from scipy.spatial.transform import Rotation
from math import sin, cos


class Frame:
    def __init__(self, t_4_4):
        self.t_4_4 = t_4_4

    def __mul__(self, other):
        if isinstance(other, Frame):
            return Frame(np.dot(self.t_4_4, other.t_4_4))
        return Frame(np.dot(self.t_4_4, other))

    def __getitem__(self, key):
        return self.t_4_4[key]

    def __str__(self):
        return self.t_4_4.__str__()

    @property
    def copy(self):
        return Frame(self.t_4_4)

    @property
    def t_3_1(self):
        return self.t_4_4[0:3, 3:4]

    @property
    def r_3_3(self):
        return self.t_4_4[0:3, 0:3]

    @property
    def q_4(self):
        return Rotation.from_matrix(self.r_3_3).as_quat()

    @property
    def r_3(self):
        return Rotation.from_matrix(self.r_3_3).as_rotvec()

    @property
    def euler_3(self):
        return Rotation.from_matrix(self.r_3_3).as_euler("ZYX", degrees=True)

    @staticmethod
    def from_r_3_3(r_3_3, t_3_1):
        t_4_4 = np.eye(4)
        t_4_4[0:3, 0:3] = r_3_3
        t_4_4[0:3, 3:4] = t_3_1
        return Frame(t_4_4)

    @staticmethod
    def from_q_4(q_4, t_3_1):
        r_3_3 = Rotation.from_quat(q_4).as_matrix()
        return Frame.from_r_3_3(r_3_3, t_3_1)

    @staticmethod
    def from_r_3(r_3, t_3_1):
        r_3_3 = Rotation.from_rotvec(r_3).as_matrix()
        return Frame.from_r_3_3(r_3_3, t_3_1)

    @staticmethod
    def from_euler_3(euler_3, t_3_1):
        r_3_3 = Rotation.from_euler("ZYX", euler_3, degrees=False).as_matrix()
        return Frame.from_r_3_3(r_3_3, t_3_1)

    @staticmethod
    def from_dh(dh_params):
        d, theta, a, alpha = dh_params
        return Frame(np.array([[cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
                               [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
                               [0., sin(alpha), cos(alpha), d],
                               [0., 0., 0., 1.]]))

    @staticmethod
    def i_4_4():
        return Frame(np.eye(4))
