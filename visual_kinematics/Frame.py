import numpy as np
from scipy.spatial.transform import Rotation
from math import sin as s, cos as c


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
        return Rotation.from_matrix(self.r_3_3).as_euler("ZYX", degrees=False)

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
        return Frame(np.array([[c(theta), -s(theta) * c(alpha), s(theta) * s(alpha), a * c(theta)],
                               [s(theta), c(theta) * c(alpha), -c(theta) * s(alpha), a * s(theta)],
                               [0., s(alpha), c(alpha), d],
                               [0., 0., 0., 1.]]))

    # for the difference between two DH parameter definitions
    # https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters
    @staticmethod
    def from_dh_modified(dh_params):
        d, theta, a, alpha = dh_params
        return Frame(np.array([[c(theta), -s(theta), 0, a],
                               [s(theta) * c(alpha), c(theta) * c(alpha), -s(alpha), -s(alpha) * d],
                               [s(theta) * s(alpha), c(theta) * s(alpha), c(alpha), c(alpha) * d],
                               [0., 0., 0., 1.]]))

    @staticmethod
    def i_4_4():
        return Frame(np.eye(4))

    def distance_to(self, other):
        return np.linalg.norm(self.t_3_1 - other.t_3_1, ord=2)
