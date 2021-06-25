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

    #  inverse of the frame
    @property
    def inv(self):
        t_4_4_new = self.t_4_4.copy()
        t_4_4_new[0:3, 3:4] = -self.t_4_4[0:3, 0:3].T.dot(self.t_4_4[0:3, 3:4])
        t_4_4_new[0:3, 0:3] = self.t_4_4[0:3, 0:3].T
        return Frame(t_4_4_new)

    @property
    def copy(self):
        return Frame(self.t_4_4)

    #  z axis vector of the frame
    @property
    def z_3_1(self):
        return self.t_4_4[0:3, 2:3]

    #  translation vector of the frame
    @property
    def t_3_1(self):
        return self.t_4_4[0:3, 3:4]

    #  rotation matrix of the frame
    @property
    def r_3_3(self):
        return self.t_4_4[0:3, 0:3]

    #  rotation in quaternion format
    @property
    def q_4(self):
        return Rotation.from_matrix(self.r_3_3).as_quat()

    #  rotation in angle-axis format
    @property
    def r_3(self):
        return Rotation.from_matrix(self.r_3_3).as_rotvec()

    #  rotation in ZYX euler angel format
    @property
    def euler_3(self):
        return Rotation.from_matrix(self.r_3_3).as_euler("ZYX", degrees=False)

    #  construct a frame using rotation matrix and translation vector
    @staticmethod
    def from_r_3_3(r_3_3, t_3_1):
        t_4_4 = np.eye(4)
        t_4_4[0:3, 0:3] = r_3_3
        t_4_4[0:3, 3:4] = t_3_1
        return Frame(t_4_4)

    #  construct a frame using quaternion and translation vector
    @staticmethod
    def from_q_4(q_4, t_3_1):
        r_3_3 = Rotation.from_quat(q_4).as_matrix()
        return Frame.from_r_3_3(r_3_3, t_3_1)

    #  construct a frame using angle-axis and translation vector
    @staticmethod
    def from_r_3(r_3, t_3_1):
        r_3_3 = Rotation.from_rotvec(r_3).as_matrix()
        return Frame.from_r_3_3(r_3_3, t_3_1)

    #  construct a frame using ZYX euler angle and translation vector
    @staticmethod
    def from_euler_3(euler_3, t_3_1):
        r_3_3 = Rotation.from_euler("ZYX", euler_3, degrees=False).as_matrix()
        return Frame.from_r_3_3(r_3_3, t_3_1)

    #  construct a frame using dh parameters
    @staticmethod
    def from_dh(dh_params):
        d, a, alpha, theta = dh_params
        return Frame(np.array([[c(theta), -s(theta) * c(alpha), s(theta) * s(alpha), a * c(theta)],
                               [s(theta), c(theta) * c(alpha), -c(theta) * s(alpha), a * s(theta)],
                               [0., s(alpha), c(alpha), d],
                               [0., 0., 0., 1.]]))

    #  construct a frame using modified dh parameters
    #  for the difference between two DH parameter definitions
    #  https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters
    @staticmethod
    def from_dh_modified(dh_params):
        d, a, alpha, theta = dh_params
        return Frame(np.array([[c(theta), -s(theta), 0, a],
                               [s(theta) * c(alpha), c(theta) * c(alpha), -s(alpha), -s(alpha) * d],
                               [s(theta) * s(alpha), c(theta) * s(alpha), c(alpha), c(alpha) * d],
                               [0., 0., 0., 1.]]))

    #  construct an identity frame
    @staticmethod
    def i_4_4():
        return Frame(np.eye(4))

    #  calculate the center distance to the other frame
    def distance_to(self, other):
        return np.linalg.norm(self.t_3_1 - other.t_3_1, ord=2)

    #  calculate the rotated angle to the other frame
    def angle_to(self, other):
        return np.linalg.norm(Rotation.from_matrix(self.r_3_3.T.dot(other.r_3_3)).as_rotvec(), ord=2)
