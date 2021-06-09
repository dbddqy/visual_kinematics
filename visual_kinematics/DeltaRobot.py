import numpy as np

from visual_kinematics.Frame import *
import matplotlib.pyplot as plt
from math import pi
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

import logging


class DeltaRobot:
    #                  r1
    #              A ==========
    #            l1 //   O    \\
    #              //          \\
    #            B \\          //
    #            l2 \\        //
    #                \\  P   //
    #                C ======
    #                  r2
    def __init__(self, r1, r2, l1, l2, theta=0.25*pi*np.ones([3, ]),
                 plot_xlim=[-0.5, 0.5], plot_ylim=[-0.5, 0.5], plot_zlim=[-1.0, 0.0]):
        self.r1 = r1
        self.r2 = r2
        self.l1 = l1
        self.l2 = l2
        self.theta = theta
        self.plot_xlim = plot_xlim
        self.plot_ylim = plot_ylim
        self.plot_zlim = plot_zlim
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection="3d")

    @property
    def phi(self):
        return np.array([0., 2.*pi/3., 4.*pi/3.])

    @property
    def oc(self):
        # ==================
        # oc [3, 3]
        # |  oc1_x  |  oc2_x  |  oc3_x  |
        # |  oc1_y  |  oc2_y  |  oc3_y  |
        # |  oc1_z  |  oc2_z  |  oc3_z  |
        # ==================
        oc = np.zeros([3, 3], dtype=np.float64)
        oc[0] = np.cos(self.phi) * self.r1
        oc[1] = np.sin(self.phi) * self.r1
        return oc

    @property
    def cb(self):
        # ==================
        # cb [3, 3]
        # |  cb1_x  |  cb2_x  |  cb3_x  |
        # |  cb1_y  |  cb2_y  |  cb3_y  |
        # |  cb1_z  |  cb2_z  |  cb3_z  |
        # ==================
        cb = np.zeros([3, 3], dtype=np.float64)
        cb[0] = np.cos(self.phi) * np.cos(self.theta) * self.l1
        cb[1] = np.sin(self.phi) * np.cos(self.theta) * self.l1
        cb[2] = - np.sin(self.theta) * self.l1
        return cb

    @property
    def ap(self):
        # ==================
        # bd [3, 3]
        # |  bd1_x  |  bd2_x  |  bd3_x  |
        # |  bd1_y  |  bd2_y  |  bd3_y  |
        # |  bd1_z  |  bd2_z  |  bd3_z  |
        # ==================
        ap = np.zeros([3, 3], dtype=np.float64)
        ap[0] = - np.cos(self.phi) * self.r2
        ap[1] = - np.sin(self.phi) * self.r2
        return ap

    @property  # ap = bd
    def bd(self):
        return self.ap

    @property
    def od(self):
        return self.oc + self.cb + self.bd

    @property
    def op(self):
        # solve for circle centroid
        od = self.od
        temp_p = np.ones([4, 4], dtype=np.float64)
        temp_p[1:4, 0:3] = od.T
        temp_a = np.zeros([3, 3])
        temp_y = np.zeros([3, 1])

        temp_a[0, 0] = np.linalg.det(np.delete(np.delete(temp_p, 0, axis=0), 0, axis=1))
        temp_a[0, 1] = - np.linalg.det(np.delete(np.delete(temp_p, 0, axis=0), 1, axis=1))
        temp_a[0, 2] = np.linalg.det(np.delete(np.delete(temp_p, 0, axis=0), 2, axis=1))
        temp_y[0, 0] = - np.linalg.det(od)

        temp_a[1, 0] = 2 * (od[0, 1] - od[0, 0])
        temp_a[1, 1] = 2 * (od[1, 1] - od[1, 0])
        temp_a[1, 2] = 2 * (od[2, 1] - od[2, 0])
        temp_y[1, 0] = np.linalg.norm(od[:, 0]) ** 2 - np.linalg.norm(od[:, 1]) ** 2

        temp_a[2, 0] = 2 * (od[0, 2] - od[0, 0])
        temp_a[2, 1] = 2 * (od[1, 2] - od[1, 0])
        temp_a[2, 2] = 2 * (od[2, 2] - od[2, 0])
        temp_y[2, 0] = np.linalg.norm(od[:, 0]) ** 2 - np.linalg.norm(od[:, 2]) ** 2

        oe = - np.linalg.inv(temp_a).dot(temp_y)
        r = np.linalg.norm(oe - od[:, 0:1])
        # r2 = np.linalg.norm(oe - od[:, 1:2])
        # r3 = np.linalg.norm(oe - od[:, 2:3])
        # return r, r2, r3
        if r > self.l2:
            logging.error("Pose cannot be reached!")
            return False, oe  # False : not reachable
        else:
            vec = np.cross(od[:, 2]-od[:, 1], od[:, 2]-od[:, 0])
            # if vec[2] > 0:
            #     print(vec)
            vec = vec / np.linalg.norm(vec) * np.sqrt(self.l2*self.l2 - r*r)
            op = oe + vec.reshape([3, 1])
            return True, op  # True : reachable

    def forward(self, theta):
        self.theta = theta
        return self.op

    def plot_settings(self):
        self.ax.set_xlim(self.plot_xlim)
        self.ax.set_ylim(self.plot_ylim)
        self.ax.set_zlim(self.plot_zlim)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    def draw(self):
        self.ax.clear()
        self.plot_settings()
        # get point coordinates
        oc = self.oc  # [3, 3]
        ob = oc + self.cb  # [3, 3]
        op = self.op[1]  # [3, 1]
        oa = op - self.ap  # [3, 3] bd = ap
        # plot top frame
        x, y, z = oc[0].tolist(), oc[1].tolist(), oc[2].tolist()
        x.append(x[0])
        y.append(y[0])
        z.append(z[0])
        self.ax.plot_wireframe(x, y, np.array([z]))
        # plot bottom frame
        x, y, z = oa[0].tolist(), oa[1].tolist(), oa[2].tolist()
        x.append(x[0])
        y.append(y[0])
        z.append(z[0])
        self.ax.plot_wireframe(x, y, np.array([z]))
        # plot three arms
        for i in range(3):
            x = [oc[0, i], ob[0, i], oa[0, i]]
            y = [oc[1, i], ob[1, i], oa[1, i]]
            z = [oc[2, i], ob[2, i], oa[2, i]]
            self.ax.plot_wireframe(x, y, np.array([z]))
        # plot two dots
        self.ax.scatter([0.], [0.], [0.], c="red", marker="o")
        self.ax.scatter(op[0], op[1], op[2], c="red", marker="o")

    def show(self):
        self.draw()
        plt.show()
