from visual_kinematics.Robot import *
from visual_kinematics.utility import simplify_angle
from math import pi, atan2, sqrt

import logging


class RobotDelta(Robot):
    # params [4, ] [r1, r2, l1, l2]
    def __init__(self, params, plot_xlim=[-0.5, 0.5], plot_ylim=[-0.5, 0.5], plot_zlim=[-1.0, 0.0],
                 ws_lim=None, ws_division=5):
        Robot.__init__(self, params, np.zeros([3, ]), plot_xlim, plot_ylim, plot_zlim, ws_lim, ws_division)
        self.is_reachable_forward = True

    # ================== Definition of r1, r2, l1, l2
    #                  r1
    #              C ========== -------
    #            l1 //   O    \\ / theta
    #              //          \\
    #            B \\---D      //
    #            l2 \\        //
    #                \\  P   //
    #                A ======
    #                  r2
    # ==================
    @property
    def r1(self):
        return self.params[0]

    @property
    def r2(self):
        return self.params[1]

    @property
    def l1(self):
        return self.params[2]

    @property
    def l2(self):
        return self.params[3]

    # ================== Definition of phi [3, ]
    #                  \
    #                   \  phi[1] = 2/3*pi
    #                    \
    #                      -----------  phi[0] = 0
    #                    /
    #   phi[2] = 4/3*pi /
    #                  /
    # ==================
    @property
    def phi(self):
        return np.array([0., 2.*pi/3., 4.*pi/3.])

    # ================== Definition of oc [3, 3]
    # |  oc1_x  |  oc2_x  |  oc3_x  |
    # |  oc1_y  |  oc2_y  |  oc3_y  |
    # |  oc1_z  |  oc2_z  |  oc3_z  |
    # ==================
    @property
    def oc(self):
        oc = np.zeros([3, 3], dtype=np.float64)
        oc[0] = np.cos(self.phi) * self.r1
        oc[1] = np.sin(self.phi) * self.r1
        return oc

    # ================== Definition of cb [3, 3]
    # |  cb1_x  |  cb2_x  |  cb3_x  |
    # |  cb1_y  |  cb2_y  |  cb3_y  |
    # |  cb1_z  |  cb2_z  |  cb3_z  |
    # ==================
    @property
    def cb(self):
        cb = np.zeros([3, 3], dtype=np.float64)
        cb[0] = np.cos(self.phi) * np.cos(self.axis_values) * self.l1
        cb[1] = np.sin(self.phi) * np.cos(self.axis_values) * self.l1
        cb[2] = - np.sin(self.axis_values) * self.l1
        return cb

    # ================== Definition of ap [3, 3]
    # |  ap1_x  |  ap2_x  |  ap3_x  |
    # |  ap1_y  |  ap2_y  |  ap3_y  |
    # |  ap1_z  |  ap2_z  |  ap3_z  |
    # ==================
    @property
    def ap(self):
        ap = np.zeros([3, 3], dtype=np.float64)
        ap[0] = - np.cos(self.phi) * self.r2
        ap[1] = - np.sin(self.phi) * self.r2
        return ap

    # ================== bd = ap [3 ,3]
    @property
    def bd(self):
        return self.ap

    # ================== od [3 ,3]
    @property
    def od(self):
        return self.oc + self.cb + self.bd

    # ================== op (is_reachable, [3, 1])
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
        if r > self.l2:
            logging.error("Pose cannot be reached!")
            self.is_reachable_inverse = False
            return oe  # False : not reachable
        else:
            vec = np.cross(od[:, 2]-od[:, 1], od[:, 2]-od[:, 0])
            vec = vec / np.linalg.norm(vec) * np.sqrt(self.l2*self.l2 - r*r)
            op = oe + vec.reshape([3, 1])
            self.is_reachable_inverse = True
            return op  # True : reachable

    @property
    def end_frame(self):
        return Frame.from_r_3_3(np.eye(3, dtype=np.float64), self.op)

    def inverse(self, end):
        op = end.t_3_1.flatten()
        # solve a*sinx + b*cosx = c
        a = 2 * self.l1 * op[2]
        theta = np.zeros([3, ], dtype=np.float64)
        self.is_reachable_inverse = True
        for i in range(3):
            oa = op - self.ap[:, i]
            b = 2 * self.l1 * (np.cos(self.phi[i]) * (self.r1 * np.cos(self.phi[i]) - oa[0])
                               + np.sin(self.phi[i]) * (self.r1 * np.sin(self.phi[i]) - oa[1]))
            c = self.l2 * self.l2 - self.l1 * self.l1 - np.linalg.norm(oa) ** 2 \
                - self.r1 * self.r1 + 2 * self.r1 * (np.cos(self.phi[i]) * oa[0] + np.sin(self.phi[i]) * oa[1])
            if a*a + b*b > c*c:
                theta[i] = simplify_angle(atan2(c, -sqrt(a*a+b*b-c*c)) - atan2(b, a))
            else:
                self.is_reachable_inverse = False
                break
        if not self.is_reachable_inverse:
            logging.error("Pose cannot be reached!")
        self.forward(theta)
        return theta

    def draw(self):
        self.ax.clear()
        self.plot_settings()
        # get point coordinates
        oc = self.oc  # [3, 3]
        ob = oc + self.cb  # [3, 3]
        op = self.op  # [3, 1]
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
