import logging

import numpy as np

from visual_kinematics.Frame import Frame
from visual_kinematics.Robot import Robot
from visual_kinematics.utility import simplify_angles


class RobotSerial(Robot):
    # ==================
    # dh_params [n, 4]
    # |  d  |  a  |  alpha  |  theta  |
    # |  x  |  x  |    x    |    x    |
    # |  x  |  x  |    x    |    x    |
    # | ... | ... |   ...   |   ...   |
    # params [n, 3] (without theta)       initial_offset [n, ]
    # |  d  |  a  |  alpha  |                  |  theta  |
    # |  x  |  x  |    x    |                  |    x    |
    # |  x  |  x  |    x    |                  |    x    |
    # | ... | ... |   ...   |                  |   ...   |
    # ==================
    def __init__(self, dh_params, dh_type="normal", tool=None, analytical_inv=None, plot_xlim=(-0.5, 0.5),
                 plot_ylim=(-0.5, 0.5),  plot_zlim=(0, 1), ws_lim=None, ws_division=5, inv_m="jac_pinv",
                 step_size=5e-1, max_iter=300, final_loss=1e-4):
        super().__init__(params=dh_params[:, 0:3], initial_offset=dh_params[:, 3], tool=tool, plot_xlim=plot_xlim,
                         plot_ylim=plot_ylim, plot_zlim=plot_zlim, ws_lim=ws_lim, ws_division=ws_division)
        self.dh_type = dh_type  # can be only normal or modified
        if dh_type != "normal" and dh_type != "modified":
            raise Exception("dh_type can only be \"normal\" or \"modified\"")
        self.analytical_inv = analytical_inv
        # inverse settings
        self.inv_m = inv_m
        if inv_m != "jac_t" and inv_m != "jac_pinv":
            raise Exception("Motion type can only be \"jac_t\" or \"jac_inv\"!")
        self.step_size = step_size
        self.max_iter = max_iter
        self.final_loss = final_loss

    @property
    def dh_params(self):
        return np.hstack((self.params, (self.axis_values + self.initial_offset).reshape([self.num_axis, 1])))

    # transformation between axes
    @property
    def ts(self):
        dh = self.dh_params
        ts = []
        for i in range(self.num_axis):
            if self.dh_type == "normal":
                ts.append(Frame.from_dh(dh[i]))
            else:
                ts.append(Frame.from_dh_modified(dh[i]))
        if self.tool is not None:
            ts.append(self.tool)
        return ts

    # base to end transformation
    @property
    def axis_frames(self):
        ts = self.ts
        fs = []
        f = Frame.i_4_4()
        for i in range(self.num_axis):
            f = f * ts[i]
            fs.append(f)
        if self.tool is not None:
            f = f * self.tool
            fs.append(f)
        return fs

    @property
    def end_frame(self):
        return self.axis_frames[-1]

    @property
    def jacobian(self):
        axis_fs = self.axis_frames
        n_rows = self.num_axis + 1 if self.tool is not None else self.num_axis
        jac = np.zeros([6, n_rows])
        if self.dh_type == "normal":
            jac[0:3, 0] = np.cross(np.array([0., 0., 1.]), axis_fs[-1].t_3_1.reshape([3, ]))
            jac[3:6, 0] = np.array([0., 0., 1.])
            for i in range(1, n_rows):
                jac[0:3, i] = np.cross(axis_fs[i - 1].z_3_1.reshape([3, ]),
                                       (axis_fs[-1].t_3_1 - axis_fs[i - 1].t_3_1).reshape([3, ]))
                jac[3:6, i] = axis_fs[i - 1].z_3_1.reshape([3, ])
        if self.dh_type == "modified":
            for i in range(0, n_rows):
                jac[0:3, i] = np.cross(axis_fs[i].z_3_1.reshape([3, ]),
                                       (axis_fs[-1].t_3_1 - axis_fs[i].t_3_1).reshape([3, ]))
                jac[3:6, i] = axis_fs[i].z_3_1.reshape([3, ])
        return jac

    def inverse(self, end_frame):
        if self.analytical_inv is not None:
            return self.inverse_analytical(end_frame, self.analytical_inv)
        else:
            return self.inverse_numerical(end_frame)

    def inverse_analytical(self, end_frame, method):
        if self.tool is None:
            self.is_reachable_inverse, theta_x = method(self.dh_params, end_frame)
        else:
            self.is_reachable_inverse, theta_x = method(self.dh_params, self.tool, end_frame)

        self.forward(theta_x)
        return theta_x

    def inverse_numerical(self, end_frame):
        last_dx = np.zeros([6, 1])
        for _ in range(self.max_iter):
            if self.inv_m == "jac_t":
                jac = self.jacobian.T
            else:
                jac = np.linalg.pinv(self.jacobian)
            end = self.end_frame
            dx = np.zeros([6, 1])
            dx[0:3, 0] = (end_frame.t_3_1 - end.t_3_1).reshape([3, ])
            diff = end.inv * end_frame
            dx[3:6, 0] = end.r_3_3.dot(diff.r_3.reshape([3, 1])).reshape([3, ])
            if np.linalg.norm(dx, ord=2) < self.final_loss or np.linalg.norm(dx - last_dx,
                                                                             ord=2) < 0.1 * self.final_loss:
                self.axis_values = simplify_angles(self.axis_values)
                self.is_reachable_inverse = True
                return self.axis_values
            dq = self.step_size * jac.dot(dx)
            self.forward(self.axis_values + dq.reshape([self.num_axis, ]))
            last_dx = dx
        logging.error("Pose cannot be reached!")
        self.is_reachable_inverse = False

    def draw(self, cylinder_relative_size: float = 0.008, orientation_relative_size: float = 0.05):
        self.ax.clear()

        # plot the arm
        x, y, z = [0.], [0.], [0.]
        axis_frames = self.axis_frames
        n_lines = self.num_axis + 1 if self.tool is not None else self.num_axis
        for i in range(n_lines):
            x.append(axis_frames[i].t_3_1[0, 0])
            y.append(axis_frames[i].t_3_1[1, 0])
            z.append(axis_frames[i].t_3_1[2, 0])
        if self.tool is None:
            self.ax.plot_wireframe(x, y, np.array([z]))
            self.ax.scatter(x[1:], y[1:], z[1:], c="red", marker="o")
        else:
            self.ax.plot_wireframe(x[:-1], y[:-1], np.array([z[:-1]]))
            self.ax.scatter(x[1:-1], y[1:-1], z[1:-1], c="red", marker="o")
            self.ax.plot_wireframe(x[-2:], y[-2:], np.array([z[-2:]]), color="orange")
            self.ax.scatter(x[-1], y[-1], z[-1], c="orange", marker="o")

        # configure plot dimensions dynamically with already printed points and get plot_size
        plot_size = self.plot_settings()

        # plot axes using cylinders
        if cylinder_relative_size > 0:
            cy_radius = plot_size * cylinder_relative_size
            cy_len = cy_radius * 7.
            cy_div = 11
            theta = np.linspace(0, 2 * np.pi, cy_div)
            cx = np.array([cy_radius * np.cos(theta)])
            cz = np.array([-0.5 * cy_len, 0.5 * cy_len])
            cx, cz = np.meshgrid(cx, cz)
            cy = np.array([cy_radius * np.sin(theta)] * 2)
            points = np.zeros([3, cy_div * 2])
            points[0] = cx.flatten()
            points[1] = cy.flatten()
            points[2] = cz.flatten()
            self.ax.plot_surface(points[0].reshape(2, cy_div), points[1].reshape(2, cy_div), points[2].reshape(2, cy_div),
                                 color="pink", rstride=1, cstride=1, linewidth=0, alpha=0.6)
            for i in range(n_lines - 1):
                f = axis_frames[i]
                points_f = f.r_3_3.dot(points) + f.t_3_1
                self.ax.plot_surface(points_f[0].reshape(2, cy_div),
                                     points_f[1].reshape(2, cy_div),
                                     points_f[2].reshape(2, cy_div),
                                     color="pink", rstride=1, cstride=1, linewidth=0, alpha=0.6)

        # plot the end frame
        if orientation_relative_size > 0:
            end_frame_scale = plot_size * orientation_relative_size
            end_pos = axis_frames[-1].t_3_1.flatten()
            end_rot = axis_frames[-1].r_3_3
            rotated_x_axis = end_rot[:, 0]
            rotated_y_axis = end_rot[:, 1]
            rotated_z_axis = end_rot[:, 2]
            self.ax.plot(*np.array([end_pos, end_pos + end_frame_scale * rotated_x_axis]).T.tolist(), color="red")
            self.ax.plot(*np.array([end_pos, end_pos + end_frame_scale * rotated_y_axis]).T.tolist(), color="green")
            self.ax.plot(*np.array([end_pos, end_pos + end_frame_scale * rotated_z_axis]).T.tolist(), color="blue")

