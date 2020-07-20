from visual_kinematics.Frame import *
import matplotlib.pyplot as plt
from math import pi
# from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

import logging


class Robot:
    # ==================
    # dh_params [n, 4]
    # |  d  |  theta  |  a  |  alpha  |
    # |  x  |    x    |  x  |    x    |
    # |  x  |    x    |  x  |    x    |
    # | ... |   ...   | ... |   ...   |
    # ==================
    def __init__(self, dh_params, dh_type="normal", analytical_inv=None
                 , plot_xlim=[-0.5, 0.5], plot_ylim=[-0.5, 0.5], plot_zlim=[0.0, 1.0]
                 , inv_m = "jac_pinv", step_size=5e-1, max_iter=300, final_loss=1e-4):
        self.dh_params = dh_params
        self.dh_type = dh_type # can be only normal or modified
        if dh_type != "normal" and dh_type != "modified":
            raise Exception("dh_type can only be \"normal\" or \"modified\"")
        self.initial_offset = self.dh_params[:, 1].copy()
        self.analytical_inv = analytical_inv
        # plot related
        self.plot_xlim = plot_xlim
        self.plot_ylim = plot_ylim
        self.plot_zlim = plot_zlim
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection="3d")
        # inverse settings
        self.inv_m = inv_m
        if inv_m != "jac_t" and inv_m != "jac_pinv":
            raise Exception("Motion type can only be \"jac_t\" or \"jac_inv\"!")
        self.step_size = step_size
        self.max_iter = max_iter
        self.final_loss = final_loss

    @property
    def num_axis(self):
        return self.dh_params.shape[0]

    @property
    def axis_values(self):
        return self.dh_params[:, 1] - self.initial_offset

    # transformation between axes
    @property
    def ts(self):
        ts = []
        for i in range(self.num_axis):
            if self.dh_type == "normal":
                ts.append(Frame.from_dh(self.dh_params[i]))
            else:
                ts.append(Frame.from_dh_modified(self.dh_params[i]))
        return ts

    # base to end transformation
    @property
    def axis_frames(self):
        ts = self.ts
        fs = []
        f = Frame.i_4_4()
        for i in range(self.num_axis):
            f = f * ts[i]
            fs.append(f.copy)
        return fs

    @property
    def end_frame(self):
        return self.axis_frames[-1]

    @property
    def jacobian(self):
        axis_fs = self.axis_frames
        jac = np.zeros([6, self.num_axis])
        if self.dh_type == "normal":
            jac[0:3, 0] = np.cross(np.array([0., 0., 1.]), axis_fs[-1].t_3_1.reshape([3, ]))
            jac[3:6, 0] = np.array([0., 0., 1.])
            for i in range(1, self.num_axis):
                jac[0:3, i] = np.cross(axis_fs[i-1].z_3_1.reshape([3, ]), (axis_fs[-1].t_3_1 - axis_fs[i-1].t_3_1).reshape([3, ]))
                jac[3:6, i] = axis_fs[i-1].z_3_1.reshape([3, ])
        if self.dh_type == "modified":
            for i in range(0, self.num_axis):
                jac[0:3, i] = np.cross(axis_fs[i].z_3_1.reshape([3, ]), (axis_fs[-1].t_3_1 - axis_fs[i].t_3_1).reshape([3, ]))
                jac[3:6, i] = axis_fs[i].z_3_1.reshape([3, ])
        return jac

    def forward(self, theta_x):
        self.dh_params[:, 1] = theta_x + self.initial_offset
        return self.end_frame

    def inverse(self, end_frame):
        if self.analytical_inv is not None:
            return self.inverse_analytical(end_frame, self.analytical_inv)
        else:
            return self.inverse_numerical(end_frame)

    def inverse_analytical(self, end_frame, method):
        theta_x = method(self.dh_params, end_frame)
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
            if np.linalg.norm(dx, ord=2) < self.final_loss or np.linalg.norm(dx - last_dx, ord=2) < 0.1*self.final_loss:
                for i in range(self.num_axis):
                    while self.dh_params[i, 1] > pi:
                        self.dh_params[i, 1] -= 2*pi
                    while self.dh_params[i, 1] < -pi:
                        self.dh_params[i, 1] += 2*pi
                return self.axis_values
            dq = self.step_size * jac.dot(dx)
            self.forward(self.axis_values + dq.reshape([self.num_axis, ]))
            last_dx = dx
        logging.error("Pose cannot be reached!")

    # ===============
    # trajectory
    # ===============

    def interpolate(self, tra, num_segs, motion="p2p", method="linear"):
        if len(tra) < 2:
            raise Exception("Please set_trajectory properly!")
        # interpolation
        # !!! currently use linear interpolation
        # !!! currently based on only the length of each segment, future work will take orientation also into consideration
        # !!! currently all segment must share same method
        length = []
        length_total = 0.
        for i in range(len(tra) - 1):
            length.append(tra[i].distance_to(tra[i + 1]))
            length_total += length[i]

        # axis angles for p2p, xyzabc for lin
        tra_array = np.zeros([len(tra), self.num_axis])
        for i in range(len(tra)):
            if motion == "p2p":
                self.inverse(tra[i])
                tra_array[i] = self.axis_values
            if motion == "lin":
                tra_array[i, 0:3] = np.array(tra[i].t_3_1.reshape([3, ]))
                tra_array[i, 3:6] = np.array(tra[i].euler_3)

        # interpolation values
        inter_values = np.zeros([num_segs + 1, self.num_axis])
        for progress in range(num_segs + 1):
            index = 0
            p_temp = progress * length_total / num_segs
            for i in range(len(length)):
                if p_temp - length[i] > 1e-5:  # prevent numerical error
                    p_temp -= length[i]
                    index += 1
                else:
                    break
            p_temp /= length[index]
            if motion == "p2p":
                inter_values[progress] = tra_array[index] * (1-p_temp) + tra_array[index+1] * p_temp
            if motion == "lin":
                xyzabc = tra_array[index] * (1-p_temp) + tra_array[index+1] * p_temp
                self.inverse(Frame.from_euler_3(xyzabc[3:6], xyzabc[0:3].reshape([3, 1])))
                inter_values[progress] = self.axis_values
        return inter_values

    # ===============
    # plot related
    # ===============

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
        # plot the arm
        x, y, z = [0.], [0.], [0.]
        axis_frames = self.axis_frames
        for i in range(self.num_axis):
            x.append(axis_frames[i].t_3_1[0, 0])
            y.append(axis_frames[i].t_3_1[1, 0])
            z.append(axis_frames[i].t_3_1[2, 0])
        self.ax.plot_wireframe(x, y, np.array([z]))
        self.ax.scatter(x[1:], y[1:], z[1:], c="red", marker="o")
        # plot the end frame
        f = axis_frames[-1].t_4_4
        self.ax.plot_wireframe(np.array([f[0, 3], f[0, 3] + 0.2 * f[0, 0]]),
                               np.array([f[1, 3], f[1, 3] + 0.2 * f[1, 0]]),
                               np.array([[f[2, 3], f[2, 3] + 0.2 * f[2, 0]]]), color="red")
        self.ax.plot_wireframe(np.array([f[0, 3], f[0, 3] + 0.2 * f[0, 1]]),
                               np.array([f[1, 3], f[1, 3] + 0.2 * f[1, 1]]),
                               np.array([[f[2, 3], f[2, 3] + 0.2 * f[2, 1]]]), color="green")
        self.ax.plot_wireframe(np.array([f[0, 3], f[0, 3] + 0.2 * f[0, 2]]),
                               np.array([f[1, 3], f[1, 3] + 0.2 * f[1, 2]]),
                               np.array([[f[2, 3], f[2, 3] + 0.2 * f[2, 2]]]), color="blue")

    def show(self):
        self.draw()
        plt.show()

    def show_trajectory(self, tra, num_segs=100, motion="p2p", method="linear"):
        # setup slider
        axamp = plt.axes([0.15, .06, 0.75, 0.02])
        samp = Slider(axamp, "progress", 0., 1., valinit=0)
        # interpolation values
        inter_values = self.interpolate(tra, num_segs, motion, method)
        # save point for drawing trajectory
        x, y, z = [], [], []
        for i in range(num_segs+1):
            self.forward(inter_values[i])
            x.append(self.end_frame.t_3_1[0, 0])
            y.append(self.end_frame.t_3_1[1, 0])
            z.append(self.end_frame.t_3_1[2, 0])
        def update(val):
            self.forward(inter_values[int(np.floor(samp.val * num_segs))])
            self.draw()
            # plot trajectory
            self.ax.plot_wireframe(x, y, np.array([z]), color="lightblue")
            self.figure.canvas.draw_idle()
        samp.on_changed(update)
        # plot initial
        self.forward(inter_values[0])
        self.draw()
        self.ax.plot_wireframe(x, y, np.array([z]), color="lightblue")
        plt.show()
        return inter_values
