from visual_kinematics.Frame import *
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

import logging


class Robot:
    # ==================
    # dh_params [6, 4]
    # |  d  |  theta  |  a  |  alpha  |
    # |  x  |    x    |  x  |    x    |
    # |  x  |    x    |  x  |    x    |
    # | ... |   ...   | ... |   ...   |
    # ==================
    def __init__(self, dh_params, analytical_inv=None):
        self.dh_params = dh_params
        self.initial_offset = self.dh_params[:, 1:2].copy()
        self.analytical_inv = analytical_inv
        # plot related
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection="3d")

    @property
    def axis_values(self):
        return self.dh_params[:, 1:2].reshape([6, ]) - self.initial_offset.reshape([6, ])

    # transformation between axes
    @property
    def ts(self):
        ts = []
        for i in range(6):
            ts.append(Frame.from_dh(self.dh_params[i]))
        return ts

    # base to end transformation
    @property
    def axis_frames(self):
        ts = self.ts
        fs = []
        f = Frame.i_4_4()
        for i in range(6):
            f = f * ts[i]
            fs.append(f.copy)
        return fs

    @property
    def end_frame(self):
        return self.axis_frames[-1]

    def forward(self, theta_6):
        self.dh_params[:, 1:2] = theta_6.reshape([6, 1]) + self.initial_offset
        return self.end_frame

    def inverse(self, end_frame):
        if self.analytical_inv is not None:
            return self.inverse_analytical(end_frame, self.analytical_inv)
        else:
            return self.inverse_numerical(end_frame)

    def inverse_analytical(self, end_frame, method):
        theta_6 = method(self.dh_params, end_frame)
        return theta_6

    def inverse_numerical(self, end_frame):
        theta_6 = self.axis_values
        ls = least_squares(Robot.cost_inverse, theta_6, args=(self, end_frame))
        if ls.cost > 0.001:
            logging.error("The pose is out of robot's reach!!!")

    @staticmethod
    def cost_inverse(x, robot, end_frame):
        end = robot.forward(x)
        return (end.t_4_4 - end_frame.t_4_4).reshape([16, ])

    # ===============
    # plot related
    # ===============
    def plot_settings(self):
        self.ax.set_xlim([-1.0, 1.0])
        self.ax.set_ylim([-1.0, 1.0])
        self.ax.set_zlim([-1.0, 1.0])
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    def draw(self):
        self.ax.clear()
        self.plot_settings()
        # plot the arm
        x, y, z = [0.], [0.], [0.]
        axis_frames = self.axis_frames
        for i in range(6):
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

    def show_trajectory(self, tra, motion="p2p"):
        if motion != "p2p" and motion != "lin":
            logging.warning("Motion type can only be \"p2p\" or \"lin\"!")
        if len(tra) < 2:
            logging.warning("Please give at least 2 frames for the trajectory!")

        # setup slider
        axamp = plt.axes([0.15, .06, 0.75, 0.02])
        samp = Slider(axamp, "progress", 0., 1., valinit=0)

        # interpolation
        # !!! currently use linear interpolation
        # !!! currently based on only the length of each segment, future work will take orientation also into consideration
        # !!! currently all segment must share same method

        tra_array = np.zeros([len(tra), 6])
        # axis angles for p2p, xyzabc for lin
        for i in range(len(tra)):
            if motion == "p2p":
                self.inverse_numerical(tra[i])
                tra_array[i] = self.axis_values
            if motion == "lin":
                tra_array[i, 0:3] = np.array(tra[i].t_3_1.reshape([3, ]))
                tra_array[i, 3:6] = np.array(tra[i].euler_3)

        length = []
        length_total = 0.
        for i in range(len(tra)-1):
            length.append(tra[i].distance_to(tra[i+1]))
            length_total += length[i]

        def move_to_interval(progress):
            index = 0
            p_temp = progress
            for i in range(len(length)):
                if p_temp > length[i]:
                    p_temp -= length[i]
                    index += 1
                else:
                    break
            p_temp /= length[index]
            if motion == "p2p":
                self.forward(tra_array[index] * (1-p_temp) + tra_array[index+1] * p_temp)
            if motion == "lin":
                xyzabc = tra_array[index] * (1-p_temp) + tra_array[index+1] * p_temp
                self.inverse_numerical(Frame.from_euler_3(xyzabc[3:6], xyzabc[0:3].reshape([3, 1])))

        # save point for drawing trajectory
        x, y, z = [], [], []
        for i in range(101):
            move_to_interval(i * length_total / 100.)
            x.append(self.end_frame.t_3_1[0, 0])
            y.append(self.end_frame.t_3_1[1, 0])
            z.append(self.end_frame.t_3_1[2, 0])

        def update(val):
            move_to_interval(samp.val * length_total)
            self.draw()
            # plot trajectory
            self.ax.plot_wireframe(x, y, np.array([z]), color="lightblue")
            self.figure.canvas.draw_idle()

        samp.on_changed(update)

        # plot initial
        self.inverse_numerical(tra[0])
        self.draw()
        self.ax.plot_wireframe(x, y, np.array([z]), color="lightblue")
        plt.show()
