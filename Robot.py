from Frame import *
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.widgets import Slider


class Robot:
    # ==================
    # dh_params [6, 4]
    # |  d  |  theta  |  a  |  alpha  |
    # |  x  |    x    |  x  |    x    |
    # |  x  |    x    |  x  |    x    |
    # | ... |   ...   | ... |   ...   |
    # ==================
    def __init__(self, dh_params):
        self.dh_params = dh_params

    @property
    def axis_values(self):
        return self.dh_params[:, 1:2].reshape([6, ])

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
        self.dh_params[:, 1:2] = theta_6.reshape([6, 1])
        return self.end_frame

    def inverse_analytical(self, end_frame, method):
        theta_6 = method(self.dh_params, end_frame)
        return theta_6

    def inverse_numerical(self, end_frame):
        theta_6 = self.axis_values
        least_squares(Robot.cost_inverse, theta_6, args=(self, end_frame))

    @staticmethod
    def cost_inverse(x, robot, end_frame):
        end = robot.forward(x)
        return (end.t_4_4 - end_frame.t_4_4).reshape([16, ])

    def show(self):
        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-1.0, 1.0])
        ax.set_zlim([-1.0, 1.0])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # plot the arm
        axis_frames = self.axis_frames
        x, y, z = [0.], [0.], [0.]
        for i in range(6):
            x.append(axis_frames[i].t_3_1[0, 0])
            y.append(axis_frames[i].t_3_1[1, 0])
            z.append(axis_frames[i].t_3_1[2, 0])
        ax.plot_wireframe(x, y, np.array([z]))
        ax.scatter(x[1:], y[1:], z[1:], c="red", marker="o")
        # plot the end frame
        f = axis_frames[-1].t_4_4
        ax.plot_wireframe(np.array([f[0, 3], f[0, 3] + 0.2 * f[0, 0]]),
                          np.array([f[1, 3], f[1, 3] + 0.2 * f[1, 0]]),
                          np.array([[f[2, 3], f[2, 3] + 0.2 * f[2, 0]]]), color="red")
        ax.plot_wireframe(np.array([f[0, 3], f[0, 3] + 0.2 * f[0, 1]]),
                          np.array([f[1, 3], f[1, 3] + 0.2 * f[1, 1]]),
                          np.array([[f[2, 3], f[2, 3] + 0.2 * f[2, 1]]]), color="green")
        ax.plot_wireframe(np.array([f[0, 3], f[0, 3] + 0.2 * f[0, 2]]),
                          np.array([f[1, 3], f[1, 3] + 0.2 * f[1, 2]]),
                          np.array([[f[2, 3], f[2, 3] + 0.2 * f[2, 2]]]), color="blue")
        plt.show()
