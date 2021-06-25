from visual_kinematics.Frame import *
from numpy import pi
from abc import abstractmethod
import matplotlib.pyplot as plt


class Robot(object):
    # ================== Definition and Kinematics
    # params: according to definition of specific robots
    # initial_offset: always [dim, ]
    # ws_lim: lower and upper bound of all axes [num_axis, 2]
    # ws_division: number of sample points of all axes
    # ==================
    def __init__(self, params, initial_offset, plot_xlim=[-0.5, 0.5], plot_ylim=[-0.5, 0.5], plot_zlim=[0.0, 1.0],
                 ws_lim=None, ws_division=5):
        self.params = params
        self.initial_offset = initial_offset
        self.axis_values = np.zeros(initial_offset.shape, dtype=np.float64)
        # is_reachable_inverse must be set everytime when inverse kinematics is performed
        self.is_reachable_inverse = True
        # plot related
        self.plot_xlim = plot_xlim
        self.plot_ylim = plot_ylim
        self.plot_zlim = plot_zlim
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection="3d")
        # workspace related
        if ws_lim is None:
            self.ws_lim = np.array([[-pi, pi]]*self.num_axis)
        else:
            self.ws_lim = ws_lim
        self.ws_division = ws_division

    @property
    def num_axis(self):
        return self.initial_offset.shape[0]

    @property
    @abstractmethod
    def end_frame(self):
        pass

    # ================== Jacobian [num_axis, 6]
    @property
    @abstractmethod
    def jacobian(self):
        pass

    def forward(self, theta_x):
        self.axis_values = theta_x
        return self.end_frame

    @abstractmethod
    def inverse(self, end_frame):
        pass

    # ================== Workspace analysis
    # sample through all possible points
    # ==================
    def workspace(self):
        num_points = self.ws_division ** self.num_axis
        lower = self.ws_lim[:, 0]
        intervals = (self.ws_lim[:, 1] - self.ws_lim[:, 0]) / (self.ws_division - 1)
        points = np.zeros([num_points, 3])
        axes_indices = np.zeros([self.num_axis, ], dtype=np.int32)
        for i in range(num_points):
            points[i] = self.forward(lower + axes_indices*intervals).t_3_1.flatten()
            axes_indices[0] += 1
            for check_index in range(self.num_axis):
                if axes_indices[check_index] >= self.ws_division:
                    if check_index >= self.num_axis-1:
                        break
                    axes_indices[check_index] = 0
                    axes_indices[check_index+1] += 1
                else:
                    break
        return points

    # ================== plot related
    # ==================
    def plot_settings(self):
        self.ax.set_xlim(self.plot_xlim)
        self.ax.set_ylim(self.plot_ylim)
        self.ax.set_zlim(self.plot_zlim)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    @abstractmethod
    def draw(self):
        pass

    def draw_ws(self):
        self.plot_settings()
        points = self.workspace()
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="red", marker="o")

    def show(self, body=True, ws=False):
        if body:
            self.draw()
        if ws:
            self.draw_ws()
        plt.show()
