from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class Robot(object):
    # ================== Definition and Kinematics
    # params: according to definition of specific robots
    # initial_offset: always [dim, ]
    # ws_lim: lower and upper bound of all axes [num_axis, 2]
    # ws_division: number of sample points of all axes
    # ==================
    def __init__(self, params, initial_offset, tool=None, plot_xlim=None, plot_ylim=None,
                 plot_zlim=None, ws_lim=None, ws_division=5):
        self.params = params
        self.initial_offset = initial_offset
        self.axis_values = np.zeros(initial_offset.shape, dtype=np.float64)
        self.tool = tool
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
            self.ws_lim = np.array([[-np.pi, np.pi]]*self.num_axis)
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
        if self.plot_xlim is not None:
            self.ax.set_xlim(self.plot_xlim)
        if self.plot_ylim is not None:
            self.ax.set_ylim(self.plot_ylim)
        if self.plot_zlim is not None:
            self.ax.set_zlim(self.plot_zlim)
        if self.plot_xlim is None and self.plot_ylim is None and self.plot_zlim is None:
            self.set_aspect_equal_3d()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        plot_size = max(np.diff(self.ax.get_xlim3d()), np.diff(self.ax.get_ylim3d()), np.diff(self.ax.get_zlim3d()))[0]

        return plot_size

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

    def set_aspect_equal_3d(self):
        """Fix equal aspect bug for 3D plots."""

        x_lim = self.ax.get_xlim3d()
        y_lim = self.ax.get_ylim3d()
        z_lim = self.ax.get_zlim3d()

        x_mean = np.mean(x_lim)
        y_mean = np.mean(y_lim)
        z_mean = np.mean(z_lim)

        plot_radius = max(
            [abs(lim - mean) for lims, mean in ((x_lim, x_mean), (y_lim, y_mean), (z_lim, z_mean)) for lim in lims]
        )

        self.ax.set_xlim3d([x_mean - plot_radius, x_mean + plot_radius])
        self.ax.set_ylim3d([y_mean - plot_radius, y_mean + plot_radius])
        self.ax.set_zlim3d([z_mean - plot_radius, z_mean + plot_radius])
