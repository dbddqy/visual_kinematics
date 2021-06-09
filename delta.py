#!/usr/bin/env python3

from visual_kinematics import DeltaRobot, Frame
import numpy as np
from math import pi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider


def main():
    np.set_printoptions(precision=3, suppress=True)

    delta = DeltaRobot.DeltaRobot(0.2, 0.2, 0.25, 0.6)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(delta.plot_xlim)
    ax.set_ylim(delta.plot_ylim)
    ax.set_zlim(delta.plot_zlim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    divide = 10
    upper_bound = 1 * pi / 2
    lower_bound = - pi / 24
    interval = (upper_bound - lower_bound) / divide

    x, y, z = [], [], []
    for i in range(divide+1):
        # if i != 0 and i != divide - 1:
        #     continue
        for j in range(divide+1):
            for k in range(divide+1):
                is_reachable, end = delta.forward(interval * np.array([i, j, k], dtype=np.float64))
                if is_reachable:
                    x.append(end[0, 0])
                    y.append(end[1, 0])
                    z.append(end[2, 0])
                else:
                    print("i: %d, j: %d, k:%d" % (i, j, k))

                if end[2, 0] > 0:
                    print("i: %d, j: %d, k:%d" % (i, j, k))

    ax.scatter(x, y, z, c="red", marker="o")
    # plt.show()

    delta.forward(interval * np.array([0, 9, 10], dtype=np.float64))
    delta.show()

    # print(np.linalg.norm(np.array([1, 1, 1])) ** 2)
    # a = np.arange(0, 16).reshape([4, 4])
    # print(a - np.ones([4, 1]))
    # temp = np.delete(np.delete(a, 0, axis=0), 1, axis=1)
    # print(temp)


if __name__ == "__main__":
    main()
