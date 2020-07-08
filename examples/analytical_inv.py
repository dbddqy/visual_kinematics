#!/usr/bin/env python3

from visual_kinematics import Robot, Frame
import numpy as np
from math import pi, sqrt, sin, cos, atan2


def main():
    np.set_printoptions(precision=3, suppress=True)

    dh_params = np.array([[0.163, 0., 0., 0.5 * pi],
                          [0., 0.5 * pi, 0.632, pi],
                          [0., 0., 0.6005, pi],
                          [0.2013, -0.5 * pi, 0., -0.5 * pi],
                          [0.1025, 0., 0., 0.5 * pi],
                          [0.094, 0., 0., 0.]])

    def aubo10_inv(dh_params, f):
        # save all 8 sets of solutions
        theta_all = np.zeros([8, 6])
        for i in range(8):
            A1 = dh_params[5, 0] * f[1, 2] - f[1, 3]
            B1 = f[0, 3] - dh_params[5, 0] * f[0, 2]
            C1 = dh_params[3, 0]
            # theta_0 : 2 sets of solutions
            if i < 4:
                theta_all[i, 0] = atan2(C1, sqrt(A1 * A1 + B1 * B1 - C1 * C1)) - atan2(A1, B1)
            else:
                theta_all[i, 0] = atan2(C1, -sqrt(A1 * A1 + B1 * B1 - C1 * C1)) - atan2(A1, B1)

            # theta_4 : 2 sets of solutions
            b = f[0, 2] * sin(theta_all[i, 0]) - f[1, 2] * cos(theta_all[i, 0])
            if i % 4 == 0 or i % 4 == 1:
                theta_all[i, 4] = atan2(sqrt(1 - b * b), b)
            else:
                theta_all[i, 4] = atan2(-sqrt(1 - b * b), b)
            # check singularity
            if abs(sin(theta_all[i, 4])) < 1e-6:
                print("singularity!!")
                return np.zeros([8, 6])

            # theta_5
            A6 = (f[0, 1] * sin(theta_all[i, 0]) - f[1, 1] * cos(theta_all[i, 0])) / sin(theta_all[i, 4])
            B6 = (f[1, 0] * cos(theta_all[i, 0]) - f[0, 0] * sin(theta_all[i, 0])) / sin(theta_all[i, 4])
            theta_all[i, 5] = atan2(A6, B6)

            # theta_1 : 2 sets of solutions
            A234 = f[2, 2] / sin(theta_all[i, 4])
            B234 = (f[0, 2] * cos(theta_all[i, 0]) + f[1, 2] * sin(theta_all[i, 0])) / sin(theta_all[i, 4])
            M = dh_params[4, 0] * A234 - dh_params[5, 0] * sin(theta_all[i, 4]) * B234 + f[0, 3] * cos(
                theta_all[i, 0]) + f[1, 3] * sin(theta_all[i, 0])
            N = -dh_params[4, 0] * B234 - dh_params[5, 0] * sin(theta_all[i, 4]) * A234 - dh_params[0, 0] + f[2, 3]
            L = (M * M + N * N + dh_params[1, 2] * dh_params[1, 2] - dh_params[2, 2] * dh_params[2, 2]) / (
                        2 * dh_params[1, 2])
            if i % 2 == 0:
                theta_all[i, 1] = atan2(N, M) - atan2(L, sqrt(M * M + N * N - L * L))
            else:
                theta_all[i, 1] = atan2(N, M) - atan2(L, -sqrt(M * M + N * N - L * L))

            # theta_2 and theta_3
            A23 = (-M - dh_params[1, 2] * sin(theta_all[i, 1])) / dh_params[2, 2]
            B23 = (N - dh_params[1, 2] * cos(theta_all[i, 1])) / dh_params[2, 2]
            theta_all[i, 2] = theta_all[i, 1] - atan2(A23, B23)
            theta_all[i, 3] = atan2(A234, B234) - atan2(A23, B23)

        # select the best solution
        diff_sum_min = 1e+5
        index = 0
        for i in range(8):
            diff_sum = 0
            for j in range(6):
                diff = theta_all[i, j] - dh_params[j, 1]
                while diff < -pi:
                    diff += 2. * pi
                while diff > pi:
                    diff -= 2. * pi
                diff_sum += abs(diff)
            if diff_sum < diff_sum_min:
                diff_sum_min = diff_sum
                index = i
        return theta_all[i]

    robot = Robot(dh_params, analytical_inv=aubo10_inv)

    # =====================================
    # trajectory
    # =====================================

    trajectory = []
    trajectory.append(Frame.from_euler_3(np.array([0.5*pi, 0., pi]), np.array([[0.28127], [0.], [1.13182]])))
    trajectory.append(Frame.from_euler_3(np.array([0.25*pi, 0., 0.75*pi]), np.array([[0.48127], [0.], [1.13182]])))
    trajectory.append(Frame.from_euler_3(np.array([0.5 * pi, 0., pi]), np.array([[0.48127], [0.], [0.63182]])))

    robot.show_trajectory(trajectory, motion="lin")


if __name__ == "__main__":
    main()
