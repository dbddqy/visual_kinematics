#!/usr/bin/env python3

from Robot import *
from math import pi, sqrt, atan2


np.set_printoptions(precision=3, suppress=True)

dh_params = np.array([[0.163,  0.,      0.,     0.5*pi],
                      [0.,     0.5*pi,  0.632,  pi],
                      [0.,     0.,      0.6005, pi],
                      [0.2013, -0.5*pi, 0.,     -0.5*pi],
                      [0.1025, 0.,      0.,     0.5*pi],
                      [0.094,  0.,      0.,     0.]])

robot = Robot(dh_params)


# =====================================
# forward
# =====================================

def solve_forward(theta):
    print("-------forward-------")
    print("axis values:")
    print(theta * 180.0 / pi)
    # add initial offset
    theta[1] += 0.5 * pi
    theta[3] += -0.5 * pi
    f = robot.forward(theta)
    print("end frame xyz:")
    print(f.t_3_1.reshape([3, ]))
    print("end frame abc:")
    print(f.euler_3)
    print("")
    robot.show()


# solve_forward(np.array([0., 0., 0.5*pi, 0., 0., 0.]))
# solve_forward(np.array([10.*pi/180., 10.*pi/180., 90.*pi/180., 10.*pi/180., 10.*pi/180., 10.*pi/180.]))
# solve_forward(np.array([20.*pi/180., 20.*pi/180., 90.*pi/180., 20.*pi/180., 20.*pi/180., 20.*pi/180.]))
# solve_forward(np.array([30.*pi/180., 30.*pi/180., 90.*pi/180., 30.*pi/180., 30.*pi/180., 30.*pi/180.]))
# solve_forward(np.array([40.*pi/180., 40.*pi/180., 90.*pi/180., 40.*pi/180., 40.*pi/180., 40.*pi/180.]))


# =====================================
# inverse
# analytical solution from https://blog.csdn.net/l1216766050/article/details/96961989
# =====================================

def analytics_aubo10(dh_params, f):
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
        b = f[0, 2]*sin(theta_all[i, 0]) - f[1, 2]*cos(theta_all[i, 0])
        if i % 4 == 0 or i % 4 == 1:
            theta_all[i, 4] = atan2(sqrt(1 - b * b), b)
        else:
            theta_all[i, 4] = atan2(-sqrt(1 - b * b), b)
        # check singularity
        if abs(sin(theta_all[i, 4])) < 1e-6:
            print("singularity!!")
            return np.zeros([8, 6])

        # theta_5
        A6 = (f[0, 1]*sin(theta_all[i, 0]) - f[1, 1]*cos(theta_all[i, 0])) / sin(theta_all[i, 4])
        B6 = (f[1, 0]*cos(theta_all[i, 0]) - f[0, 0]*sin(theta_all[i, 0])) / sin(theta_all[i, 4])
        theta_all[i, 5] = atan2(A6, B6)

        # theta_1 : 2 sets of solutions
        A234 = f[2, 2] / sin(theta_all[i, 4])
        B234 = (f[0, 2]*cos(theta_all[i, 0]) + f[1, 2]*sin(theta_all[i, 0])) / sin(theta_all[i, 4])
        M = dh_params[4, 0] * A234 - dh_params[5, 0] * sin(theta_all[i, 4]) * B234 + f[0, 3] * cos(theta_all[i, 0]) + f[1, 3] * sin(theta_all[i, 0])
        N = -dh_params[4, 0] * B234 - dh_params[5, 0] * sin(theta_all[i, 4]) * A234 - dh_params[0, 0] + f[2, 3]
        L = (M*M+N*N+dh_params[1, 2]*dh_params[1, 2]-dh_params[2, 2]*dh_params[2, 2]) / (2*dh_params[1, 2])
        if i % 2 == 0:
            theta_all[i, 1] = atan2(N, M) - atan2(L, sqrt(M * M + N * N - L * L))
        else:
            theta_all[i, 1] = atan2(N, M) - atan2(L, -sqrt(M * M + N * N - L * L))

        # theta_2 and theta_3
        A23 = (-M-dh_params[1, 2]*sin(theta_all[i, 1])) / dh_params[2, 2]
        B23 = (N-dh_params[1, 2]*cos(theta_all[i, 1])) / dh_params[2, 2]
        theta_all[i, 2] = theta_all[i, 1] - atan2(A23, B23)
        theta_all[i, 3] = atan2(A234, B234) - atan2(A23, B23)

        # add initial offset
        theta_all[i, 1] += 0.5 * pi
        theta_all[i, 3] += -0.5 * pi

        # modify theta into (-pi, pi)
        for j in range(6):
            if theta_all[i, j] < -pi:
                theta_all[i, j] += 2. * pi
            if theta_all[i, j] > pi:
                theta_all[i, j] -= 2. * pi
    return theta_all


def solve_inverse(xyz, abc):
    print("-------inverse-------")
    print("end frame xyz:")
    print(xyz)
    print("end frame abc:")
    print(abc.reshape([3, ]))
    end = Frame.from_euler_3(xyz, abc)
    theta_all = robot.inverse_analytical(end, method=analytics_aubo10)
    for i in range(theta_all.shape[0]):
        print("axis values of solution %d: " % i)
        print(theta_all[i] * 180.0 / pi)
        robot.forward(theta_all[i])
        robot.show()


# solve_inverse(np.array([0.5 * pi, 0., pi]), np.array([[0.28127], [0.], [1.13182]]))
# solve_inverse(np.array([0.5 * pi, 0., pi]), np.array([[0.41323], [0.], [1.00785]]))
# solve_inverse(np.array([0.5 * pi, 0., pi]), np.array([[0.54519], [0.], [0.88387]]))
# solve_inverse(np.array([0.5 * pi, 0., pi]), np.array([[0.67715], [0.], [0.7599]]))
solve_inverse(np.array([0.5 * pi, 0., pi]), np.array([[0.8091], [0.], [0.63592]]))
