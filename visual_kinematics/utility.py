import numpy as np
from math import pi


# ================== constrain angle between -pi and pi
def simplify_angle(angle):
    while angle > pi:
        angle -= 2 * pi
    while angle < -pi:
        angle += 2 * pi
    return angle


# ================== constrain angles[n, ] between -pi and pi
def simplify_angles(angles):
    for i in range(angles.shape[0]):
        angles[i] = simplify_angle(angles[i])
    return angles
