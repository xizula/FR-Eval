import numpy as np
from scipy.optimize import brentq


def difference(x_value, x_array, y1_array, y2_array):
    y1_interp = np.interp(x_value, x_array, y1_array)
    y2_interp = np.interp(x_value, x_array, y2_array)
    return y1_interp - y2_interp


def calculate_eer(x, y1, y2):
    intersection_x = brentq(difference, x[0], x[-1], args=(x, y1, y2))
    intersection_y = np.interp(intersection_x, x, y1)

    return intersection_x, intersection_y