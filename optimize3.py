import numpy as np
from scipy import optimize


def fun(x):
    return (-1) * (550 * x[0] + 600 * x[1] + 400 * x[3] + 200 * x[4]) + 350 * x[2]


