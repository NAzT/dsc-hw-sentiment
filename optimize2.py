import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
from scipy import optimize


def fun(x):
    return (- 1) * (550 * x[0] + 600 * x[1] + 350 * x[2] + 400 * x[3] + 200 * x[4])


def con1(x):
    return 288 - (12 * x[0] + 20 * x[1] + 0 * x[2] + 25 * x[3] + 15 * x[4])


def con2(x):
    return 192 - (10 * x[0] + 8 * x[1] + 16 * x[2] + 0 * x[3] + 0 * x[4])


def con3(x):
    return 384 - (20 * x[0] + 20 * x[1] + 20 * x[2] + 20 * x[3] + 20 * x[4])


x0 = [0, 0, 0, 0, 0]

con = [{'type': 'ineq', 'fun': con1}, {'type': 'ineq', 'fun': con2}, {'type': 'ineq', 'fun': con3}]
result = optimize.minimize(fun, x0, constraints=con)

print(result)
