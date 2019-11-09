import numpy as np
# from scipy.optimize import minimize
import scipy.optimize as optimize
import matplotlib.pyplot as plt


#
#
def parabola(x):
    return (x - 6) * (x - 6)


x0 = [0.0]

result = optimize.minimize_scalar(parabola, bounds=(2, 4), method='bounded')
print(result)

#
#
# myX = np.arange(0, 10, 0.0001)
# myY = func(x)
# myConfiguration = 1
# myMin = func(myConfiguration)
# for i in myX:
#     if myMin > func(i):
#         myMin = func(i)
#         myConfiguration = i
# print("The best solution is", myConfiguration, " with value of ", myMin)
