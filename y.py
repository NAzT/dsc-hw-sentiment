import numpy as np


def calculateSurface(x, y, z):
    return 2 * x * y + 2 * x * z + 2 * z * y


myX = np.arange(1.0, 100.0, 0.01)
myY = np.arange(1.0, 100.0, 0.01)
myZ = np.arange(1.0, 100.0, 0.01)

myConfigurationX = 10.0
myConfigurationY = 10.0
myConfigurationZ = 10.0
myMin = calculateSurface(myConfigurationX, myConfigurationY, myConfigurationZ)

for i in myX:
    for j in myY:
        for k in myZ:
            if i * j * k == 1000:
                if calculateSurface(i, j, k) < myMin:
                    myMin = calculateSurface(i, j, k)
                    myConfigurationX = i
                    myConfigurationY = j
                    myConfigurationZ = k

print("The best solution is (", myConfigurationX, " ", myConfigurationY, " ", myConfigurationZ, ") with value of ",
      myMin)
