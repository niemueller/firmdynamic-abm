import timeit

import numpy as np
import scipy.optimize as opt
import math
import matplotlib

# min(timeit.repeat)
from matplotlib import pyplot as plt


def log_utility(effort, a, b, beta, theta, endowment, effort_others, number_employees):
    return -(theta * np.log(
        (a * (effort + effort_others) + b * (effort + effort_others) ** beta) / number_employees) + (
                     1 - theta) * np.log(endowment - effort))


a1 = 0.3
b1 = 1
beta1 = 1.5
theta1 = 0.5
endowment1 = 1
effort_others1 = 0.4
number_employees1 = 3
params = (a1, b1, beta1, theta1, endowment1, effort_others1, number_employees1)

x = np.arange(0, 1, 0.01)
y = (theta1 * np.log(
    (a1 * (x + effort_others1) + b1 * (x + effort_others1) ** beta1) / number_employees1) + (
             1 - theta1) * np.log(endowment1 - x))

y2 = (((a1 * (x + effort_others1) + b1 * (x + effort_others1) ** beta1) / number_employees1) ** theta1 * (
        endowment1 - x) ** (1 - theta1))

print(x)
print(y2)
plt.plot(x, y2)
plt.show()


def optimization_output(utility):
    return opt.minimize_scalar(utility, bounds=(0, 0.9), args=params, method="bounded")


timeit.timeit(lambda: optimization_output(log_utility))
