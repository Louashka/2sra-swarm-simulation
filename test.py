import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.interpolate import UnivariateSpline

def get_r(a, b, theta):
    e = np.sqrt(1 - (b**2) / (a**2))
    r = b / np.sqrt(1 - (e * np.cos(theta))**2)

    return r

def func(var, *data):
    x1, y1, x2, y2, r = data
    eq1 = (x1 - var[0])**2 + (y1 - var[1])**2 - r**2
    eq2 = (x2 - var[0])**2 + (y2 - var[1])**2 - r**2

    return [eq1, eq2]

def get_curvature(x, y=None, error=0.1):
    if y is None:
        x, y = x.real, x.imag

    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))

    xˈ = fx.derivative(1)(t)
    xˈˈ = fx.derivative(2)(t)
    yˈ = fy.derivative(1)(t)
    yˈˈ = fy.derivative(2)(t)
    curvature = (xˈ* yˈˈ - yˈ* xˈˈ) / np.power(xˈ** 2 + yˈ** 2, 3 / 2)
    return curvature


centre = [0, 0]
a = 1
b = 0.6
theta = np.linspace(0, 2 * np.pi, 50)

contour = [a * np.cos(theta), b * np.sin(theta)]
k = get_curvature(contour[0], contour[1])

start = 20
end = 30

theta1 = theta[start]
theta2 = theta[end]

# r1 = get_r(a, b, theta1)
# r2 = get_r(a, b, theta2)

# r_avrg = np.abs(r1 + r1) / 2

k_avrg = np.average(k[start:end+1])
r_avrg = 1 / np.abs(k_avrg)

x1, y1 = [a * np.cos(theta1), b * np.sin(theta1)]
x2, y2 = [a * np.cos(theta2), b * np.sin(theta2)]

delta_theta = theta2 - theta1

result = opt.fsolve(func, (0, 0), args = (x1, y1, x2, y2, r_avrg))
contour_new = [result[0] + r_avrg * np.cos(theta), result[1] + r_avrg * np.sin(theta)]


plt.plot(contour[0], contour[1])
plt.plot(contour_new[0], contour_new[1], '--')
plt.plot(x1, y1, 'ro')
plt.plot(x2, y2, 'ro')

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

plt.show()
