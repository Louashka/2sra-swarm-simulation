import shape
from pyefd import elliptic_fourier_descriptors, plot_efd
import cv2 as cv
import numpy as np

class Manipulandum(object):

    def __init__(self) -> None:
        self.__contour = self.__generate_contour()
        self.__phi = 0

    def __generate_contour(self):
        contour_original = shape.get_bezier_curve()

        contour_original = np.array(contour_original[:2])
        contour_original -= self.__com(contour_original)

        m = 10
        coeffs = elliptic_fourier_descriptors(contour_original.T, order=m)

        z = []
        s_array = np.linspace(0, 1, 500)
        for i in range(m):
            z_k = np.zeros((len(s_array),2))
            j = 0
            coef = coeffs[i,:].reshape(2, 2)
            for s in s_array:
                arg = 2 * np.pi * (i + 1) * s
                exp = np.array([[np.cos(arg)], [np.sin(arg)]])
                z_k[j, :] = np.matmul(coef, exp).T
                j += 1
            z.append(z_k)

        contour_fourier = sum(z).T
        contour_fourier -= self.__com(contour_fourier)

        self.__com = self.__com(contour_fourier)

        return contour_fourier

    def __com(self, contour):
        points = []
        for i in range(contour[0].shape[0]):
            x = int(contour[0, i] * 1000)
            y = int(contour[1, i] * 1000)

            point = [x, y]
            points.append([point])

        M = cv.moments(np.array(points))
        x = int(M["m10"] / M["m00"]) / 1000
        y = int(M["m01"] / M["m00"]) / 1000

        com = np.array([[x, y]]).T

        return com

    def __assign_frame(self):
        R = 0.1
        alpha = rnd.uniform(0, 2 * np.pi)

        x_dx = R * np.cos(alpha)
        x_dy = R * np.sin(alpha)

        beta = (alpha + np.pi / 2) % (2 * np.pi)

        y_dx = R * np.cos(beta)
        y_dy = R * np.sin(beta)

        return [x_dx, x_dy], [y_dx, y_dy]

    @property
    def com(self) -> list:
        return self.__com

    @property
    def phi(self) -> float:
        return self.__phi

    @property
    def contour(self) -> object:
        return self.__contour


