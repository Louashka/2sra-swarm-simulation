import shape
import cv2 as cv

class Manipulandum(object):

    def __init__(self) -> None:
        self.contour = shape.get_bezier_curve()
        self.com = self.com()

    def com():
        # x = sum(contour[0]) / len(contour[0])
        # y = sum(contour[1]) / len(contour[1])
        points = []
        for i in range(len(self.contour[0])):
            x = int(self.contour[0][i] * 1000)
            y = int(self.contour[1][i] * 1000)

            point = [x, y]
            points.append([point])

        M = cv.moments(np.array(points))
        x = int(M["m10"] / M["m00"]) / 1000
        y = int(M["m01"] / M["m00"]) / 1000

        return [x, y]

    def assign_frame():
        R = 0.1
        alpha = rnd.uniform(0, 2 * np.pi)

        x_dx = R * np.cos(alpha)
        x_dy = R * np.sin(alpha)

        beta = (alpha + np.pi / 2) % (2 * np.pi)

        y_dx = R * np.cos(beta)
        y_dy = R * np.sin(beta)

        return [x_dx, x_dy], [y_dx, y_dy]


# contour = shape.get_bezier_curve()
# centre = com(contour)

# x_axis, y_axis = assign_frame(centre)
