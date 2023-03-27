import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from graph import Graph
import mas
import ori
import graphics


# ORIGINAL SHAPE

centre = [0, 0]
a = 1
b = 0.6
theta = np.linspace(0, 2 * np.pi, 50)

contour = [a * np.cos(theta), b * np.sin(theta)]

axis_r = 0.2
alpha = 0

x_axis = [axis_r * np.cos(alpha), axis_r * np.sin(alpha)]
y_axis = [axis_r * np.cos(alpha + np.pi / 2), axis_r * np.sin(alpha + np.pi / 2)]

# GENERATE A PATH

dt = 0.05
t = 0

q_0 = np.array(centre + [alpha])

path = []
q_current = q_0
path.append(q_current)

velocity = [0, 1, -0.3]
# velocity = [rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-0.8, 0.8)]
# print(velocity)

while  t < 3:
    R = np.array([[np.cos(q_current[2]), -np.sin(q_current[2]), 0], [np.sin(q_current[2]), np.cos(q_current[2]), 0], [0, 0, 1]])

    q_dot = R.dot(velocity)

    q_current = q_current + q_dot * dt

    path.append(q_current)
    t += dt

centre_d = [q_current[0], q_current[1]]
alpha_d = q_current[2]

x_axis_d = [axis_r * np.cos(alpha_d), axis_r * np.sin(alpha_d)]
y_axis_d = [axis_r * np.cos(alpha_d + np.pi / 2), axis_r * np.sin(alpha_d + np.pi / 2)]

R_d = np.array([[np.cos(alpha_d), -np.sin(alpha_d)], [np.sin(alpha_d), np.cos(alpha_d)]])
contour_d = np.array([centre_d]).T + R_d.dot(np.array(contour))

path = np.array(path)
# print(path)

# OPTIMISATION PROBLEM

n = 2
cp_n = 3
swarm = Graph(n)

theta = []
L = []
q = []
flags = []

q_current = q_0

grasp_model = ori.GraspModel(swarm, cp_n, a, b, q_current, path[1,:])

for q_d in path[1:,:]:
    grasp_model.update(q_current, q_d)
    result = grasp_model.solve()

    q_current = result[2]

    theta.append(result[0])
    L.append(result[1])
    q.append(q_current)
    # flags.append(result[3])

# for l in L:
#     print([l[0], l[1]])

# for flag, th in zip(flags, theta):
#     print([abs(th[0] - th[1]), abs(th[1] - th[2]), abs(th[0] - th[2])])
#     print(flag)

for th in theta:
    print(th)
#     print(D.dot(th))

# PLOT THE RESULT

graphics.plot_motion(swarm, cp_n, a, b, path[::5], q, theta)


# tr = mas.rendezvous(swarm)
# tr = mas.form_regular_polygon(swarm, 0.15)
# mas.form_regular_polygon(swarm)
# print(swarm.all_edges)
# mas.show_motion(swarm, tr)




