import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from graph import Graph
import mas
import ori
import graphics
import manipulandum as mp


# ORIGINAL SHAPE

obj = mp.Manipulandum()

# GENERATE A PATH

dt = 0.05
t = 0

q_0 = np.append(obj.com, obj.phi)

path = []
q_current = q_0
path.append(q_current)

velocity = [0, 0.6, -0.3]
# velocity = [rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-0.8, 0.8)]
# print(velocity)

while  t < 5:
    R = np.array([[np.cos(q_current[2]), -np.sin(q_current[2]), 0], [np.sin(q_current[2]), np.cos(q_current[2]), 0], [0, 0, 1]])

    q_dot = R.dot(velocity)

    q_current = q_current + q_dot * dt

    path.append(q_current)
    t += dt

path = np.array(path)
# print(path)

# OPTIMISATION PROBLEM

n = 2
cp_n = 3
swarm = Graph(n)

vss = 0.04

s = []
L = []
q = []

q.append(q_0)
s.append([0] * n * cp_n)

q_current = q_0

# grasp_model = ori.GraspModel(swarm, cp_n, com_fourier, contour_fourier, vsf, q_current, path[1,:])

# for q_d in path[1:,:]:
#     grasp_model.update(q_current, q_d)
#     result = grasp_model.solve()

#     q_current = result[2]

#     s.append(result[0])
#     L.append(result[1])
#     q.append(q_current)

# for l in L:
#     print([l[0], l[1]])

# for flag, th in zip(flags, theta):
#     print([abs(th[0] - th[1]), abs(th[1] - th[2]), abs(th[0] - th[2])])
#     print(flag)

# for th in theta:
# '[]'    print(th)
#     print(D.dot(th))

# PLOT THE RESULT

graphics.plot_motion(swarm, cp_n, obj, path[::5], q, s)


# tr = mas.rendezvous(swarm)
# tr = mas.form_regular_polygon(swarm, 0.15)
# mas.form_regular_polygon(swarm)
# print(swarm.all_edges)
# mas.show_motion(swarm, tr)




