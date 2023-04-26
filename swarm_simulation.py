import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from graph import Graph
import mas
import ori
import graphics
import manipulandum as mp
import pickle


# ORIGINAL SHAPE

def generate_path(obj):
    path = []

    dt = 0.05
    t = 0
    velocity = [0, 1, 0.8]

    q_current = np.append(obj.com, obj.phi)
    path.append(q_current)
    while  t < 5:

        if t > 2.5:
            velocity[-1] = -0.5

        R = np.array([[np.cos(q_current[2]), -np.sin(q_current[2]), 0], [np.sin(q_current[2]), np.cos(q_current[2]), 0], [0, 0, 1]])

        q_dot = R.dot(velocity)

        q_current = q_current + q_dot * dt

        path.append(q_current)
        t += dt


    return np.array(path)

def load_data():
    db_file = open('manipulandum_data', 'rb')
    obj, path = pickle.load(db_file)

    db_file.close()

    return obj, path

def store_data(obj, path):
    dbfile = open('manipulandum_data', 'wb')
    db = [obj, path]

    pickle.dump(db, dbfile)
    dbfile.close()



try:
    obj, path = load_data()
except (OSError, IOError) as e:
    obj = mp.Manipulandum()
    path = generate_path(obj)

    store_data(obj, path)


# OPTIMISATION PROBLEM

n = 2
cp_n = 3
swarm = Graph(n)

vsf = 0.02

s = []
L = []
q = []

# q.append(q_0)
# s = [0] * n * cp_n

q_current = path[0,:]
s_current = [0] * n * cp_n

grasp_model = ori.GraspModel(swarm, cp_n, obj, vsf, q_current, path[1,:], s_current)

for q_d in path[1:,:]:
    grasp_model.update(q_current, q_d, s_current)
    result = grasp_model.solve()

    s_current = result[0]
    q_current = result[2]

    s.append(s_current)
    L.append(result[1])
    q.append(q_current)


# PLOT THE RESULT

graphics.plot_motion(swarm, cp_n, obj, path[::5], q, s)

# s_array = np.linspace(0, 1, 30)

# plt.plot(obj.contour[0], obj.contour[1])

# r = 0.05

# for s in s_array:
#     p0 = obj.get_point(s)
#     theta = obj.get_x_hat_direc(s)
#     p1 = [p0[0] + r * np.cos(theta), p0[1] + r * np.sin(theta)]

#     plt.plot(p0[0], p0[1], 'k.')
#     plt.plot([p0[0], p1[0]], [p0[1], p1[1]], color='red')

# plt.show()


# tr = mas.rendezvous(swarm)
# tr = mas.form_regular_polygon(swarm, 0.15)
# mas.form_regular_polygon(swarm)
# print(swarm.all_edges)
# mas.show_motion(swarm, tr)




