import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from graph import Graph
import mas
import ori
import graphics
import manipulandum as mp
import pickle
import pandas as pd


# ORIGINAL SHAPE

def generate_path(obj):
    path = []

    dt = 0.05
    t = 0
    velocity = [0, rnd.uniform(0.5, 1), rnd.uniform(-1, 1)]

    q_current = np.append(obj.com, obj.phi)
    path.append(q_current)

    flag1 = True
    flag2 = True

    while  t < 6:

        if t > 2 and flag1:
            velocity[2] = rnd.uniform(-1, 1)
            flag1 = False
        elif t > 4 and flag2:
            velocity[2] = rnd.uniform(-1, 1)
            flag2 = False

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

def experiment():

    n_robots = 2
    swarm = Graph(n_robots)
    cp_n = 3
    vsf = 0.02

    n_shapes = 1
    w_span = np.linspace(0, 0.2, 21)
    # w_span = [0.05, 0.1, 0.15]
    shape_id = 10

    df = pd.DataFrame()

    id_data = []
    time_data = []
    path_data = []
    q_data = []
    L_data = []
    s_data = []
    w_data = []

    for n in range(n_shapes):

        obj = mp.Manipulandum()
        path = generate_path(obj)

        for w3 in w_span:
            q_current = path[0,:]
            s_current = [0] * swarm.n * cp_n
            w = [1, 1, w3]

            grasp_model = ori.GraspModel(swarm, cp_n, obj, vsf, q_current, path[1,:], s_current, w)

            # s = []
            # L = []
            # q = []
            t = 1

            for q_d in path[1:,:]:
                grasp_model.update(q_current, q_d, s_current)
                result = grasp_model.solve()

                s_current = result[0]
                q_current = result[2]

                # s.append(s_current)
                # L.append(result[1])
                # q.append(q_current)

                id_data.append(shape_id)
                time_data.append(t)
                path_data.append(q_d.tolist())
                q_data.append(q_current)
                L_data.append(result[1])
                s_data.append(s_current)
                w_data.append(w3)

                t += 1

        shape_id += 1

    data = {"id": id_data, "time": time_data, "target_pose": path_data, "pose": q_data, "force": L_data, "contact_locations": s_data, "weight": w_data}
    df = pd.DataFrame(data)
    df.to_csv('weights_experiment.csv', index=False, mode='a')


experiment()
# w_span = np.linspace(0, 0.4, 41)
# print(w_span)


# try:
#     obj, path = load_data()
# except (OSError, IOError) as e:
#     obj = mp.Manipulandum()
#     path = generate_path(obj)

#     store_data(obj, path)


# # OPTIMISATION PROBLEM

# n = 2
# cp_n = 3
# swarm = Graph(n)

# vsf = 0.02

# s = []
# L = []
# q = []

# # q.append(q_0)
# # s = [0] * n * cp_n

# q_current = path[0,:]
# s_current = [0] * n * cp_n

# w = [1, 1, 0]

# grasp_model = ori.GraspModel(swarm, cp_n, obj, vsf, q_current, path[1,:], s_current, w)

# for q_d in path[1:,:]:
#     grasp_model.update(q_current, q_d, s_current)
#     result = grasp_model.solve()

#     s_current = result[0]
#     q_current = result[2]

#     s.append(s_current)
#     L.append(result[1])
#     q.append(q_current)


# # PLOT THE RESULT

# graphics.plot_motion(swarm, cp_n, obj, path[::5], q, s)




