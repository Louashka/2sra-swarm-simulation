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
from pyomo.opt import SolverStatus, TerminationCondition


# ORIGINAL SHAPE

vsf = 0.02
n_robots = 2
swarm = Graph(vsf, n_robots)
cp_n = 3

def generate_path(obj):
    path = []

    dt = 0.25
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

def experiment(shape_id):

    # w_span = np.linspace(0, 0.2, 21)
    # w_span = np.linspace(0.21, 0.5, 30)
    w_span = np.linspace(0, 0.5, 51)
    # w_span = [0.02]

    df = pd.DataFrame()

    # try:
    #     obj, path = load_data()
    # except (OSError, IOError) as e:
    #     obj = mp.Manipulandum()
    #     path = generate_path(obj)

    #     store_data(obj, path)

    obj = mp.Manipulandum()
    path = generate_path(obj)
    shape_status = True

    grasp_model = ori.GraspModel(swarm, cp_n, obj, path, [0.5, 0.5, 0])

    s_data = []
    L_data = []
    q_data = []

    for w3 in w_span:
        print(str(w3) + " ...")
        w = [(1-w3)/2, (1-w3)/2, w3]

        grasp_model.update(w)
        solution_status = grasp_model.solve()

        if solution_status:
            results, s, L, q = grasp_model.parse_results()

            if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
                s_data.append(s)
                L_data.append(L)
                q_data.append(q)
                print("Success!")
            else:
                print("Terminated! Start again!")
                shape_status = False
                experiment(shape_id)
        else:
            print("Restoration failed! Start again!")
            shape_status = False
            experiment(shape_id)

    if shape_status:
        id_data = [shape_id] * len(w_span)
        path_data = [path.tolist()] * len(w_span)

        data = {"id": id_data, "target_pose": path_data, "pose": q_data, "force": L_data, "contact_locations": s_data, "weight": w_span}
        df = pd.DataFrame(data)
        df.to_csv('weights_experiment.csv', index=False, mode='a')

        print("Data is saved!")

        if shape_id < 30:
            print("Shape id: " + str(shape_id))
            experiment(shape_id + 1)


experiment(1)
# w_span = np.linspace(0, 0.5, 51)
# print(w_span)


# try:
#     obj, path = load_data()
# except (OSError, IOError) as e:
#     obj = mp.Manipulandum()
#     path = generate_path(obj)

#     store_data(obj, path)

# print(path.shape)


# # OPTIMISATION PROBLEM

# vsf = 0.02
# n = 2
# swarm = Graph(vsf, n)

# cp_n = 3

# # s = []
# # L = []
# # q = []

# # q.append(q_0)
# # s = [0] * n * cp_n

# q_current = path[0,:]
# s_current = [0] * n * cp_n

# w = [0.5, 0.5, 0]

# grasp_model = ori.GraspModel(swarm, cp_n, obj, path, w)
# res, s, L, q = grasp_model.solve()

# # for q_d in path[1:,:]:
# #     grasp_model.update(q_current, q_d, s_current)
# #     result = grasp_model.solve()

# #     s_current = result[0]
# #     q_current = result[2]

# #     s.append(s_current)
# #     L.append(result[1])
# #     q.append(q_current)


# # PLOT THE RESULT
# # print(q)

# graphics.plot_motion(swarm, cp_n, obj, path, q, s)




