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

def create_shape():
    obj = mp.Manipulandum()
    path = generate_path(obj)

    store_data(obj, path)

    return obj, path

def get_shape(restore=False):
    if restore:
        try:
            return load_data()
        except (OSError, IOError) as e:
            return create_shape()
    else:
        return create_shape()

def oap(obj, path, w_span):
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
        else:
            print("Restoration failed! Start again!")
            shape_status = False

        return shape_status, s_data, L_data, q_data


def experiment(shape_id):
    w_span = np.linspace(0, 0.5, 51)
    df = pd.DataFrame()

    obj, path = get_shape()
    shape_status, s_data, L_data, q_data = oap(obj, path, w_span)

    if shape_status:
        id_data = [shape_id] * len(w_span)
        path_data = [path.tolist()] * len(w_span)

        data = {"id": id_data, "target_pose": path_data, "pose": q_data, "force": L_data, "contact_locations": s_data, "weight": w_span}
        df = pd.DataFrame(data)
        df.to_csv('weights_experiment.csv', index=False, mode='a')

        print("Shape id: " + str(shape_id))
        print("Data is saved!")

        if shape_id < 30:
            experiment(shape_id + 1)

    else:
        xperiment(shape_id)


# experiment(1)

obj, path = get_shape()
shape_status, s, L, q = oap(obj, path, [0.28])

while True:
    if shape_status:
        break
    else:
        obj, path = get_shape()
        shape_status, s, L, q = oap(obj, path, [0.28])


# # PLOT THE RESULT

graphics.plot_motion(swarm, cp_n, obj, path, q[0], s[0])




