import numpy as np
import globals_
from graph import *
import kinematics
import graphictools

def formation(n, agent_type="dot") -> object:
    graph = Graph(n, agent_type)
    return graph

def __reach_consensus(formation, ksi) -> list:
    trajectories = []

    q_current = formation.state
    q_0 = q_current
    trajectories.append(q_current)

    norm = np.linalg.norm(q_current)
    norm_prev = -norm

    dist = np.sum(norm)
    t = globals_.DT

    s1 = False
    s2 = False

    while norm != norm_prev:
        q_dot = np.matmul(-formation.laplacian, q_current - ksi)

        if formation.type == "2SR":
            u = []
            q_dot_ik = []

            for i in range(formation.n):

                jacobian = kinematics.hybrid_jacobian(q_0[i], q_current[i], [s1, s2])
                u_i = np.matmul(np.linalg.pinv(jacobian), q_dot[i, :])

                u.append(u_i)
                q_dot_ik.append(np.matmul(jacobian, u_i))

            q_current = q_current + np.array(q_dot_ik) * t * globals_.DT
        else:
            q_current = q_current + q_dot * t * globals_.DT

        trajectories.append(q_current)

        norm_prev = norm
        norm = np.linalg.norm(q_current)
        t += globals_.DT

    return trajectories

def rendezvous(formation) -> list:

    if not formation.is_connected:
        formation.cycle()

    ksi = np.zeros((formation.n, formation.config_dim))
    trajectories = __reach_consensus(formation, ksi)

    return trajectories

def form_circle(formation, R) -> list:

    if not formation.is_cycle or formation.is_complete:
        formation.cycle()

    order = formation.sort_nodes_by_angles().argsort()
    theta = np.array([2*np.pi / formation.n * x - np.pi/2 for x in range(formation.n)])
    theta = theta[order]
    ksi = R * np.column_stack((np.cos(theta),np.sin(theta)))

    if formation.type == "oriented":
        ksi = np.column_stack((ksi, theta))
    elif formation.type == "2SR":
        ksi = np.column_stack((ksi, theta))
        ksi = np.column_stack((ksi, np.zeros((formation.n, 2))))

    trajectories = __reach_consensus(formation, ksi)

    return trajectories

def form_regular_polygon(formation):
    if not formation.is_connected:
        formation.min_rigid()

def show_motion(formation, trajectories):

    graphictools.anim(formation, trajectories)


