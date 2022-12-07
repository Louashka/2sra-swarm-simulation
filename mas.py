import numpy as np
import globals_
from graph import *
import graphictools

def formation(n, agent_type="dot"):
    graph = Graph(n, agent_type)
    return graph

def __reach_consensus(formation, ksi):
    trajectories = []

    q_current = formation.state
    trajectories.append(q_current)

    norm = np.linalg.norm(q_current)
    norm_prev = -norm

    dist = np.sum(norm)
    t = globals_.DT

    while norm != norm_prev:
        q_dot = np.matmul(-formation.laplacian, q_current - ksi)
        q_current = q_current + q_dot * t * globals_.DT
        trajectories.append(q_current)

        norm_prev = norm
        norm = np.linalg.norm(q_current)
        t += globals_.DT

    return trajectories

def rendezvous(formation):

    if not formation.is_connected:
        formation.cycle()

    ksi = np.zeros((formation.n, formation.config_dim))
    trajectories = __reach_consensus(formation, ksi)

    return trajectories

def form_circle(formation, R):

    if not formation.is_cycle or formation.is_complete:
        formation.cycle()

    order = formation.sort_nodes_by_angles().argsort()
    theta = np.array([2*np.pi / formation.n * x - np.pi for x in range(formation.n)])
    theta = theta[order]
    ksi = R * np.column_stack((np.cos(theta),np.sin(theta)))

    if formation.type == "oriented":
        ksi = np.column_stack((ksi, theta))

    trajectories = __reach_consensus(formation, ksi)

    return trajectories

def show_motion(formation, trajectories):

    graphictools.anim(formation, trajectories)


