import numpy as np
import sa_kinematics as kinematics
import graphics
import random as rnd
import globals_
import pandas as pd
from scipy.optimize import minimize

class Control:

    def __init__(self, q_start):
        self.q_start = q_start  # initial configuration

    def getCircleTargetPoints(self, scale):
        FULL_LEGTH = globals_.L_LINK + 2 * (globals_.L_VSS + globals_.L_LU)
        R = scale * 3 / (2 * np.pi) * FULL_LEGTH

        centroid_x = 0
        centroid_y = 0

        for q_0 in self.q_start:
            centroid_x += q_0[0]
            centroid_y += q_0[1]

        centroid = [centroid_x/len(self.q_start), centroid_y/len(self.q_start)]

        cons = ({'type': 'eq', 'fun': lambda p_t:  np.linalg.norm(np.array([p_t[4],p_t[5]]) - np.array([p_t[0],p_t[1]]))- 3/np.sqrt(3) * R},
            {'type': 'eq', 'fun': lambda p_t: np.linalg.norm(np.array([p_t[0],p_t[1]]) - np.array([p_t[2],p_t[3]])) - 3/np.sqrt(3) * R},
            {'type': 'eq', 'fun': lambda p_t:  np.linalg.norm(centroid - np.array([p_t[0],p_t[1]])) - R},
            {'type': 'eq', 'fun': lambda p_t:  np.linalg.norm(centroid - np.array([p_t[2],p_t[3]])) - R},
            {'type': 'eq', 'fun': lambda p_t:  np.linalg.norm(centroid - np.array([p_t[4],p_t[5]])) - R})

        p_t_guess = []
        for i in range(3):
            p_t_guess.append(centroid[0] + R * np.cos(2*np.pi/3*i))
            p_t_guess.append(centroid[1] + R * np.sin(2*np.pi/3*i))

        res = minimize(self.fun, p_t_guess, constraints=cons)
        points = np.reshape(res.x, (-1, 2))

        print(centroid)
        print(points)

        delta_x = points[:,0] - centroid[0]
        delta_y = points[:,1] - centroid[1]
        angles = np.divide(-delta_x, delta_y, out=np.zeros_like(delta_x) + np.pi/2, where=delta_y!=0)
        print(angles)

        return points

    def fun(self, p_t):
        p_t = np.reshape(p_t, (-1, 2))
        f = 0
        for i in range(len(self.q_start)):
            f += np.linalg.norm(self.q_start[i][:2] - p_t[i,:])

        return f


    def shapeFormationPlanner(self, target_points):

        return



if __name__ == "__main__":

    # SIMULATION PARAMETERS

    sim_time = 10  # simulation time
    t = np.arange(globals_.DT, sim_time + globals_.DT, globals_.DT)  # span

    # INITIAL SWARM CONFIGURATION

    n = 3 # number of agents

    q_start = []
    for i in range(n):
        q_start.append([rnd.uniform(-0.3, 0.3), rnd.uniform(-0.3, 0.3), rnd.uniform(-0.5, 0.5),
                   0, 0])

    s_start = [[0, 0], [0, 0], [0, 0]]

    # TARGET SWARM SHAPE PARAMETERS
    scale = 1

    # EXECUTE SHAPE FORMATION

    control = Control(q_start)
    target_points = control.getCircleTargetPoints(scale)

    config_states = [q_start]
    stiffness_states = [s_start]


    # Animation of the 2SRR motion towards the target
    graphics.plotMotion(config_states, stiffness_states, scale, target_points, scale)
