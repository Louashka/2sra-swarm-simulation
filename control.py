import numpy as np
import sa_kinematics as kinematics
import graphics
import random as rnd
import globals_
import pandas as pd
from scipy.optimize import minimize
import itertools

class Control:

    def __init__(self, q_start, scale):
        self.q_start = q_start  # initial configuration

        FULL_LEGTH = globals_.L_LINK + 2 * (globals_.L_VSS + globals_.L_LU)
        self.R = scale * 3 / (2 * np.pi) * FULL_LEGTH

        centroid_x = 0
        centroid_y = 0

        for q_0 in q_start:
            centroid_x += q_0[0]
            centroid_y += q_0[1]

        self.centroid = [centroid_x/len(q_start), centroid_y/len(q_start)]

    def getCircleTargetPoints(self):

        cons = ({'type': 'eq', 'fun': lambda p_t:  np.linalg.norm(np.array([p_t[4],p_t[5]]) - np.array([p_t[0],p_t[1]]))- 3/np.sqrt(3) * self.R},
            {'type': 'eq', 'fun': lambda p_t: np.linalg.norm(np.array([p_t[0],p_t[1]]) - np.array([p_t[2],p_t[3]])) - 3/np.sqrt(3) * self.R},
            {'type': 'eq', 'fun': lambda p_t:  np.linalg.norm(self.centroid - np.array([p_t[0],p_t[1]])) - self.R},
            {'type': 'eq', 'fun': lambda p_t:  np.linalg.norm(self.centroid - np.array([p_t[2],p_t[3]])) - self.R},
            {'type': 'eq', 'fun': lambda p_t:  np.linalg.norm(self.centroid - np.array([p_t[4],p_t[5]])) - self.R})

        p_t_guess = []
        for i in range(3):
            p_t_guess.append(self.centroid[0] + self.R * np.cos(2*np.pi/3*i))
            p_t_guess.append(self.centroid[1] + self.R * np.sin(2*np.pi/3*i))

        res = minimize(self.fun, p_t_guess, constraints=cons)
        points = np.reshape(res.x, (-1, 2))

        ind = list(itertools.permutations([0, 1, 2]))
        min_f = self.fun(points[ind[0],:].flatten().tolist())
        min_i = 0

        for i in range(1, len(ind)):
            f = self.fun(points[ind[i],:].flatten().tolist())
            if f < min_f:
                min_f = f
                min_i = i

        points_sorted = points[ind[min_i],:]

        delta_x = points_sorted[:,0] - self.centroid[0]
        delta_y = points_sorted[:,1] - self.centroid[1]
        angles = np.arctan(np.divide(-delta_x, delta_y, out=np.zeros_like(delta_x), where=delta_y!=0))


        return points_sorted[:,:2], angles

    def fun(self, p_t):
        p_t = np.reshape(p_t, (-1, 2))
        f = 0
        for i in range(len(self.q_start)):
            f += np.linalg.norm(self.q_start[i][:2] - p_t[i,:])

        return f


    def shapeFormationPlanner(self, target_points, target_angles):

        # A set of possible stiffness configurations
        s = [[0, 0], [0, 1], [1, 0], [1, 1]]
        # Initialize a sequence of VSB stiffness values
        s_list = []
        # 2SRR always starts from the rigid state
        s_list.append(s[0])
        # Initialize the number of stiffness transitions
        switch_counter = 0

        # Initialise swarm trajectories
        config_states = []
        stiffness_states = []

        # A set of possible configurations
        q_ = [None] * len(s)
        v_ = [None] * len(s)

        # Define target configuration
        target_angles = np.array([target_angles])
        kappa = 1/self.R * np.ones((3,2)) + 8
        for i in range(3):
            if target_points[i,1] > self.centroid[1]:
                kappa[i,:] = -kappa[i,:]

        q_t = np.concatenate((target_points, target_angles.T, kappa), axis=1)

        # Euclidean distance between current and target configurations (error)
        dist = np.linalg.norm(q_start - q_t, axis=1)

        # feedback gain
        velocity_coeff = np.array([0.8, 0.8, 1, 1, 1])

        for i in range(3):

            t = globals_.DT  # current time
            # Index of the current stiffness configuration
            current_i = None

            # Initialize a trajectory
            q_list = []
            q = self.q_start[i] # current configuration
            q_list.append(q)

            while dist[i] > 0:

                flag = False  # indicates whether VSB stiffness has changed

                # INVERSE KINEMATICS

                q_tilda = (q_t[i,:] - q) * t
                for j in range(len(s)):
                    # Jacobian matrix
                    J = kinematics.hybridJacobian(self.q_start[i], q, s[j])
                    # velocity input commands
                    v_[j] = np.matmul(np.linalg.pinv(J), q_tilda)
                    q_dot = np.matmul(J, v_[j])
                    q_[j] = q + (1 - np.exp(-1 * t)) * q_dot * globals_.DT

                # Determine the stiffness configuration that promotes
                # faster approach to the target
                dist_ = np.linalg.norm(q_ - q_t[i,:], axis=1)
                min_i = np.argmin(dist_)

                # The extent of the configuration change
                delta_q_ = np.linalg.norm(q - np.array(q_), axis=1)

                # Stiffness transition is committed only if the previous
                # stiffness configuration does not promote further motion
                if min_i != current_i and current_i is not None:
                    if delta_q_[current_i] > 10**(-17):
                        min_i = current_i
                    else:
                        flag = True

                q = q_[min_i]  # update current configuration
                dist[i] = np.linalg.norm(q - q_t[i])  # update error
                current_i = min_i  # update current stiffness
                if (delta_q_[current_i] > 10 ** (-5)):
                    q_list.append(q)
                    s_list.append(s[current_i])

                t += globals_.DT  # increment time

            config_states.append(q_list)
            stiffness_states.append(s_list)

        return config_states, stiffness_states



if __name__ == "__main__":

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

    control = Control(q_start, scale)
    target_points, target_angles = control.getCircleTargetPoints()
    config_states, stiffness_states = control.shapeFormationPlanner(target_points, target_angles)


    frames = max(len(config_states[0]), len(config_states[1]), len(config_states[2])) + 50

    for i in range(3):
        config_states[i] += [config_states[i][-1]] * (frames - len(config_states[i]))
        stiffness_states[i] += [stiffness_states[i][-1]] * (frames - len(stiffness_states[i]))


    # Animation of the 2SRR motion towards the target
    graphics.plotMotion(config_states, stiffness_states, scale, target_points, frames)
