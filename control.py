import numpy as np
import sa_kinematics as kinematics
import graphics
import random as rnd
import globals_
import pandas as pd

class Control:

    def __init__(self, q_0):
        self.q_0 = q_0  # initial configuration

    def shapeFormation():
        return

    def motionPlanner(self, q_target):
        # A set of possible stiffness configurations
        s = [[0, 0], [0, 1], [1, 0], [1, 1]]
        # Initialize a sequence of VSB stiffness values
        s_list = []
        # 2SRR always starts from the rigid state
        s_list.append(s[0])
        # Initialize the number of stiffness transitions
        switch_counter = 0

        # Initialize a trajectory
        q_list = []
        q = self.q_0  # current configuration
        q_list.append(q)

        # A set of possible configurations
        q_ = [None] * len(s)
        v_ = [None] * len(s)

        q_t = np.array(q_target)
        # Euclidean distance between current and target configurations (error)
        dist = np.linalg.norm(q - q_t)

        t = globals_.DT  # current time
        # feedback gain
        velocity_coeff = np.ones((5,), dtype=int)
        # Index of the current stiffness configuration
        current_i = None

        while dist > 0:

            flag = False  # indicates whether VSB stiffness has changed

            # INVERSE KINEMATICS

            q_tilda = 1 * (q_t - q) * t
            for i in range(len(s)):
                # Jacobian matrix
                J = kinematics.hybridJacobian(self.q_0, q, s[i])
                # velocity input commands
                v_[i] = np.matmul(np.linalg.pinv(J), q_tilda)
                q_dot = np.matmul(J, v_[i])
                q_[i] = q + (1 - np.exp(-1 * t)) * q_dot * globals_.DT

            # Determine the stiffness configuration that promotes
            # faster approach to the target
            dist_ = np.linalg.norm(q_ - q_t, axis=1)
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
            dist = np.linalg.norm(q - q_t)  # update error
            current_i = min_i  # update current stiffness
            if (delta_q_[current_i] > 10 ** (-5)):
                q_list.append(q)
                s_list.append(s[current_i])
            # print(s_list[current_i])

            if flag:
                switch_counter += 1

            t += globals_.DT  # increment time

        return q_list, s_list, switch_counter


def phase_transition(s1, s2):
    if s1 == 0:
        return s2
    if s2 == 0:
        return s1
    return s1 * s2


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


    config_states = [q_start]
    stiffness_states = [s_start]

    # Animation of the 2SRR motion towards the target
    graphics.plotMotion(config_states, stiffness_states, 1)
