import numpy as np
import sa_kinematics as kinematics
import graphics
import random as rnd
import globals_
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Swarm:

    def __init__(self, n):
        self.n = n

        self.laplacian = np.diag(np.full(self.n,self.n-1))
        self.laplacian[self.laplacian == 0] = -1

        self.fig, self.ax = plt.subplots()

        self.robots = []
        for i in range(n):
            self.robots.append(self.ax.plot([], [], lw=3, marker="o", color="black"))

        self.edges = []
        self.edge_lines = []

        self.trajectories = []

    def buildEdges(self):
        self.edge_lines = []

        edges_n = len(self.edges)

        for i in range(edges_n):
            self.edge_lines.append(self.ax.plot([], [], lw=1, linestyle="dashed", color="black"))


    def genRandomStart(self):
        q_start = []
        for i in range(self.n):
            q_start.append([rnd.uniform(-0.5, 0.5), rnd.uniform(-0.5, 0.5)])

        return q_start

    def getMassCentre(self, q):
        q = np.array(q)
        massCentre = np.mean(q, axis=0)

        return massCentre

    def computeAnglesFromCentre(self, centre, q):
        q_c = np.array(q) - centre
        angles = np.arctan2(q_c[:,1], q_c[:,0])

        return angles

    def connectCycleGraph(self, q):

        q_id = np.array([x for x in range(self.n)])

        massCentre = self.getMassCentre(q)
        alpha = self.computeAnglesFromCentre(massCentre, q)
        r = np.linalg.norm(massCentre - q, axis=1)

        nodes = np.column_stack((q_id, alpha ,r))
        nodes = nodes[nodes[:,2].argsort()]
        nodes = nodes[nodes[:,1].argsort(kind='mergesort')]

        connections = [-1] * self.n
        for i in range(self.n-1):
            connections[int(nodes[i,0])] = int(nodes[i+1,0])

        connections[int(nodes[-1,0])] = int(nodes[0,0])

        self.laplacian = np.diag(np.full(self.n,2))
        for i in range(len(connections)):
            self.laplacian[i, connections[i]] = -1
            self.laplacian[connections[i], i] = -1

            self.edges.append((i, connections[i]))

        return nodes

    def reachConsensus(self, q_0, ksi):
        trajectories = []

        q_current = q_0
        trajectories.append(q_current)

        norm = np.linalg.norm(q_current)
        norm_prev = -norm

        dist = np.sum(norm)
        t = globals_.DT

        while norm != norm_prev:
            q_dot = np.matmul(-self.laplacian,q_current - ksi)
            q_current = q_current + q_dot * t * globals_.DT
            trajectories.append(q_current)

            norm_prev = norm
            norm = np.linalg.norm(q_current)
            t += globals_.DT


        return trajectories

    def rendezvous(self, q_0):

        ksi = np.zeros((self.n, 2))
        trajectories = self.reachConsensus(q_0, ksi)

        return trajectories


    def formCircle(self, q_0, R):
        nodes_ordered = self.connectCycleGraph(q_0)

        order = nodes_ordered[:,0].argsort()
        theta = np.array([2*np.pi / self.n * x - np.pi for x in range(self.n)])
        theta = theta[order]
        ksi = R * np.column_stack((np.cos(theta),np.sin(theta)))

        trajectories = self.reachConsensus(q_0, ksi)


        return trajectories


    def defineRange(self):
        margin = 0.1

        x_min, y_min = self.trajectories.min(axis=1)[0]
        x_max, y_max = self.trajectories.max(axis=1)[0]

        ax_range = max(x_max - x_min, y_max - y_min) + margin

        x_range = (x_min - margin, x_min + ax_range)
        y_range = (y_min - margin, y_min + ax_range)

        return x_range, y_range

    def anim_init(self):
        x_range, y_range = self.defineRange()
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(y_range)
        self.ax.set_aspect("equal")

        self.buildEdges()

    def anim_update(self,i):
        q_i = self.trajectories[i,:,:]
        for j in range(self.n):
            robot, = self.robots[j]

            x = q_i[j,0]
            y = q_i[j,1]

            robot.set_data(x, y)

        for j in range(len(self.edges)):
            node1 = q_i[self.edges[j][0]]
            node2 = q_i[self.edges[j][1]]

            x = [node1[0], node2[0]]
            y = [node1[1], node2[1]]

            edge_line, = self.edge_lines[j]
            edge_line.set_data(x, y)


    def anim(self, trajectories):

        self.trajectories = np.array(trajectories)

        frames = len(trajectories)
        anim = FuncAnimation(self.fig, self.anim_update, frames, init_func=self.anim_init, interval=50, repeat=True)
        plt.show()




if __name__ == "__main__":
    n = 8

    swarm = Swarm(n)
    q_0 = swarm.genRandomStart()
    # tr = swarm.rendezvous(q_0)

    tr = swarm.formCircle(q_0, 0.2)
    swarm.anim(tr)
