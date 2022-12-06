import random as rnd
import numpy as np
from agent import *
import globals_

class SwarmGraph:

    def __init__(self, n, agent_type) -> None:
        self.__n = n
        self.__type = agent_type
        self.__graph_dict = {}
        self.__collection = []

        self.__laplacian = np.zeros((self.n, self.n))

        self.__generate_nodes()

    @property
    def n(self) -> float:
        return self.__n

    @property
    def type(self) -> float:
        return self.__type

    @property
    def collection(self) -> list:
        return self.__collection

    @property
    def laplacian(self) -> object:
        return self.__laplacian

    def __generate_nodes(self) -> None:
        for i in range(self.n):
            x = rnd.uniform(-0.5, 0.5)
            y = rnd.uniform(-0.5, 0.5)
            phi = rnd.uniform(-0.5, 0.5)
            k1 = rnd.uniform(-np.pi/(2*globals_.L_VSS), np.pi/(2*globals_.L_VSS))
            k2 = rnd.uniform(-np.pi/(2*globals_.L_VSS), np.pi/(2*globals_.L_VSS))

            if self.type == "dot":
                agent = DotAgent(i, x, y)
            elif self.type == "oriented":
                agent = OrientedAgent(i, x, y, phi)
            elif self.type == "2SR":
                agent = TwoSRAgent(i, x, y, phi, k1, k2)
            else:
                raise Exception("Invalid robot type!")

            self.collection.append(agent)
            self.__graph_dict[agent.node_id] = set()

    def update(self, agents_id, states) -> list:
        for agent_id, state in zip(agents_id, states):
            self.collection[agent_id].update(state)

        return self.collection

    def neighbourhood(self, node) -> list:
        """ returns a list of all the edges of a node"""
        return self.__graph_dict[node]

    def all_nodes(self) -> set:
        """ returns the nodes of a graph as a set """
        return set(self.__graph_dict.keys())

    def all_edges(self) -> list:
        """ returns the edges of a graph """
        return self.__generate_edges()

    def __generate_edges(self) -> list:
        """ A static method generating the edges of the
            graph. Edges are represented as sets
            with one (a loop back to the node) or two
            nodes
        """
        edges = []
        for node in self.__graph_dict:
            for neighbour in self.neighbourhood(node):
                if {neighbour, node} not in edges and {node, neighbour} not in edges:
                    edges.append({node, neighbour})
        return edges

    def add_edge(self, agent1, agent2):
        self.__graph_dict[agent1.node_id].add(agent2.node_id)
        self.__graph_dict[agent2.node_id].add(agent1.node_id)

    def cycle_graph(self) -> object:
        position_coordinates = np.array([self.collection[i].position for i in self.all_nodes()])

        massCentre = np.mean(position_coordinates, axis=0)
        diff = position_coordinates - massCentre
        angles = np.arctan2(diff[:,1], diff[:,0])
        r = np.linalg.norm(massCentre - position_coordinates, axis=1)

        nodes = np.column_stack((list(self.all_nodes()), angles ,r))
        nodes = nodes[nodes[:,2].argsort()]
        nodes = nodes[nodes[:,1].argsort(kind='mergesort')]

        self.__laplacian = np.diag(np.full(self.n,2))

        for i in range(self.n-1):
            id1 = int(nodes[i,0])
            id2 = int(nodes[i+1,0])

            node1 = self.collection[id1]
            node2 = self.collection[id2]

            self.add_edge(node1, node2)

            self.__laplacian[id1, id2] = -1
            self.__laplacian[id2, id1] = -1

        self.add_edge(node2, self.collection[int(nodes[0,0])])

        self.__laplacian[id2, int(nodes[0,0])] = -1
        self.__laplacian[int(nodes[0,0]), id2] = -1

        return self

    def complete_graph(self) -> object:
        for i in range(self.n-1):
            node1 = self.collection[i]
            for j in range(i+1, self.n):
                node2 = self.collection[j]
                self.add_edge(node1, node2)

        self.__laplacian = np.diag(np.full(self.n,self.n-1))
        self.__laplacian[self.laplacian == 0] = -1

        return self


    def __str__(self):
        out = ""
        for key in self.__graph_dict.keys():
            out += "agent " + str(key) + ": " +  str(self.__graph_dict[key]) + "\n"
        return out


