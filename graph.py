import random as rnd
import numpy as np
from queue import Queue
from agent import *
import globals_

class Graph:

    def __init__(self, n, agent_type="dot") -> None:

        if agent_type not in {"dot", "oriented", "2SR"}:
            raise Exception("Invalid robot type!")

        self.__n = n
        self.__type = agent_type
        self.__graph_dict = {}
        self.__collection = []

        self.__laplacian = np.zeros((self.n, self.n))

        self.__is_connected = False
        self.__is_complete = False
        self.__is_cycle = False

        self.__generate_nodes()

    @property
    def n(self) -> float:
        return self.__n

    @property
    def type(self) -> float:
        return self.__type

    @property
    def config_dim(self) -> int:
        dim = 2
        if self.type == "oriented":
            dim = 3
        elif self.type == "2SR":
            dim = 5

        return dim

    @property
    def collection(self) -> list:
        return self.__collection

    @property
    def state(self) -> list:
        current_state = [self.collection[i].state for i in self.all_nodes]

        return current_state

    @property
    def laplacian(self) -> object:
        return self.__laplacian

    @property
    def all_nodes(self) -> set:
        """ returns the nodes of a graph as a set """
        return set(self.__graph_dict.keys())

    @property
    def all_edges(self) -> list:
        """ returns the edges of a graph """
        return self.__generate_edges()

    @property
    def is_connected(self) -> bool:
        return self.__is_connected

    @property
    def is_complete(self) -> bool:
        return self.__is_complete

    @property
    def is_cycle(self) -> bool:
        return self.__is_cycle

    def __generate_nodes(self) -> None:
        for i in range(self.n):
            x = rnd.uniform(-0.5, 0.5)
            y = rnd.uniform(-0.5, 0.5)
            phi = rnd.uniform(0, 2* np.pi)
            k1 = rnd.uniform(-np.pi/(2*globals_.L_VSS), np.pi/(2*globals_.L_VSS))
            k2 = rnd.uniform(-np.pi/(2*globals_.L_VSS), np.pi/(2*globals_.L_VSS))

            if self.type == "dot":
                agent = DotAgent(i, x, y)
            elif self.type == "oriented":
                agent = OrientedAgent(i, x, y, phi)
            elif self.type == "2SR":
                agent = TwoSRAgent(i, x, y, phi, k1, k2)

            self.collection.append(agent)
            self.__graph_dict[agent.node_id] = set()

    def neighbourhood(self, node) -> list:
        """ returns a list of all the edges of a node"""
        return self.__graph_dict[node]

    def update(self, agents_id, states) -> list:
        for agent_id, state in zip(agents_id, states):
            self.collection[agent_id].update(state)

        return self.collection

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

    def sort_nodes_by_angles(self) -> object:
        position_coordinates = np.array([self.collection[i].position for i in self.all_nodes])

        massCentre = np.mean(position_coordinates, axis=0)
        diff = position_coordinates - massCentre
        angles = np.arctan2(diff[:,1], diff[:,0])
        r = np.linalg.norm(massCentre - position_coordinates, axis=1)

        nodes = np.column_stack((list(self.all_nodes), angles ,r))
        nodes_sorted = nodes[nodes[:,2].argsort()]
        nodes_sorted = nodes_sorted[nodes_sorted[:,1].argsort(kind='mergesort')]

        return nodes_sorted[:,0].astype(int)

    def complete(self) -> object:
        for i in range(self.n-1):
            node1 = self.collection[i]
            for j in range(i+1, self.n):
                node2 = self.collection[j]
                self.add_edge(node1, node2)

        self.__laplacian = np.diag(np.full(self.n,self.n-1))
        self.__laplacian[self.laplacian == 0] = -1

        self.__is_connected = True
        self.__is_complete = True
        self.__is_cycle = True

        return self

    def cycle(self) -> list:
        nodes = self.sort_nodes_by_angles()
        print(nodes[0])

        self.__laplacian = np.diag(np.full(self.n,2))

        for i in range(self.n-1):
            id1 = nodes[i]
            id2 = nodes[i+1]

            node1 = self.collection[id1]
            node2 = self.collection[id2]

            self.add_edge(node1, node2)

            self.__laplacian[id1, id2] = -1
            self.__laplacian[id2, id1] = -1

        self.add_edge(node2, self.collection[nodes[0]])

        self.__laplacian[id2, nodes[0]] = -1
        self.__laplacian[nodes[0], id2] = -1

        self.__is_connected = True
        self.__is_complete = False
        self.__is_cycle = True

        return nodes

    def min_rigid(self) -> object:
        nodes = self.cycle()

        if len(nodes) > 3:
            id1 = nodes[0]
            node1 = self.collection[id1]

            for i in range(2, len(nodes)-1) :
                id2 = nodes[i]
                node2 = self.collection[id2]

                self.add_edge(node1, node2)

                self.__laplacian[id1, id2] = -1
                self.__laplacian[id2, id1] = -1

                self.__laplacian[id1, id1] += 1
                self.__laplacian[id2, id2] += 1

        print(self.laplacian)

        return self

    def __dfs(self, visited, node) -> list:
        if node not in visited:
            visited.append(node)
            for neighbour in self.neighbourhood(node):
                self.dfs(visited, neighbour)

        return visited

    def dfs_traversal(self, start_node_id) -> list:
        visited = list()
        visited = self.__dfs(visited, start_node_id)

        return visited

    def __str__(self):
        out = ""
        for key in self.__graph_dict.keys():
            out += "agent " + str(key) + ": " +  str(self.__graph_dict[key]) + "\n"
        return out


