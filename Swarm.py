import random as rnd
from Agent import *
import globals_

class Swarm:

    def __init__(self, n, agent_type) -> None:
        self.__n = n
        self.__type = agent_type
        self.__collection = []

        self.__generate_agents()

    @property
    def n(self) -> float:
        return self.__n

    @property
    def type(self) -> float:
        return self.__type

    @property
    def collection(self) -> list:
        return self.__collection

    def __generate_agents(self) -> None:
        x = rnd.uniform(-0.5, 0.5)
        y = rnd.uniform(-0.5, 0.5)
        phi = rnd.uniform(-0.5, 0.5)
        k1 = rnd.uniform(-np.pi/(2*globals_.L_VSS), np.pi/(2*globals_.L_VSS))
        k2 = rnd.uniform(-np.pi/(2*globals_.L_VSS), np.pi/(2*globals_.L_VSS))

        for i in range(self.n):
            if self.type == "dot":
                agent = DotAgent(x, y)
            elif self.type == "oriented":
                agent = OrientedAgent(x, y, phi)
            elif self.type == "2SR":
                agent = TwoSRAgent(x, y, phi, k1, k2)
            else:
                raise Exception("Invalid robot type!")

            self.collection.append(agent)

    def update(self, agents_id, states) -> list:
        for agent_id, state in zip(agents_id, states):
            self.collection[agent_id].update(state)

        return self.collection


