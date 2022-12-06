import numpy as np

class DotAgent(object):

    def __init__(self, id_, x, y) -> None:
        self.__node_id = id_
        self.__x = x
        self.__y = y

    @property
    def node_id(self) -> int:
        return self.__node_id

    @node_id.setter
    def node_id(self, value) -> None:
        self.__node_id = value

    @property
    def x(self) -> float:
        return self.__x

    @x.setter
    def x(self, value) -> None:
        self.__x = value

    @property
    def y(self) -> float:
        return self.__y

    @y.setter
    def y(self, value) -> None:
        self.__y = value

    @property
    def position(self) -> list:
        return [self.x, self.y]

    @property
    def state(self) -> list:
        current_state = [self.x, self.y]

        return current_state

    @property
    def type(self) -> str:
        return "dot"

    def update(self, params) -> object:
        self.__x = params[0]
        self.__y = params[1]

        return self


class OrientedAgent(DotAgent, object):

    def __init__(self, id_, x, y, phi):
        super().__init__(id_, x, y)
        self.__phi = phi

    @property
    def phi(self) -> float:
        return self.__phi

    @phi.setter
    def phi(self, value) -> None:
        self.__phi = value % (2 * np.pi)

    @property
    def state(self) -> list:
        current_state = super().state() + [self.phi]

        return current_state

    @property
    def type(self) -> str:
        return "oriented"

    def update(self, params) -> object:
        super().update(params[:2])
        self.__phi = params[2]

        return self

class TwoSRAgent(OrientedAgent, object):

    def __init__(self, id_, x, y, phi, k1, k2):
        super().__init__(id_, x, y, phi)
        self.__k1 = k1
        self.__k2 = k2

    @property
    def k1(self) -> float:
        return self.__k1

    @k1.setter
    def k1(self, value) -> None:
        self.__k1 = value

    @property
    def k2(self) -> float:
        return self.__k2

    @k2.setter
    def k2(self, value) -> None:
        self.__k2 = value

    @property
    def state(self) -> list:
        current_state = super().state() + [self.k1, self.k2]

        return current_state

    @property
    def type(self) -> str:
        return "2SR"

    def update(self, params) -> object:
        super().update(params[:3])
        self.__k1 = params[3]
        self.__k2 = params[4]

        return self

