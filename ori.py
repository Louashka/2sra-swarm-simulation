import pyomo.environ as pe
import pyomo.opt as po
import numpy as np

class GraspModel():
    def __init__(self, swarm, a, b, q_0, q_d) -> None:

        self.__n = swarm.n
        self.__a = a
        self.__b = b

        self.__dt = 1

        self.__model = pe.ConcreteModel()
        self.__solver = po.SolverFactory('ipopt')

        self.__build_model(q_0, q_d)

    def __build_model(self, q_0, q_d):

        # Define sets

        self.__model.dof = pe.RangeSet(1, 3)
        self.__model.robots = pe.RangeSet(1, self.__n)
        self.__model.forces = pe.RangeSet(1, 2 * self.__n)

        # Define parameters

        D = {}
        for i in range(self.__n):
            for j in range(self.__n):
                value = 0
                if i == j:
                    value = 1
                elif j == i + 1 or i == self.__n-1 and j == 0:
                    value = -1
                D.update({(i+1, j+1): value})

        self.__model.a = self.__a
        self.__model.b = self.__b
        self.__model.c = pe.Param(self.__model.dof, initialize={1: -0.981, 2: 0.196, 3: -0.49})
        self.__model.D = pe.Param(self.__model.robots, self.__model.robots, initialize=D)
        self.__model.hat_theta = pe.Param(self.__model.robots, initialize={1: 0, 2: 2 * np.pi, 3: 0})
        self.__model.epsilon = pe.Param(self.__model.robots, initialize={1: np.pi / 10, 2: np.pi / 10, 3: np.pi / 10})

        self.__model.q_d = pe.Param(self.__model.dof, initialize={1: q_d[0], 2: q_d[1], 3: q_d[2]}, mutable=True)
        self.__model.q_0 = pe.Param(self.__model.dof, initialize={1: q_0[0], 2: q_0[1], 3: q_0[2]}, mutable=True)

        # Define variables

        self.__model.theta = pe.Var(self.__model.robots, domain=pe.NonNegativeReals, bounds=(0, 2*np.pi))
        self.__model.L = pe.Var(self.__model.forces, domain=pe.NonNegativeReals)
        self.__model.q = pe.Var(self.__model.dof, domain=pe.Reals)

        # Define the objective

        self.__model.obj = pe.Objective(rule = self.__obj_rule())

        # Define constraints

        self.__model.ContraintX = pe.Constraint(rule = self.__constraint_x())
        self.__model.ContraintY = pe.Constraint(rule = self.__constraint_y())
        self.__model.ContraintTh = pe.Constraint(rule = self.__constraint_theta())
        # self.__model.ContraintAngleUpper = pe.Constraint(rule = self.__constraint_angle_upper())
        # self.__model.ContraintAngleLower = pe.Constraint(rule = self.__constraint_angle_lower())

        self.__model.avoid_collisions = pe.ConstraintList()
        for r in self.__model.robots:
            lhs = sum(self.__model.D[r, i] * self.__model.theta[i] + self.__model.hat_theta[r] for i in self.__model.robots)
            rhs = self.__model.epsilon[r]
            self.__model.avoid_collisions.add(lhs >= rhs)

    def __obj_rule(self):

        tracking_error = sum((self.__model.q_d[j] - self.__model.q[j])**2 for j in self.__model.dof)
        contact_forces = sum(self.__model.L[j]**2 for j in self.__model.forces)

        return tracking_error + contact_forces

    def __constraint_x(self):
        return self.__model.q[1] == self.__model.q_0[1] + self.__dt * sum(self.__model.c[1] * (self.__model.L[j*2-1] + self.__model.L[j*2]) * pe.cos(self.__model.q_0[3] +
            self.__model.theta[j]) + self.__model.c[2] * (self.__model.L[j*2-1] + self.__model.L[j*2]) * pe.sin(self.__model.q_0[3] + self.__model.theta[j]) for j in self.__model.robots)

    def __constraint_y(self):
        return self.__model.q[2] == self.__model.q_0[2] + self.__dt * sum(self.__model.c[2] * (self.__model.L[j*2] - self.__model.L[j*2-1]) * pe.cos(self.__model.q_0[3] +
            self.__model.theta[j]) + self.__model.c[1] * (self.__model.L[j*2-1] + self.__model.L[j*2]) * pe.sin(self.__model.q_0[3] + self.__model.theta[j]) for j in self.__model.robots)

    def __constraint_theta(self):
        return self.__model.q[3] == self.__model.q_0[3] + self.__dt * sum(self.__model.c[2] * (self.__model.L[j*2] - self.__model.L[j*2-1]) * (self.__model.a * pe.cos(self.__model.theta[j])**2 +
            self.__model.b * pe.sin(self.__model.theta[j])**2) + self.__model.c[3] * (self.__model.L[j*2-1] + self.__model.L[j*2]) * (self.__model.a -
            self.__model.b) * pe.sin(2 * self.__model.theta[j]) for j in self.__model.robots)

    def __constraint_angle_upper(self):
        return self.__model.q[3] <= 2 * np.pi

    def __constraint_angle_lower(self):
        return self.__model.q[3] >= 0

    def update(self, q_0, q_d):
        for i in self.__model.dof:
            self.__model.q_0[i] = q_0[i-1]
            self.__model.q_d[i] = q_d[i-1]

    def solve(self):
        result = self.__solver.solve(self.__model, tee=True)

        theta = [pe.value(self.__model.theta[key]) for key in self.__model.robots]
        L = [pe.value(self.__model.L[key]) for key in self.__model.forces]
        q = [pe.value(self.__model.q[key]) for key in self.__model.dof]

        return theta, L, q




