import pyomo.environ as pe
import pyomo.opt as po
import numpy as np

class GraspModel():
    def __init__(self, swarm, a, b, q_0, q_d) -> None:

        self.__n = swarm.n
        self.__a = a
        self.__b = b

        self.__dt = 1
        self.__e = np.pi / 8

        self.__model = pe.ConcreteModel()
        self.__solver = po.SolverFactory('ipopt')
        # self.__solver = po.SolverFactory('mindtpy')

        self.__build_model(q_0, q_d)

    def __build_model(self, q_0, q_d):

        # Define sets

        self.__model.dof = pe.RangeSet(1, 3)
        self.__model.robots = pe.RangeSet(1, self.__n)
        self.__model.forces = pe.RangeSet(1, 2 * self.__n)

        # Define parameters

        D = {}
        for i in self.__model.robots:
            for j in self.__model.robots:
                value = 0
                if i == j or i == self.__n and j == 1:
                    value = 1
                elif j == i + 1 or i == self.__n and j == self.__n:
                    value = -1
                D.update({(i, j): value})

        # hat_theta_array = [i * 4 * np.pi / (self.__n - 1) for i in range(self.__n)]
        # for i in range(self.__n - 1, self.__n - 1 - int(self.__n / 2), -1):
        #     hat_theta_array[i] = hat_theta_array[self.__n - 1 - i]
        # print(hat_theta_array)

        # hat_theta_array = [0, 0, 0]

        # hat_theta_array = [np.pi, 0, 0]
        # hat_theta_array = [0, np.pi, 0]
        # hat_theta_array = [0, 0, np.pi]
        # hat_theta_array = [np.pi, np.pi, 0]
        # hat_theta_array = [np.pi, 0, np.pi]
        hat_theta_array = [0] * self.__n
        hat_theta_array[self.__n - 1] = 2 * np.pi
        if self.__n > 2:
            hat_theta_array[self.__n - 2] = 2 * np.pi
        # hat_theta_array = [0, np.pi, np.pi]
        # hat_theta_array = [np.pi, np.pi, np.pi]

        # hat_theta_array = [-np.pi, 0, 0]
        # hat_theta_array = [0, -np.pi, 0]
        # hat_theta_array = [0, 0, -np.pi]
        # hat_theta_array = [-np.pi, -np.pi, 0]
        # hat_theta_array = [-np.pi, 0, -np.pi]
        # hat_theta_array = [0, -np.pi, -np.pi]
        # hat_theta_array = [-np.pi, -np.pi, -np.pi]

        # hat_theta_array = [2 * np.pi, 0, 0]
        # hat_theta_array = [0, 2 * np.pi, 0]
        # hat_theta_array = [0, 0, 2 * np.pi]
        # hat_theta_array = [2 * np.pi, 2 * np.pi, 0]
        # hat_theta_array = [2 * np.pi, 0, 2 * np.pi]
        # hat_theta_array = [0, 2 * np.pi, 2 * np.pi]
        # hat_theta_array = [2 * np.pi, 2 * np.pi, 2 * np.pi]

        # hat_theta_array = [-2 * np.pi, 0, 0]
        # hat_theta_array = [0, -2 * np.pi, 0]
        # hat_theta_array = [0, 0, -2 * np.pi]
        # hat_theta_array = [-2 * np.pi, -2 * np.pi, 0]
        # hat_theta_array = [-2 * np.pi, 0, -2 * np.pi]
        # hat_theta_array = [0, -2 * np.pi, -2 * np.pi]
        # hat_theta_array = [-2 * np.pi, -2 * np.pi, -2 * np.pi]

        hat_theta = {}
        epsilon = {}

        for i in range(self.__n):
            hat_theta.update({i+1: hat_theta_array[i]})
            epsilon.update({i+1: self.__e})

        self.__model.a = self.__a
        self.__model.b = self.__b
        self.__model.c = pe.Param(self.__model.dof, initialize={1: -0.981, 2: 0.196, 3: -0.49})
        # self.__model.c = pe.Param(self.__model.dof, initialize={1: 0.196, 2: -0.981, 3: -0.098})
        self.__model.D = pe.Param(self.__model.robots, self.__model.robots, initialize=D)
        self.__model.hat_theta = pe.Param(self.__model.robots, initialize=hat_theta)
        self.__model.epsilon = pe.Param(self.__model.robots, initialize=epsilon)

        self.__model.q_d = pe.Param(self.__model.dof, initialize={1: q_d[0], 2: q_d[1], 3: q_d[2]}, mutable=True)
        self.__model.q_0 = pe.Param(self.__model.dof, initialize={1: q_0[0], 2: q_0[1], 3: q_0[2]}, mutable=True)

        # Define variables

        self.__model.theta = pe.Var(self.__model.robots, domain=pe.Reals, bounds=(-np.pi, np.pi))
        self.__model.L = pe.Var(self.__model.forces, domain=pe.NonNegativeReals)
        self.__model.q = pe.Var(self.__model.dof, domain=pe.Reals)
        self.__model.flag = pe.Var(self.__model.robots, within=pe.Binary)

        # Define the objective

        self.__model.obj = pe.Objective(rule = self.__obj_rule())

        # Define constraints

        self.__model.constraint_x = pe.Constraint(rule = self.__constraint_x())
        self.__model.constraint_y = pe.Constraint(rule = self.__constraint_y())
        self.__model.constraint_th = pe.Constraint(rule = self.__constraint_theta())

        self.__model.set_flag1 = pe.ConstraintList()
        for r in self.__model.robots:
            lhs = (1 - self.__model.flag[r]) * np.pi
            rhs = 2 * np.pi - abs(sum(self.__model.D[r, i] * self.__model.theta[i] for i in self.__model.robots))
            # self.__model.set_flag1.add(lhs <= rhs)

        self.__model.set_flag2 = pe.ConstraintList()
        for r in self.__model.robots:
            lhs = self.__model.flag[r] * np.pi
            rhs = abs(sum(self.__model.D[r, i] * self.__model.theta[i] for i in self.__model.robots))
            # self.__model.set_flag2.add(lhs <= rhs)

        self.__model.avoid_collisions = pe.ConstraintList()
        for r in self.__model.robots:
            # lhs = np.abs(self.__model.flag[r] * 2 * np.pi - np.abs(sum(self.__model.D[r, i] * self.__model.theta[i] + self.__model.hat_theta[r] for i in self.__model.robots)))
            lhs = sum(self.__model.D[r, i] * self.__model.theta[i] + self.__model.hat_theta[r] for i in self.__model.robots)
            rhs = self.__model.epsilon[r]
            self.__model.avoid_collisions.add(lhs >= rhs)

    def __obj_rule(self):

        tracking_error = sum((self.__model.q_d[j] - self.__model.q[j])**2 for j in self.__model.dof)
        contact_forces = sum(self.__model.L[j]**2 for j in self.__model.forces)

        return tracking_error + contact_forces

    def __constraint_x(self):
        return self.__model.q[1] == self.__model.q_0[1] + sum(self.__model.c[1] * (self.__model.L[j*2-1] + self.__model.L[j*2]) * pe.cos(self.__model.q_0[3] +
            self.__model.theta[j]) + self.__model.c[2] * (self.__model.L[j*2-1] - self.__model.L[j*2]) * pe.sin(self.__model.q_0[3] + self.__model.theta[j]) for j in self.__model.robots)
        # return self.__model.q[1] == self.__model.q_0[1] + sum(self.__model.c[1] * (- self.__model.L[j*2-1] + self.__model.L[j*2]) * pe.cos(self.__model.q_0[3] +
        #     self.__model.theta[j]) - self.__model.c[2] * (self.__model.L[j*2-1] + self.__model.L[j*2]) * pe.sin(self.__model.q_0[3] + self.__model.theta[j]) for j in self.__model.robots)

    def __constraint_y(self):
        return self.__model.q[2] == self.__model.q_0[2] + sum(self.__model.c[2] * (self.__model.L[j*2] - self.__model.L[j*2-1]) * pe.cos(self.__model.q_0[3] +
            self.__model.theta[j]) + self.__model.c[1] * (self.__model.L[j*2-1] + self.__model.L[j*2]) * pe.sin(self.__model.q_0[3] + self.__model.theta[j]) for j in self.__model.robots)
        # return self.__model.q[2] == self.__model.q_0[2] + sum(self.__model.c[1] * (- self.__model.L[j*2-1] + self.__model.L[j*2]) * pe.sin(self.__model.q_0[3] +
        #     self.__model.theta[j]) + self.__model.c[2] * (self.__model.L[j*2-1] + self.__model.L[j*2]) * pe.cos(self.__model.q_0[3] + self.__model.theta[j]) for j in self.__model.robots)

    def __constraint_theta(self):
        return self.__model.q[3] == self.__model.q_0[3] + sum(self.__model.c[2] * (self.__model.L[j*2] - self.__model.L[j*2-1]) * (self.__model.a * pe.cos(self.__model.theta[j])**2 +
            self.__model.b * pe.sin(self.__model.theta[j])**2) + self.__model.c[3] * (self.__model.L[j*2-1] + self.__model.L[j*2]) * (self.__model.a -
            self.__model.b) * pe.sin(2 * self.__model.theta[j]) for j in self.__model.robots)
        # return self.__model.q[3] == self.__model.q_0[3] + sum(self.__model.c[2] * (self.__model.L[j*2-1] + self.__model.L[j*2]) * (self.__model.a * pe.cos(self.__model.theta[j])**2 +
        #     self.__model.b * pe.sin(self.__model.theta[j])**2) + self.__model.c[3] * (self.__model.L[j*2-1] - self.__model.L[j*2]) * (self.__model.a -
        #     self.__model.b) * pe.sin(2 * self.__model.theta[j]) for j in self.__model.robots)

    def update(self, q_0, q_d):
        for i in self.__model.dof:
            self.__model.q_0[i] = q_0[i-1]
            self.__model.q_d[i] = q_d[i-1]

    def solve(self):
        result = self.__solver.solve(self.__model, tee=True)

        theta = [pe.value(self.__model.theta[key]) for key in self.__model.robots]
        L = [pe.value(self.__model.L[key]) for key in self.__model.forces]
        q = [pe.value(self.__model.q[key]) for key in self.__model.dof]
        # flags = [pe.value(self.__model.flag[key]) for key in self.__model.robots]

        return theta, L, q


