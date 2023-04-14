import pyomo.environ as pe
import pyomo.opt as po
import numpy as np
import shape

class GraspModel():
    def __init__(self, swarm, cp_n, com, contour, vsf, q_0, q_d) -> None:

        self.__n = swarm.n
        self.__cp_n = cp_n
        self.__N = cp_n * swarm.n
        self.__com = com
        self.__contour = contour
        self.__vsf = vsf

        self.__e = np.pi / 12

        self.__model = pe.ConcreteModel()
        self.__solver = po.SolverFactory('ipopt')
        # self.__solver = po.SolverFactory('mindtpy')

        self.__build_model(q_0, q_d)

    def __build_model(self, q_0, q_d):

        # Define sets

        self.__model.dof = pe.RangeSet(1, 3)
        self.__model.p_dof = pe.RangeSet(1, 2)
        self.__model.contact_points = pe.RangeSet(1, self.__N)
        self.__model.forces = pe.RangeSet(1, 2 * self.__N)
        self.__model.joints = pe.Set(initialize=[i for i in self.__model.contact_points if i % self.__cp_n != 0])
        self.__model.borders = pe.Set(initialize=[i for i in self.__model.contact_points if i % self.__cp_n == 0])

        # Define parameters

        hat_theta_array = [0] * self.__N
        hat_theta_array[-1] = 2 * np.pi
        if self.__N > 2:
            hat_theta_array[-2] = 2 * np.pi

        def __get_neighbours(m, i, j):
            if i == j:
                return 1
            elif j == i + 1 or i == self.__N and j == 1:
                return -1
            else:
                return 0

        self.__model.length = self.__length
        self.__model.com = pe.Param(self.__model.p_dof, initialize={1: self.__com[0], 2: self.__com[1]})
        self.__model.c = pe.Param(self.__model.dof, initialize={1: -0.981, 2: 0.196, 3: 0.981})

        self.__model.D = pe.Param(self.__model.contact_points, self.__model.contact_points, initialize=__get_neighbours)
        self.__model.hat_theta = pe.Param(self.__model.contact_points, initialize=dict(zip(self.__model.contact_points, hat_theta_array)))
        self.__model.epsilon = pe.Param(self.__model.contact_points, initialize=dict(zip(self.__model.contact_points, [self.__e] * self.__model.contact_points[-1])))

        self.__model.q_d = pe.Param(self.__model.dof, initialize={1: q_d[0], 2: q_d[1], 3: q_d[2]}, mutable=True)
        self.__model.q_0 = pe.Param(self.__model.dof, initialize={1: q_0[0], 2: q_0[1], 3: q_0[2]}, mutable=True)

        # Define variables

        self.__s = pe.Var(self.__model.contact_points, domain=pe.Reals, bounds=(0, 1))
        self.__model.L = pe.Var(self.__model.forces, domain=pe.NonNegativeReals)
        self.__model.q = pe.Var(self.__model.dof, domain=pe.Reals)

        self.__model.theta = pe.Var(self.__model.contact_points, domain=pe.Reals, bounds=(-np.pi, np.pi))
        self.__x = pe.Var(self.__model.contact_points, domain=pe.Reals)
        self.__y = pe.Var(self.__model.contact_points, domain=pe.Reals)

        # Define the objective

        @self.__model.Objective()
        def __obj_rule(m):

            tracking_error = sum((m.q_d[j] - m.q[j])**2 for j in m.dof)
            contact_forces = sum(m.L[j]**2 for j in m.forces)

            return tracking_error + contact_forces

        # Define constraints

        @self.__model.Constraint()
        def __constraint_x(m):
            return m.q[1] == m.q_0[1] + sum(m.c[2] * (m.L[j*2-1] - m.L[j*2]) * pe.sin(m.q_0[3] + m.theta[j]) + m.c[1] * (m.L[j*2-1] +
                m.L[j*2]) * pe.cos(m.q_0[3] + m.theta[j]) for j in m.contact_points)

        @self.__model.Constraint()
        def __constraint_y(m):
            return m.q[2] == m.q_0[2] + sum(m.c[1] * (m.L[j*2-1] + m.L[j*2]) * pe.cos(m.q_0[3] + m.theta[j]) + m.c[2] * (m.L[j*2-1] -
                m.L[j*2]) * pe.sin(m.q_0[3] + m.theta[j]) for j in m.contact_points)

        @self.__model.Constraint()
        def __constraint_theta(m):
            return m.q[3] == m.q_0[3] + sum(m.c[2] * (m.L[j*2] - m.L[j*2-1]) * ((m.com[1] - m.x[j]) * pe.sin(m.theta[j]) + (-m.com[2] + m.y[j]) * pe.cos(m.theta[j])) +
                m.c[3] * (m.L[j*2-1] + m.L[j*2]) * ((m.com[1] - m.x[j]) * pe.cos(m.theta[j]) + (m.com[2] - m.y[j]) * pe.sin(m.theta[j])) for j in m.contact_points)

        @self.__model.Constraint(self.__model.borders)
        def __constraint_collisions(m, i):
            lhs = sum(m.D[i, j] * m.theta[j] + m.hat_theta[i] for j in m.contact_points)
            rhs = m.epsilon[i]
            return lhs >= rhs

        @self.__model.Constraint(self.__model.joints)
        def __constraint_joints(m, i):
            lhs = sum(m.D[i, j] * m.theta[j] for j in m.contact_points)
            rhs = np.pi / 8
            return lhs == rhs


    def update(self, q_0, q_d):
        for i in self.__model.dof:
            self.__model.q_0[i] = q_0[i-1]
            self.__model.q_d[i] = q_d[i-1]

    def solve(self):
        result = self.__solver.solve(self.__model, tee=True)

        theta = [pe.value(self.__model.theta[key]) for key in self.__model.contact_points]
        L = [pe.value(self.__model.L[key]) for key in self.__model.forces]
        q = [pe.value(self.__model.q[key]) for key in self.__model.dof]
        # flags = [pe.value(self.__model.flag[key]) for key in self.__model.robots]

        return theta, L, q


