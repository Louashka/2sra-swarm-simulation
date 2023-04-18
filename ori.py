import pyomo.environ as pe
import pyomo.opt as po
import numpy as np
import shape

class GraspModel():
    def __init__(self, swarm, cp_n, obj, vsf, q_0, q_d) -> None:

        self.__n = swarm.n
        self.__cp_n = cp_n
        self.__N = cp_n * swarm.n
        self.__obj = obj
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
        self.__model.borders = pe.Set(initialize=[i for i in self.__model.contact_points if i % self.__cp_n == 0 and i != self.__N])

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

        def __get_point(s):
            x = 0
            y = 0

            for h in range(self.__obj.m):
                arg = 2 * np.pi * (h + 1) * s
                exp = [pe.cos(arg), pe.sin(arg)]

                coef = self.__obj.coeffs[h,:]
                x += coef[0] * exp[0] + coef[1] * exp[1]
                y += coef[2] * exp[0] + coef[3] * exp[1]

            return [x, y]

        def __get_tangent(s):
            dx = 0
            dy = 0

            for h in range(self.__obj.m):
                c = 2 * np.pi * (h + 1)
                arg = c * s
                exp = [-c * pe.sin(arg),  c * pe.cos(arg)]

                coef = self.__obj.coeffs[h,:]
                dx += coef[0] * exp[0] + coef[1] * exp[1]
                dy += coef[2] * exp[0] + coef[3] * exp[1]

            theta = pe.atan(dy/dx)

            return theta

        def __get_x_hat_direc(s):
            theta = __get_tangent(s)

            p1 = __get_point(s)
            p2 = [p1[0] + pe.cos(theta), p1[1] + pe.sin(theta)]

            p3 = [p1[0] + pe.cos(theta - np.pi/2), p1[1] + pe.sin(theta - np.pi/2)]

            cross_prod1 = (p1[0] - p2[0]) * (self.__obj.com[1] - p2[1]) - (p1[1] - p2[1]) * (self.__obj.com[0] - p2[0])
            cross_prod2 = (p1[0] - p2[0]) * (p3[1] - p2[1]) - (p1[1] - p2[1]) * (p3[0] - p2[0])

            condition = (abs(cross_prod1 * cross_prod2) - cross_prod1 * cross_prod2) / (-2 * cross_prod1 * cross_prod2)

            return theta + condition * np.pi

        self.__model.vsf = self.__vsf
        self.__model.com = pe.Param(self.__model.p_dof, initialize={1: self.__obj.com[0], 2: self.__obj.com[1]})
        self.__model.c = pe.Param(self.__model.dof, initialize={1: 0.196, 2: -0.981, 3: 0.981})

        self.__model.D = pe.Param(self.__model.contact_points, self.__model.contact_points, initialize=__get_neighbours)
        self.__model.hat_theta = pe.Param(self.__model.contact_points, initialize=dict(zip(self.__model.contact_points, hat_theta_array)))
        self.__model.epsilon = pe.Param(self.__model.contact_points, initialize=dict(zip(self.__model.contact_points, [self.__e] * self.__model.contact_points[-1])))

        self.__model.q_d = pe.Param(self.__model.dof, initialize={1: q_d[0], 2: q_d[1], 3: q_d[2]}, mutable=True)
        self.__model.q_0 = pe.Param(self.__model.dof, initialize={1: q_0[0], 2: q_0[1], 3: q_0[2]}, mutable=True)

        # Define variables

        self.__model.point_theta = pe.Var(self.__model.contact_points, domain=pe.Reals)
        self.__model.point_x = pe.Var(self.__model.contact_points, domain=pe.Reals)
        self.__model.point_y = pe.Var(self.__model.contact_points, domain=pe.Reals)

        self.__model.s = pe.Var(self.__model.contact_points, domain=pe.Reals, bounds=(0, 1))
        self.__model.L = pe.Var(self.__model.forces, domain=pe.NonNegativeReals)
        self.__model.q = pe.Var(self.__model.dof, domain=pe.Reals)

        # Define the objective

        @self.__model.Objective()
        def __obj_rule(m):

            tracking_error = sum((m.q_d[j] - m.q[j])**2 for j in m.dof)
            contact_forces = sum(m.L[j]**2 for j in m.forces)

            return tracking_error + contact_forces

        # Define constraints

        @self.__model.Constraint()
        def __constraint_q_x(m):
            return m.q[1] == m.q_0[1] + sum(m.c[1] * (m.L[j*2-1] - m.L[j*2]) * pe.cos(m.q_0[3] + m.point_theta[j]) - m.c[2] * (m.L[j*2-1] +
                m.L[j*2]) * pe.sin(m.q_0[3] + m.point_theta[j]) for j in m.contact_points)

        @self.__model.Constraint()
        def __constraint_q_y(m):
            return m.q[2] == m.q_0[2] + sum(m.c[1] * (m.L[j*2-1] - m.L[j*2]) * pe.sin(m.q_0[3] + m.point_theta[j]) + m.c[2] * (m.L[j*2-1] +
                m.L[j*2]) * pe.cos(m.q_0[3] + m.point_theta[j]) for j in m.contact_points)

        @self.__model.Constraint()
        def __constraint_q_theta(m):
            return m.q[3] == m.q_0[3] + sum(m.c[1] * (m.L[j*2] - m.L[j*2-1]) * ((m.com[1] - m.point_x[j]) * pe.sin(m.point_theta[j]) - (m.com[2] - m.point_y[j]) * pe.cos(m.point_theta[j])) +
                m.c[3] * (m.L[j*2-1] + m.L[j*2]) * ((m.com[1] - m.point_x[j]) * pe.cos(m.point_theta[j]) + (m.com[2] - m.point_y[j]) * pe.sin(m.point_theta[j])) for j in m.contact_points)

        @self.__model.Constraint(self.__model.contact_points)
        def __constraint_point_x(m, i):
            return m.point_x[i] == __get_point(m.s[i])[0]

        @self.__model.Constraint(self.__model.contact_points)
        def __constraint_point_y(m, i):
            return m.point_y[i] == __get_point(m.s[i])[1]

        @self.__model.Constraint(self.__model.contact_points)
        def __constraint_point_theta(m, i):
            return m.point_theta[i] == __get_x_hat_direc(m.s[i])


        @self.__model.Constraint(self.__model.borders)
        def __constraint_collisions(m, i):
            lhs = sum(m.D[i, j] * m.s[j] for j in m.contact_points)
            rhs = 0.05
            return lhs >= rhs

        @self.__model.Constraint(self.__model.joints)
        def __constraint_joints(m, i):
            lhs = sum(m.D[i, j] * m.s[j] for j in m.contact_points)
            rhs = 0.05
            return lhs == rhs


    def update(self, q_0, q_d):
        for i in self.__model.dof:
            self.__model.q_0[i] = q_0[i-1]
            self.__model.q_d[i] = q_d[i-1]

    def solve(self):
        result = self.__solver.solve(self.__model, tee=True)

        s = [pe.value(self.__model.s[key]) for key in self.__model.contact_points]
        L = [pe.value(self.__model.L[key]) for key in self.__model.forces]
        q = [pe.value(self.__model.q[key]) for key in self.__model.dof]
        # flags = [pe.value(self.__model.flag[key]) for key in self.__model.robots]

        return s, L, q


