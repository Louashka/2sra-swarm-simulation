import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
import mas
import globals_
import shape
import cv2 as cv
import sympy
import pyomo.environ as pe
import pyomo.opt as po

def com(contour):
    # x = sum(contour[0]) / len(contour[0])
    # y = sum(contour[1]) / len(contour[1])
    points = []
    for i in range(len(contour[0])):
        x = int(contour[0][i] * 1000)
        y = int(contour[1][i] * 1000)

        point = [x, y]
        points.append([point])

    M = cv.moments(np.array(points))
    x = int(M["m10"] / M["m00"]) / 1000
    y = int(M["m01"] / M["m00"]) / 1000

    return [x, y]

def assign_frame(p0):
    R = 0.1
    alpha = rnd.uniform(0, 2 * np.pi)

    x_dx = R * np.cos(alpha)
    x_dy = R * np.sin(alpha)

    beta = (alpha + np.pi / 2) % (2 * np.pi)

    y_dx = R * np.cos(beta)
    y_dy = R * np.sin(beta)

    return [x_dx, x_dy], [y_dx, y_dy]


# contour = shape.get_bezier_curve()
# centre = com(contour)

# x_axis, y_axis = assign_frame(centre)

# n = 5
# swarm = mas.formation(n, "dot")

# ORIGINAL SHAPE

centre = [0, 0]

axis_r = 0.2
alpha = 0

x_axis = [axis_r * np.cos(alpha), axis_r * np.sin(alpha)]
y_axis = [axis_r * np.cos(alpha + np.pi / 2), axis_r * np.sin(alpha + np.pi / 2)]

theta = np.linspace(0, 2 * np.pi, 50)
a = 1
b = 0.6
contour = [a * np.cos(theta), b * np.sin(theta)]

# GENERATE A PATH

dt = 0.5
t = 0

q_0 = np.array(centre + [alpha])

path = []
q_current = q_0

path.append(q_current)

while  t < 3:
    R = np.array([[np.cos(q_current[2]), -np.sin(q_current[2]), 0], [np.sin(q_current[2]), np.cos(q_current[2]), 0], [0, 0, 1]])

    velocity = [0, 1, -0.3]
    q_dot = R.dot(velocity)

    q_current = q_current + q_dot * dt

    path.append(q_current)
    t += dt

centre_d = [q_current[0], q_current[1]]
alpha_d = q_current[2]

x_axis_d = [axis_r * np.cos(alpha_d), axis_r * np.sin(alpha_d)]
y_axis_d = [axis_r * np.cos(alpha_d + np.pi / 2), axis_r * np.sin(alpha_d + np.pi / 2)]

R_d = np.array([[np.cos(alpha_d), -np.sin(alpha_d)], [np.sin(alpha_d), np.cos(alpha_d)]])
contour_d = np.array([centre_d]).T + R_d.dot(np.array(contour))

path = np.array(path)
# print(path)

# OPTIMISATION PROBLEM

n = 3
gen_c = {'x', 'y', 'theta'}
nu = 0.2

hat_theta = np.arctan(nu)
R_y_1 = np.array([[np.cos(hat_theta), 0, -np.sin(hat_theta)], [0, 1, 0], [np.sin(hat_theta), 0, np.cos(hat_theta)]])
R_y_2 = np.array([[np.cos(-hat_theta), 0, -np.sin(-hat_theta)], [0, 1, 0], [np.sin(-hat_theta), 0, np.cos(-hat_theta)]])
hat_z = np.array([0, 0, 1])

hat_f_1 = R_y_1.dot(hat_z)[[0,2]]
hat_f_2 = R_y_2.dot(hat_z)[[0,2]]

hat_F_c_i = np.column_stack((hat_f_1, hat_f_2))
hat_F_c_block = np.kron(np.eye(n,dtype=int), hat_F_c_i)

hat_F = {}

for i in range(hat_F_c_block.shape[0]):
    for j in range(hat_F_c_block.shape[1]):
        hat_F.update({(i+1, j+1): hat_F_c_block[i, j]})

R_0 = {(1, 1): np.cos(q_0[2]), (1, 2): -np.sin(q_0[2]), (2, 1): np.sin(q_0[2]), (2, 2): np.cos(q_0[2])}

D = {}

for i in range(n):
    for j in range(n):
        value = 0
        if i == j:
            value = 1
        elif j == i + 1 or i == n-1 and j == 0:
            value = -1
        D.update({(i+1, j+1): value})

q_d = {}
i = 1
for row in path:
    q_d.update({(i, 'x'): row[0]})
    q_d.update({(i, 'y'): row[1]})
    q_d.update({(i, 'theta'): row[2]})

    i += 1

model = pe.ConcreteModel()

model.gen_c = pe.Set(initialize=gen_c)
model.robots = pe.RangeSet(1, n)
model.forces = pe.RangeSet(1, 2*n)
model.path_states = pe.RangeSet(1, len(path))
model.path_states_next = pe.RangeSet(2, len(path))
model.dimen = pe.RangeSet(1, 2)
model.dof = pe.RangeSet(1, 3)

model.a = a
model.b = b
model.c = pe.Param(model.dof, initialize={1: -0.981, 2: 0.196, 3: -0.49})
model.q_d = pe.Param(model.gen_c, initialize={'x': path[1, 0], 'y': path[1, 1], 'theta': path[1, 2]})
model.q_0 = pe.Param(model.gen_c, initialize={'x': q_0[0], 'y': q_0[1], 'theta': q_0[2]})
model.D = pe.Param(model.robots, model.robots, initialize=D)
model.hat_theta = pe.Param(model.robots, initialize={1: 0, 2: 2 * np.pi, 3: 0})
model.epsilon = pe.Param(model.robots, initialize={1: 0.5, 2: 0.5, 3: 0.5})

model.theta = pe.Var(model.robots, domain=pe.NonNegativeReals, bounds=(0, 2*np.pi))
model.L = pe.Var(model.forces, domain=pe.NonNegativeReals)
model.q = pe.Var(model.gen_c, domain=pe.Reals)

def obj_rule(model):

    tracking_error = sum((model.q_d[coord] - model.q[coord])**2 for coord in model.gen_c)
    contact_forces = sum(model.L[j]**2 for j in model.forces)

    return tracking_error + contact_forces

def constraint_x(m):
    return m.q['x'] == m.q_0['x'] + sum(m.c[1] * (m.L[j*2-1] + m.L[j*2]) * pe.cos(m.q_0['theta'] + m.theta[j]) + m.c[2] * (m.L[j*2-1] + m.L[j*2]) * pe.sin(m.q_0['theta'] + m.theta[j]) for j in m.robots)

def constraint_y(m):
    return m.q['y'] == m.q_0['y'] + sum(m.c[2] * (m.L[j*2] - m.L[j*2-1]) * pe.cos(m.q_0['theta'] + m.theta[j]) + m.c[1] * (m.L[j*2-1] + m.L[j*2]) * pe.sin(m.q_0['theta'] + m.theta[j]) for j in m.robots)

def constraint_theta(m):
    return m.q['theta'] == m.q_0['theta'] + sum(m.c[2] * (m.L[j*2] - m.L[j*2-1]) * (m.a * pe.cos(m.theta[j])**2 + m.b * pe.sin(m.theta[j])**2) + m.c[3] * (m.L[j*2-1] + m.L[j*2]) * (m.a - m.b) * pe.sin(2 * m.theta[j]) for j in m.robots)


model.obj = pe.Objective(rule=obj_rule)

model.ContraintX = pe.Constraint(rule=constraint_x)
model.ContraintY = pe.Constraint(rule=constraint_y)
model.ContraintTh = pe.Constraint(rule=constraint_theta)

model.avoid_collisions = pe.ConstraintList()
for r in model.robots:
    lhs = sum(model.D[r, i] * model.theta[i] + model.hat_theta[r] for i in model.robots)
    rhs = model.epsilon[r]
    model.avoid_collisions.add(lhs >= rhs)

solver = po.SolverFactory('ipopt')
results = solver.solve(model, tee=True)

df = pd.DataFrame()
df['x'] = [pe.value(model.theta[key]) for key in model.robots]

print(df)


# PLOT THE RESULT

fig, ax = plt.subplots()
ax.set_aspect("equal")

plt.plot(contour[0], contour[1])
plt.plot(centre[0], centre[1], "*")
plt.arrow(centre[0], centre[1], x_axis[0], x_axis[1], width = 0.005)
plt.arrow(centre[0], centre[1], y_axis[0], y_axis[1], width = 0.005, color='red')

plt.plot(contour_d[0], contour_d[1], '--')
plt.plot(centre_d[0], centre_d[1], "*")
plt.arrow(centre_d[0], centre_d[1], x_axis_d[0], x_axis_d[1], linestyle = 'dashed', width = 0.005)
plt.arrow(centre_d[0], centre_d[1], y_axis_d[0], y_axis_d[1], linestyle = 'dashed', width = 0.005, color='red')

plt.scatter(path[:,0], path[:,1], color = 'black')

plt.show()

# tr = mas.rendezvous(swarm)
# tr = mas.form_regular_polygon(swarm, 0.15)
# mas.form_regular_polygon(swarm)
# print(swarm.all_edges)
# mas.show_motion(swarm, tr)




