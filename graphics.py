import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
import numpy as np

obj = None

swarm = None
cp_n = 1
path = []
q = []
s = []

l_axis = 0.2

fig, ax = plt.subplots()

centroid, = ax.plot([], [], lw=2, marker='*', color='red')
manip, = ax.plot([], [], lw=1, color='blue')

x_axis, = ax.plot([], [], lw=1, color='blue')
y_axis, = ax.plot([], [], lw=1, color='red')

centroid_d, = ax.plot([], [], lw=2, marker='*', color='red')
manip_d, = ax.plot([], [], lw=1, linestyle='dashed', color='green')

x_axis_d, = ax.plot([], [], lw=1, color='blue')
y_axis_d, = ax.plot([], [], lw=1, color='red')

path_curve = None

agents = []

agent1, = ax.plot([], [], lw=2, marker='o', color='magenta')
agent2, = ax.plot([], [], lw=2, marker='o', color='magenta')
agent3, = ax.plot([], [], lw=2, marker='o', color='magenta')


def defineRange():
    margin = 1

    x_min, y_min = path[:, :2].min(axis=0)
    x_max, y_max = path[:, :2].max(axis=0)

    ax_range = max(x_max - x_min, y_max - y_min) + margin

    x_range = (x_min - ax_range/2, x_min + ax_range)
    y_range = (y_min - ax_range/3, y_min + ax_range)

    return x_range, y_range


def init():
    global ax, path_curve

    x_range, y_range = defineRange()
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")

    for i in range(cp_n * swarm.n):
        agents.append(ax.plot([], [], markersize=3, marker='o', color='magenta'))

    path_curve = ax.scatter(path[:,0], path[:,1], lw=2, marker='.', color='black')

    centre_d = [path[-1,0], path[-1,1]]
    centroid_d.set_data(centre_d[0], centre_d[1])

    R_d = np.array([[np.cos(path[-1,2]), -np.sin(path[-1,2])], [np.sin(path[-1,2]), np.cos(path[-1,2])]])
    contour_d = np.array([centre_d]).T + R_d.dot(obj.contour)

    manip_d.set_data(contour_d[0], contour_d[1])

    x_axis_d.set_data([centre_d[0], centre_d[0] + l_axis * np.cos(path[-1,2])], [centre_d[1], centre_d[1] + l_axis * np.sin(path[-1,2])])
    y_axis_d.set_data([centre_d[0], centre_d[0] + l_axis * np.cos(path[-1,2] + np.pi / 2)], [centre_d[1], centre_d[1] + l_axis * np.sin(path[-1,2] + np.pi / 2)])


    return path_curve, centroid_d, manip_d, x_axis_d, y_axis_d,


def update(i):
    # global links, arcs1, arcs2, centres, centroid, circle, target_nodes

    centre = [q[i][0], q[i][1]]
    centroid.set_data(centre[0], centre[1])

    R = np.array([[np.cos(q[i][2]), -np.sin(q[i][2])], [np.sin(q[i][2]), np.cos(q[i][2])]])
    contour = np.array([centre]).T + R.dot(obj.contour)

    manip.set_data(contour[0], contour[1])

    # for j in range(cp_n * swarm.n):
    #     agent_position = R.dot(np.array([[a * np.cos(theta[i][j]), b * np.sin(theta[i][j])]]).T)
    #     agent, = agents[j]
    #     agent.set_data(centre[0] + agent_position[0], centre[1] + agent_position[1])

    return centroid, manip,


def plot_motion(swarm_, cp_n_, obj_, path_, q_, s_):
    global swarm, cp_n, obj, path, q, s

    swarm = swarm_
    cp_n = cp_n_
    obj = obj_
    path = path_
    q = q_
    s = s_

    frames = len(q)


    anim = FuncAnimation(fig, update, frames,
                         init_func=init, interval=200, repeat=True)

    # Save animation
    # mywriter = FFMpegWriter(fps=30)
    # anim.save('grasping1.mp4', writer=mywriter, dpi=300)

    plt.show()
