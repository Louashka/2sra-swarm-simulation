import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import globals_


link_width = globals_.D_BRIDGE
link_length = globals_.L_LINK

swarm = None
trajectories = []

fig, ax = plt.subplots()

agents = []
agents_orientation = []
edge_lines = []
agents_vss1 = []
agents_vss2 = []

def define_range():
    margin = 0.08
    x_min, y_min = trajectories[:,:,:2].min(axis=1)[0]
    x_max, y_max = trajectories[:,:,:2].max(axis=1)[0]

    ax_range = max(x_max - x_min, y_max - y_min) + margin

    x_range = (x_min - margin, x_min + ax_range)
    y_range = (y_min - margin, y_min + ax_range)

    return x_range, y_range

def gen_arc(q, seg):
    s = np.linspace(0, globals_.L_VSS, 50)

    flag = -1 if seg == 1 else 1

    gamma_array = q[2] + flag * q[2 + seg] * s

    x_0 = q[0] + flag * np.cos(q[2]) * link_length / 2
    y_0 = q[1] + flag * np.sin(q[2]) * link_length / 2

    if q[2 + seg] == 0:
        x = x_0 + [0, flag * globals_.L_VSS * np.cos(q[2])]
        y = y_0 + [0, flag * globals_.L_VSS * np.sin(q[2])]
    else:
        x = x_0 + np.sin(gamma_array) / \
            q[2 + seg] - np.sin(q[2]) / q[2 + seg]
        y = y_0 - np.cos(gamma_array) / \
            q[2 + seg] + np.cos(q[2]) / q[2 + seg]

    return [x, y]

def anim_init():
    x_range, y_range = define_range()
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")

    for i in range(swarm.n):
        agents.append(ax.plot([], [], marker="o", markersize=5, color="None", markeredgecolor="black"))
        agents_orientation.append(ax.plot([], [], lw=1, color="magenta"))
        agents_vss1.append(ax.plot([], [], lw=1, color="blue"))
        agents_vss2.append(ax.plot([], [], lw=1, color="blue"))

    for i in range(len(swarm.all_edges)):
        edge_lines.append(ax.plot([], [], lw=1, linestyle="dashed", color="black"))

def anim_update(i):
    q_i = trajectories[i,:,:]

    for j in range(swarm.n):
        q_i_j = q_i[j,:]
        agent, = agents[j]
        agent_orientation, = agents_orientation[j]
        agent_vss1, = agents_vss1[j]
        agent_vss2, = agents_vss2[j]

        x = q_i_j[0]
        y = q_i_j[1]

        agent.set_data(x, y)

        if j == 0:
            agent.set_color("magenta")

        if swarm.type in {"oriented", "2SR"}:
            # phi = q_i_j[2] + np.pi/2
            phi = q_i_j[2]
            ro_x = [x, x + 0.05 * np.cos(phi)]
            ro_y = [y, y + 0.05 * np.sin(phi)]

            agent_orientation.set_data(ro_x, ro_y)

        if swarm.type == "2SR":
            [seg1_x, seg1_y] = gen_arc(q_i_j, 1)
            [seg2_x, seg2_y] = gen_arc(q_i_j, 2)

            agent_vss1.set_data(seg1_x, seg1_y)
            agent_vss2.set_data(seg2_x, seg2_y)

    for edge, j in zip(swarm.all_edges, range(len(swarm.all_edges))):
        edge = list(edge)
        node1 = edge[0]
        node2 = edge[1]

        x = [trajectories[i,node1,0], trajectories[i,node2,0]]
        y = [trajectories[i,node1,1], trajectories[i,node2,1]]

        edge_line, = edge_lines[j]
        edge_line.set_data(x, y)


def anim(swarm_, trajectories_):
    global swarm, trajectories

    swarm = swarm_
    trajectories = np.array(trajectories_)

    frames = len(trajectories)
    anim = FuncAnimation(fig, anim_update, frames, init_func=anim_init, interval=25, repeat=True)

    # # Save animation
    # mywriter = FFMpegWriter(fps=30)
    # anim.save('circle_formation.mp4', writer=mywriter, dpi=300)

    plt.show()
