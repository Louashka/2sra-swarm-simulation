import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

swarm = None
trajectories = []

fig, ax = plt.subplots()

agents = []
agents_orientation = []
edge_lines = []

def defineRange():
    margin = 0.15
    x_min, y_min = trajectories[:,:,:2].min(axis=1)[0]
    x_max, y_max = trajectories[:,:,:2].max(axis=1)[0]

    ax_range = max(x_max - x_min, y_max - y_min) + margin

    x_range = (x_min - margin, x_min + ax_range)
    y_range = (y_min - margin, y_min + ax_range)

    return x_range, y_range

def anim_init():
    x_range, y_range = defineRange()
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")

    for i in range(swarm.n):
        agents.append(ax.plot([], [], marker="o", markersize=10, color="None", markeredgecolor="black"))
        agents_orientation.append(ax.plot([], [], lw=1, color="magenta"))

    for i in range(len(swarm.all_edges)):
        edge_lines.append(ax.plot([], [], lw=1, linestyle="dashed", color="black"))

def anim_update(i):
    q_i = trajectories[i,:,:]

    for j in range(swarm.n):
        agent, = agents[j]
        agent_orientation, = agents_orientation[j]

        x = q_i[j,0]
        y = q_i[j,1]

        agent.set_data(x, y)

        if swarm.type == "oriented":
            phi = q_i[j,2]
            ro_x = [x, x + 0.05 * np.cos(phi)]
            ro_y = [y, y + 0.05 * np.sin(phi)]

            agent_orientation.set_data(ro_x, ro_y)

            # if j == 0:
            #     target_nodes.set_color("magenta")

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
    anim = FuncAnimation(fig, anim_update, frames, init_func=anim_init, interval=50, repeat=True)

    # # Save animation
    # mywriter = FFMpegWriter(fps=30)
    # anim.save('circle_formation.mp4', writer=mywriter, dpi=300)

    plt.show()
