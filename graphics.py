import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import globals_


link_width = globals_.D_BRIDGE
link_length = globals_.L_LINK

LINK_DIAG = ((link_length / 2)**2 + (link_width / 2)**2)**(1 / 2)

font_size = 22
fig, ax = plt.subplots()
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)

links = []
arcs1 = []
arcs2 = []
centres = []

for i in range(3):
    links.append(Rectangle((0, 0), 0, 0, fc='y'))
    arcs1.append(ax.plot([], [], lw=2, color="blue"))
    arcs2.append(ax.plot([], [], lw=2, color="blue"))
    centres.append(ax.plot([], [], lw=1, marker=".", color="black"))

swarm_config_states = []
swarm_stiffness_states = []

x_range = 0
y_range = 0


# ax.axis('off')


def init():
    global ax, x_range, y_range

    x_range, y_range = defineRange()
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")

    for link in links:
        ax.add_patch(link)


def defineRange():
    margin = 0.06

    q_array = np.array(swarm_config_states)
    x_min, y_min = q_array[:, :, :2].min(axis=1)[0]
    x_max, y_max = q_array[:, :, :2].max(axis=1)[0]

    ax_range = max(x_max - x_min, y_max - y_min) + margin

    x_range = (x_min - margin, x_min + ax_range)
    y_range = (y_min - margin, y_min + ax_range)

    return x_range, y_range


def genArc(q, seg):
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


def update(i):
    global links, arcs1, arcs2, centres

    swarm_config = swarm_config_states[i]
    swarm_stiffness = swarm_config_states[i]

    for j in range(3):

        q = swarm_config[j]
        s = swarm_stiffness[j]
        link = links[j]
        arc1, = arcs1[j]
        arc2, = arcs2[j]
        centre,= centres[j]

        x = q[0]
        y = q[1]
        phi = q[2]

        x0 = x - link_length / 2
        y0 = y - link_width / 2

        link.set_width(link_length)
        link.set_height(link_width)
        link.set_xy([x0, y0])

        transform = mpl.transforms.Affine2D().rotate_around(
            x, y, phi) + ax.transData
        link.set_transform(transform)

        seg1 = genArc(q, 1)
        seg2 = genArc(q, 2)

        arc1.set_data(seg1[0], seg1[1])
        arc2.set_data(seg2[0], seg2[1])

        if s[0] == 0:
            arc1.set_color("blue")
        else:
            arc1.set_color("red")

        if s[1] == 0:
            arc2.set_color("blue")
        else:
            arc2.set_color("red")

        centre.set_data(x, y)

    return links, arcs1, arcs2, centres,


def plotMotion(config_states, stiffness_states, frames):
    global swarm_config_states, swarm_stiffness_states

    swarm_config_states = config_states
    swarm_stiffness_states = stiffness_states

    anim = FuncAnimation(fig, update, frames,
                         init_func=init, interval=1, repeat=True)

    # Save animation
    # mywriter = FFMpegWriter(fps=30)
    # anim.save('Animation/sim_for_video_5.mp4', writer=mywriter, dpi=300)

    plt.show()
