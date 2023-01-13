import numpy as np
import mas

swarm = mas.formation(5, "dot")

# tr = mas.rendezvous(swarm)
# tr = mas.form_circle(swarm, 0.15)
mas.form_regular_polygon(swarm)
# mas.show_motion(swarm, tr)


print(swarm.all_edges)

