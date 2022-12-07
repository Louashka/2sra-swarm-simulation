import numpy as np
import mas

swarm = mas.formation(11, "oriented")

# tr = mas.rendezvous(swarm)
tr = mas.form_circle(swarm, 0.3)

mas.show_motion(swarm, tr)


