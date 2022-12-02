from Swarm import *

swarm = Swarm(7, "oriented")

swarm.collection[3].update([0, 0, 0])

print(swarm.collection[3].y)
