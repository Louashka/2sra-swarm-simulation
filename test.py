from swarmgraph import *

swarm = SwarmGraph(7, "oriented")
print(swarm.laplacian)
swarm.complete_graph()

print(swarm)
print(swarm.laplacian)
