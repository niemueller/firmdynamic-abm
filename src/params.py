# Base case config of computational model Parameters needed to run the model
# Model params
# number of agents
# compensation rule
# number of neighbors v
# activation regime
# probability of agent activation/period
# time calibration: one model period
# initial condition:

# firm params
# constant returns coefficient a
# increasing returns coefficient b
# increasing returns exponent beta

# agent params
# distribution of preferences
# endowments w
import networkx as nx

N = 10
P = 0.4
graph = nx.fast_gnp_random_graph(N, P)
