from tqdm import tqdm

"""
Import Model

base_model.py
Original implementation with basic functionality

base_model_grid.py
Added grid optimization, many times faster than scipy.optimize algorithm, which accounts for around 80 percent of overall
runtime. Simulation results relative robust, even if agents make more errors.

base_model_async_random.py
Sequential activation of agents as in Axtell (99)
Direct calculation used, fastest implementation but only applicable for beta = 2
"""
from src.base_model_async_random import BaseModel
# import vis_test
import logging
import random
# Reporter needed to write simulation results to csv file
from src.reporter import Reporter

logging.basicConfig(level=logging.INFO)

# define seed for whole run
random.seed(1231)
"""
Run BaseModel x times
Write output in results folder
"""

# Meta Data
run_id = 500
out_dir = "../results/axtell_99"
# Firm Parameters
CONSTANT_RETURNS_COEF_A = 1
INCREASING_RETURNS_COEF_B = 1
INCREASING_RETURNS_EXP_BETA = 2
# Worker Parameters

# Model Parameters
number_of_steps = 1000
number_of_agents = 1000
number_of_active_agents = 1
"""
1 = Simultaneous
2 = Random, asynchroneous,
3 = Random, asynchroneous, agents can be active more than once each period

Not every activation type is implemented in each model scripts
"""

ACTIVATION_TYPE = 3

"""
1 = bounded scipy algorithm: slow
2 = grid search 0.1 interval: fastesr
3 = grid search 0.01 interval: faster than 1
4 = grid search 0.001 interval: slowest

For async_random script, e* and U* are directly calculated (can only be done if beta = 2)
"""
optimization = 1

"""
Average node degree (number of neighbors)
Erdos Renyi Random Graph
self.G = nx.fast_gnp_random_graph(n=self.num_agents, p=prob)
number of neighbors varies, but average as specified

Regular Graph
self.G = nx.random_regular_graph(avg_node_degree, num_agents)
Every agents has exaclty avg_node_degree neighbors

Cycle Graph with every agent having 2 neighbors
self.G = nx.cycle_graph(num_agents)
Special case of regular graph with every agent having 2 neighbors and all connected through a cycle.
"""
AVERAGE_NODE_DEGREE = 2

# Specification on what attributes should be saved and written to csv by reporter module
attribute_worker_tuple = ("effort", "wealth", "income", "job_event", "tenure", "preference")
attribute_firm_tuple = ("age", "number_employees", "total_effort", "output", "average_pref")
attributes_model_tuple = ("total_firms", "numb_new_firms", "numb_dead_firms")

# Create Model with (n) agents
if type(optimization) == int:
    model = BaseModel(number_of_agents,
                      CONSTANT_RETURNS_COEF_A,
                      INCREASING_RETURNS_COEF_B,
                      INCREASING_RETURNS_EXP_BETA,
                      optimization,
                      number_of_active_agents,
                      ACTIVATION_TYPE,
                      AVERAGE_NODE_DEGREE)

    agent_reporter = Reporter("agent",
                              run_id,
                              out_dir,
                              model,
                              attribute_worker_tuple,
                              attribute_firm_tuple,
                              attributes_model_tuple,
                              optimization)

    # Run Model (i) times
    for i in tqdm(range(number_of_steps)):
        model.step()
        agent_reporter.on_step(attribute_worker_tuple, attribute_firm_tuple, attributes_model_tuple)
        model.reset_stats()
    agent_reporter.close()

else:
    for x in optimization:
        model = BaseModel(number_of_agents,
                          CONSTANT_RETURNS_COEF_A,
                          INCREASING_RETURNS_COEF_B,
                          INCREASING_RETURNS_EXP_BETA,
                          x,
                          number_of_active_agents,
                          ACTIVATION_TYPE,
                          AVERAGE_NODE_DEGREE)

        agent_reporter = Reporter("agent",
                                  run_id,
                                  out_dir,
                                  model,
                                  attribute_worker_tuple,
                                  attribute_firm_tuple,
                                  attributes_model_tuple,
                                  x)

        # Run Model (i) times
        for i in tqdm(range(number_of_steps)):
            model.step()
            agent_reporter.on_step(attribute_worker_tuple, attribute_firm_tuple, attributes_model_tuple)
            model.reset_stats()

        agent_reporter.close()
