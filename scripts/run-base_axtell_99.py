from tqdm import tqdm

from src.base_model_async_random import BaseModel
# import vis_test
import logging
import random
from src.reporter import Reporter

# import src.visualisation
logging.basicConfig(level=logging.INFO)
# folder to load config file
# CONFIG_PATH = "config/"


# Function to load yaml configuration file
# def load_config(config_name):
#     with open(os.path.join(CONFIG_PATH, config_name)) as file:
#         config = yaml.safe_load(file)
#
#     return config


# config = load_config("config.yaml")
# define seed for whole run
random.seed(100)
"""
Run BaseModel x times
Write output in results folder
"""

# Meta Data
run_id = 313
out_dir = "../results/axtell_99"
# Firm Parameters
CONSTANT_RETURNS_COEF_A = 1
INCREASING_RETURNS_COEF_B = 1
INCREASING_RETURNS_EXP_BETA = 2
# Worker Parameters
# DIST_PREFERENCES_THETA = random.uniform(0, 1)
# Model Parameters
number_of_steps = 10000
number_of_agents = 1000
number_of_active_agents = 1
# activation type 1 = simultaneous, 2 = asynchroneous (random)
ACTIVATION_TYPE = 3

# optimization type
optimization = 1

# Average node degree (number of neighbors)
AVERAGE_NODE_DEGREE = 2

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
