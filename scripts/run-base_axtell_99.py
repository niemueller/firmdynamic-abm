from tqdm import tqdm

from src.base_model_grid import BaseModel
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
random.seed(1112)
"""
Run BaseModel x times
Write output in results folder
"""

# Meta Data
run_id = 200
out_dir = "../results/axtell_99"
# Firm Parameters
CONSTANT_RETURNS_COEF_A = 1
INCREASING_RETURNS_COEF_B = 1
INCREASING_RETURNS_EXP_BETA = 2
# Worker Parameters
# DIST_PREFERENCES_THETA = random.uniform(0, 1)
# Model Parameters
number_of_steps = 1000
number_of_agents = 100
number_of_active_agents = 1
# activation type 1 = simultaneous, 2 = asynchroneous (random)
ACTIVATION_TYPE = 2

attribute_worker_tuple = ("effort", "wealth", "income", "job_event", "tenure", "preference")
attribute_firm_tuple = ("age", "number_employees", "total_effort", "output", "average_pref")
attributes_model_tuple = ("total_firms", "numb_new_firms", "numb_dead_firms")
optimization = (1, 2)
# Create Model with (n) agents
if type(optimization) == int:
    model = BaseModel(number_of_agents,
                      CONSTANT_RETURNS_COEF_A,
                      INCREASING_RETURNS_COEF_B,
                      INCREASING_RETURNS_EXP_BETA,
                      optimization,
                      number_of_active_agents,
                      ACTIVATION_TYPE)

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
                          optimization,
                          number_of_active_agents,
                          ACTIVATION_TYPE)

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

# model_vars = model.datacollector.get_model_vars_dataframe()
# agent_vars = model.datacollector.get_agent_vars_dataframe()
#
# # create csv tables for model vars and agent vars and save it in results folder
# #model_vars.to_csv("C:/Users/41782/Documents/MasterThesis/firmdynamic-abm/results/model_vars.csv", encoding="utf-8", index=False)
# model_vars.to_csv("../results/model_vars.csv", encoding="utf-8",
#                   index=False)
