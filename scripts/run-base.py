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
random.seed(1111)
"""
Run BaseModel x times
Write output in results folder
"""

run_id = "4mio"
number_of_steps = 1000
number_of_agents = 4000000
out_dir = "../results"
number_of_active_agents = 0.04
attribute_worker_tuple = ("effort", "wealth", "income", "job_event", "tenure", "preference")
attribute_firm_tuple = ("age", "number_employees", "total_effort", "output", "average_pref")
optimization = 2
# Create Model with (n) agents
if type(optimization) == int:
    model = BaseModel(number_of_agents, optimization, number_of_active_agents)

    agent_reporter = Reporter("agent", run_id, out_dir, model, attribute_worker_tuple, attribute_firm_tuple,
                              optimization)

    # Run Model (i) times
    for i in tqdm(range(number_of_steps)):
        model.step()
        agent_reporter.on_step(attribute_worker_tuple, attribute_firm_tuple)
    agent_reporter.close()

else:
    for x in optimization:
        model = BaseModel(number_of_agents, x, number_of_active_agents)

        agent_reporter = Reporter("agent", run_id, out_dir, model, attribute_worker_tuple, attribute_firm_tuple, x)

        # Run Model (i) times
        for i in tqdm(range(number_of_steps)):
            model.step()
            agent_reporter.on_step(attribute_worker_tuple, attribute_firm_tuple)

        agent_reporter.close()

# model_vars = model.datacollector.get_model_vars_dataframe()
# agent_vars = model.datacollector.get_agent_vars_dataframe()
#
# # create csv tables for model vars and agent vars and save it in results folder
# #model_vars.to_csv("C:/Users/41782/Documents/MasterThesis/firmdynamic-abm/results/model_vars.csv", encoding="utf-8", index=False)
# model_vars.to_csv("../results/model_vars.csv", encoding="utf-8",
#                   index=False)
