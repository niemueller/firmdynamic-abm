from tqdm import tqdm

from src.base_model import BaseModel
import logging
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

"""
Run BaseModel x times
Write output in results folder
"""

run_id = "120_bounded_1"
out_dir = "../results"
attribute_worker_tuple = ("effort", "job_event", "wealth", "income", "job_event", "tenure", "preference")
attribute_firm_tuple = ("age", "number_employees", "total_effort", "output", "average_pref")
# Create Model with (n) agents
model = BaseModel(120000)

agent_reporter = Reporter("agent", run_id, out_dir, model, attribute_worker_tuple, attribute_firm_tuple)

# Run Model (i) times
for i in tqdm(range(3000)):
    model.step()
    agent_reporter.on_step(attribute_worker_tuple, attribute_firm_tuple)

# model_vars = model.datacollector.get_model_vars_dataframe()
# agent_vars = model.datacollector.get_agent_vars_dataframe()
#
# # create csv tables for model vars and agent vars and save it in results folder
# #model_vars.to_csv("C:/Users/41782/Documents/MasterThesis/firmdynamic-abm/results/model_vars.csv", encoding="utf-8", index=False)
# model_vars.to_csv("../results/model_vars.csv", encoding="utf-8",
#                   index=False)

agent_reporter.close()
