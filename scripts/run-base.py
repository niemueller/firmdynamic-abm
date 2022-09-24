import src.base_model
from src.base_model import BaseModel
import yaml
import os

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
Run BaseModel a 100 times
Write output in results folder
"""


def main():
    # Create Model with (n) agents
    model = BaseModel(100)

    # Run Model (i) times
    for i in range(1):
        model.step()

    model_vars = model.datacollector.get_model_vars_dataframe()
    agent_vars = model.datacollector.get_agent_vars_dataframe()

    # create csv tables for model vars and agent vars and save it in results folder
    #model_vars.to_csv("C:/Users/41782/Documents/MasterThesis/firmdynamic-abm/results/model_vars.csv", encoding="utf-8", index=False)
    model_vars.to_csv("../results/model_vars.csv", encoding="utf-8",
                      index=False)

    return model, model_vars, agent_vars


if __name__ == "__main__":
    main()
