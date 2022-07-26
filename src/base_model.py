import random
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid


class Worker(Agent):
    """
    An individual agent represented by a node in a network
    With initial endowment 1 and preference e between 0 and 1
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.endowment = 1
        self.preference = random.uniform(0, 1)
        # production function OUTPUT = a*(Sum of Efforts) + b(Sum of Efforts)^beta

    def step(self):
        # The agent's step will go here
        # Test purpose print agent's unique_id
        print("Hi, I am agent " + str(self.unique_id) + ".")
        neigh = self.model.grid.get_neighbors(self.pos)
        print(neigh)


class Firm(Agent):
    """Firms in the model"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


class BaseModel(Model):
    """A model with N agents connected in a network"""

    def __init__(self, N, graph):
        # Number of agents
        self.num_agents = N
        # Set network
        self.graph = graph
        self.grid = NetworkGrid(graph)
        self.schedule = RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = Worker(i, self)
            self.schedule.add(a)
            # Add agent to a node
            self.grid.place_agent(a, i)

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
