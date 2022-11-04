from mesa import Agent
from mesa.time import RandomActivationByType
from typing import Dict, Iterator, List, Type, Union, Iterable
import random
import numpy as np

class SimultaneousActivationByType(RandomActivationByType):
    '''
    by type is only implemented in randomactivation
    '''

    def step_type(self, type_class: Type[Agent], shuffle_agents: bool = True) -> None:
        """
        Shuffle order and run all agents of a given type.
        This method is equivalent to the NetLogo 'ask [breed]...'.

        Args:
            type_class: Class object of the type to run.
        """

        agent_keys = self.agents_by_type[type_class].keys()
        for agent_key in agent_keys:
            self.agents_by_type[type_class][agent_key].step()

        for agent_key in agent_keys:
            self.agents_by_type[type_class][agent_key].advance()


        # agent_keys: List[int] = list(self.agents_by_type[type_class].keys())
        # if shuffle_agents:
        #     self.model.random.shuffle(agent_keys)
        # for agent_key in agent_keys:
        #     self.agents_by_type[type_class][agent_key].step()
        # # We recompute the keys because some agents might have been removed in
        # # the previous loop.
        # #agent_keys: List[int] = list(self.agents_by_type[type_class].keys())
        # for agent_key in agent_keys:
        #     self.agents_by_type[type_class][agent_key].advance()

class PoissonActiveByType(RandomActivationByType):

    def step(self, shuffle_types: bool = True, shuffle_agents: bool = True) -> None:
        """
        Executes the step of each agent type, one at a time, in random order.

        Args:
            shuffle_types: If True, the order of execution of each types is
                           shuffled.
            shuffle_agents: If True, the order of execution of each agents in a
                            type group is shuffled.
        """
        type_keys = list(self.agents_by_type.keys())
        worker = type_keys[0]
        firm = type_keys[1]

        # asynchroneous activation of N agents
        worker_keys: Iterable[int] = self.agents_by_type[worker].keys()

        # Using poisson() method
        # numb_agents = len(worker_keys)
        # list_agents = list(worker_keys)
        # random.shuffle(list_agents)
        # poisson_dist = np.random.poisson(1, numb_agents)
        # activated_workers = random.choices(list_agents, weights=poisson_dist, k=numb_agents)

        #Uniform random draw
        numb_agents = len(worker_keys)
        list_agents = list(worker_keys)
        random.shuffle(list_agents)
        activated_workers = random.choices(list_agents, k=numb_agents)
        for active in activated_workers:
            self.agents_by_type[worker][active].move()
        # step for workers
        self.step_type(worker, shuffle_agents=shuffle_agents)
        # step for all firms
        self.step_type(firm, shuffle_agents=shuffle_agents)

        self.steps += 1
        self.time += 1

    def step_type(self, type_class: type[Agent], shuffle_agents: bool = True) -> None:
        """
        Shuffle order and run all agents of a given type.
        This method is equivalent to the NetLogo 'ask [breed]...'.

        Args:
            type_class: Class object of the type to run.
        """
        agent_keys: Iterable[int] = self.agents_by_type[type_class].keys()
        if shuffle_agents:
            agent_keys = list(agent_keys)
            self.model.random.shuffle(agent_keys)
        for agent_key in agent_keys:
            self.agents_by_type[type_class][agent_key].step()