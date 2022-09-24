from mesa import Agent
from mesa.time import RandomActivationByType
from typing import Dict, Iterator, List, Type, Union

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
        agent_keys: List[int] = list(self.agents_by_type[type_class].keys())
        if shuffle_agents:
            self.model.random.shuffle(agent_keys)
        for agent_key in agent_keys:
            self.agents_by_type[type_class][agent_key].step()
        # We recompute the keys because some agents might have been removed in
        # the previous loop.
        agent_keys: List[int] = list(self.agents_by_type[type_class].keys())
        for agent_key in agent_keys:
            self.agents_by_type[type_class][agent_key].advance()
