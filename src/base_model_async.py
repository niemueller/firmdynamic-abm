import logging
import random

import pandas as pd
import scipy.optimize as opt
import networkx as nx
from mesa import Agent, Model
from mesa.time import RandomActivationByType
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import numpy as np
import math
import sys
from operator import itemgetter

# would be useful for defining intervals (open intervals) (intervals renamed to portion but could not install)
# import portion as P not used, delete package
from src.mesa_ext import SimultaneousActivationByType


def firm_output(a, b, beta, total_effort):
    return a * total_effort + b * total_effort ** beta


def utility_func(effort, a, b, beta, theta, endowment, effort_others, number_employees):
    return ((a * (effort + effort_others) + b * (effort + effort_others) ** beta) / number_employees) ** theta * (
            endowment - effort) ** (1 - theta)


def log_utility(effort, a, b, beta, theta, endowment, effort_others, number_employees):
    return (-(theta * np.log((a * (effort + effort_others) + b * (effort + effort_others) ** beta)
                             / number_employees) + (1 - theta) * np.log(endowment - effort)))


def e_star(a, b, theta, endowment, effort_others):
    return max(0, (-a-2*b*(effort_others-theta)+(a**2+4*a*b*theta**2*(endowment+effort_others)+4*b**2*theta**2*(1+effort_others)**2)**(1/2))/(2*b*(1+theta)))


class MyAgent(Agent):

    def __init__(self, unique_id, model, type):
        super().__init__(unique_id, model)
        self.type = type

    @staticmethod
    def getResultsHeader(*args):
        converted_list = [str(arg) for arg in args]
        joined_string = ",".join(converted_list)
        return joined_string

    # @staticmethod
    # def getResultsHeader(attribute):
    #     return ['"' + attribute + '"']

    def getStepResults(self, attribute_tuple):
        a_list = []
        for arg in attribute_tuple:
            x = getattr(self, arg)
            # if type(x) != str:
            #     a_list.append(round(getattr(self, arg), 3))
            # else:
            #     a_list.append(getattr(self, arg))
            a_list.append(getattr(self, arg))

        return a_list


class Worker(MyAgent):
    """
    An individual agent represented by a node in a network
    With endowment 1 and preference e between 0 and 1
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, "W")
        self.model = model
        self.endowment = 1
        self.preference = random.uniform(0, 1)
        # create open-open interval (random.uniform is [,))
        while self.preference == 0.0:
            self.preference = random.uniform(0, 1)
        # Add current firm variable
        self.currentFirm = None
        self.newFirm = None
        self.effort = 0
        self.oldeffort = 0
        self.job_event = None
        self.active = False
        self.wealth = 0
        self.income = None
        self.tenure = 0

        # @staticmethod
        # def getResultsHeader(attribute):
        #     return ['"'+attribute+'"']

    # def getStepResults(self):
    #     return [str(self.wealth)]

    @property
    def endowment(self):
        return self._endowment

    @endowment.setter
    def endowment(self, value):
        if value <= 0:
            raise ValueError("Endowment cannot be 0 or below")
        self._endowment = value

    @property
    def effort(self):
        return self._effort

    @effort.setter
    def effort(self, value):
        if value < 0:
            raise ValueError("Effort cannot be negative")
        self._effort = value

    # def create_parameter_table(self):
    #     col_names = ["firm_id", "move_type", "param_tuple", "effort", "utility"]
    #     df = pd.DataFrame(columns=col_names)
    #     return df


    def get_effort_others_current_firm(self) -> float:
        firm = self.currentFirm
        effort_others = firm.total_effort - self.oldeffort
        return effort_others

    def get_employee_count_plusone(self, firm) -> int:
        employee_plus_one = firm.number_employees + 1
        return employee_plus_one

    def get_fixed_param_tuple_current(self) -> tuple:
        current_firm = self.currentFirm
        a = current_firm.constantReturnCoef
        b = current_firm.increasingReturnCoef
        beta = current_firm.increasingReturnExp
        theta = self.preference
        endowment = self.endowment
        if current_firm.number_employees == 1:
            effort_others = 0
        else:
            effort_others = self.get_effort_others_current_firm()
        number_employees = current_firm.number_employees
        if effort_others < 0:
            raise ValueError("Effort others negative")

        param_tuple = (a, b, beta, theta, endowment, effort_others, number_employees)
        return param_tuple

    def get_fixed_param_tuple_other(self, firm) -> tuple:
        a = firm.constantReturnCoef
        b = firm.increasingReturnCoef
        beta = firm.increasingReturnExp
        theta = self.preference
        endowment = self.endowment
        effort_others = firm.total_effort
        number_employees = self.get_employee_count_plusone(firm)
        if effort_others < 0:
            raise ValueError("Effort others negative")

        param_tuple = (a, b, beta, theta, endowment, effort_others, number_employees)
        return param_tuple

    def get_fixed_param_tuple_startup(self, firm):
        a = firm.constantReturnCoef
        b = firm.increasingReturnCoef
        beta = firm.increasingReturnExp
        theta = self.preference
        endowment = self.endowment
        effort_others = 0
        number_employees = 1
        param_tuple = (a, b, beta, theta, endowment, effort_others, number_employees)
        return param_tuple

    def utility_max_object(self, params: tuple):
        params = params
        effort_others = params[5]
        endowment = params[4]
        epsilon = sys.float_info.epsilon
        # if all others
        if effort_others == 0:
            bnds = (epsilon, endowment - epsilon)
        else:
            bnds = (0, endowment - epsilon)

        effort_ana = e_star(params[0], params[1], params[3], 1, params[5])
        utility_ana = utility_func(effort_ana, *params)
        # optimization_output = opt.minimize_scalar(log_utility, args=params, method="bounded", bounds=bnds)
        # if not optimization_output.success:
        #     raise ValueError("Optimization not successful")
        # else:
        #     effort_star = optimization_output.x
        #     utility_star = -optimization_output.fun
        # print(effort_ana, effort_star)
        # print(utility_ana, utility_star)
        return effort_ana, utility_ana

    def utility_max_object_grid(self, gridsize: int, params: tuple):
        effort_others = params[5]
        endowment = params[4]
        grid_step = endowment/gridsize
        # if all others
        if effort_others == 0:
            grid = np.arange(grid_step, 1, grid_step).tolist()
            initial_effort = 0.1
        else:
            grid = np.arange(0, 1, grid_step).tolist()
            initial_effort = 0

        initial_utility = 0

        for i in grid:
            utility_temp = -log_utility(i, params[0], params[1], params[2],
                                        params[3], params[4], params[5], params[6])
            if utility_temp >= initial_utility:
                initial_utility = utility_temp
                initial_effort = i
        return initial_effort, initial_utility

    def get_neighbors(self):
        return self.model.grid.get_neighbors(self.pos, include_center=False)

    def get_firms_in_network(self) -> list:
        # does not include current firm
        firm_network = []
        neighbor_nodes = self.get_neighbors()
        for agent in self.model.grid.get_cell_list_contents(neighbor_nodes):
            if agent.currentFirm != self.currentFirm:
                firm_network.append(agent.currentFirm)
        res = []
        [res.append(x) for x in firm_network if x not in res]
        return res

    def network_firm_maximization(self, firm_object):
        params = self.get_fixed_param_tuple_other(firm_object)
        # choose optimization type:
        if self.model.OPTIMIZATION == 1:
            utility_object = self.utility_max_object(params)
        elif self.model.OPTIMIZATION == 2:
            utility_object = self.utility_max_object_grid(gridsize=10, params=params)
        elif self.model.OPTIMIZATION == 3:
            utility_object = self.utility_max_object_grid(gridsize=100, params=params)
        elif self.model.OPTIMIZATION == 4:
            utility_object = self.utility_max_object_grid(gridsize=1000, params=params)
        else:
            utility_object = self.utility_max_object(params)

        max_effort = utility_object[0]
        max_utility = utility_object[1]
        # return tuple (len4)
        return firm_object, "join_other", max_effort, max_utility

    def maximization_over_network_firms(self, firm_list):
        if not firm_list:
            logging.debug("No other firms found in agents network")
            return False
        else:
            list_max_tuples = []
            for firm in firm_list:
                list_max_tuples.append(self.network_firm_maximization(firm))
            return list_max_tuples

    def current_firm_maximization(self):
        firm_object = self.currentFirm
        params = self.get_fixed_param_tuple_current()
        # choose optimization type:
        if self.model.OPTIMIZATION == 1:
            utility_object = self.utility_max_object(params)
        elif self.model.OPTIMIZATION == 2:
            utility_object = self.utility_max_object_grid(gridsize=10, params=params)
        elif self.model.OPTIMIZATION == 3:
            utility_object = self.utility_max_object_grid(gridsize=100, params=params)
        elif self.model.OPTIMIZATION == 4:
            utility_object = self.utility_max_object_grid(gridsize=1000, params=params)
        else:
            utility_object = self.utility_max_object(params)

        max_effort = utility_object[0]
        max_utility = utility_object[1]
        # return tuple (len4)
        return firm_object, "stay", max_effort, max_utility

    def create_startup(self):
        startup = Firm(self.model.next_id(), self.model)
        return startup

    def startup_maximization(self):
        startup = self.create_startup()
        params = self.get_fixed_param_tuple_startup(startup)
        # choose optimization type:
        if self.model.OPTIMIZATION == 1:
            utility_object = self.utility_max_object(params)
        elif self.model.OPTIMIZATION == 2:
            utility_object = self.utility_max_object_grid(gridsize=10, params=params)
        elif self.model.OPTIMIZATION == 3:
            utility_object = self.utility_max_object_grid(gridsize=100, params=params)
        elif self.model.OPTIMIZATION == 4:
            utility_object = self.utility_max_object_grid(gridsize=1000, params=params)
        else:
            utility_object = self.utility_max_object(params)

        startup_effort = utility_object[0]
        startup_utility = utility_object[1]
        # return tuple (len4)
        return startup, "startup", startup_effort, startup_utility

    def get_total_max_list(self):
        join_other_max_list = self.maximization_over_network_firms(self.get_firms_in_network())
        startup_max_tuple = self.startup_maximization()
        current_max_tuple = self.current_firm_maximization()
        # check if join_other_max_list is empty
        if join_other_max_list:
            all_max_list = join_other_max_list
            all_max_list.append(startup_max_tuple)
            all_max_list.append(current_max_tuple)
        else:
            all_max_list = [current_max_tuple, startup_max_tuple]
        return all_max_list

    def get_max_tuple(self, list_of_tuples) -> tuple:
        return max(list_of_tuples, key=itemgetter(3))

    def increase_tenure(self):
        self.tenure += 1

    def reset_tenure(self):
        self.tenure = 0

    def step(self):

        if random.random() <= self.model.activate:
            self.active = True
            # The agent's step will go here

            # safe last period's effort
            self.oldeffort = self.effort
            self.currentFirm = self.newFirm
            # maximization
            max_tuple = self.get_max_tuple(self.get_total_max_list())
            self.newFirm = max_tuple[0]
            self.job_event = max_tuple[1]
            self.effort = max_tuple[2]

            # subtract last period's effort from current firm
            # self.currentFirm.minus_effort(self.oldeffort)
            self.currentFirm.minus_employee()
            self.currentFirm.remove_employee_list(self)
            self.currentFirm.calc_total_effort()

            # add agents attributes to new firm
            self.newFirm.add_employee_list(self)
            self.newFirm.calc_total_effort()

            # self.newFirm.add_employee_list(self)

            if self.effort >= self.endowment:
                print("Effort bigger than endowment")
                sys.exit()

            if self.job_event == "stay":
                self.model.current_id -= 1
                self.newFirm.plus_employee()

            elif self.job_event == "startup":
                self.model.schedule.add(self.newFirm)
                self.model.add_new_firm()

            elif self.job_event == "join_other":
                # add to new firm (not needed for startup as founder already in employee count)
                self.model.current_id -= 1
                self.newFirm.plus_employee()

            else:
                print("There should not be a 4th option")

            # set new firm as current firm
            self.currentFirm = self.newFirm

        else:
            self.active = False

            # safe last period's effort
            self.oldeffort = self.effort
            self.currentFirm = self.newFirm

            # maximization
            max_tuple = self.current_firm_maximization()
            self.newFirm = self.currentFirm
            self.job_event = "not_active"
            self.effort = max_tuple[2]

            # subtract last period's effort from current firm
            # self.currentFirm.plus_effort(self.effort)
            # self.currentFirm.minus_effort(self.oldeffort)
            self.currentFirm.calc_total_effort()


    def advance(self):
        self.currentFirm = self.newFirm
        if self.job_event == "stay" or self.job_event == "not_active":
            self.increase_tenure()
        else:
            self.reset_tenure()


class Firm(MyAgent):
    """Heterogeneous Firms in the model with random production function coefficients"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, "F")

        # Random a,b and beta for each firm, todo: rewrite hardcoded part and define them in params script
        #     self.constantReturnCoef = random.uniform(0, 0.5)
        #     self.increasingReturnCoef = random.uniform(3 / 4, 5 / 4)
        #     self.increasingReturnExp = random.uniform(3 / 2, 2)
        self.constantReturnCoef = self.model.a
        self.increasingReturnCoef = self.model.b
        self.increasingReturnExp = self.model.beta
        # self.increasingReturnExp = 1.3
        # Store all employee's in a list
        self.employeeList = []
        # Attributes needed for statistics
        self.age = 0  # start with 0 or 1?
        self.total_effort = 0
        self.output = 0
        self.number_employees = 1
        self.average_pref = None

    # @staticmethod
    # def getResultsHeader(attribute):
    #     return ['"' + attribute + '"']

    # def getStepResults(self):
    #     return [str(self.output)]

    @property
    def total_effort(self):
        return self._total_effort

    @total_effort.setter
    def total_effort(self, value):
        if value < 0:
            raise ValueError(f"Total effort cannot be negative: {self._total_effort}")
        self._total_effort = value

    def calc_total_effort(self):
        total = 0
        for w in self.employeeList:
            total += w.effort
        self.total_effort = total

    def minus_effort(self, effort):
        self.total_effort -= effort

    def plus_effort(self, effort):
        self.total_effort += effort

    def minus_employee(self):
        self.number_employees -= 1

    def plus_employee(self):
        self.number_employees += 1

    def add_employee_list(self, worker):
        self.employeeList.append(worker)

    def remove_employee_list(self, worker):
        self.employeeList.remove(worker)

    def get_unique_id(self) -> int:
        return self.unique_id

    def compute_average_preference(self):
        pref_list = []
        for a in self.employeeList:
            pref_list.append(a.effort)
        return np.mean(pref_list)

    def update_output(self):
        if self.total_effort > 0:
            output = firm_output(self.constantReturnCoef, self.increasingReturnCoef, self.increasingReturnExp,
                                 self.total_effort)
        else:
            output = 0
        self.output = output

    def step(self):
        self.age += 1
        # self.employeeList = self.new_employeeList
        logging.debug(f"Age:{self.age}\tNumEmployees: {self.number_employees}")
        if self.employeeList:
            self.update_output()
            self.average_pref = self.compute_average_preference()
            output_share = self.output / self.number_employees
            for agent in self.employeeList:
                agent.wealth += output_share
                agent.income = output_share
        else:
            self.output = 0
            self.model.dead_firms.append(self)


class BaseModel(Model):
    """A model with N agents connected in a network"""

    def __init__(self, num_agents, a, b, beta,  optimization: int, activate: float, activation_type, avg_node_degree):
        # Number of agents
        self.num_agents = num_agents  # equals number of nodes (each agent on one node)
        # Optimization used:
        # 1 = bounded scipy algorithm: slow
        # 2 = grid search 0.1 interval: fastest
        # 3 = grid search 0.01 interval: faster than 1
        # 4 = grid search 0.001 interval
        self.OPTIMIZATION = optimization
        # number of active agents per period (floating decimal, corresponds to monthly job searches)
        self.activate = activate
        # activation type (1 = Simultaneous, 2 = Random (asynchroneous))
        self.activation_type = activation_type
        # Firm parameters
        self.a = a
        self.b = b
        self.beta = beta
        # worker parameter
        # self.theta = theta
        # Set network
        prob = avg_node_degree / self.num_agents
        logging.info("create graph")
        # Erdos Renyi Random Graph
        # self.G = nx.fast_gnp_random_graph(n=self.num_agents, p=prob)
        # Regular Graph with avg_node_degree = # of neighbors
        self.G = nx.random_regular_graph(avg_node_degree, num_agents)
        # Cycle Graph with every agent having 2 neighbors
        # self.G = nx.cycle_graph(num_agents)
        self.grid = NetworkGrid(self.G)
        if self.activation_type == 1:
            self.schedule = SimultaneousActivationByType(self)
        elif self.activation_type == 2:
            self.schedule = RandomActivationByType(self)
        else:
            print("no valid activation type set")

        self.current_id = 0
        self.dead_firms = []
        self.numb_new_firms = 0
        self.total_firms = num_agents
        self.numb_dead_firms = 0
        logging.info("graph done")

        # Create agents
        for i in range(self.num_agents):
            worker = Worker(self.next_id(), self)
            firm = Firm(self.next_id(), self)
            # Initial condition, every agent in a singleton firm
            worker.currentFirm = firm
            worker.newFirm = firm
            # Add current employee to firms employee list (initial condition len != 1 for all firms)
            firm.employeeList.append(worker)
            self.schedule.add(worker)
            self.schedule.add(firm)
            # Add agent to a node
            self.grid.place_agent(worker, i)

        logging.info("initialization done")

    def count_dead_firms(self):
        self.numb_dead_firms = len(self.dead_firms)

    def add_new_firm(self):
        self.numb_new_firms += 1

    def count_total_firms(self):
        self.total_firms = self.total_firms - self.numb_dead_firms + self.numb_new_firms

    def step(self):
        """Advance the model by one step."""
        self.schedule.step(shuffle_types=False, shuffle_agents=True)

        self.count_dead_firms()
        self.count_total_firms()
        for x in self.dead_firms:
            self.schedule.remove(x)

    def reset_stats(self):

        self.dead_firms = []
        self.numb_dead_firms = 0
        self.numb_new_firms = 0

    @staticmethod
    def getResultsHeader(*args):
        converted_list = [str(arg) for arg in args]
        joined_string = ",".join(converted_list)
        return joined_string

    # @staticmethod
    # def getResultsHeader(attribute):
    #     return ['"' + attribute + '"']

    def getStepResults(self, attribute_tuple):
        a_list = []
        for arg in attribute_tuple:
            x = getattr(self, arg)
            if type(x) != str:
                a_list.append(round(getattr(self, arg), 3))
            else:
                a_list.append(getattr(self, arg))

        return a_list
