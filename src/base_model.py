import random
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


def firm_output(a, b, beta, total_effort):
    return a * total_effort + b * total_effort ** beta


def utility(effort, a, b, beta, theta, endowment, effort_others, number_employees):
    return (-(((a * (effort + effort_others) + b * (effort + effort_others) ** beta) / number_employees) ** theta * (
            endowment - effort) ** (1 - theta)))


def log_utility(effort, a, b, beta, theta, endowment, effort_others, number_employees):
    return -(theta * math.log10(
        (a * (effort + effort_others) + b * (effort + effort_others) ** beta) / number_employees) + (
                     1 - theta) * math.log10(endowment - effort))


class Worker(Agent):
    """
    An individual agent represented by a node in a network
    With initial endowment 1 and preference e between 0 and 1
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.endowment = 1
        self.preference = random.uniform(0, 1)
        # create open-open interval (random.uniform is [,))
        while self.preference == 0.0:
            self.preference = random.uniform(0, 1)
        # Add current firm variable
        self.currentFirm = None
        self.newFirm = None
        self.effort = 0
        self.job_event = None
        self.active = False
        self.wealth = 0
        self.income = None

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

    def get_effort_others_current_firm(self) -> float:
        firm = self.currentFirm
        effort_others = firm.total_effort - self.effort
        return effort_others

    def get_employee_count_plusone(self) -> int:
        firm = self.currentFirm
        employee_plus_one = firm.get_employee_count() + 1
        return employee_plus_one

    def get_fixed_param_tuple_current(self) -> tuple:
        current_firm = self.currentFirm
        a = current_firm.constantReturnCoef
        b = current_firm.increasingReturnCoef
        beta = current_firm.increasingReturnExp
        theta = self.preference
        endowment = self.endowment
        effort_others = self.get_effort_others_current_firm()
        number_employees = current_firm.get_employee_count()
        if effort_others < 0:
            print("Effort others negative")
            sys.exit()
        param_tuple = (a, b, beta, theta, endowment, effort_others, number_employees)
        return param_tuple

    def get_fixed_param_tuple_other(self, firm) -> tuple:
        a = firm.constantReturnCoef
        b = firm.increasingReturnCoef
        beta = firm.increasingReturnExp
        theta = self.preference
        endowment = self.endowment
        effort_others = firm.total_effort
        number_employees = self.get_employee_count_plusone()
        if effort_others < 0:
            print("Effort others negative")
            sys.exit()
        param_tuple = (a, b, beta, theta, endowment, effort_others, number_employees)
        return param_tuple

    def get_fixed_param_tuple_startup(self, current_firm):
        a = current_firm.constantReturnCoef
        b = current_firm.increasingReturnCoef
        beta = current_firm.increasingReturnExp
        theta = self.preference
        endowment = self.endowment
        effort_others = 0
        number_employees = 1
        param_tuple = (a, b, beta, theta, endowment, effort_others, number_employees)
        return param_tuple

    def utility_max_object(self, params: tuple):
        effort_others = params[5]
        endowment = params[4]
        epsilon = sys.float_info.epsilon
        # if all others
        if effort_others == 0:
            bnds = (epsilon, endowment - epsilon)
        else:
            bnds = (0, endowment - epsilon)

        optimization_output = opt.minimize_scalar(log_utility, args=params, method="bounded", bounds=bnds)
        if not optimization_output.success:
            print("Optimization not successful")
            sys.exit()
        else:
            return optimization_output

    def effort_star(self, optimization_output):
        return optimization_output.x

    def utility_star(self, optimization_output):
        return -optimization_output.fun

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
        utility_object = self.utility_max_object(params)
        max_utility = self.utility_star(utility_object)
        max_effort = self.effort_star(utility_object)
        # return tuple (len4)
        return firm_object, "join_other", max_effort, max_utility

    def maximization_over_network_firms(self, firm_list):
        if not firm_list:
            print("No other firms found in agents network")
            return False
        else:
            list_max_tuples = []
            for firm in firm_list:
                list_max_tuples.append(self.network_firm_maximization(firm))
            return list_max_tuples

    def current_firm_maximization(self):
        firm_object = self.currentFirm
        params = self.get_fixed_param_tuple_current()
        utility_object = self.utility_max_object(params)
        max_effort = self.effort_star(utility_object)
        max_utility = self.utility_star(utility_object)
        # return tuple (len4)
        return firm_object, "stay", max_effort, max_utility

    def create_startup(self):
        startup = Firm(self.model.next_id(), self.model)
        return startup

    def startup_maximization(self):
        startup = self.create_startup()
        params = self.get_fixed_param_tuple_startup(startup)
        utility_object = self.utility_max_object(params)
        startup_effort = self.effort_star(utility_object)
        startup_utility = self.utility_star(utility_object)
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
            all_max_list = [startup_max_tuple, current_max_tuple]
        return all_max_list

    def get_max_tuple(self, list_of_tuples) -> tuple:
        return max(list_of_tuples, key=itemgetter(3))

    def step(self):

        # activate agent with certain probability (4% of agents are activated each period on average)
        if random.random() <= 1:
            self.active = True
            # The agent's step will go here
            max_tuple = self.get_max_tuple(self.get_total_max_list())
            print(max_tuple)
            self.newFirm = max_tuple[0]
            self.job_event = max_tuple[1]
            self.effort = max_tuple[2]

            if self.effort >= self.endowment:
                print("Effort bigger than endowment")
                sys.exit()

            # self.endowment -= self.effort

            if self.job_event != "startup":
                self.model.current_id -= 1
            elif self.job_event == "startup":
                self.model.schedule.add(self.newFirm)
            else:
                print("There should not be a third option")

            # update Firm Agent
            self.newFirm.new_employeeList.append(self)
        else:
            self.active = False
            max_tuple = self.current_firm_maximization()
            self.newFirm = self.currentFirm
            self.job_event = "not_active"
            self.effort = max_tuple[2]
            self.newFirm.new_employeeList.append(self)

    def advance(self):
        self.currentFirm = self.newFirm


class Firm(Agent):
    """Heterogeneous Firms in the model with random production function coefficients"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # Random a,b and beta for each firm, todo: rewrite hardcoded part and define them in params script
        self.constantReturnCoef = random.uniform(0, 0.5)
        self.increasingReturnCoef = random.uniform(3 / 4, 5 / 4)
        self.increasingReturnExp = random.uniform(3 / 2, 2)
        # Store all employee's in a list
        self.employeeList = []
        self.new_employeeList = []
        # Attributes needed for statistics
        self.age = 0  # start with 0 or 1?
        self.total_effort = 0
        self.output = 0

    @property
    def total_effort(self):
        return self._total_effort

    @total_effort.setter
    def total_effort(self, value):
        if value < 0:
            raise ValueError("Total effort cannot be negative")
        self._total_effort = value

    def get_unique_id(self) -> int:
        return self.unique_id

    def get_sum_effort(self):
        sum_effort = 0.0
        for agent in self.employeeList:
            sum_effort += agent.effort
        return sum_effort

    def update_total_effort(self):
        self.total_effort = self.get_sum_effort()

    def get_employee_count(self):
        return len(self.employeeList)

    def reset_new_employeeList(self):
        self.new_employeeList = []

    def update_output(self):
        if self.total_effort > 0:
            output = firm_output(self.constantReturnCoef, self.increasingReturnCoef, self.increasingReturnExp,
                                 self.total_effort)
        else:
            output = 0
        self.output = output

    def step(self):
        self.age += 1
        self.employeeList = self.new_employeeList
        print(self.employeeList)

        if self.employeeList:
            self.update_total_effort()
            self.update_output()
            output_share = self.output / self.get_employee_count()
            print(output_share)
            for agent in self.employeeList:
                agent.wealth += output_share
                agent.income = output_share
        else:
            self.output = 0
            self.model.dead_firms.append(self)

    def advance(self):
        self.reset_new_employeeList()


class BaseModel(Model):
    """A model with N agents connected in a network"""

    def __init__(self, num_agents, avg_node_degree=4):
        # Number of agents
        self.num_agents = num_agents  # equals number of nodes (each agent on one node)
        # Set network
        prob = avg_node_degree / self.num_agents
        self.G = nx.erdos_renyi_graph(n=self.num_agents, p=prob)
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivationByType(self)
        self.current_id = 0
        self.dead_firms = []
        self.firm_distr = []
        self.datacollector = DataCollector(
            # model_reporters= {"Firm Size Distribution": self.get_firm_size_distribution()}
            # {"Average_Output": average_output}
            #                 },
            agent_reporters={"wealth": lambda w: getattr(w, "wealth", None),
                             "output": lambda o: getattr(o, "output", None)}
        )

        # Create agents
        for i in range(self.num_agents):
            worker = Worker(self.next_id(), self)
            firm = Firm(self.next_id(), self)
            # Initial condition, every agent in a singleton firm
            worker.currentFirm = firm
            # Add current employee to firms employee list (initial condition len != 1 for all firms)
            firm.employeeList.append(worker)
            self.schedule.add(worker)
            self.schedule.add(firm)
            # Add agent to a node
            self.grid.place_agent(worker, i)

    def get_firm_size_distribution(self):
        firm_sizes = [firm.get_employee_count() for firm in self.schedule.agents_by_type[Firm].values()]
        x = sorted(firm_sizes)
        return x

    def step(self):
        """Advance the model by one step."""
        print(self.schedule.agents_by_type[Firm].values())
        self.firm_distr = self.get_firm_size_distribution()
        self.datacollector.collect(self)
        self.schedule.step(shuffle_types=False)

        for x in self.dead_firms:
            self.schedule.remove(x)
            self.dead_firms = []


# Helper functions

# Data collector functions
def get_firm_size_distribution(model):
    firm_sizes = [firm.get_employee_count() for firm in model.schedule.agents_by_type[Firm]]
    x = sorted(firm_sizes)
    return x


def average_output(model):
    firms_output = [firm.output for firm in model.schedule.agents_by_type[Firm]]
    if firms_output:
        return np.mean(firms_output)
    else:
        return 0.0
