import random
import scipy.optimize as opt
import networkx as nx
from mesa import Agent, Model
from mesa.time import RandomActivationByType
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import pandas as pd
import math
import sys
# would be useful for defining intervals (open intervals) (intervals renamed to portion but could not install)
# import portion as P not used, delete package


def firm_output(a, b, beta, total_effort):
    return a * total_effort + b * total_effort ** beta


def utility(effort, a, b, beta, theta, wealth, effort_others, number_employees):
    return (-(((a * (effort + effort_others) + b * (effort + effort_others) ** beta) / number_employees) ** theta * (
                wealth - effort) ** (1 - theta)))


def log_utility(effort, a, b, beta, theta, wealth, effort_others, number_employees):
    return (-(theta*math.log10(a*(effort+effort_others)+b*(effort+effort_others)**beta)
            - theta*math.log10(number_employees)
            + (1-theta)*math.log10(wealth-effort)))


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
        # production function OUTPUT = a*(Sum of Efforts) + b(Sum of Efforts)^beta
        # Add current firm variable
        self.currentFirm = None
        self.effort = 0

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

    def get_fixed_param_tuple(self, current_firm):
        a = current_firm.constantReturnCoef
        b = current_firm.increasingReturnCoef
        beta = current_firm.increasingReturnExp
        theta = self.preference
        wealth = self.endowment
        current_firm.update_total_effort()
        effort_others = current_firm.total_effort - self.effort
        number_employees = len(current_firm.employeeList)
        param_tuple = (a, b, beta, theta, wealth, effort_others, number_employees)
        return param_tuple

    def utility_max_object(self, current_firm):
        params = self.get_fixed_param_tuple(current_firm)
        if current_firm.get_employee_count() == 1:
            # open interval as logUtility not defined on e = 0 and e = w
            bnds = [0+sys.float_info.epsilon, self.endowment-sys.float_info.epsilon]
        # closed-open interval as logUtility not defined e = w
        elif current_firm.get_employee_count() > 1:
            bnds = [0, self.endowment-sys.float_info.epsilon]
        else:
            print("Current firm has no employee")
            sys.exit()

        optimization_output = opt.minimize_scalar(log_utility, args=params, method="bounded", bounds=bnds)

        if not optimization_output.success:
            print("Optimization not successful")
            sys.exit()
        else:
            return optimization_output

    def effort_star(self, optimization_output):
        return optimization_output.x

    def utility_star(self, optimization_output):
        return (-optimization_output.fun)

    def get_neighbors(self):
        return self.model.grid.get_neighbors(self.pos, include_center=True)

    def get_firms_in_network(self):
        firm_network = []
        neighbor_nodes = self.get_neighbors()
        for agent in self.model.grid.get_cell_list_contents(neighbor_nodes):
            firm_network.append(agent.currentFirm)
        res = []
        [res.append(x) for x in firm_network if x not in res]
        return res

    def optimization_over_firms_in_network(self):
        firm_list = []
        effort_list = []
        utility_list = []
        for firm in self.get_firms_in_network():
            firm_list.append(firm)
            utility_object = self.utility_max_object(firm)
            util = self.utility_star(utility_object)
            effort = self.effort_star(utility_object)
            utility_list.append(util)
            effort_list.append(effort)

        optimization_df = pd.DataFrame()
        optimization_df["firm"] = firm_list
        optimization_df["effort"] = effort_list
        optimization_df["utility"] = utility_list
        return optimization_df

    def create_startup(self):
        startup = Firm(self.model.next_id(), self.model)
        startup.employeeList.append(self)
        return startup

    def startup_maximization(self, startup):
        startup = startup
        params = self.get_fixed_param_tuple(startup)
        utility_object = self.utility_max_object(startup)
        startup_effort = self.effort_star(utility_object)
        startup_utility = self.utility_star(utility_object)
        startup_df = {"firm": startup, "effort": startup_effort, "utility": startup_utility}
        return startup_df

    def optimal_values(self):
        df = self.optimization_over_firms_in_network()
        return df.iloc[[df["utility"].idxmax()]]

    def step(self):
        # The agent's step will go here
        print(self.optimization_over_firms_in_network())
        optimal_values = self.optimal_values()

        if optimal_values["effort"].item() >= self.endowment:
            print("Effort bigger than endowment")
            sys.exit()

        # Join existing firm or start new firm
        startup = self.create_startup()
        startup_df = self.startup_maximization(startup)
        startup_utility = startup_df["utility"]
        if optimal_values["utility"].item() >= startup_utility:
            self.model.current_id -= 1
            optimal_firm = optimal_values["firm"].item()
            print(optimal_firm)
            self.effort = optimal_values["effort"].item()
            print(self.effort)
            self.endowment -= self.effort
            if self.currentFirm != optimal_firm:
                old_firm = self.currentFirm
                old_firm.employeeList.remove(self)
                optimal_firm.employeeList.append(self)
                self.currentFirm = optimal_firm
        else:
            optimal_firm = startup_df["firm"]
            print(optimal_firm.unique_id)
            self.effort = startup_df["effort"]
            self.endowment -= self.effort
            old_firm = self.currentFirm
            old_firm.employeeList.remove(self)
            self.currentFirm = optimal_firm
            self.model.schedule.add(optimal_firm)


class Firm(Agent):
    """Heterogeneous Firms in the model with random production function coefficients"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # Random a,b and beta for each firm, todo: rewrite hardcoded part and define them in params script
        self.constantReturnCoef = random.uniform(0, 0.5)
        self.increasingReturnCoef = random.uniform(3/4, 5/4)
        self.increasingReturnExp = random.uniform(2, 3/2)
        # Store all employee's in a list
        self.employeeList = []
        # Attributes needed for statistics
        self.age = 0  # start with 0 or 1?
        self.total_effort = 0

    @property
    def total_effort(self):
        return self._total_effort

    @total_effort.setter
    def total_effort(self, value):
        if value < 0:
            raise ValueError("Total effort cannot be negative")
        self._total_effort = value

    def get_sum_effort(self):
        sum_effort = 0
        for agent in self.employeeList:
            sum_effort += agent.effort
        return sum_effort

    def update_total_effort(self):
        self.total_effort = self.get_sum_effort()

    def get_employee_count(self):
        return len(self.employeeList)

    def step(self):
        self.age += 1
        print(self.employeeList)

        if self.employeeList:
            self.total_effort = self.get_sum_effort()
            output = firm_output(self.constantReturnCoef, self.increasingReturnCoef, self.increasingReturnExp,
                                 self.total_effort)
            output_share = output / len(self.employeeList)
            for agent in self.employeeList:
                agent.endowment += output_share
        else:
            self.model.dead_firms.append(self)


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
        self.datacollector = DataCollector(
            # model_reporters={"Size": self.get_firm_size_distribution()},
            agent_reporters={"wealth": lambda w: getattr(w, "endowment", None)}
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

    # Data collector functions
    def get_firm_size_distribution(self):
        firm_sizes = [len(firm.employeeList) for firm in self.schedule.agents_by_type[Firm]]
        x = sorted(firm_sizes)
        return x

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step(shuffle_types=False)

        for x in self.dead_firms:
            self.schedule.remove(x)
            self.dead_firms = []
