import random
import scipy.optimize as opt
import networkx as nx
from mesa import Agent, Model
from mesa.time import RandomActivationByType
from mesa.space import NetworkGrid
import pandas as pd


def firm_output(a, b, beta, total_effort):
    return a*total_effort + b*total_effort**beta


def utility(effort, a, b, beta, theta, wealth, effort_others, number_employees):
    return -(((a*(effort + effort_others) + b*(effort + effort_others)**beta)/number_employees)**theta*(wealth - effort)**(1-theta))


def utility_2(effort, p):
    a = p[0]
    b = p[1]
    beta = p[2]
    theta = p[3]
    wealth = p[4]
    effort_others = p[5]
    number_employees = p[6]
    return -(((a*(effort + effort_others) + b*(effort + effort_others)**beta)/number_employees)**theta*(wealth - effort)**(1-theta))


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
        # Add current firm variable
        self.currentFirm = None
        self.effort = 0
        print(self.endowment, self.preference, self.currentFirm)

    def get_fixed_param_tuple(self, current_firm):
        a = current_firm.constantReturnCoef
        b = current_firm.increasingReturnCoef
        beta = current_firm.increasingReturnExp
        theta = self.preference
        wealth = self.endowment
        effort_others = current_firm.total_effort - self.effort
        number_employees = len(current_firm.employeeList)
        param_tuple = (a, b, beta, theta, wealth, effort_others, number_employees)
        return param_tuple

    def utility_max_object(self, current_firm):
        params = self.get_fixed_param_tuple(current_firm)
        optimization_output = opt.minimize_scalar(utility, args=params, method="bounded", bounds=[0, self.endowment])
        return optimization_output

    def effort_star(self, optimization_output):
        return optimization_output.x

    def utility_star(self, optimization_output):
        return -optimization_output.fun

    def get_neighbors(self):
        return self.model.grid.get_neighbors(self.pos, include_center=True)

    def get_firms_in_network(self):
        firm_network = []
        neighbor_nodes = self.get_neighbors()
        for agent in self.model.grid.get_cell_list_contents(neighbor_nodes):
            firm_network.append(agent.currentFirm)
        return firm_network

    def optimization_over_firms_in_network(self):
        firm_list = []
        effort_list = []
        utility_list = []
        for firm in self.get_firms_in_network():
            firm_list.append(firm)
            utility_object = self.utility_max_object(firm)
            effort_list.append(self.effort_star(utility_object))
            utility_list.append(self.utility_star(utility_object))
        optimization_df = pd.DataFrame()
        optimization_df["firm"] = firm_list
        optimization_df["effort"] = effort_list
        optimization_df["utility"] = utility_list
        return optimization_df

    def optimal_values(self):
        df = self.optimization_over_firms_in_network()
        return df.iloc[[df["utility"].idxmax()]]

    def step(self):
        # The agent's step will go here
        # Test purpose print agent's unique_id
        print("Hi, I am agent " + str(self.unique_id) + ".")
        neigh = self.model.grid.get_neighbors(self.pos)
        print(neigh)
        #utility_object = self.utility_max_object()
        #print(self.effort_star(utility_object))
        #print(self.utility_star(utility_object))
        print(self.optimization_over_firms_in_network())
        print(self.optimal_values())
        optimal_values = self.optimal_values()
        optimal_firm = optimal_values["firm"].item()
        self.effort = optimal_values["effort"].item()
        print(optimal_firm)
        print(self.effort)
        self.endowment -= self.effort
        if self.currentFirm != optimal_firm:
            old_firm = self.currentFirm
            old_firm.employeeList.remove(self)
            optimal_firm.employeeList.append(self)
            self.currentFirm = optimal_firm


class Firm(Agent):
    """Heterogeneous Firms in the model with random production function coefficients"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # Random a,b and beta for each firm, todo: rewrite hardcoded part and define them in params script
        self.constantReturnCoef = random.uniform(0, 0.5)
        self.increasingReturnCoef = random.uniform(3/4, 5/4)
        self.increasingReturnExp = random.uniform(3/2, 2)
        # Store all employee's in a list
        self.employeeList = []
        # Attributes needed for statistics
        self.age = 0    # start with 0 or 1?
        self.total_effort = 0

    def get_sum_effort(self):
        sum_effort = 0
        for agent in self.employeeList:
            sum_effort += agent.effort
        return sum_effort

    def step(self):
        self.age += 1
        print(self.employeeList)

        if self.employeeList:
            self.total_effort = self.get_sum_effort()
            output = firm_output(self.constantReturnCoef, self.increasingReturnCoef, self.increasingReturnExp, self.total_effort)
            output_share = output/len(self.employeeList)
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
        self.dead_firms = ()

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

    def step(self,):
        """Advance the model by one step."""
        self.schedule.step()
        for x in self.dead_firms:
            self.schedule.remove(x)
            self.dead_firms.remove(x)
