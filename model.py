import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


def get_bidder_id(model):
    try:
        return model.outstanding_bidder.unique_id
    except AttributeError:
        return -1


def get_asker_id(model):
    try:
        return model.outstanding_asker.unique_id
    except AttributeError:
        return -1

def compute_efficiency(model):
    agent_surpluses = np.array([agent.surplus for agent in model.schedule.agents])


class ZeroIntelligence(Agent):
    """Zero-Intelligence strategy from Gode and Sunder (1993)"""
    def __init__(self, unique_id, model, role, value):
        super().__init__(unique_id, model)
        self.role = role  # buyer or seller
        self.value = value  # value or cost

        self.surplus = 0  # profit
        self.right = 5  # right to buy
        self.good = 5  # good owned

    def bid(self, price):
        # Submit a bid to the CDA.
        self.model.update_ob(self, price)

    def ask(self, price):
        # Submit an ask to the CDA.
        self.model.update_oa(self, price)

    def buy(self, price):
        self.good += 1
        self.right -= 1
        self.surplus += self.value - price

    def sell(self, price):
        self.good -= 1
        self.right += 1
        self.surplus += price - self.value

    def step(self):
        if self.role == "buyer":
            price = random.randint(1, self.value)
            if (self.right > 0):
                self.bid(price)
        else:
            price = random.randint(self.value, 200)
            if (self.good > 0):
                self.ask(price)
        self.model.datacollector.collect(self.model)
        self.model.next_tick()


class CDAmodel(Model):
    """Continuous Double Auction model with some number of agents."""
    def __init__(self, demand, supply):
        self.period = 1
        self.tick = 1
        self.num_buyers = len(demand)
        self.num_sellers = len(supply)
        self.initialize_spread()
        self.market_price = None
        # How agents are activated at each step
        self.schedule = RandomActivation(self)
        # Create agents
        for i, value in enumerate(demand):
            self.schedule.add(
                ZeroIntelligence(i, self, "buyer", value)
            )
        for i, cost in enumerate(supply):
            self.schedule.add(
                ZeroIntelligence(self.num_buyers + i, self, "seller", cost)
            )

        # Collecting data
        self.datacollector = DataCollector(
            model_reporters={"Period": "period",
                             "Tick": "tick",
                             "OB": "outstanding_bid",
                             "OBer": get_bidder_id,
                             "OA": "outstanding_ask",
                             "OAer": get_asker_id,
                             "Market Price": "market_price",
                             "Traded": "traded"},
            agent_reporters={"Period": lambda x: x.model.period,
                             "Tick": lambda x: x.model.tick,
                             "Role": "role",
                             "Value": "value",
                             "Good": "good",
                             "Right": "right",
                             "Surplus": "surplus"}
        )

    def initialize_spread(self):
        # Initialize outstanding bid and ask
        self.outstanding_bid = 0
        self.outstanding_bidder = None
        self.outstanding_ask = math.inf
        self.outstanding_asker = None
        self.traded = 0
        self.market_price = None

    def update_ob(self, bidder, price):
        if price > self.outstanding_bid:
            # Update the outstanding bid
            self.outstanding_bid = price
            self.outstanding_bidder = bidder
        if price > self.outstanding_ask:
            self.execute_contract(price)

    def update_oa(self, asker, price):
        if price < self.outstanding_ask:
            # Update the outstanding ask
            self.outstanding_ask = price
            self.outstanding_asker = asker
        if price < self.outstanding_bid:
            self.execute_contract(price)

    def execute_contract(self, contract_price):
        self.outstanding_bidder.buy(contract_price)
        self.outstanding_asker.sell(contract_price)
        self.market_price = contract_price
        self.traded = 1

    def next_tick(self):
        if self.traded == 1:
            self.initialize_spread()
        self.tick += 1

    def step(self):
        self.schedule.step()
        self.period += 1
        self.tick = 1


class Supply:
    def __init__(self, N, q, p_min, p_max, steps):
        if N % steps != 0:
            raise ValueError("Number of agents must be divisible by number of steps")

        self.num_agents = N  # number of buyers
        self.quantity_per_agent = q  # number of units held by each buyer

        num_per_step = self.num_agents // steps
        prices = np.linspace(p_min, p_max, steps)
        self.price_schedule = np.array(
            [p for p in prices for i in range(num_per_step)]
        )
        self.quantity_supplied = np.cumsum(
            np.repeat(self.quantity_per_agent, self.num_agents)
        )

    def graph(self):
        plt.step(np.append(0, self.quantity_supplied),
                 np.append(self.price_schedule[0], self.price_schedule))
        plt.show()

    def market_graph(self, other):
        plt.step(np.append(0, self.quantity_supplied),
                 np.append(self.price_schedule[0], self.price_schedule))
        plt.step(np.append(0, other.quantity_supplied),
                 np.append(other.price_schedule[0], other.price_schedule))
        plt.show()


class Demand(Supply):
    def __init__(self, N, q, p_min, p_max, steps):
        super().__init__(N, q, p_min, p_max, steps)
        self.price_schedule = np.flip(self.price_schedule, 0)


class Supply2(Supply):
    def __init__(self, num_in, num_ex, q, p_min, p_eqb, num_per_step):
        if num_in % num_per_step != 0:
            raise ValueError("Number of intramarginal agents \
            must be divisible by number of agents per step")
        if num_ex % num_per_step != 0:
            raise ValueError("Number of extramarginal agents \
            must be divisible by number of agents per step")

        self.num_agents = num_in + num_ex  # number of buyers
        self.quantity_per_agent = q  # number of units held by each buyer

        steps_in = num_in // num_per_step
        prices_in = np.linspace(p_min, p_eqb, steps_in)
        delta = (p_eqb - p_min) / (steps_in - 1)
        steps_ex = num_ex // num_per_step
        prices_out = np.array(
            [p_eqb + i * delta for i in range(1, steps_ex + 1)]
        )
        prices = np.append(prices_in, prices_out)
        self.price_schedule = np.array(
            [p for p in prices for i in range(num_per_step)]
        )
        self.quantity_supplied = np.cumsum(
            np.repeat(self.quantity_per_agent, self.num_agents)
        )


# equilibrium price is 110
# equilibrium quantity is 24
demand = Demand(14, 3, 50, 170, 7)
supply = Supply(14, 3, 50, 170, 7)

model = CDAmodel(demand.price_schedule, supply.price_schedule)
for i in range(100):
    model.step()

data_model = model.datacollector.get_model_vars_dataframe()
data_model = data_model[data_model.Traded == 1]

data_agent = model.datacollector.get_agent_vars_dataframe()
