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
                 np.append(0, self.price_schedule))
        plt.show()


demand = np.linspace(150, 50, 6)
supply = np.linspace(50, 150, 6)

model = CDAmodel(demand, supply)
for i in range(100):
    model.step()
