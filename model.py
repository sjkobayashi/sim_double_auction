import random
import math
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

# ------------------------------ scheduler ------------------------------


# ------------------------------ Data functions ------------------------------


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
    producer_surplus = model.supply.theoretical_surplus
    consumer_surplus = model.demand.theoretical_surplus
    total_surplus = producer_surplus + consumer_surplus
    actual_surplus = np.sum(np.fromiter(
        (agent.surplus for agent in model.schedule.agents), float
    ))
    return actual_surplus / total_surplus

# ------------------------------ Traders ------------------------------


class Trader(Agent):
    def __init__(self, unique_id, model, role, value, q):
        super().__init__(unique_id, model)
        self.role = role  # buyer or seller
        self.value = value  # value or cost

        if role == "seller":
            self.good = q  # good owned
            self.right = 0
            self.strategy = self._sell_strategy
        elif role == "buyer":
            self.good = 0
            self.right = q  # right to buy
            self.strategy = self._buy_strategy
        else:
            raise ValueError("Role must be 'seller' or 'buyer'.")

        self.surplus = 0  # profit
        self.active = True

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
        if self.right == 0:
            # if buyer's rights become zero, it deactivates.
            self.active = False

    def sell(self, price):
        self.good -= 1
        self.right += 1
        self.surplus += price - self.value
        if self.good == 0:
            # if seller's units become zero, it deactivates.
            self.active = False

    def _sell_strategy(self):
        pass

    def _buy_strategy(self):
        pass

    def step(self):
        if self.active:
            self.strategy()
            self.model.datacollector.collect(self.model)
            self.model.next_tick()


class ZI(Trader):
    """Zero-Intelligence strategy from Gode and Sunder (1993)"""
    def _sell_strategy(self):
        price = random.randint(self.value, 200)
        self.ask(price)

    def _buy_strategy(self):
        price = random.randint(1, self.value)
        self.bid(price)


class ZIP(Trader):
    """Zero-Intelligence-Plus strategy from Cliff (1997)"""
    def _sell_strategy(self):
        a

# ------------------------------ CDA ------------------------------
Order = namedtuple('Order', ['type', 'price', 'bidder', 'asker'])


class CDAmodel(Model):
    """Continuous Double Auction model with some number of agents."""
    def __init__(self, supply, demand):
        self.period = 1
        self.supply = supply
        self.demand = demand
        self.num_sellers = supply.num_agents
        self.num_buyers = demand.num_agents
        self.initialize_spread()
        self.market_price = None

        # How agents are activated at each step
        self.schedule = RandomActivation(self)
        # Create agents
        for i, value in enumerate(demand.price_schedule):
            self.schedule.add(
                ZI(i, self, "buyer", value, demand.q_per_agent)
            )
        for i, cost in enumerate(supply.price_schedule):
            j = self.num_buyers + i
            self.schedule.add(
                ZI(j, self, "seller", cost, supply.q_per_agent)
            )

        # Collecting data
        self.datacollector = DataCollector(
            model_reporters={"Period": "period",
                             "Tick": "tick",
                             "OB": "outstanding_bid",
                             "OBer": get_bidder_id,
                             "OA": "outstanding_ask",
                             "OAer": get_asker_id,
                             "MarketPrice": "market_price",
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
        self.tick = 1
        self.schedule.step()
        self.period += 1

    def plot_model(self):
        data = self.datacollector.get_model_vars_dataframe()
        data = data[data.Traded == 1]
        f = plt.figure(1)
        ax = f.add_subplot(111)
        plt.plot(data.MarketPrice)
        plt.axhline(y=self.supply.equilibrium_price,
                    color='black',
                    linestyle='dashed'
                    )
        plt.text(0.8, 0.9, round(compute_efficiency(self), 3), fontsize=20,
                 transform=ax.transAxes)
        plt.show()

# ------------------------------ Supply & Demand ------------------------------


class Supply:
    def __init__(self, num_in, num_ex, q, p_min, p_eqb, num_per_step):
        self._init_common(num_in, num_ex, q, p_min, p_eqb, num_per_step)

        steps_in = num_in // num_per_step
        prices_in = np.linspace(p_min, p_eqb, steps_in)
        delta = (p_eqb - p_min) / (steps_in - 1)
        steps_ex = num_ex // num_per_step
        prices_ex = np.fromiter(
            (p_eqb + i * delta for i in range(1, steps_ex + 1)), float
        )

        self._init_prices(prices_in, delta, prices_ex, num_per_step)
        self.comp_theoretical_surplus()

    def _init_common(self, num_in, num_ex, q, p_min, p_eqb, num_per_step):
        if num_in % num_per_step != 0:
            raise ValueError("Number of intramarginal agents \
            must be divisible by number of agents per step")
        if num_ex % num_per_step != 0:
            raise ValueError("Number of extramarginal agents \
            must be divisible by number of agents per step")

        self.num_agents = num_in + num_ex  # number of agents
        self.q_per_agent = q  # number of units held by each agent
        self.equilibrium_price = p_eqb

    def _init_prices(self, prices_in, delta, prices_ex, num_per_step):
        self.price_schedule_in = np.fromiter(
            (p for p in prices_in for i in range(num_per_step)), float
        )
        self.price_schedule_ex = np.fromiter(
            (p for p in prices_ex for i in range(num_per_step)), float
        )
        self.price_schedule = np.append(self.price_schedule_in,
                                        self.price_schedule_ex)
        self.cumulative_quantity = np.cumsum(
            np.repeat(self.q_per_agent, self.num_agents)
        )

    def graph(self):
        plt.step(np.append(0, self.cumulative_quantity),
                 np.append(self.price_schedule[0], self.price_schedule))
        plt.show()

    def market_graph(self, other):
        plt.step(np.append(0, self.cumulative_quantity),
                 np.append(self.price_schedule[0], self.price_schedule))
        plt.step(np.append(0, other.cumulative_quantity),
                 np.append(other.price_schedule[0], other.price_schedule))
        plt.show()

    def comp_theoretical_surplus(self):
        surplus = self.equilibrium_price - self.price_schedule_in
        iterable = (p * self.q_per_agent for p in surplus)
        self.theoretical_surplus = sum(np.fromiter(iterable, float))


class Demand(Supply):
    def __init__(self, num_in, num_ex, q, p_max, p_eqb, num_per_step):
        self._init_common(num_in, num_ex, q, p_max, p_eqb, num_per_step)

        steps_in = num_in // num_per_step
        prices_in = np.linspace(p_max, p_eqb, steps_in)
        delta = (p_max - p_eqb) / (steps_in - 1)
        steps_ex = num_ex // num_per_step
        prices_ex = np.fromiter(
            (p_eqb - i * delta for i in range(1, steps_ex + 1)), float
        )

        self._init_prices(prices_in, delta, prices_ex, num_per_step)
        self.comp_theoretical_surplus()

    def comp_theoretical_surplus(self):
        surplus = self.price_schedule_in - self.equilibrium_price
        iterable = (p * self.q_per_agent for p in surplus)
        self.theoretical_surplus = sum(np.fromiter(iterable, float))

# ------------------------------ Simulation ------------------------------


# equilibrium price is 110
supply = Supply(10, 4, 3, 10, 110, 2)
demand = Demand(10, 4, 3, 210, 110, 2)

model = CDAmodel(supply, demand)
for i in range(100):
    model.step()

data_model = model.datacollector.get_model_vars_dataframe()
data_model = data_model[data_model.Traded == 1]

data_agent = model.datacollector.get_agent_vars_dataframe()

model.plot_model()
