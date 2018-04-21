import random
import math
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector

# ------------------------------ scheduler ------------------------------


class RandomChoiceActivation(BaseScheduler):
    """A scheduler which activates a randomly picked agent once per step."""
    def step(self):
        agent = random.choice(self.agents)
        agent.step()
        self.steps += 1
        self.time += 1
        for i in self.agents:
            i.response()


# ------------------------------ Data functions ------------------------------


def get_bidder_id(model):
    # Displays the unique ID of the outstanding bidder.
    try:
        return model.outstanding_bidder.unique_id
    except AttributeError:
        return -1


def get_asker_id(model):
    # Displays the unique ID of the outstanding asker.
    try:
        return model.outstanding_asker.unique_id
    except AttributeError:
        return -1


def get_latest_order(model):
    # Displays the latest order from the history
    return model.history[-1]


def compute_efficiency(model):
    # Compute the efficiency of traders' surplus
    # in contrast to the theoretical maximum surplus.
    producer_surplus = model.supply.theoretical_surplus
    consumer_surplus = model.demand.theoretical_surplus
    total_surplus = producer_surplus + consumer_surplus
    actual_surplus = np.sum(np.fromiter(
        (agent.surplus for agent in model.schedule.agents), float
    ))
    return actual_surplus / total_surplus

# ------------------------------ Traders ------------------------------


def uniform_dollar(a, b):
    # Uniform discrete distribution for dollar values.
    A = round(a, 2)
    B = round(b, 2)
    interval = np.arange(A, B, 0.01)
    x = random.choice(interval)
    return round(x, 2)


class Trader(Agent):
    """A base class for traders"""
    def __init__(self, unique_id, model, role, value, q):
        super().__init__(unique_id, model)
        self.role = role  # buyer or seller
        self.value = value  # value or cost

        if role == "seller":
            self.good = q  # good owned
            self.right = 0
            self.strategy = self._sell_strategy
            self.response = self._sell_response
        elif role == "buyer":
            self.good = 0
            self.right = q  # right to buy
            self.strategy = self._buy_strategy
            self.response = self._buy_response
        else:
            raise ValueError("Role must be 'seller' or 'buyer'.")

        self.surplus = 0  # profit
        self.active = True

    def bid(self, price):
        """Submit a bid to the CDA."""
        self.model.update_ob(self, price)

    def ask(self, price):
        """Submit an ask to the CDA."""
        self.model.update_oa(self, price)

    def buy(self, price):
        """Buy a good from someone"""
        self.good += 1
        self.right -= 1
        self.surplus += self.value - price
        if self.right == 0:
            # if buyer's rights become zero, it deactivates.
            self.active = False

    def sell(self, price):
        """Sell a good to someone"""
        self.good -= 1
        self.right += 1
        self.surplus += price - self.value
        if self.good == 0:
            # if seller's units become zero, it deactivates.
            self.active = False

    def _sell_strategy(self):
        """Strategy for selling."""
        pass

    def _buy_strategy(self):
        """Strategy for buying."""
        pass

    def _sell_response(self):
        """How a seller responds to someone's action."""
        pass

    def _buy_response(self):
        """How a buyer responds to someone's action."""
        pass

    def do_nothing(self):
        """Send a message to the history that the trader did nothing."""
        self.model.history.append(
            Order(type=None,
                  price=None,
                  bidder=self.unique_id,
                  asker=self.unique_id)
        )

    def step(self):
        """How the trader acts when he is picked by the scheduler."""
        if self.active:
            self.strategy()
        else:
            self.do_nothing()


class ZI(Trader):
    """Zero-Intelligence strategy from Gode and Sunder (1993)."""
    def _sell_strategy(self):
        """Strategy for selling. Pick a value from uniform discrete \
        distribution (v_i, 200)."""
        price = random.randint(self.value, 200)
        self.ask(price)

    def _buy_strategy(self):
        """Strategy for buying. Pick a value from uniform discrete \
        distribution (0, v_i)."""
        price = random.randint(1, self.value)
        self.bid(price)


class ZIP(Trader):
    """Zero-Intelligence-Plus strategy from Cliff (1997)."""
    def __init__(self, unique_id, model, role, value, q):
        super().__init__(unique_id, model, role, value, q)

        self.beta = np.random.uniform(0.1, 0.5)
        self.gamma = np.random.uniform(0, 0.1)

        if role == "seller":
            self.margins = [np.random.uniform(0.05, 0.35)]
        elif role == "buyer":
            self.margins = [np.random.uniform(-0.35, -0.05)]
        else:
            raise ValueError("Role must be 'seller' or 'buyer'.")

        self.planned_shout = [self.value * (self.margins[0] + 1)]
        self.change = [0]
        self.target = []

    def _sell_strategy(self):
        """Submit the planned ask."""
        self.ask(self.planned_shout[-1])

    def _buy_strategy(self):
        """Submit the planned bid."""
        self.bid(self.planned_shout[-1])

    def _sell_response(self):
        """Observe the last shout and change the planned ask."""
        last_order = self.model.history[-1]
        if (last_order.type == 'Accept Bid' or
                last_order.type == 'Accept Ask'):
            if self.planned_shout[-1] <= last_order.price:
                # transaction price is higher than what I planned
                # increase the margin
                self.increase_target(last_order.price)
                self._sell_update_margin()
            else:
                if (last_order.type == 'Accept Ask' and
                        self.active):
                    # bid was accepted with a smaller price than my planned ask
                    # decrease the margin
                    self.decrease_target(last_order.price)
                    self._sell_update_margin()
        else:
            # no transaction
            if (last_order.type == 'Ask' and
                    self.planned_shout[-1] > last_order.price):
                # ask has a smaller price than my planned ask
                # decrease the margin
                self.decrease_target(last_order.price)
                self._sell_update_margin()
            else:
                # ask has a higher price than my planned ask
                # or there was a bid
                # target at the smallest possible ask, and update the margin
                # stub order
                pass

    def _buy_response(self):
        """Observe the last shout and change the planned bid."""
        last_order = self.model.history[-1]
        if (last_order.type == 'Accept Bid' or
                last_order.type == 'Accept Ask'):
            if self.planned_shout[-1] >= last_order.price:
                # transaction price is smaller than what I planned
                # increase the margin
                self.decrease_target(last_order.price)
                self._buy_update_margin()
            else:
                if (last_order.type == 'Accept Bid' and
                        self.active):
                    # bid was accepted with a higher price than my planned bid
                    # decrease the margin
                    self.increase_target(last_order.price)
                    self._buy_update_margin()
        else:
            # no transaction
            if (last_order.type == 'Bid' and
                    self.planned_shout[-1] < last_order.price):
                # bid has a higher price than my planned bid
                # decrease the margin
                self.increase_target(last_order.price)
                self._buy_update_margin()
            else:
                # bid has a smaller price than my planned bid
                # or there was an ask
                # target at the highest possible bid, and update the margin
                # stub order
                pass

    def _sell_update_margin(self):
        """Update the margin and planned ask."""
        delta = self.beta * (self.target[-1] - self.planned_shout[-1])
        new_change = self.gamma * self.change[-1] + (1 - self.gamma) * delta
        self.change.append(new_change)

        next_shout = self.planned_shout[-1] + self.change[-1]

        if next_shout >= self.value:
            self.planned_shout.append(next_shout)
            new_margin = next_shout / self.value - 1
            self.margins.append(new_margin)

    def _buy_update_margin(self):
        """Update the margin and planned bid."""
        delta = self.beta * (self.target[-1] - self.planned_shout[-1])
        new_change = self.gamma * self.change[-1] + (1 - self.gamma) * delta
        self.change.append(new_change)

        next_shout = self.planned_shout[-1] + self.change[-1]

        if next_shout <= self.value:
            self.planned_shout.append(next_shout)
            new_margin = next_shout / self.value - 1
            self.margins.append(new_margin)

    def increase_target(self, last_price):
        """Compute the target for increasing the planned ask."""
        R = np.random.uniform(1, 1.05)
        A = np.random.uniform(0, 0.05)
        new_target = R * last_price + A
        self.target.append(new_target)

    def decrease_target(self, last_price):
        """Compute the target for increasing the planned bid."""
        R = np.random.uniform(0.95, 1)
        A = np.random.uniform(-0.05, 0)
        new_target = R * last_price + A
        self.target.append(new_target)


class GD(Trader):
    """Gjerstad-Dickhaut strategy from Gjerstad and Dickhaut (1998)"""
    def __init__(self, unique_id, model, role, value, q):
        super().__init__(unique_id, model, role, value, q)
        self.eta = 0.001
        self.highest_ask = 10
        self.lowest_bid = 0

    def _sell_belief(self, ask):
        """Compute the seller's belief that an ask will be accepted."""
        if ask >= self.model.outstanding_ask:
            return 0

        def num_geq(history, price):
            return len([d for d in history if d >= price])

        def num_leq(history, price):
            return len([d for d in history if d <= price])

        history = self.model.history
        asks = history.get_asks()
        accepted_asks = history.get_accepted_asks()
        accepted_bids = history.get_accepted_bids()

        TAG = num_geq(accepted_asks, ask)
        TBG = num_geq(accepted_bids, ask)
        RAL = num_leq(asks, ask) - num_leq(accepted_asks, ask)

        if TAG + TBG == 0:
            return self.eta
        else:
            return (TAG + TBG) / (TAG + TBG + RAL)

    def _buy_belief(self, bid):
        """Compute the buyer's belief that an bid will be accepted."""
        if bid <= self.model.outstanding_bid:
            return 0

        def num_geq(history, price):
            return len([d for d in history if d >= price])

        def num_leq(history, price):
            return len([d for d in history if d <= price])

        history = self.model.history
        bids = history.get_bids()
        accepted_asks = history.get_accepted_asks()
        accepted_bids = history.get_accepted_bids()

        TBL = num_leq(accepted_bids, bid)
        TAL = num_leq(accepted_asks, bid)
        RBG = num_geq(bids, bid) - num_geq(accepted_bids, bid)

        if TBL + TAL == 0:
            return self.eta
        else:
            return (TBL + TAL) / (TBL + TAL + RBG)

    def _sell_surplus_maximizer(self, prices):
        """Given a list of prices, find the price that maximizes \
        the seller's expected surplus."""
        def exp_surplus(price):
            return (price - self.value) * self._sell_belief(price)

        exp_surpluses = [exp_surplus(price) for price in prices]
        max_index, max_es = max(enumerate(exp_surpluses), key=lambda p: p[1])
        return max_index

    def _buy_surplus_maximizer(self, prices):
        """Given a list of prices, find the price that maximizes \
        the buyer's expected surplus."""
        def exp_surplus(price):
            return (self.value - price) * self._buy_belief(price)

        exp_surpluses = [exp_surplus(price) for price in prices]
        max_index, max_es = max(enumerate(exp_surpluses), key=lambda p: p[1])
        return max_index

    def _sell_strategy(self):
        """Randomize the expected surplus maximizing price \
        and submit it an ask."""
        if self.value >= self.model.outstanding_ask:
            self.do_nothing()
            return

        if self.model.num_traded == 0:
            # If no trade has occurred, be a smarter ZI.
            a = self.value
            b = min(self.model.outstanding_ask, self.highest_ask)
        else:
            prices = self.model.history.get_prices()
            prices.append(self.lowest_bid)
            prices.append(self.highest_ask)
            prices.sort()
            max_index = self._sell_surplus_maximizer(prices)
            max_price = prices[max_index]

            if max_index == 0:
                smallest_gap = prices[max_index + 1] - max_price
            elif max_index == len(prices) - 1:
                smallest_gap = max_price - prices[max_index - 1]
            else:
                smallest_gap = min(max_price - prices[max_index - 1],
                                   prices[max_index + 1] - max_price)
            # max_price - smallest_gap results in the outstanding ask
            # if eta is given for ask >= oa
            a = max(max_price - smallest_gap, self.value)
            b = min(max_price + smallest_gap, self.model.outstanding_ask)

        planned_ask = np.random.uniform(a, b)
        self.ask(planned_ask)
        self.model.unif = (a, b)

    def _buy_strategy(self):
        """Randomize the expected surplus maximizing price \
        and submit it a bid."""

        if self.value <= self.model.outstanding_bid:
            self.do_nothing()
            return

        if self.model.num_traded == 0:
            # If no trade has occurred, be a smarter ZI.
            a = max(0, self.model.outstanding_bid)
            b = self.value
        else:
            prices = self.model.history.get_prices()
            prices.append(self.lowest_bid)
            prices.append(self.highest_ask)
            prices.sort()
            max_index = self._buy_surplus_maximizer(prices)
            max_price = prices[max_index]

            if max_index == 0:
                smallest_gap = prices[max_index + 1] - max_price
            elif max_index == len(prices) - 1:
                smallest_gap = max_price - prices[max_index - 1]
            else:
                smallest_gap = min(max_price - prices[max_index - 1],
                                   prices[max_index + 1] - max_price)
            # max_price - smallest_gap results in the outstanding ask
            # if eta is given for ask >= oa
            a = max(max_price - smallest_gap, self.model.outstanding_bid)
            b = min(max_price + smallest_gap, self.value)

        planned_bid = np.random.uniform(a, b)
        self.bid(planned_bid)
        self.model.unif = (a, b)

# ------------------------------ CDA ------------------------------


Order = namedtuple('Order',
                   ['type', 'price', 'bidder', 'asker'])


# exclude outstanding spread?
class Order_history(list):
    def get_prices(self):
        asks = self.get_asks()  # without outstanding ask
        bids = self.get_bids()  # without outstanding bid
        prices = list(set(asks + bids))
        return prices

    def get_asks(self):
        # an ask can be accepted ask. Double counted?
        asks_and_accepts = [order for order in self if order.type == "Ask"
                            or order.type == "Accept Ask"]
        try:
            if asks_and_accepts[-1].type == "Accept Ask":
                # Was the recent ask accepted? Then, there is no outstanding ask.
                return [order.price for order in self if order.type == "Ask"]
            else:
                # There is an outstanding ask, so remove it.
                return [order.price for order in self if order.type == "Ask"][:-1]
        except IndexError:
            return []

    def get_bids(self):
        bids_and_accepts = [order for order in self if order.type == "Bid"
                            or order.type == "Accept Bid"]
        try:
            if bids_and_accepts[-1].type == "Accept Bid":
                return [order.price for order in self if order.type == "Bid"]
            else:
                return [order.price for order in self if order.type == "Bid"][:-1]
        except IndexError:
            return []

    def get_accepted_asks(self):
        accepted_asks = [order.price for order in self
                         if order.type == "Accept Ask"]
        return accepted_asks

    def get_accepted_bids(self):
        accepted_bids = [order.price for order in self
                         if order.type == "Accept Bid"]
        return accepted_bids


class CDAmodel(Model):
    """Continuous Double Auction model with some number of agents."""
    def __init__(self, supply, demand):
        self.unif = (0, 0)
        self.supply = supply
        self.demand = demand
        self.num_sellers = supply.num_agents
        self.num_buyers = demand.num_agents
        self.initialize_spread()
        self.market_price = None
        self.history = Order_history()
        self.num_traded = 0
        # history records an order as a bid or ask only if it updates
        # the spread

        # How agents are activated at each step
        self.schedule = RandomChoiceActivation(self)
        # Create agents
        for i, value in enumerate(demand.price_schedule):
            self.schedule.add(
                GD(i, self, "buyer", value, demand.q_per_agent)
            )

        for i, cost in enumerate(supply.price_schedule):
            j = self.num_buyers + i
            self.schedule.add(
                GD(j, self, "seller", cost, supply.q_per_agent)
            )

        # Collecting data
        self.datacollector = DataCollector(
            model_reporters={"OB": "outstanding_bid",
                             "OBer": get_bidder_id,
                             "OA": "outstanding_ask",
                             "OAer": get_asker_id,
                             "MarketPrice": "market_price",
                             "Traded": "traded",
                             "Order": get_latest_order,
                             "Unif": "unif"},
            agent_reporters={"Type": lambda x: type(x),
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
                # a transaction happens
                contract_price = self.outstanding_ask
                self.execute_contract(contract_price)
                self.history.append(
                    Order(type='Accept Ask',
                          price=contract_price,
                          bidder=bidder.unique_id,
                          asker=self.outstanding_asker.unique_id)
                )
            else:
                self.history.append(
                    Order(type='Bid',
                          price=price,
                          bidder=bidder.unique_id,
                          asker=None)
                )
        else:
            # null order
            self.history.append(
                Order(type=None,
                      price=price,
                      bidder=bidder.unique_id,
                      asker=None)
                )

    def update_oa(self, asker, price):
        if price < self.outstanding_ask:
            # Update the outstanding ask
            self.outstanding_ask = price
            self.outstanding_asker = asker

            if price < self.outstanding_bid:
                contract_price = self.outstanding_bid
                self.execute_contract(contract_price)
                self.history.append(
                    Order(type='Accept Bid',
                          price=contract_price,
                          bidder=self.outstanding_bidder.unique_id,
                          asker=asker.unique_id)
                )
            else:
                # only updated the outstanding ask
                self.history.append(
                    Order(type='Ask',
                          price=price,
                          bidder=None,
                          asker=asker.unique_id)
                )
        else:
            # null order
            self.history.append(
                Order(type=None,
                      price=price,
                      bidder=None,
                      asker=asker.unique_id)
            )

    def execute_contract(self, contract_price):
        self.outstanding_bidder.buy(contract_price)
        self.outstanding_asker.sell(contract_price)
        self.market_price = contract_price
        self.traded = 1
        self.num_traded += 1

    def step(self):
        if self.traded == 1:
            self.initialize_spread()
        self.schedule.step()
        self.datacollector.collect(self)

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


# ZIP
supply = Supply(6, 5, 3, 75, 200, 1)
demand = Demand(6, 5, 3, 325, 200, 1)

# GD
supply = Supply(7, 5, 5, 1.45, 2.50, 1)
demand = Demand(7, 5, 5, 3.55, 2.50, 1)

model = CDAmodel(supply, demand)
for i in range(1000):
    model.step()

data_model = model.datacollector.get_model_vars_dataframe()
data_traded_model = data_model[data_model.Traded == 1]

data_agent = model.datacollector.get_agent_vars_dataframe()

model.plot_model()


def get_agent(unique_id):
    return [i for i in model.schedule.agents if i.unique_id == unique_id][0]
