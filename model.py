import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mesa import Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
from agents import *
from supply_demand import *

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
    A = int(round(a, 2) * 100)
    B = int(round(b, 2) * 100)
    interval = np.arange(A, B + 1)
    x = random.choice(interval)
    return x / 100


class GD(Trader):
    """Gjerstad-Dickhaut strategy from Gjerstad and Dickhaut (1998)"""
    def __init__(self, unique_id, model, role, value, q):
        super().__init__(unique_id, model, role, value, q)
        self.eta = 0.001
        self.highest_ask = 10.0
        self.lowest_bid = 0.0
        self.mem_length = 3

    def _sell_belief(self, ask):
        """Compute the seller's belief that an ask will be accepted."""
        if ask >= self.model.outstanding_ask:
            return 0

        def num_geq(history, price):
            return len([d for d in history if d >= price])

        def num_leq(history, price):
            return len([d for d in history if d <= price])

        history = self.model.history
        asks = history.get_asks(length=self.mem_length)
        accepted_asks = history.get_accepted_asks(length=self.mem_length)
        accepted_bids = history.get_accepted_bids(length=self.mem_length)

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
        bids = history.get_bids(length=self.mem_length)
        accepted_asks = history.get_accepted_asks(length=self.mem_length)
        accepted_bids = history.get_accepted_bids(length=self.mem_length)

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
        print("Belief:", [self._sell_belief(p) for p in prices])
        print("ES:", exp_surpluses)
        max_index, max_es = max(enumerate(exp_surpluses), key=lambda p: p[1])
        return max_index

    def _buy_surplus_maximizer(self, prices):
        """Given a list of prices, find the price that maximizes \
        the buyer's expected surplus."""
        def exp_surplus(price):
            return (self.value - price) * self._buy_belief(price)

        exp_surpluses = [exp_surplus(price) for price in prices]
        print("Belief:", [self._buy_belief(p) for p in prices])
        print("ES:", exp_surpluses)
        #@@@
        # If there are multiple instances of the maximum,
        # get the highest price with the maximum.
        max_index, max_es = max(reversed(list(enumerate(exp_surpluses))),
                                key=lambda p: p[1])
        return max_index

    def _sell_strategy(self):
        """Randomize the expected surplus maximizing price \
        and submit it an ask."""
        print("seller:", self.unique_id, "value:", self.value)
        if round(self.value, 2) >= self.model.outstanding_ask:
            self.do_nothing()
            print()
            return

        if self.model.num_traded == 0:
            # If no trade has occurred, be a smarter ZI.
            a = self.value
            b = min(self.model.outstanding_ask, self.highest_ask)
        else:
            prices = self.model.history.get_prices(length=self.mem_length)

            if self.lowest_bid not in prices:
                prices.append(self.lowest_bid)
            if self.highest_ask not in prices:
                prices.append(self.highest_ask)

            prices.sort()
            print("prices:", prices)
            max_index = self._sell_surplus_maximizer(prices)
            max_price = prices[max_index]
            print("max:", max_price)

            if max_index == 0:
                smallest_gap = prices[max_index + 1] - max_price
            elif max_index == len(prices) - 1:
                smallest_gap = max_price - prices[max_index - 1]
            else:
                smallest_gap = min(max_price - prices[max_index - 1],
                                   prices[max_index + 1] - max_price)
            # max_price - smallest_gap results in the outstanding ask
            # if eta is given for ask >= oa
            b = min(max_price + smallest_gap, self.model.outstanding_ask - 0.01)
            a = max(b - 2 * smallest_gap, self.value)

        print("OA:", self.model.outstanding_ask)
        print("(a, b):", (a, b))
        self.model.unif = (a, b)
        planned_ask = uniform_dollar(a, b)
        print("Ask:", planned_ask)
        print()
        self.ask(planned_ask)

    def _buy_strategy(self):
        """Randomize the expected surplus maximizing price \
        and submit it a bid."""
        print("buyer:", self.unique_id, "value:", self.value)
        if round(self.value, 2) <= self.model.outstanding_bid:
            self.do_nothing()
            print()
            return

        if self.model.num_traded == 0:
            # If no trade has occurred, be a smarter ZI.
            a = max(0, self.model.outstanding_bid)
            b = self.value
        else:
            prices = self.model.history.get_prices(length=self.mem_length)

            if self.lowest_bid not in prices:
                prices.append(self.lowest_bid)
            if self.highest_ask not in prices:
                prices.append(self.highest_ask)

            prices.sort()
            print("prices:", prices)
            max_index = self._buy_surplus_maximizer(prices)
            max_price = prices[max_index]
            print("max:", max_price)

            if max_index == 0:
                smallest_gap = prices[max_index + 1] - max_price
            elif max_index == len(prices) - 1:
                smallest_gap = max_price - prices[max_index - 1]
            else:
                smallest_gap = min(max_price - prices[max_index - 1],
                                   prices[max_index + 1] - max_price)
            # max_price - smallest_gap results in the outstanding ask
            # if eta is given for ask >= oa
            a = max(max_price - smallest_gap, self.model.outstanding_bid + 0.01)
            b = min(a + 2 * smallest_gap, self.value) #@@@

        print("OB:", self.model.outstanding_bid)
        print("(a, b):", (a, b))
        self.model.unif = (a, b)
        planned_bid = uniform_dollar(a, b)
        print("Bid:", planned_bid)
        print()
        self.bid(planned_bid)

# ------------------------------ CDA ------------------------------


class Order_history:
    def __init__(self):
        self.orders = list()
        self.actions = list()
        self.num_trades = 0
        self.oa_index = None
        self.ob_index = None
        self.trade_indices = list()

    def submit_ask(self, asker, price):
        order = Order(type='Ask',
                      price=price,
                      bidder=None,
                      asker=asker.unique_id)
        self.orders.append(order)
        self.oa_index = len(self.orders) - 1

        self.actions.append(order)

    def submit_bid(self, bidder, price):
        order = Order(type='Bid',
                      price=price,
                      bidder=bidder.unique_id,
                      asker=None)
        self.orders.append(order)
        self.ob_index = len(self.orders) - 1

        self.actions.append(order)

    def accept_bid(self, bidder, asker, price):
        order = Order(type='Accept Ask',
                      price=price,
                      bidder=bidder.unique_id,
                      asker=asker.unique_id)
        self.orders.append(order)
        self.num_trades += 1
        self.trade_indices.append(len(self.orders) - 1)
        self.oa_index = None
        self.ob_index = None

        self.actions.append(order)

    def accept_ask(self, bidder, asker, price):
        order = Order(type='Accept Bid',
                      price=price,
                      bidder=bidder.unique_id,
                      asker=asker.unique_id)
        self.orders.append(order)
        self.num_trades += 1
        self.trade_indices.append(len(self.orders) - 1)
        self.oa_index = None
        self.ob_index = None

        self.actions.append(order)

    def submit_null(self, bidder, asker, price):
        """Used for tracking all actions."""
        if bidder is not None:
            b_id = bidder.unique_id
        else:
            b_id = None

        if asker is not None:
            a_id = asker.unique_id
        else:
            a_id = None

        self.actions.append(
            Order(type=None,
                  price=price,
                  bidder=b_id,
                  asker=a_id)
        )

    def get_non_outstanding_orders(self, length=None):
        """Return non-outstanding orders."""
        non_outst_orders = [order for i, order in enumerate(self.orders)
                            if i not in (self.oa_index, self.ob_index)]

        if (length is None) or (length >= self.num_trades):
            return non_outst_orders
        else:
            start_point = self.trade_indices[-(length + 1)] + 1
            return non_outst_orders[start_point:]

    def get_prices(self, length=None):
        past_orders = self.get_non_outstanding_orders(length)
        return list({order.price for order in past_orders})

    def get_asks(self, length=None):
        past_orders = self.get_non_outstanding_orders(length)
        asks = [order.price for order in past_orders if order.type == "Ask"
                or order.type == "Accept Ask"]
        return asks

    def get_bids(self, length=None):
        past_orders = self.get_non_outstanding_orders(length)
        bids = [order.price for order in past_orders if order.type == "Bid"
                or order.type == "Accept Bid"]
        return bids

    def get_accepted_asks(self, length=None):
        past_orders = self.get_non_outstanding_orders(length)
        accepted_asks = [order.price for order in past_orders
                         if order.type == "Accept Ask"]
        return accepted_asks

    def get_accepted_bids(self, length=None):
        past_orders = self.get_non_outstanding_orders(length)
        accepted_bids = [order.price for order in past_orders
                         if order.type == "Accept Bid"]
        return accepted_bids

    def get_last_action(self):
        return self.actions[-1]


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
        self.period = 0
        # history records an order as a bid or ask only if it updates
        # the spread

        # How agents are activated at each step
        self.schedule = RandomChoiceActivation(self)
        # Create agents
        for i, value in enumerate(demand.price_schedule):
            self.schedule.add(
                GDtest(i, self, "buyer", value, demand.q_per_agent)
            )

        for i, cost in enumerate(supply.price_schedule):
            j = self.num_buyers + i
            self.schedule.add(
                GDtest(j, self, "seller", cost, supply.q_per_agent)
            )

        # Collecting data
        self.datacollector = DataCollector(
            model_reporters={"OB": "outstanding_bid",
                             "OBer": get_bidder_id,
                             "OA": "outstanding_ask",
                             "OAer": get_asker_id,
                             "MarketPrice": "market_price",
                             "Traded": "traded",
                             "Order": lambda x: x.history.get_last_action(),
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
                self.history.accept_bid(bidder, self.outstanding_asker,
                                        contract_price)
            else:
                self.history.submit_bid(bidder, price)
        else:
            # null order
            self.history.submit_null(bidder, None, price)

    def update_oa(self, asker, price):
        if price < self.outstanding_ask:
            # Update the outstanding ask
            self.outstanding_ask = price
            self.outstanding_asker = asker

            if price < self.outstanding_bid:
                contract_price = self.outstanding_bid
                self.execute_contract(contract_price)
                self.history.accept_ask(self.outstanding_bidder, asker,
                                        contract_price)
            else:
                # only updated the outstanding ask
                self.history.submit_ask(asker, price)
        else:
            # null order
            self.history.submit_null(None, asker, price)

    def execute_contract(self, contract_price):
        self.outstanding_bidder.buy(contract_price)
        self.outstanding_asker.sell(contract_price)
        self.market_price = contract_price
        self.traded = 1
        self.num_traded += 1

    def step(self):
        if self.traded == 1:
            self.initialize_spread()
        print("step:", self.period)
        self.schedule.step()
        self.period += 1
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

# ------------------------------ Logger ----------------------------------

# logging.basicconfig(level="debug", filename='simulation.log',
#                     filemode='w')
# logger = logging.getlogger("model")
# logger.debug("starting a simulation.")


# ------------------------------ Simulation ------------------------------

# ZIP
supply = Supply(6, 5, 3, 75, 200, 1)
demand = Demand(6, 5, 3, 325, 200, 1)

# GD
supply = Supply(6, 5, 2, 1.45, 2.50, 1)
demand = Demand(6, 5, 2, 3.55, 2.50, 1)

model = CDAmodel(supply, demand)
for i in range(200):
    model.step()

data_model = model.datacollector.get_model_vars_dataframe()
data_traded_model = data_model[data_model.Traded == 1]

data_agent = model.datacollector.get_agent_vars_dataframe()

model.plot_model()


def get_agent(unique_id):
    return [i for i in model.schedule.agents if i.unique_id == unique_id][0]
