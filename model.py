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
    total_surplus = (producer_surplus + consumer_surplus) * model.num_period
    actual_surplus = np.sum(np.fromiter(
        (agent.surplus for agent in model.schedule.agents), float
    ))
    return actual_surplus / total_surplus

# ---------------------------------------------------------------


class MGD(GD):
    def _sell_belief(self, ask):
        """Compute the seller's belief that an ask will be accepted."""
        if ask >= self.model.outstanding_ask:
            return 0

        try:
            if ask >= self.model.history.max_last_period:
                return self.eta
        except TypeError:
            pass

        try:
            if ask >= self.model.history.max_current_period:
                return self.eta
        except TypeError:
            pass

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

        try:
            if bid <= self.model.history.min_last_period:
                return self.eta
        except TypeError:
            pass

        try:
            if bid <= self.model.history.min_current_period:
                return self.eta
        except TypeError:
            pass

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


# ------------------------------ CDA ------------------------------


class Order_history:
    def __init__(self):
        self.orders = list()
        self.actions = list()
        self.num_trades = 0
        self.oa_index = None
        self.ob_index = None
        self.trade_indices = list()

        self.starting_step = 0
        self.max_last_period = None
        self.min_last_period = None
        self.max_current_period = None
        self.min_current_period = None

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

        self.update_min_max_current(price)

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

        self.update_min_max_current(price)

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

    def next_period(self):
        prices_this_period = [order.price for order in
                              self.orders[self.starting_step:]
                              if order.type == "Accept Ask" or
                              order.type == "Accept Bid"]
        self.max_last_period = max(prices_this_period)
        self.min_last_period = min(prices_this_period)
        self.starting_step = len(self.orders)
        self.max_current_period = None
        self.min_current_period = None

    def update_min_max_current(self, price):
        if self.max_current_period is None:
            self.max_current_period = price
        elif self.max_current_period < price:
            self.max_current_period = price

        if self.min_current_period is None:
            self.min_current_period = price
        elif self.min_current_period > price:
            self.min_current_period = price

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

    def get_prices_above(self, value, length=None):
        past_orders = self.get_non_outstanding_orders(length)
        return list({order.price for order in past_orders
                     if order.price >= value})

    def get_prices_below(self, value, length=None):
        past_orders = self.get_non_outstanding_orders(length)
        return list({order.price for order in past_orders
                     if order.price <= value})

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
        self.supply = supply
        self.demand = demand
        self.num_sellers = supply.num_agents
        self.num_buyers = demand.num_agents
        self.initialize_spread()
        self.market_price = None
        self.history = Order_history()
        self.num_traded = 0
        self.num_step = 0
        # history records an order as a bid or ask only if it updates
        # the spread

        self.num_period = 1
        self.loc_period = [0]  # where each period happens

        # How agents are activated at each step
        self.schedule = RandomChoiceActivation(self)
        # Create agents
        for i, value in enumerate(demand.price_schedule):
            self.schedule.add(
                MGD(i, self, "buyer", value, demand.q_per_agent)
            )

        for i, cost in enumerate(supply.price_schedule):
            j = self.num_buyers + i
            self.schedule.add(
                MGD(j, self, "seller", cost, supply.q_per_agent)
            )

        # Collecting data
        self.datacollector = DataCollector(
            model_reporters={"Period": "num_period",
                             "OB": "outstanding_bid",
                             "OBer": get_bidder_id,
                             "OA": "outstanding_ask",
                             "OAer": get_asker_id,
                             "MarketPrice": "market_price",
                             "Traded": "traded",
                             "Order": lambda x: x.history.get_last_action(),
                             "Efficiency": compute_efficiency},
            agent_reporters={"Period": lambda x: x.model.num_period,
                             "Type": lambda x: type(x),
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

    def next_period(self):
        # Make sure the schedule is in the same order as it was initialized.
        model.schedule.agents.sort(key=lambda x: x.unique_id)
        for i, _ in enumerate(self.demand.price_schedule):
            self.schedule.agents[i].right = self.demand.q_per_agent
            self.schedule.agents[i].active = True

        for i, _ in enumerate(self.demand.price_schedule):
            j = self.num_buyers + i
            self.schedule.agents[j].good = self.supply.q_per_agent
            self.schedule.agents[j].active = True

        self.num_period += 1
        self.loc_period.append(self.num_step)
        self.history.next_period()

    def step(self):
        if self.traded == 1:
            self.initialize_spread()
        # print("step:", self.num_step)
        self.schedule.step()
        self.num_step += 1
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
        for i in range(self.num_period):
            plt.axvline(x=self.loc_period[i],
                        color='black',
                        linestyle='dashed'
                        )
        plt.show()

# ------------------------------ Logger ----------------------------------

# logging.basicconfig(level="debug", filename='simulation.log',
#                     filemode='w')
# logger = logging.getlogger("model")
# logger.debug("starting a simulation.")


# ------------------------------ Simulation ------------------------------

# ZIP
supply = Supply(6, 5, 1, 75, 200, 1)
demand = Demand(6, 5, 1, 325, 200, 1)

# GD
supply = Supply(6, 5, 1, 1.45, 2.50, 1)
demand = Demand(6, 5, 1, 3.55, 2.50, 1)

# test
supply = Supply(6, 5, 1, 14.50, 25.00, 1)
demand = Demand(6, 5, 1, 35.50, 25.00, 1)

model = CDAmodel(supply, demand)

for i in range(300):
    model.step()

for j in range(9):
    model.next_period()
    for i in range(300):
        model.step()

data_model = model.datacollector.get_model_vars_dataframe()
data_traded_model = data_model[data_model.Traded == 1]

data_agent = model.datacollector.get_agent_vars_dataframe()

model.plot_model()


def get_agent(unique_id):
    return [i for i in model.schedule.agents if i.unique_id == unique_id][0]
