import random
import math
import numpy as np
import pandas as pd
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
        while not agent.active:
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


def compute_actual_surplus(model):
    actual_surplus = np.sum(np.fromiter(
        (agent.surplus for agent in model.schedule.agents), float
    ))
    return actual_surplus


def compute_theoretical_surplus(model):
    if model.shifted:
        return (model.new_supply.theoretical_surplus +
                model.new_demand.theoretical_surplus)
    else:
        return (model.supply.theoretical_surplus +
                model.demand.theoretical_surplus)


def compute_acc_theoretical_surplus(model):
    """Compute accumulated theoretical surplus"""
    if model.shifted:
        init_producer_surplus = model.supply.theoretical_surplus
        init_consumer_surplus = model.demand.theoretical_surplus
        init_total_surplus = ((init_producer_surplus + init_consumer_surplus) *
                              (model.shifted_period - 1))

        shift_prod_surplus = model.new_supply.theoretical_surplus
        shift_cons_surplus = model.new_demand.theoretical_surplus
        shift_total_surplus = ((shift_prod_surplus + shift_cons_surplus) *
                               (model.num_period - model.shifted_period + 1))
        total_surplus = init_total_surplus + shift_total_surplus
    else:
        producer_surplus = model.supply.theoretical_surplus
        consumer_surplus = model.demand.theoretical_surplus
        total_surplus = ((producer_surplus + consumer_surplus) *
                         model.num_period)
    return total_surplus


def compute_efficiency(model):
    # Compute the efficiency of traders' surplus
    # in contrast to the theoretical maximum surplus.
    total_surplus = compute_acc_theoretical_surplus(model)
    actual_surplus = compute_actual_surplus(model)
    return actual_surplus / total_surplus


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

        if len(prices_this_period) <= 1:
            self.max_last_period = None
            self.min_last_period = None
        else:
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
    def __init__(self, supply, demand, s_strategy, b_strategy,
                 highest_ask=100, lowest_ask=0):
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

        # Sometimes trade does not happen within a period,
        # so we need variables to indicate them.
        self.no_trade = False

        # When a shift happens, I need to know it so that
        # I can calculate efficiency properly.
        self.shifted = False
        self.shifted_period = -1
        self.new_supply = None
        self.new_demand = None

        # How agents are activated at each step
        self.schedule = RandomChoiceActivation(self)
        # Create agents
        for i, cost in enumerate(self.supply.price_schedule):
            self.schedule.add(
                b_strategy(i, self, "seller", cost, supply.q_per_agent,
                           highest_ask, lowest_ask)
            )
        for i, value in enumerate(self.demand.price_schedule):
            j = self.num_sellers + i
            self.schedule.add(
                s_strategy(j, self, "buyer", value, demand.q_per_agent,
                           highest_ask, lowest_ask)
            )

        # Collecting data
        # self.datacollector = DataCollector(
        #     model_reporters={"Period": "num_period",
        #                      "OB": "outstanding_bid",
        #                      "OBer": get_bidder_id,
        #                      "OA": "outstanding_ask",
        #                      "OAer": get_asker_id,
        #                      "MarketPrice": "market_price",
        #                      "Traded": "traded",
        #                      "Order": lambda x: x.history.get_last_action(),
        #                      "Efficiency": compute_efficiency},
        #     agent_reporters={"Period": lambda x: x.model.num_period,
        #                      "Type": lambda x: type(x),
        #                      "Role": "role",
        #                      "Value": "value",
        #                      "Good": "good",
        #                      "Right": "right",
        #                      "Surplus": "surplus"}
        # )
        self.datacollector = DataCollector(
            model_reporters={"Step": "num_step",
                             "Period": "num_period",
                             "TransNum": "num_traded",
                             "OB": "outstanding_bid",
                             "OA": "outstanding_ask",
                             "MarketPrice": "market_price",
                             "Traded": "traded",
                             "CumulativeActualSurplus": compute_actual_surplus,
                             "TheoreticalSurplus": compute_theoretical_surplus,
                             "CumulativeTS": compute_acc_theoretical_surplus,
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

    def next_period(self, new_supply=None, new_demand=None):

        if self.num_traded == 0:
            self.no_trade = True

        if new_supply and new_demand:
            self.new_supply = new_supply
            self.new_demand = new_demand
            self.shifted = True
            self.shifted_period = self.num_period + 1

        if self.shifted:
            supply = self.new_supply
            demand = self.new_demand
        else:
            supply = self.supply
            demand = self.demand

        # Making sure the schedule is ordered as it was initialized.
        self.schedule.agents.sort(key=lambda x: x.unique_id)

        for i, cost in enumerate(supply.price_schedule):
            self.schedule.agents[i].good = supply.q_per_agent
            self.schedule.agents[i].value = cost
            self.schedule.agents[i].active = True

        for i, value in enumerate(demand.price_schedule):
            j = self.num_sellers + i
            self.schedule.agents[j].right = demand.q_per_agent
            self.schedule.agents[j].value = value
            self.schedule.agents[j].active = True

        self.num_period += 1
        self.num_traded = 0
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


# ----------------------------- Sim -----------------------------

# ZIP
#supply = Supply(6, 5, 1, 75, 200, 1)
#demand = Demand(6, 5, 1, 325, 200, 1)

# GD
#supply = Supply(6, 5, 1, 1.45, 2.50, 1)
#demand = Demand(6, 5, 1, 3.55, 2.50, 1)

# test
# supply = Supply(6, 5, 1, 24.00, 25.00, 1)
# demand = Demand(6, 5, 1, 35.50, 25.00, 1)

# num_batches = 0
# while num_batches <= 9:
#     print()
#     print(num_batches)
#     model = CDAmodel(supply, demand, ZIP, ZIP, 100, 0)
#     for j in range(10):
#         print("Period", model.num_period)
#         for i in range(500):
#             model.step()
#         model.next_period()
#         if model.no_trade:
#             print("No Trade")
#             break
#     else:
#         num_batches += 1



# data_model = model.datacollector.get_model_vars_dataframe()
# data_t_model = data_model[data_model.Traded == 1].drop(
#             columns='Traded')


def get_agent(unique_id):
    return [i for i in model.schedule.agents if i.unique_id == unique_id][0]


# supply and demand shock test
supply = Supply(6, 5, 1, 15.00, 25.00, 1)
demand = Demand(6, 5, 1, 35.00, 25.00, 1)

def supply_shift(supply, demand, delta_q):
    """Shift a given supply by delta_q * change in price per quantity.
    demand object is needed for adjustment of market price and surplus."""
    # works only for the case of 1 quantity per agent.
    delta_price = ((supply.equilibrium_price - supply.minimum_price)
                   / (supply.num_in - 1)) * 2 * delta_q
    delta_eqb_price = ((supply.equilibrium_price - supply.minimum_price)
                       / (supply.num_in - 1)) * delta_q
    shifted_supply = Supply(supply.num_in + delta_q,
                            supply.num_ex - delta_q, 1,
                            supply.minimum_price - delta_price,
                            supply.equilibrium_price - delta_eqb_price, 1)
    adjusted_demand = Demand(demand.num_in + delta_q,
                             demand.num_ex - delta_q, 1,
                             demand.maximum_price,
                             demand.equilibrium_price - delta_eqb_price, 1)
    return shifted_supply, adjusted_demand


def demand_shift(supply, demand, delta_q):
    """Shift a given supply by delta_q * change in price per quantity.
    demand object is needed for adjustment of market price and surplus."""
    # works only for the case of 1 quantity per agent.
    delta_price = ((demand.maximum_price - demand.equilibrium_price)
                   / (demand.num_in - 1)) * 2 * delta_q
    delta_eqb_price = ((demand.maximum_price - demand.equilibrium_price)
                       / (demand.num_in - 1)) * delta_q
    shifted_demand = Demand(demand.num_in + delta_q,
                            demand.num_ex - delta_q, 1,
                            demand.maximum_price + delta_price,
                            demand.equilibrium_price + delta_eqb_price, 1)
    adjusted_supply = Supply(supply.num_in + delta_q,
                             supply.num_ex - delta_q, 1,
                             supply.minimum_price,
                             supply.equilibrium_price + delta_eqb_price, 1)
    return adjusted_supply, shifted_demand

supply2, demand2 = supply_shift(supply, demand, 2)


model = CDAmodel(supply, demand, ZIP, ZIP, 100, 0)
for j in range(4):
    print("Period", model.num_period)
    for i in range(500):
        model.step()
    model.next_period()

print("Period", model.num_period)
for i in range(500):
    model.step()
model.next_period(new_supply=supply2, new_demand=demand2)
for i in range(500):
    model.step()
model.next_period()

for j in range(4):
    print("Period", model.num_period)
    for i in range(500):
        model.step()
    model.next_period()
    if model.no_trade:
        print("No Trade")
        break

data_model = model.datacollector.get_model_vars_dataframe()
data_t_model = data_model[data_model.Traded == 1].drop(
                columns='Traded')
