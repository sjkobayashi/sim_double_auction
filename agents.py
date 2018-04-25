import random
from collections import namedtuple
import numpy as np
from mesa import Agent

Order = namedtuple('Order',
                   ['type', 'price', 'bidder', 'asker'])


def uniform_dollar(a, b):
    # Uniform discrete distribution for dollar values.
    A = int(round(a, 2) * 100)
    B = int(round(b, 2) * 100)
    interval = np.arange(A, B + 1)
    x = random.choice(interval)
    return x / 100


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
        self.right -= 1
        self.surplus += self.value - price
        if self.right == 0:
            # if buyer's rights become zero, it deactivates.
            self.active = False

    def sell(self, price):
        """Sell a good to someone"""
        self.good -= 1
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
        self.model.history.submit_null(self, self, None)

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
        last_order = self.model.history.get_last_action()
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
        last_order = self.model.history.get_last_action()
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
            if new_margin > 0.0:
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
            if new_margin < 0.0:
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
        self.highest_ask = 100.0
        self.lowest_bid = 0.0
        self.mem_length = None

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

        # @@@
        # exclude prices that would result in negative surpluses
        profitable_prices = [(i, price) for i, price in enumerate(prices)
                             if price >= self.value]
        exp_surpluses = [(i, exp_surplus(price)) for i, price in
                         profitable_prices]
        # print("Belief:", [self._sell_belief(p) for p in prices])
        # print("ES:", exp_surpluses)
        max_index, max_es = max(exp_surpluses, key=lambda p: p[1])
        return max_index

    def _buy_surplus_maximizer(self, prices):
        """Given a list of prices, find the price that maximizes \
        the buyer's expected surplus."""
        def exp_surplus(price):
            return (self.value - price) * self._buy_belief(price)

        # @@@
        # exclude prices that would result in negative surpluses
        profitable_prices = [(i, price) for i, price in enumerate(prices)
                             if price <= self.value]
        exp_surpluses = [(i, exp_surplus(price)) for i, price in
                         profitable_prices]
        # print("Belief:", [self._buy_belief(p) for p in prices])
        # print("ES:", exp_surpluses)
        # If there are multiple instances of the maximum,
        # get the highest price with the maximum.
        max_index, max_es = max(reversed(exp_surpluses),
                                key=lambda p: p[1])
        return max_index

    def _sell_strategy(self):
        """Randomize the expected surplus maximizing price \
        and submit it an ask."""
        # print("seller:", self.unique_id, "value:", self.value)
        if round(self.value, 2) >= self.model.outstanding_ask:
            self.do_nothing()
            # print()
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
            # print("prices:", prices)
            max_index = self._sell_surplus_maximizer(prices)
            max_price = prices[max_index]
            # print("max:", max_price)

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
            a = max(b - 2 * smallest_gap, self.value)  # @@@

        # print("OA:", self.model.outstanding_ask)
        # print("(a, b):", (a, b))
        planned_ask = uniform_dollar(a, b)
        # print("Ask:", planned_ask)
        # print()
        self.ask(planned_ask)

    def _buy_strategy(self):
        """Randomize the expected surplus maximizing price \
        and submit it a bid."""
        # print("buyer:", self.unique_id, "value:", self.value)
        if round(self.value, 2) <= self.model.outstanding_bid:
            self.do_nothing()
            # print()
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
            # print("prices:", prices)
            max_index = self._buy_surplus_maximizer(prices)
            max_price = prices[max_index]
            # print("max:", max_price)

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

        # print("OB:", self.model.outstanding_bid)
        # print("(a, b):", (a, b))
        planned_bid = uniform_dollar(a, b)
        # print("Bid:", planned_bid)
        # print()
        self.bid(planned_bid)
