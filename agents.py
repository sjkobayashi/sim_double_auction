import random
from collections import namedtuple
import numpy as np
from mesa import Agent

Order = namedtuple('Order',
                   ['type', 'price', 'bidder', 'asker'])


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
