import random
import math
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


class ZeroIntelligence(Agent):
    """Zero-Intelligence strategy from Gode and Sunder (1993)"""
    def __init__(self, unique_id, model, role):
        super().__init__(unique_id, model)
        self.role = role
        self.money = 1000
        self.good = 5

    def bid(self, price):
        # Submit a bid to the CDA.
        self.model.update_ob(self, price)

    def ask(self, price):
        # Submit an ask to the CDA.
        self.model.update_oa(self, price)

    def step(self):
        price = random.randint(1, 200)
        if self.role == "buyer":
            if price <= self.money:
                self.bid(price)
        else:
            if self.good > 0:
                self.ask(price)
        self.model.datacollector.collect(self.model)
        self.model.next_tick()

class CDAmodel(Model):
    """Continuous Double Auction model with some number of agents."""
    def __init__(self, N):
        self.period = 1
        self.tick = 1
        self.num_agents = N
        self.initialize_spread()
        self.market_price = None
        # How agents are activated at each step
        self.schedule = RandomActivation(self)
        # Create agents
        for i in range(self.num_agents):
            self.schedule.add(ZeroIntelligence(i, self, "buyer"))
            self.schedule.add(ZeroIntelligence(N+i, self, "seller"))

        # Collecting data
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

        self.datacollector = DataCollector(
            model_reporters={"Period": "period",
                             "Tick": "tick",
                             "OB": "outstanding_bid",
                             "OBer": get_bidder_id,
                             "OA": "outstanding_ask",
                             "OAer": get_asker_id,
                             "Market Price": "market_price"},
            agent_reporters={"Period": lambda x: x.model.period,
                             "Tick": lambda x: x.model.tick,
                             "Role": "role", "Good": "good", "Money": "money"}
        )

    def initialize_spread(self):
        # Initialize outstanding bid and ask
        self.outstanding_bid = 0
        self.outstanding_bidder = None
        self.outstanding_ask = math.inf
        self.outstanding_asker = None

    def update_ob(self, bidder, price):
        if price > self.outstanding_bid:
            # Update the outstanding bid
            self.outstanding_bidder = bidder
            self.outstanding_bid = price
        if price > self.outstanding_ask:
            self.execute_contract(price)

    def update_oa(self, asker, price):
        if price < self.outstanding_ask:
            # Update the outstanding ask
            self.outstanding_asker = asker
            self.outstanding_ask = price
        if price < self.outstanding_bid:
            self.execute_contract(price)

    def execute_contract(self, contract_price):
        self.outstanding_asker.good -= 1
        self.outstanding_asker.money += contract_price
        self.outstanding_bidder.good += 1
        self.outstanding_bidder.money -= contract_price
        self.market_price = contract_price

    def next_tick(self):
        if self.market_price is not None:
            self.market_price = None
            self.initialize_spread()
        self.tick += 1

    def step(self):
        self.schedule.step()
        self.period += 1
        self.tick = 1


model = CDAmodel(2)
for i in range(10):
    model.step()
