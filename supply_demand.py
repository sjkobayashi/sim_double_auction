import numpy as np
import matplotlib.pyplot as plt


class Supply:
    """A class representing a supply curve."""
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
        """Compute the maximum (competitive) surplus."""
        surplus = self.equilibrium_price - self.price_schedule_in
        iterable = (p * self.q_per_agent for p in surplus)
        self.theoretical_surplus = sum(np.fromiter(iterable, float))


class Demand(Supply):
    """A class representing a demand curve."""
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
        """Compute the maximum (competitive) surplus."""
        surplus = self.price_schedule_in - self.equilibrium_price
        iterable = (p * self.q_per_agent for p in surplus)
        self.theoretical_surplus = sum(np.fromiter(iterable, float))

