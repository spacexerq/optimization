import numpy as np


class TransportVS:
    """
    a - Providers resource;
    b - Consumers wish;
    c - Rate matrix;
    """

    def __init__(self, a: np.array, b: np.array, c: np.array):
        self.has_fict_column = None
        self.has_fict_row = None
        self.a = a
        self.b = b
        self.c = c

    @property
    def m(self):
        """Number of providers."""
        return len(self.a)

    @property
    def n(self):
        """Number of consumers."""
        return len(self.b)

    def get_supply_demand_difference(self):
        """Solvability. Difference between supply and demand. """
        return sum(self.a) - sum(self.b)

    # It is possible to use fake consumer and provider, could be  realised.

    def calculate_cost(self, x):
        """Aim function"""
        return np.sum(self.c * np.nan_to_num(x))

    def calculate_potentials(self, x):
        """Potentials"""
        res = {'a': [np.inf for _ in range(self.m)], 'b': [np.inf for _ in range(self.n)]}
        res['a'][0] = 0.0

        while np.inf in res['a'] or np.inf in res['b']:
            for i in range(self.m):
                for j in range(self.n):
                    if x[i][j] != 0:
                        if res['a'][i] != np.inf:
                            res['b'][j] = self.c[i][j] - res['a'][i]
                        elif res['b'][j] != np.inf:
                            res['a'][i] = self.c[i][j] - res['b'][j]

        return res

    def is_plan_optimal(self, x, p):
        for i, j in zip(*np.nonzero(x == 0)):
            if p['a'][i] + p['b'][j] > self.c[i][j]:
                return False
        return True

    def get_best_free_cell(self, x, p):
        """Finds the best cell to recalculate potentials"""
        free_cells = tuple(zip(*np.nonzero(x == 0)))
        return free_cells[np.argmax([p['a'][i] + p['b'][j] - self.c[i][j] for i, j in free_cells])]

    def add_supplier(self, volume) -> None:
        e = np.ones(self.n)
        self.c = np.row_stack((self.c, e))
        self.a = np.append(self.a, volume)
        if np.all(e == 0):
            self.has_fict_row = True

    def add_customer(self, volume) -> None:
        e = np.ones(self.m)
        self.c = np.column_stack((self.c, e))
        self.b = np.append(self.b, volume)
        if np.all(e == 0):
            self.has_fict_column = True
