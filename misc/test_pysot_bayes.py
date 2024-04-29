# %%

import numpy as np
import proplot as pplt

from poap.controller import SerialController
from pySOT.optimization_problems import OptimizationProblem
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy, EIStrategy
from pySOT.surrogate import RBFInterpolant, GPRegressor

max_evals = 100


def f(x: np.ndarray) -> float:
    return x[0] * np.sin(x[0] * x[1]) * x[1]


class Obj(OptimizationProblem):
    def __init__(self):
        self.lb = np.array([-2, -2])
        self.ub = np.array([2, 2])
        self.dim = 2
        self.cont_var = np.arange(0, self.dim)
        self.int_var = np.array([])

    def eval(self, x):
        if x.shape[0] != self.dim:
            raise ValueError("Dimension mismatch")
        return f(x)


obj = Obj()

surrogate = GPRegressor(dim=obj.dim, lb=obj.lb, ub=obj.ub)

n_initial = 2 * (obj.dim + 1)
sampling = SymmetricLatinHypercube(dim=obj.dim, num_pts=n_initial)

# Create a strategy and a controller
controller = SerialController(obj.eval)
controller.strategy = EIStrategy(
    max_evals=max_evals,
    opt_prob=obj,
    exp_design=sampling,
    surrogate=surrogate,
    asynchronous=True,
    batch_size=1,
    use_restarts=True,
)
# controller = CheckpointController(controller, fname=fname)

result = controller.run()
# result = controller.resume()

x_opt = result.params
y_opt = result.value


# %% Plot the results

x = [record.params[0] for record in controller.fevals]

# convert x to matrix 
x = np.array(x)

y = [record.value for record in controller.fevals]

ids = np.arange(1, len(y) + 1)
id_opt = np.argmin(y) + 1

fig, ax = pplt.subplots(nrows=2, sharex=True, sharey=False, figsize=(5,5), tight_layout=True)
ax[0].plot(ids, y, ".")
ax[0].set_xlabel("Evaluations")
ax[0].set_ylabel("Objective")
ax[0].vlines(n_initial, min(y), max(y), color="k", linestyle="--")
ax[0].vlines(id_opt, min(y), max(y), color="r", linestyle="--")

for i in range(obj.dim):
    ax[1].plot(ids, x[:, i], ".", label="x{}".format(i + 1))

# add vline at n_initial, from max lb to max ub
ax[1].vlines(n_initial, min(obj.lb), max(obj.ub), color="k", linestyle="--")
ax[1].vlines(id_opt, min(obj.lb), max(obj.ub), color="r", linestyle="--")

ax[1].set_xlabel("Evaluations")
ax[1].set_ylabel("DV")


# %%
