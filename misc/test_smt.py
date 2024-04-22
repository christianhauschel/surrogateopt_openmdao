# %%

import numpy as np
from smt.applications import EGO
from smt.surrogate_models import KRG
from smt.utils.design_space import DesignSpace
import proplot as pplt
import matplotlib.pyplot as plt


def function_test_1d(x):
    x = np.reshape(x, (-1,))
    y = np.zeros(x.shape)
    #y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
    y = x * np.sin(x)
    return y.reshape((-1, 1))


# %%

n_iter = 10
xlimits = np.array([[0, 25.0]])

n_x = 1
n_doe = n_x * 2 + 1

random_state = 42  # for reproducibility
random_state = None
design_space = DesignSpace(xlimits, random_state=random_state)
# xdoe = np.atleast_2d([0, 7, 25]).T
# n_doe = xdoe.size

criterion = "EI"  #'EI' or 'SBO' or 'LCB'

ego = EGO(
    n_iter=n_iter,
    criterion=criterion,
    # xdoe=xdoe,
    n_max_optim=n_doe,
    surrogate=KRG(design_space=design_space, print_global=False),
    random_state=random_state,
)

x_opt, y_opt, _, x_data, y_data = ego.optimize(fun=function_test_1d)
print("Minimum in x={:.1f} with f(x)={:.1f}".format(float(x_opt), float(y_opt)))

x_plot = np.atleast_2d(np.linspace(0, 25, 100)).T
y_plot = function_test_1d(x_plot)

fig = plt.figure(figsize=[10, 10])
for i in range(n_iter):
    k = n_doe + i
    x_data_k = x_data[0:k]
    y_data_k = y_data[0:k]
    ego.gpr.set_training_values(x_data_k, y_data_k)
    ego.gpr.train()

    y_gp_plot = ego.gpr.predict_values(x_plot)
    y_gp_plot_var = ego.gpr.predict_variances(x_plot)
    y_ei_plot = -ego.EI(x_plot)

    ax = fig.add_subplot((n_iter + 1) // 2, 2, i + 1)
    ax1 = ax.twinx()
    (ei,) = ax1.plot(x_plot, y_ei_plot, color="red")

    (true_fun,) = ax.plot(x_plot, y_plot)
    (data,) = ax.plot(
        x_data_k, y_data_k, linestyle="", marker="o", color="orange"
    )
    if i < n_iter - 1:
        (opt,) = ax.plot(
            x_data[k], y_data[k], linestyle="", marker="*", color="r"
        )
    (gp,) = ax.plot(x_plot, y_gp_plot, linestyle="--", color="g")
    sig_plus = y_gp_plot + 3 * np.sqrt(y_gp_plot_var)
    sig_moins = y_gp_plot - 3 * np.sqrt(y_gp_plot_var)
    un_gp = ax.fill_between(
        x_plot.T[0], sig_plus.T[0], sig_moins.T[0], alpha=0.3, color="g"
    )
    lines = [true_fun, data, gp, un_gp, opt, ei]
    fig.suptitle("EGO optimization of $f(x) = x \sin{x}$")
    fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.8)
    ax.set_title("iteration {}".format(i + 1))
    fig.legend(
        lines,
        [
            "f(x)=xsin(x)",
            "Given data points",
            "Kriging prediction",
            "Kriging 99% confidence interval",
            "Next point to evaluate",
            "Expected improvment function",
        ],
    )
plt.show()
# Check the optimal point is x_opt=18.9, y_opt =-15.1

# %%

from smt.problems import Sphere, NdimRobotArm, Rosenbrock

fun = Rosenbrock(ndim=1)