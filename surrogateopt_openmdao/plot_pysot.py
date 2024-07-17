# %%
import proplot as pplt
import dill
from pathlib import Path
import yaml
import numpy as np


def calc_grid(n: int) -> tuple:
    """
    Calc number of cols/rows for subplots.
    """

    def nearest_square(num):
        num1 = np.floor(np.sqrt(num)) ** 2
        return np.sqrt(num1)

    n_rows = int(nearest_square(n))
    n_cols = n_rows

    while True:
        if n_cols * n_rows < n:
            n_cols += 1
        else:
            break
    return n_rows, n_cols


def flattened_names(names, shapes):
    flattened = []
    for i, name in enumerate(names):
        if (shapes[i][0] == 1 and len(shapes[i]) == 1) or (
            shapes[i][0] == 1 and shapes[i][1] == 1 and len(shapes[i]) == 2
        ):
            flattened.append(name)
        elif (shapes[i][0] > 1 and len(shapes[i]) == 1) or (
            shapes[i][0] > 1 and shapes[i][1] == 1 and len(shapes[i]) == 2
        ):
            for i in range(shapes[i][0]):
                flattened.append(f"{name}_{i}")

        # if 2d matrix
        elif shapes[i][0] > 1 and shapes[i][1] > 1 and len(shapes[i]) == 2:
            for i in range(shapes[i][0]):
                for j in range(shapes[i][1]):
                    flattened.append(f"{name}_{i}_{j}")

        else:
            raise ValueError("Unsupported shape")
    return flattened


def plot_pysot(
    fname, figsize=(6, 4), fname_plot=None, dpi=300, show=False, plot_dv_separate=True
):

    if type(fname) is str:
        fname = Path(fname)

    fname_info = fname.parent / "info.yaml"

    with open(fname_info, "r") as f:
        info = yaml.safe_load(f)

    dv_names = info["dv"]["name"]
    dv_shapes = [(val,) for val in info["dv"]["shape"]]
    dv_names = flattened_names(dv_names, dv_shapes)

    with open(fname, "rb") as f:
        strategy = dill.load(f)

    x = strategy.X
    y = strategy.fX
    n = len(x)
    lb = strategy.surrogate.lb
    ub = strategy.surrogate.ub

    # range from 0 to num_evals
    ids = range(n)
    i_min = np.argmin(y)
    id_min = ids[i_min]
    y_min = y[i_min]
    

    if plot_dv_separate:
        n_rows, n_cols = calc_grid(x.shape[1] + 1)
    else:
        n_rows, n_cols = 1, 2

    fig, ax = pplt.subplots(ncols=n_cols, nrows=n_rows, figsize=figsize, sharey=False)
    ax[0].plot(ids, y)
    ax[0].plot(id_min, y_min, ".", c="C1", label="Minimum")
    ax[0].plot(0, y[0], "x", c="k", label="Initial")
    ax[0].legend()
    ax[0].set(
        ylabel="Objective",
        xlabel="Iterations",
    )

    ax.format(
        suptitle="Optimization History",
    )

    if plot_dv_separate:
        for i in range(x.shape[1]):
            ax[i + 1].plot(ids, x[:, i], "-")
            ax[i + 1].set(
                ylabel=f"{dv_names[i]}",
                xlabel="Iterations",
            )
            ax[i + 1].hlines(lb[i], 0, n-1, color="C2", linestyle="--", lw=1)
            ax[i + 1].hlines(ub[i], 0, n-1, color="C2", linestyle="--", lw=1)
            ax[i + 1].plot(id_min, x[i_min, i], ".", c="C1")
            ax[i + 1].plot(0, x[0, i], "x", c="k")
            
    else:
        for i in range(x.shape[1]):
            ax[1].plot(ids, x[:, i], label=dv_names[i])
        ax[1].set(
            ylabel="DV",
            xlabel="Iterations",
        )
        ax[1].legend()

    if show:
        pplt.show()

    if fname_plot is not None:
        fig.savefig(fname_plot, dpi=dpi)

    return fig


# %%
