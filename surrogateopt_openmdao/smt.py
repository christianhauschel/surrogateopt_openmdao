from pySOT import *

import numpy as np

from openmdao.core.driver import Driver

import sys

import numpy as np

from openmdao.core.driver import Driver, RecordingDebugging

from smt.applications import EGO
from smt.surrogate_models import KRG
from smt.utils.design_space import DesignSpace



class SMTDriver(Driver):
    """
    Driver wrapper for the SMT-based global optimizer.

    The EGO optimizer is a Bayesian optimizer that uses a Gaussian process surrogate model to
    approximate the objective function. The optimizer is based on the Efficient Global Optimization
    (EGO) algorithm.

    References
    ----------
    https://smt.readthedocs.io/en/stable/_src_docs/applications/ego.html

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Driver options.
    """

    def __init__(self, **kwargs):
        """
        Initialize the SMT optimization driver.
        """
        super().__init__(**kwargs)

        # What we support
        self.supports["optimization"] = True
        self.supports["inequality_constraints"] = False
        self.supports["equality_constraints"] = False
        self.supports["two_sided_constraints"] = False
        self.supports["linear_constraints"] = False
        self.supports["simultaneous_derivatives"] = False

        # What we don't support
        self.supports["multiple_objectives"] = False
        self.supports["active_set"] = False
        self.supports["integer_design_vars"] = False
        self.supports["distributed_design_vars"] = False
        self.supports._read_only = True

        # The user places optimizer-specific settings in here.
        self.opt_settings = {}

        self.result = None
        self._grad_cache = None
        self._con_cache = None
        self._con_idx = {}
        self._obj_and_nlcons = None
        self._dvlist = None
        self._lincongrad_cache = None
        self.fail = False
        self.iter_count = 0
        self._check_jac = False
        self._exc_info = None
        self._total_jac_format = "array"


    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare("optimizer", "EGO", desc="SMT optimization.")
        self.options.declare("criterion", "EI", desc="EGO criterion.")
        self.options.declare(
            "maxiter", 200, lower=0, desc="Maximum number of iterations."
        )
        self.options.declare(
            "n_init", None, lower=0, types=int, desc="Number of points for initial DOE."
        )
        self.options.declare("disp", False, types=bool, desc="Print optimization info.")
        self.options.declare(
            "random_state",
            None,
            types=int,
            allow_none=True,
        )

    def _get_name(self):
        """
        Get name of current optimizer.

        Returns
        -------
        str
            The name of the current optimizer.
        """
        return "SMT_" + self.options["optimizer"] + "_" + self.options["criterion"]

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer
        """
        super()._setup_driver(problem)
        opt = self.options["optimizer"]

        self.supports._read_only = False
        self.supports["gradients"] = False  #  opt in _gradient_optimizers
        self.supports["inequality_constraints"] = False  # opt in _constraint_optimizers
        self.supports["two_sided_constraints"] = False  # opt in _constraint_optimizers
        self.supports["equality_constraints"] = (
            False  # opt in _eq_constraint_optimizers
        )
        self.supports._read_only = True
        # self._check_jac = self.options['singular_jac_behavior'] in ['error', 'warn']

        # # Raises error if multiple objectives are not supported, but more objectives were defined.
        # if not self.supports['multiple_objectives'] and len(self._objs) > 1:
        #     msg = '{} currently does not support multiple objectives.'
        #     raise RuntimeError(msg.format(self.msginfo))

        # # Since COBYLA does not support bounds, we need to add to the _cons metadata
        # # for any bounds that need to be translated into a constraint
        # if opt == 'COBYLA':
        #     for name, meta in self._designvars.items():
        #         lower = meta['lower']
        #         upper = meta['upper']
        #         if isinstance(lower, np.ndarray) or lower > -INF_BOUND \
        #                 or isinstance(upper, np.ndarray) or upper < INF_BOUND:
        #             self._cons[name] = meta.copy()
        #             self._cons[name]['equals'] = None
        #             self._cons[name]['linear'] = True

    def get_driver_objective_calls(self):
        """
        Return number of objective evaluations made during a driver run.

        Returns
        -------
        int
            Number of objective evaluations made during a driver run.
        """
        return self.iter_count

    def get_driver_derivative_calls(self):
        """
        Return number of derivative evaluations made during a driver run.

        Returns
        -------
        int
            Number of derivative evaluations made during a driver run.
        """
        return 0

    def run(self):
        """
        Optimize the problem using selected Scipy optimizer.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        problem = self._problem()
        optimizer = self.options["optimizer"]
        criterion = self.options["criterion"]
        model = problem.model
        self.iter_count = 0
        self._total_jac = None

        self._check_for_missing_objective()
        self._check_for_invalid_desvar_values()



        self._con_cache = self.get_constraint_values()
        desvar_vals = self.get_design_var_values()
        self._dvlist = list(self._designvars)

        # # maxiter and disp get passed into scipy with all the other options.
        # if 'maxiter' not in self.opt_settings:  # lets you override the value in options
        #     self.opt_settings['maxiter'] = self.options['maxiter']
        # self.opt_settings['disp'] = self.options['disp']

        # Size Problem
        ndesvar = 0
        for desvar in self._designvars.values():
            size = desvar["global_size"] if desvar["distributed"] else desvar["size"]
            ndesvar += size
        x_init = np.empty(ndesvar)

        # Initial Design Vars
        i = 0

        lb = []
        ub = []

        for name, meta in self._designvars.items():
            size = meta["global_size"] if meta["distributed"] else meta["size"]
            x_init[i : i + size] = desvar_vals[name]
            i += size

            # Bounds if our optimizer supports them
            meta_low = meta["lower"]
            meta_high = meta["upper"]
            for j in range(size):

                if isinstance(meta_low, np.ndarray):
                    p_low = meta_low[j]
                else:
                    p_low = meta_low

                if isinstance(meta_high, np.ndarray):
                    p_high = meta_high[j]
                else:
                    p_high = meta_high

                lb.append(p_low)
                ub.append(p_high)

        # Set up the optimization problem
        # obj = self._objfunc

        def obj(x):
            n = x.shape[0]
            y = np.zeros(n)

            for i in range(n):
                y[i] = self._objfunc(x[i, :])
            return y

        xlimits = np.array([lb, ub]).T
        n_x = ndesvar

        n_init = self.options["n_init"]
        if n_init is None:
            n_init = n_x * 2 + 1

        n_iter = self.options["maxiter"]

        random_state = self.options["random_state"]
        design_space = DesignSpace(xlimits, random_state=random_state)

        if optimizer == "EGO":
            ego = EGO(
                n_iter=n_iter,
                criterion=criterion,
                n_doe=n_init,
                surrogate=KRG(
                    design_space=design_space, print_global=self.options["disp"]
                ),
                random_state=random_state,
            )
        else:
            raise ValueError(f"Optimizer {optimizer} not supported!")

        x_opt, y_opt, _, x_data, y_data = ego.optimize(fun=obj)

        # import proplot as pplt
        # import matplotlib.pyplot as plt

        # x_plot = np.atleast_2d(np.linspace(lb[0], ub[0], 100)).T
        # y_plot = obj(x_plot)

        # fig = plt.figure(figsize=[10, 10])
        # for i in range(n_iter):
        #     k = n_init + i
        #     x_data_k = x_data[0:k]
        #     y_data_k = y_data[0:k]
        #     ego.gpr.set_training_values(x_data_k, y_data_k)
        #     ego.gpr.train()

        #     y_gp_plot = ego.gpr.predict_values(x_plot)
        #     y_gp_plot_var = ego.gpr.predict_variances(x_plot)
        #     y_ei_plot = -ego.EI(x_plot)

        #     ax = fig.add_subplot((n_iter + 1) // 2, 2, i + 1)
        #     ax1 = ax.twinx()
        #     (ei,) = ax1.plot(x_plot, y_ei_plot, color="red")

        #     (true_fun,) = ax.plot(x_plot, y_plot)
        #     (data,) = ax.plot(
        #         x_data_k, y_data_k, linestyle="", marker="o", color="orange"
        #     )
        #     if i < n_iter - 1:
        #         (opt,) = ax.plot(
        #             x_data[k], y_data[k], linestyle="", marker="*", color="r"
        #         )
        #     (gp,) = ax.plot(x_plot, y_gp_plot, linestyle="--", color="g")
        #     sig_plus = y_gp_plot + 3 * np.sqrt(y_gp_plot_var)
        #     sig_moins = y_gp_plot - 3 * np.sqrt(y_gp_plot_var)
        #     un_gp = ax.fill_between(
        #         x_plot.T[0], sig_plus.T[0], sig_moins.T[0], alpha=0.3, color="g"
        #     )
        #     lines = [true_fun, data, gp, un_gp, opt, ei]
        #     fig.suptitle("EGO optimization of $f(x) = x \sin{x}$")
        #     fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.8)
        #     ax.set_title("iteration {}".format(i + 1))
        #     fig.legend(
        #         lines,
        #         [
        #             "f(x)=xsin(x)",
        #             "Given data points",
        #             "Kriging prediction",
        #             "Kriging 99% confidence interval",
        #             "Next point to evaluate",
        #             "Expected improvment function",
        #         ],
        #     )
        # plt.show()

    
        # Re-run the model with the optimal design point, s.t. the last point 
        # is the optimal one
        # TODO: remove this
        obj(np.array([x_opt]))

        self.result = y_opt

        return False

    def _objfunc(self, x_new):
        """
        Evaluate and return the objective function.

        Model is executed here.

        Parameters
        ----------
        x_new : ndarray
            Array containing input values at new design point.

        Returns
        -------
        float
            Value of the objective function evaluated at the new design point.
        """
        model = self._problem().model

        try:

            # Pass in new inputs
            i = 0

            for name, meta in self._designvars.items():
                size = meta["size"]
                self.set_design_var(name, x_new[i : i + size])
                i += size

            with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
                self.iter_count += 1
                model.run_solve_nonlinear()

            # Get the objective function evaluations
            for obj in self.get_objective_values().values():
                f_new = obj
                break

        except Exception as msg:
            self._exc_info = sys.exc_info()
            return 0

        return f_new
