from pySOT import *

import numpy as np

from openmdao.core.driver import Driver
from collections import OrderedDict

import sys
from packaging.version import Version

import numpy as np
from scipy import __version__ as scipy_version
from scipy.optimize import minimize

from openmdao.core.constants import INF_BOUND
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.class_util import WeakMethodWrapper
from openmdao.utils.mpi import MPI

from poap.controller import SerialController
from pySOT.optimization_problems import OptimizationProblem
from pySOT.experimental_design import SymmetricLatinHypercube, LatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import RBFInterpolant


CITATIONS = """
...
"""


class PySOTDriver(Driver):
    """
    Driver wrapper for the scipy.optimize.minimize family of local optimizers.

    Inequality constraints are supported by COBYLA and SLSQP,
    but equality constraints are only supported by SLSQP. None of the other
    optimizers support constraints.

    ScipyOptimizeDriver supports the following:
        equality_constraints
        inequality_constraints

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Driver options.

    Attributes
    ----------
    fail : bool
        Flag that indicates failure of most recent optimization.
    iter_count : int
        Counter for function evaluations.
    result : OptimizeResult
        Result returned from scipy.optimize call.
    opt_settings : dict
        Dictionary of solver-specific options. See the scipy.optimize.minimize documentation.
    _check_jac : bool
        Used internally to control when to perform singular checks on computed total derivs.
    _con_cache : dict
        Cached result of constraint evaluations because scipy asks for them in a separate function.
    _con_idx : dict
        Used for constraint bookkeeping in the presence of 2-sided constraints.
    _grad_cache : {}
        Cached result of nonlinear constraint derivatives because scipy asks for them in a separate
        function.
    _exc_info : 3 item tuple
        Storage for exception and traceback information.
    _obj_and_nlcons : list
        List of objective + nonlinear constraints. Used to compute total derivatives
        for all except linear constraints.
    _dvlist : list
        Copy of _designvars.
    _lincongrad_cache : np.ndarray
        Pre-calculated gradients of linear constraints.
    """

    def __init__(self, **kwargs):
        """
        Initialize the ScipyOptimizeDriver.
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

        self.cite = CITATIONS

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare("optimizer", "RBF")
        # self.options.declare('tol', 1.0e-6, lower=0.0,
        #                      desc='Tolerance for termination. For detailed '
        #                      'control, use solver-specific options.')
        self.options.declare('maxiter', 200, lower=0,
                             desc='Maximum number of iterations.')
        self.options.declare('n_init', None, lower=0, desc='Number of initial points.')
        self.options.declare("kwargs_surrogate", {}, types=dict, desc="Surrogate options")
        self.options.declare("use_restarts", True, types=bool, desc="Use restarts")
        # self.options.declare('disp', True, types=bool,
        #                      desc='Set to False to prevent printing of Scipy convergence messages')


    def _get_name(self):
        """
        Get name of current optimizer.

        Returns
        -------
        str
            The name of the current optimizer.
        """
        return "PySOT_" + self.options["optimizer"]

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


    def get_driver_objective_calls(self):
        """
        Return number of objective evaluations made during a driver run.

        Returns
        -------
        int
            Number of objective evaluations made during a driver run.
        """
        if self.result and hasattr(self.result, "nfev"):
            return self.result.nfev
        else:
            return None

    def get_driver_derivative_calls(self):
        """
        Return number of derivative evaluations made during a driver run.

        Returns
        -------
        int
            Number of derivative evaluations made during a driver run.
        """
        if self.result and hasattr(self.result, "njev"):
            return self.result.njev
        else:
            return None

    def run(self):
        """
        Optimize the problem using selected Scipy optimizer.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        problem = self._problem()
        opt = self.options["optimizer"]
        model = problem.model
        self.iter_count = 0
        self._total_jac = None

        self._check_for_missing_objective()
        self._check_for_invalid_desvar_values()

        # Initial Run
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            model.run_solve_nonlinear()
            self.iter_count += 1

        self._con_cache = self.get_constraint_values()
        desvar_vals = self.get_design_var_values()
        self._dvlist = list(self._designvars)

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
        obj = self._objfunc

        class Obj(OptimizationProblem):
            def __init__(self):
                self.dim = ndesvar
                self.lb = np.array(lb)
                self.ub = np.array(ub)
                self.int_var = np.array([])
                self.cont_var = np.arange(0, ndesvar)
                self.info = "obj"

            def eval(self, x):
                return obj(x)[0]

        problem = Obj()


        kwargs_surrogate = self.options["kwargs_surrogate"]

        # Surrogate Model
        surrogate = RBFInterpolant(
            dim=problem.dim,
            lb=problem.lb,
            ub=problem.ub,
            **kwargs_surrogate,
            # kernel=CubicKernel(),
            # tail=LinearTail(problem.dim),
        )

        controller = SerialController(problem.eval, skip=False)

        if self.options["n_init"] is None:
            n_init = 2 * (problem.dim + 1)
        else:
            n_init = self.options["n_init"]
        sampling = LatinHypercube(dim=problem.dim, num_pts=n_init)

        maxiter = self.options["maxiter"]

        strategy = SRBFStrategy(
            opt_prob=problem,
            exp_design=sampling,
            surrogate=surrogate,
            asynchronous=False,
            max_evals=maxiter,
            batch_size=1,
            use_restarts=self.options["use_restarts"],
            # extra_points=np.array([x0]) if config["optim"]["enable_x0"] else None,
            # extra_vals=np.array([[y0]]) if config["optim"]["enable_x0"] else None,
        )

        controller.strategy = strategy

        try:
            res = controller.run()
        except Exception as e:
            # print warning 
            print(f"Warning: PySOT error\n\t {e}")
            self.fail = True
            return self.fail
        
        x = [record.params for record in controller.fevals]
        x = np.array(x)
        y = [record.value for record in controller.fevals]

        y_opt = min(y)

        self.result = y_opt

        # return self.fail

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

            self._con_cache = self.get_constraint_values()

        except Exception as msg:
            self._exc_info = sys.exc_info()
            return 0

        return f_new

    def _con_val_func(self, x_new, name, dbl, idx):
        """
        Return the value of the constraint function requested in args.

        The lower or upper bound is **not** subtracted from the value. Used for optimizers,
        which take the bounds of the constraints (e.g. trust-constr)

        Parameters
        ----------
        x_new : ndarray
            Array containing input values at new design point.
        name : str
            Name of the constraint to be evaluated.
        dbl : bool
            True if double sided constraint.
        idx : float
            Contains index into the constraint array.

        Returns
        -------
        float
            Value of the constraint function.
        """
        return self._con_cache[name][idx]

    def _confunc(self, x_new, name, dbl, idx):
        """
        Return the value of the constraint function requested in args.

        Note that this function is called for each constraint, so the model is only run when the
        objective is evaluated.

        Parameters
        ----------
        x_new : ndarray
            Array containing input values at new design point.
        name : str
            Name of the constraint to be evaluated.
        dbl : bool
            True if double sided constraint.
        idx : float
            Contains index into the constraint array.

        Returns
        -------
        float
            Value of the constraint function.
        """
        if self._exc_info is not None:
            self._reraise()

        cons = self._con_cache
        meta = self._cons[name]

        # Equality constraints
        equals = meta["equals"]
        if equals is not None:
            if isinstance(equals, np.ndarray):
                equals = equals[idx]
            return cons[name][idx] - equals

        # Note, scipy defines constraints to be satisfied when positive,
        # which is the opposite of OpenMDAO.
        upper = meta["upper"]
        if isinstance(upper, np.ndarray):
            upper = upper[idx]

        lower = meta["lower"]
        if isinstance(lower, np.ndarray):
            lower = lower[idx]

        if dbl or (lower <= -INF_BOUND):
            return upper - cons[name][idx]
        else:
            return cons[name][idx] - lower

    # def _gradfunc(self, x_new):
    #     """
    #     Evaluate and return the gradient for the objective.

    #     Gradients for the constraints are also calculated and cached here.

    #     Parameters
    #     ----------
    #     x_new : ndarray
    #         Array containing input values at new design point.

    #     Returns
    #     -------
    #     ndarray
    #         Gradient of objective with respect to input array.
    #     """
    #     try:
    #         grad = self._compute_totals(of=self._obj_and_nlcons, wrt=self._dvlist,
    #                                     return_format=self._total_jac_format)
    #         self._grad_cache = grad

    #         # First time through, check for zero row/col.
    #         if self._check_jac:
    #             raise_error = self.options['singular_jac_behavior'] == 'error'
    #             self._total_jac.check_total_jac(raise_error=raise_error,
    #                                             tol=self.options['singular_jac_tol'])
    #             self._check_jac = False

    #     except Exception as msg:
    #         self._exc_info = sys.exc_info()
    #         return np.array([[]])

    #     # print("Gradients calculated for objective")
    #     # print('   xnew', x_new)
    #     # print('   grad', grad[0, :])

    #     return grad[0, :]

    # def _congradfunc(self, x_new, name, dbl, idx):
    #     """
    #     Return the cached gradient of the constraint function.

    #     Note, scipy calls the constraints one at a time, so the gradient is cached when the
    #     objective gradient is called.

    #     Parameters
    #     ----------
    #     x_new : ndarray
    #         Array containing input values at new design point.
    #     name : str
    #         Name of the constraint to be evaluated.
    #     dbl : bool
    #         Denotes if a constraint is double-sided or not.
    #     idx : float
    #         Contains index into the constraint array.

    #     Returns
    #     -------
    #     float
    #         Gradient of the constraint function wrt all inputs.
    #     """
    #     if self._exc_info is not None:
    #         self._reraise()

    #     meta = self._cons[name]

    #     if meta['linear']:
    #         grad = self._lincongrad_cache
    #     else:
    #         grad = self._grad_cache
    #     grad_idx = self._con_idx[name] + idx

    #     # print("Constraint Gradient returned")
    #     # print('   xnew', x_new)
    #     # print('   grad', name, 'idx', idx, grad[grad_idx, :])

    #     # Equality constraints
    #     if meta['equals'] is not None:
    #         return grad[grad_idx, :]

    #     # Note, scipy defines constraints to be satisfied when positive,
    #     # which is the opposite of OpenMDAO.
    #     lower = meta['lower']
    #     if isinstance(lower, np.ndarray):
    #         lower = lower[idx]

    #     if dbl or (lower <= -INF_BOUND):
    #         return -grad[grad_idx, :]
    #     else:
    #         return grad[grad_idx, :]

    # def _reraise(self):
    #     """
    #     Reraise any exception encountered when scipy calls back into our method.
    #     """
    #     raise self._exc_info[1].with_traceback(self._exc_info[2])


# def signature_extender(fcn, extra_args):
#     """
#     Closure function, which appends extra arguments to the original function call.

#     The first argument is the design vector. The possible extra arguments from the callback
#     of :func:`scipy.optimize.minimize` are not passed to the function.

#     Some algorithms take a sequence of :class:`~scipy.optimize.NonlinearConstraint` as input
#     for the constraints. For this class it is not possible to pass additional arguments.
#     With this function the signature will be correct for both scipy and the driver.

#     Parameters
#     ----------
#     fcn : callable
#         Function, which takes the design vector as the first argument.
#     extra_args : tuple or list
#         Extra arguments for the function.

#     Returns
#     -------
#     callable
#         The function with the signature expected by the driver.
#     """
#     def closure(x, *args):
#         return fcn(x, *extra_args)

#     return closure
