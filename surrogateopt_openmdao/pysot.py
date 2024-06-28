from pySOT import *

import numpy as np

from openmdao.core.driver import Driver

import sys


import numpy as np
import os
from openmdao.core.driver import Driver, RecordingDebugging

from poap.controller import BasicWorkerThread, ThreadController, SerialController
from pySOT.controller import CheckpointController
from pySOT.optimization_problems import OptimizationProblem
from pySOT.experimental_design import LatinHypercube
from pySOT.strategy import *
from pySOT.surrogate import *

from pathlib import Path
import yaml

from threading import Lock

lock = Lock()


class PySOTDriver(Driver):
    """
    Driver wrapper for pySOT surrogate optimizer.

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

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare(
            "optimizer",
            default="SRBF",
            desc="Optimiziation strategy: SRBF (default), EI, LCB, and DYCOR.",
        )
        self.options.declare(
            "surrogate",
            default="RBF",
            desc="Surrogate models: RBF (default), GP, MARS, and Poly.",
        )
        self.options.declare(
            "maxiter", 200, lower=0, desc="Maximum number of iterations."
        )
        self.options.declare(
            "n_init", default=None, lower=0, desc="Number of initial points."
        )
        self.options.declare(
            "kwargs_strategy", {}, types=dict, desc="Strategy options."
        )
        self.options.declare(
            "kwargs_surrogate", {}, types=dict, desc="Surrogate options."
        )
        self.options.declare(
            "asynchronous",
            False,
            types=bool,
            desc="Asynchronous optimization (default: True).",
        )
        self.options.declare(
            "use_restarts",
            True,
            types=bool,
            desc="Allow strategy to restart if it gets stuck.",
        )
        self.options.declare(
            "extra_points",
            None,
            types=np.ndarray,
            desc="Extra points to add to the design.",
        )
        self.options.declare(
            "extra_vals",
            None,
            types=np.ndarray,
            desc="Extra values to add to the design.",
        )
        self.options.declare(
            "checkpoint_file",
            None,
            types=str,
            desc="Checkpoint file (makes it possible to restart a manually terminated optimization).",
        )
        self.options.declare(
            "batch_size",
            1,
            types=int,
            desc="Number of points to evaluate in parallel.",
        )

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
        opt = self.options

        # Raise error if batch_size > 1 or asynchronous is True
        if opt["batch_size"] > 1 or opt["asynchronous"]:
            raise ValueError("PySOTDriver does not support batch_size > 1 or asynchronous=True.")

        model = problem.model
        self.iter_count = 0
        self._total_jac = None

        self._check_for_missing_objective()
        self._check_for_invalid_desvar_values()

        # Initial Run
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            model.run_solve_nonlinear()
            self.iter_count += 1
            
        # get initial objective and initial DVs
        y0 = self.get_objective_values()
        x0 = self.get_design_var_values()


        self._con_cache = self.get_constraint_values()
        desvar_vals = self.get_design_var_values()
        self._dvlist = list(self._designvars)

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

        # Surrogate Model
        kwargs_surrogate = opt["kwargs_surrogate"]
        if opt["surrogate"] == "RBF":
            surrogate_fct = RBFInterpolant
        elif opt["surrogate"] == "GP":
            surrogate_fct = GPRegressor
        elif opt["surrogate"] == "MARS":
            surrogate_fct = MARSInterpolant
        elif opt["surrogate"] == "Poly":
            surrogate_fct = PolyRegressor
        else:
            raise ValueError("Surrogate model not recognized.")
        surrogate = surrogate_fct(
            dim=problem.dim,
            lb=problem.lb,
            ub=problem.ub,
            **kwargs_surrogate,
        )

        if opt["batch_size"] > 1:
            controller = ThreadController()
        else:
            controller = SerialController(problem.eval)

        restart = "n"
        if opt["checkpoint_file"] is not None:
            if os.path.exists(opt["checkpoint_file"]):
                restart = input("Checkpoint file exists. Resume optimization? (y/n): ")
                if restart == "y":
                    None
                else:
                    os.remove(opt["checkpoint_file"])

            # -------------------------
            # Save meta data
            # -------------------------

            i = 0
            dv_names = []
            dv_shapes = []
            for name, meta in self._designvars.items():
                size = meta["size"]
                i += size
                dv_names.append(name)
                dv_shapes.append(size)

            dict_info = {
                "dv": {
                    "name": dv_names,
                    "shape": dv_shapes,
                }
            }

            dir_out = Path(opt["checkpoint_file"]).parent
            fname_yaml = dir_out / "info.yaml"
            with open(fname_yaml, "w") as f:
                yaml.dump(dict_info, f)

        if restart != "y":
            print("Fresh optimization...")
            n_init_min = 2 * (problem.dim + 1)
            if opt["n_init"] is None:
                n_init = n_init_min
            else:
                n_init = opt["n_init"]
                if n_init < n_init_min:
                    print(
                        f"Warning: n_init is less than {n_init_min}. Setting n_init to {n_init_min}"
                    )
                    n_init = n_init_min
            sampling = LatinHypercube(dim=problem.dim, num_pts=n_init)

            if opt["optimizer"] == "SRBF":
                strategy_fct = SRBFStrategy
            elif opt["optimizer"] == "SRBF_Failsafe":
                try:
                    strategy_fct = SRBFFailsafeStrategy
                except:
                    print("Strategy `SRBF_Failsafe` not available.")
                    return
            elif opt["optimizer"] == "EI":
                strategy_fct = EIStrategy
            elif opt["optimizer"] == "LCB":
                strategy_fct = LCBStrategy
            elif opt["optimizer"] == "DYCOR":
                strategy_fct = DYCORSStrategy
            elif opt["optimizer"] == "SOP":
                strategy_fct = SOPStrategy
            else:
                raise ValueError("Strategy not recognized.")
            
            if x0 is not None and y0 is not None:
                
                
                # concat points from arrays in dict x0 into an array _x0 
                _x0 = np.array([])
                for name, meta in self._designvars.items():
                    size = meta["size"]
                    _x0 = np.hstack((_x0, x0[name]))
                
                # concat points from arrays in dict y0 into an array _y0
                _y0 = np.array([])
                for name, val in y0.items():
                    _y0 = np.hstack((_y0, val))
                    
                
                # insert extra points at beginning of opt["extra_points"] (n_pts x nx)
                if opt["extra_points"] is not None:
                    extra_points = opt["extra_points"]
                    extra_vals = opt["extra_vals"]
                else:
                    extra_points = np.zeros((1, problem.dim))
                    extra_vals = np.zeros((1, 1))
                    
                    extra_points[0,:] = _x0
                    extra_vals[0,0] = _y0[0]
                    
                    
            else:
                extra_points = opt["extra_points"]
                extra_vals = opt["extra_vals"]
            
            strategy = strategy_fct(
                opt_prob=problem,
                exp_design=sampling,
                surrogate=surrogate,
                asynchronous=opt["asynchronous"],
                max_evals=opt["maxiter"],
                use_restarts=opt["use_restarts"],
                extra_points=extra_points,
                extra_vals=extra_vals,
                batch_size=opt["batch_size"],
                **opt["kwargs_strategy"],
            )
            controller.strategy = strategy

        if opt["batch_size"] > 1:
            for _ in range(opt["batch_size"]):
                worker = BasicWorkerThread(controller, problem.eval)
                controller.launch_worker(worker)


        if opt["checkpoint_file"] is not None:
            controller = CheckpointController(controller, opt["checkpoint_file"])

        if restart != "y":
            _ = controller.run()
        else:
            _ = controller.resume()

        # x = [record.params for record in controller.fevals]
        # x = np.array(x)
        # y = [record.value for record in controller.fevals]

        # Run best design again, to register it in OpenMDAO's recorder as last value
        # (this is not registered in the checkpoint file though)
        # x_opt = x[np.argmin(y)]
        # y_opt = obj(x_opt)

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

            # Set DVs
            i = 0
            for name, meta in self._designvars.items():
                size = meta["size"]
                self.set_design_var(name, x_new[i : i + size])
                i += size

            # Run Model
            with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
                self.iter_count += 1
                model.run_solve_nonlinear()

            # Get Objective
            for obj in self.get_objective_values().values():
                f_new = obj
                break

        except Exception as msg:
            self._exc_info = sys.exc_info()
            return 0

        return f_new
