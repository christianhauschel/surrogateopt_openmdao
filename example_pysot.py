# %%
from surrogateopt_openmdao import PySOTDriver, plot_pysot
import numpy as np 
from pathlib import Path

dir_out = Path("out/pysot")
if not dir_out.exists():
    dir_out.mkdir(parents=True, exist_ok=True)


import openmdao.api as om

prob = om.Problem()

class ObjComp(om.ExplicitComponent):
    def setup(self):
        self.add_input("y", val=0.0)
        self.add_input("x", shape=(2,), val=np.zeros(2))
        self.add_output("f")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]

        outputs["f"] = x[0]**2 + x[1]**2 + y**2

prob.model.add_subsystem("obj", ObjComp())

# setup the optimization
prob.driver = PySOTDriver()
prob.driver.options["optimizer"] = "SRBF_Failsafe"
prob.driver.options["surrogate"] = "RBF"
prob.driver.options["maxiter"] = 10
prob.driver.options["n_init"] = 4
prob.driver.options["run_initial_design"] = True
prob.driver.options["batch_size"] = 1
prob.driver.options["asynchronous"] = False
prob.driver.options["checkpoint_file"] = str(dir_out / "checkpoint.pysot")
# prob.driver.options["debug_print"] = ["objs", "desvars"]

prob.model.add_design_var("obj.x", lower=np.array([-1,-1]), upper=np.array([1,1]))
prob.model.add_design_var('obj.y', lower=-1, upper=1)
prob.model.add_objective("obj.f")


# %%

prob.setup()
prob.run_driver()


# %% Plot 

plot_pysot(dir_out / "checkpoint.pysot", figsize=(6, 4), dpi=300, show=True)


# %%