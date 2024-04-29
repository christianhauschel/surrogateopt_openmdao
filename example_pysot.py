# %%

from surrogateopt_openmdao import SMTDriver, PySOTDriver
import numpy as np 
import proplot as pplt
import openmdao.api as om

# build the model
prob = om.Problem()

#prob.model.add_subsystem("obj", om.ExecComp("f = x * sin(x * y) * y"))
prob.model.add_subsystem("obj", om.ExecComp("f = x**2 + y**2"))

# setup the optimization
prob.driver = PySOTDriver()
prob.driver.options["optimizer"] = "RBF"
prob.driver.options["maxiter"] = 100
prob.driver.options["n_init"] = 5

prob.model.add_design_var("obj.x", lower=-1, upper=1)
prob.model.add_design_var('obj.y', lower=-1, upper=1)
prob.model.add_objective("obj.f")

# Create a recorder
fname_recorder = "reports/cases.sql"
recorder = om.SqliteRecorder(fname_recorder)
prob.add_recorder(recorder)
prob.driver.add_recorder(recorder)

prob.setup()
prob.run_driver()

# %% Print the results

print("DVs")
print(prob.get_val("obj.x"))
print(prob.get_val("obj.y"))

print("Objective")
print(prob.get_val("obj.f"))


# %% Load the recorded data

# Instantiate your CaseReader
cr = om.CaseReader(fname_recorder)


cases = cr.list_cases("driver", recurse=False, out_stream=None)
n_cases = len(cases)

iterations = np.arange(1, n_cases + 1)
objectives = np.zeros(n_cases)
constraints = np.zeros(n_cases)

for i, case in enumerate(cases):
    objectives[i] = cr.get_case(case).get_objectives()["obj.f"][0]


driver_cases = cr.list_cases("driver", recurse=False)


fig, ax = pplt.subplots(ncols=1, nrows=1, figsize=(5, 3), sharey=False)
ax[0].plot(iterations, objectives)
ax.format(
    suptitle="Optimization History",
    xlabel="Iterations",
)
ax[0].set(ylabel="obj") 
pplt.show()


# %% Visualize the optimization history in iPython

# try:
#     om.CaseViewer(fname_recorder)
# except:
#     pass

# %%
