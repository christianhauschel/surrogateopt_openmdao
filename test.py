# %%

from surrogateopt_openmdao import PySOTOptimizer, SMTOptimizer
import numpy as np 
import proplot as pplt
import openmdao.api as om

# build the model
prob = om.Problem()

prob.model.add_subsystem("obj", om.ExecComp("f = x * sin(x)"))

# setup the optimization
prob.driver = SMTOptimizer()
prob.driver.options["optimizer"] = "EI"
prob.driver.options["optimizer"] = "SBO"
prob.driver.options["maxiter"] = 10
prob.driver.options["n_init"] = 5

prob.model.add_design_var("obj.x", lower=0, upper=10)
# prob.model.add_design_var('obj.y', lower=-50, upper=50)
prob.model.add_objective("obj.f")

# Create a recorder
fname_recorder = "reports/cases.sql"
recorder = om.SqliteRecorder(fname_recorder)
prob.add_recorder(recorder)
prob.driver.add_recorder(recorder)


prob.setup()

# Set initial values.
# prob.set_val('obj.x', 3.0)
# prob.set_val('obj.y', -4.0)

# run the optimization
prob.run_driver()

# %%

# print the results
print("DVs")
print(prob.get_val("obj.x"))

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

# %%


# fig, ax = pplt.subplots()
# x = np.linspace(0, 10, 100)
# ax.plot(x, x * np.sin(x))
# ax.plot(prob.get_val("obj.x"), prob.get_val("obj.f"), "o")
# ax.format(
#     xlabel="x",
#     ylabel="f(x)",
#     title="Objective Function"
# )
# pplt.show()

# %%

#om.CaseViewer(fname_recorder)

# %%
