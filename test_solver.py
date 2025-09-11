from pyomo.environ import *

# Define a simple model
model = ConcreteModel()

# Variables
model.x = Var(within=NonNegativeReals)
model.y = Var(within=NonNegativeReals)

# Objective: Maximize 3x + 4y
model.obj = Objective(expr=3 * model.x + 4 * model.y, sense=maximize)

# Constraint: x + 2y <= 8
model.constraint = Constraint(expr=model.x + 2 * model.y <= 8)

# Solve with CBC
solver = SolverFactory("cbc")
result = solver.solve(model, tee=True)

# Print results
print(f"x = {model.x():.2f}")
print(f"y = {model.y():.2f}")
print(f"Objective = {model.obj():.2f}")