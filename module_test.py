# Import module
from afccp.core.problem_class import CadetCareerProblem
import numpy as np

# Import problem instance
instance = CadetCareerProblem("2023b")
instance.import_default_value_parameters()
instance.export_to_excel()

# instance.solve_vft_pyomo_model({"provide_executable": False})
# instance.full_vft_model_solve({"populate": True, "provide_executable": False, "iterate_from_quota": False})
# instance.solve_vft_pyomo_model({"provide_executable": False})

# instance.solve_original_pyomo_model({"max_time": 60, "provide_executable": False})

# instance = CadetCareerProblem("2023b")
# instance.set_instance_value_parameters()
# instance.show_value_function()
# instance.export_to_excel()
# instance.export_value_parameters_as_defaults()
# instance.set_instance_solution("Test")
# instance.solve_for_constraints({"provide_executable": False, "skip_quota_constraint": False})
# instance.set_instance_solution("Original")
# instance.export_to_excel(aggregate=False)
# instance.solve_vft_pyomo_model({"provide_executable": False})
# instance.export_to_excel()

