# Get directory path
import os
global dir_path
dir_path = os.getcwd() + '/'

# Import main problem class
from afccp.core.problem_class import CadetCareerProblem

instance = CadetCareerProblem('Realistic', printing=True)
instance.import_default_value_parameters(no_constraints=True)
instance.solve_vft_pyomo_model(max_time=10)
instance.export_to_excel()
