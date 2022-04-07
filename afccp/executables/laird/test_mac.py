# Get directory path
import os
dir_path = os.getcwd() + '/'

# Get main afccp folder path
index = dir_path.find('afccp')
dir_path = dir_path[:index + 6]

# Update working directory
os.chdir(dir_path)

# Import main problem class
from afccp.core.problem_class import CadetCareerProblem

instance = CadetCareerProblem('C', printing=True)
instance.set_instance_value_parameters()
# instance.solve_vft_pyomo_model(provide_executable=True, max_time=10)
# instance.set_instance_solution('GP')
# instance.export_to_excel()
instance.full_vft_model_solve(ga_max_time=120, ga_printing=True)


