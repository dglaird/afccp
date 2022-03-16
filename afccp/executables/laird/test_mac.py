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
instance.set_instance_solution()
print(instance.solution_dict.keys())
instance.solve_gp_pyomo_model(max_time=20)
instance.export_to_excel()