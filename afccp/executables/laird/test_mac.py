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
instance.full_vft_model_solve(ga_max_time=30, ga_printing=True, percent_step=15)
print(instance.solution_dict.keys())

