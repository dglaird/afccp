# Get directory path
import os
dir_path = os.getcwd() + '/'

# Get main afccp folder path
index = dir_path.find('afccp')
dir_path = dir_path[:index + 6]

# Update working directory
os.chdir(dir_path)

# Import compiler class
from afccp.core.problem_class import CadetCareerProblem

instance = CadetCareerProblem('2021', printing=True)
instance.set_instance_value_parameters()
instance.full_vft_model_solve(ga_max_time=60, ga_printing=True)
# instance.vft_to_gp_parameters(get_new_rewards_penalties=True, provide_executable=True)
# instance.export_to_excel()

