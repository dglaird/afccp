# Get directory path
import os
dir_path = os.getcwd() + '/'

# Get main afccp folder path
index = dir_path.find('afccp')
dir_path = dir_path[:index + 6]

# Update working directory
os.chdir(dir_path)
#
# Import problem class
from afccp.core.problem_class import CadetCareerProblem

instance = CadetCareerProblem('2023', printing=True)
instance.set_instance_value_parameters()
instance.full_vft_model_solve(ga_printing=True, percent_step=5)
instance.export_to_excel()
