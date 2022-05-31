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
# instance.solve_vft_pyomo_model(max_time=10)
# instance.full_vft_model_solve(ga_printing=True, percent_step=10, ga_max_time=60 * 2)
instance.export_to_excel()
