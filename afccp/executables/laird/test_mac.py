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

instance = CadetCareerProblem('2015', printing=True)
# instance.import_default_value_parameters()
# instance.set_instance_value_parameters()
# instance.solve_vft_pyomo_model(max_time=10)
# instance.vft_to_gp_parameters(get_new_rewards_penalties=True)
# instance.solve_gp_pyomo_model()
instance.export_to_excel()
