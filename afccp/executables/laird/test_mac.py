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

instance = CadetCareerProblem('2021', printing=True)
instance.set_instance_value_parameters('VP_2')
# instance.set_instance_solution('A-VFT')
instance.solve_vft_pyomo_model(max_time=10)
instance.export_to_excel(aggregate=False)
