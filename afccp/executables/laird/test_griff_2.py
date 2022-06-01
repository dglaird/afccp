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
instance.solve_original_pyomo_model()
# instance.set_instance_solution("A-VFT_2")
# instance.solve_vft_pyomo_model(max_time=10)
# instance.import_default_value_parameters()
# instance.export_to_excel(aggregate=False)
instance.export_to_excel()