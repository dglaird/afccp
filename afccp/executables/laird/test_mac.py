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
print(instance.vp_dict.keys())
instance.set_instance_value_parameters()
print('before', instance.value_parameters['objective_weight'][11, :])
instance.value_parameters['objective_weight'][11, 0] += 0.02
instance.value_parameters['objective_weight'][11, 2] -= 0.02
print('after', instance.value_parameters['objective_weight'][11, :])
print(instance.vp_name)
instance.save_new_value_parameters_to_dict()
print(instance.vp_dict.keys())
print(instance.vp_name)