# Get directory path
import os
dir_path = os.getcwd() + '/'

print(dir_path)

# # Get main afccp folder path
# index = dir_path.find('afccp')
# dir_path = dir_path[:index + 6]

# # Update working directory
# os.chdir(dir_path)

# # Import main problem class
# from afccp.core.problem_class import CadetCareerProblem

# instance = CadetCareerProblem('2021', printing=True)
# instance.set_instance_value_parameters()
# instance.set_instance_solution()
# instance.export_value_parameters_as_defaults()
