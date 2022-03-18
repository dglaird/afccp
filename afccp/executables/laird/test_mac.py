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
instance.import_default_value_parameters()

