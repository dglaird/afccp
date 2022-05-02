# Get directory path
import os
dir_path = os.getcwd() + '/'

# Get main afccp folder path
index = dir_path.find('afccp')
dir_path = dir_path[:index + 6]

# Update working directory
os.chdir(dir_path)

# Import compiler class
from afccp.processing.compiler import DataAggregator

data = DataAggregator()
data.compile_problem_instance_file(printing=True)
