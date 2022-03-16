# Get directory path
import os
global dir_path
dir_path = os.getcwd() + '/'

# Import main problem class
from afccp.core.problem_class import CadetCareerProblem

instance = CadetCareerProblem('C', printing=True)
instance.set_instance_value_parameters()
instance.set_instance_solution()
instance.display_results_graph(graph='USAFA Proportion')
