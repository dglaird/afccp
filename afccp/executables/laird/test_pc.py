# Get directory path
import os
global dir_path
dir_path = os.getcwd() + '/'

# Import main problem class
from afccp.core.problem_class import CadetCareerProblem

instance = CadetCareerProblem('2020', printing=True)
instance.set_instance_value_parameters()
gp_var = instance.solve_gp_pyomo_model(con_term='T', max_time=60*4)
print(gp_var)
# instance.set_instance_solution()
# instance.display_results_graph(graph='USAFA Proportion')
