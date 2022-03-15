# Import Research Libraries
from afccp.research.research_functions_other import *
from afccp.research.research_functions_real_comparison import *
from afccp.research.research_functions_data_generation import *
from afccp.research.research_functions_solver_performance import *

instance = CadetCareerProblem(data_name='2021', printing=True)
instance.display_data_graph(graph='USAFA Proportion')
instance.set_instance_value_parameters()
instance.set_instance_solution()
instance.solve_gp_pyomo_model(max_time=10)
print(instance.solution_dict.keys())