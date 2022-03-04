# Import Research Libraries
from afccp.research.research_functions_other import *
from afccp.research.research_functions_real_comparison import *
from afccp.research.research_functions_data_generation import *
from afccp.research.research_functions_solver_performance import *

instance = CadetCareerProblem(data_name='C', printing=True)
instance.create_aggregate_file()
# chart = instance.display_data_graph(graph='Eligible Quota', alpha=0.5, num=100, figsize=(22, 14))
# instance.import_value_parameters()
# instance.solve_vft_pyomo_model(max_time=10)