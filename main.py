# Import Research Libraries
from afccp.research.research_functions_other import *
from afccp.research.research_functions_real_comparison import *
from afccp.research.research_functions_data_generation import *
from afccp.research.research_functions_solver_performance import *

instance = CadetCareerProblem("2021", printing=True)
instance.import_default_value_parameters(no_constraints=True)
instance.genetic_algorithm(stopping_time=20)
instance.full_vft_model_solve()
instance.export_to_excel()