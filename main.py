# Import Research Libraries
from afccp.research.research_functions_other import *
from afccp.research.research_functions_real_comparison import *
from afccp.research.research_functions_data_generation import *
from afccp.research.research_functions_solver_performance import *

instance = CadetCareerProblem("C", printing=True)
instance.create_aggregate_file()