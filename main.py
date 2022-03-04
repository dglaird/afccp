# Import Research Libraries
from afccp.research.research_functions_other import *
from afccp.research.research_functions_real_comparison import *
from afccp.research.research_functions_data_generation import *
from afccp.research.research_functions_solver_performance import *

instance = CadetCareerProblem("2021", printing=True)
instance.value_parameters = instance.vp_dict['VP_2']
metrics_dict = {'Original': instance.metrics_dict['VP_2']['Original'],
                'VFT_2': instance.metrics_dict['VP_2']['VFT_2']}

# instance.display_data_graph(graph='Eligible Quota', save=True, figsize=(24, 10), num=100)
instance.display_results_graph(graph='Combined Quota', figsize=(24, 10), save=True, metrics_dict=metrics_dict,
                               dot_size=200, display_title=False, y_max=1.25)