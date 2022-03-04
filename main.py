# Import Research Libraries
import copy

from afccp.research.research_functions_other import *
from afccp.research.research_functions_real_comparison import *
from afccp.research.research_functions_data_generation import *
from afccp.research.research_functions_solver_performance import *

instance = CadetCareerProblem(data_name='C', printing=True)
# instance.create_aggregate_file()
# chart = instance.display_data_graph(graph='Eligible Quota', alpha=0.5, num=100, figsize=(22, 14))
vp_1 = instance.import_value_parameters()
vp_2 = copy.deepcopy(vp_1)
vp_2['cadets_overall_weight'] = 0.55
vp_2['afscs_overall_weight'] = 0.45
for j, afsc in enumerate(instance.parameters['afsc_vector']):
    swing_weights = vp_2['objective_weight'][j, :]
    swing_weights = swing_weights / max(swing_weights)
    for k in vp_2['K^A'][j]:
        objective = vp_2['objectives'][k]
        if objective in ['Merit', 'USAFA Proportion']:
            swing_weights[k] = 0.05
    vp_2['objective_weight'][j, :] = swing_weights / sum(swing_weights)

instance.vp_dict = {'VP': vp_1, 'VP_2': vp_2}
instance.create_aggregate_file()
# instance.solve_vft_pyomo_model(max_time=10)