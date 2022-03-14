# Import Research Libraries
from afccp.research.research_functions_other import *
from afccp.research.research_functions_real_comparison import *
from afccp.research.research_functions_data_generation import *
from afccp.research.research_functions_solver_performance import *
from afccp.sensitive.data_cleaning_main import *

<<<<<<< Updated upstream
instance = CadetCareerProblem("2021", printing=True)
instance.set_instance_value_parameters(vp_name='VP_3')
instance.set_instance_solution(solution_name='AG-VFT')
# instance.import_default_value_parameters(no_constraints=True)
# instance.genetic_algorithm(stopping_time=20)
# instance.full_vft_model_solve()
instance.export_to_excel(aggregate=False)
print('hello')
=======
# aggregate_historic_data()
ctgan_train(epochs=3)
>>>>>>> Stashed changes
