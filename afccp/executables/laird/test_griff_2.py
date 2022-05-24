# Get directory path
import os
dir_path = os.getcwd() + '/'

# Get main afccp folder path
index = dir_path.find('afccp')
dir_path = dir_path[:index + 6]

# Update working directory
os.chdir(dir_path)

# Import problem class
# from afccp.core.problem_class import CadetCareerProblem

# instance = CadetCareerProblem('2023', printing=True)
# instance.set_instance_value_parameters()
# # instance.adjust_qualification_matrix(use_matrix=False)
# # instance.import_default_value_parameters(num_breakpoints=24)
# # instance.stable_matching()
# instance.solve_vft_pyomo_model(max_time=10)
# # instance.export_value_parameters_as_defaults()
# # chart = instance.show_value_function()
# # chart.show()
# instance.export_to_excel()

from afccp.core.more_graphs import test_graph
test_graph()