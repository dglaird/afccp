# Get directory path
import os
dir_path = os.getcwd() + '/'

# Get main afccp folder path
index = dir_path.find('afccp')
dir_path = dir_path[:index + 6]

# Update working directory
os.chdir(dir_path)

# Import problem class
from afccp.core.problem_class import CadetCareerProblem

instance = CadetCareerProblem('2023', printing=True)
instance.import_default_value_parameters()
# instance.set_instance_value_parameters()
# instance.set_instance_solution("A-VFT")
instance.solve_for_constraints()
# instance.solve_vft_pyomo_model(max_time=10)
# instance.export_to_excel()
# from afccp.core.more_graphs import test_graph
# test_graph()

# from afccp.research.laird.research_functions_other import *
#
# functions = ["USAFA Proportion", "Merit", "Combined Quota", "AFOCD"]
# for func in ["USAFA Proportion"]:
#
#     if func == "Merit":
#         for actual in [0.4, 0.5, 0.6]:
#             value_function_examples(function=func, actual=actual)
#     elif func == "USAFA Proportion":
#         for actual in [0.1, 0.25, 0.4]:
#             value_function_examples(function=func, actual=actual)
