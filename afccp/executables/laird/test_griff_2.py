# Get directory path
import os
dir_path = os.getcwd() + '/'

# Get main afccp folder path
index = dir_path.find('afccp')
dir_path = dir_path[:index + 6]

# Update working directory
os.chdir(dir_path)
#
# Import problem class
from afccp.core.problem_class import CadetCareerProblem

# Genetic algorithm stuff
# instance = CadetCareerProblem('2023', printing=True)
# instance.set_instance_value_parameters("VP")
# initial_solutions = list(instance.solution_dict.keys())[7:]
# print(initial_solutions)
# instance.genetic_algorithm({"solution_names": initial_solutions, "ga_max_time": 60 * 40})
# instance.export_to_excel()

# Chart stuff
instance = CadetCareerProblem('2023', printing=True)
instance.set_instance_value_parameters("VP")
instance.set_instance_solution("Solution H")
solution_names = ["Solution A", "Solution B", "Solution C", "Solution D",
                  "Solution E", "Solution F", "Solution G", "Solution H"]
instance.display_results_graph({"graph": "Multi-Criteria Comparison", "compare_solutions": True,
                                "solution_names": solution_names})
# instance.display_all_results_graphs()
# instance.display_results_graph({"objective": "Extra", "version": "AFOCD_proportion"})

# print(instance.value_parameters["cadet_value_min"][0])
# instance.export_value_parameters_as_defaults()
# instance.set_instance_solution("Solution G")
# comparison_afscs = ["14N", "21A", "38F", "63A", "21R", "64P", "21M", "13M"]
# comparison_afscs = instance.parameters["afsc_vector"]
# instance.display_results_graph({"graph": "Multi-Criteria Comparison", "compare_solutions": True})
# instance.display_results_graph({"graph": "Multi-Criteria Comparison", "comparison_afscs": comparison_afscs})
# instance.mdl_p["solution_names"] = list(instance.solution_dict.keys())[2:]

# print(instance.mdl_p["solution_names"])

# Get some more starting weights
# for cw in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
#     instance.value_parameters["cadets_overall_weight"] = cw
#     instance.value_parameters["afscs_overall_weight"] = 1 - cw
#     instance.solve_vft_pyomo_model()

# Change it back
# instance.value_parameters["cadets_overall_weight"] = 0.65
# instance.value_parameters["afscs_overall_weight"] = 1 - 0.65
# instance.export_to_excel()
# instance.set_instance_solution("Solution A")
# instance.display_weight_function(cadets=True, save=True)
# instance.initial_overall_weights_pareto_analysis()

# instance = CadetCareerProblem('2023', printing=True)
# instance.set_instance_value_parameters()
# instance.genetic_overall_weights_pareto_analysis({"ga_max_time": 60 * 5})
# instance.show_pareto_chart()
# instance.export_value_parameters_as_defaults()
# for afsc in ["32EXC", "32EXF", "62EXB"]:
#     instance.show_value_function(afsc, "USAFA Quota", save=True)
# instance.vft_to_gp_parameters(get_new_rewards_penalties=True)
# instance.solve_gp_pyomo_model(max_time=60 * 10)
# instance.solve_vft_pyomo_model(max_time=10)
# solution_names = ["A-VFT_" + str(n + 2) for n in range(5)]
# instance.genetic_algorithm({"ga_max_time": 60 * 5, "ga_printing": True})
# instance.export_to_excel()
# instance.set_instance_solution("Solution E")
# instance.display_all_results_graphs()
# instance.display_results_graph({"save": True, "graph": "Utility vs. Merit"})
# instance.export_to_excel(aggregate=False)
