# Get directory path
import os
dir_path = os.getcwd() + '/'

# Get main afccp folder path
index = dir_path.find('afccp')
dir_path = dir_path[:index + 6]

# Update working directory
os.chdir(dir_path)

# Generate qual matrix
# from afccp.core.preprocessing import generate_cip_to_qual_matrix
# generate_cip_to_qual_matrix()


# # Import problem class
from afccp.core.problem_class import CadetCareerProblem

instance = CadetCareerProblem('2023', printing=True)
# instance.adjust_qualification_matrix(use_matrix=False)
# instance.import_default_value_parameters(no_constraints=True)
instance.set_instance_solution("A-VFT")
instance.set_instance_value_parameters()
# instance.solve_vft_pyomo_model(max_time=10)
# instance.stable_matching()
# print(instance.value_parameters["afscs_overall_weight"])
# instance.set_instance_value_parameters()
# instance.solve_original_pyomo_model(max_time=60)
# instance.full_vft_model_solve(ga_max_time=60, ga_printing=True)
# instance.vft_to_gp_parameters(get_new_rewards_penalties=True, provide_executable=True)
instance.export_to_excel(aggregate=False)

