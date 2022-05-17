# Get directory path
import os
dir_path = os.getcwd() + '/'  # on ploty enterprise, the directory path is "/workspace/"

# Get main afccp folder path
dir_path = "/workspace/afccp/"

# Update working directory
os.chdir(dir_path)

# If this is the first time we run this script, we'll get an error saying "No module named 'afccp'" so we need to add it to the path
try:

    # Import main problem class
    from afccp.core.problem_class import CadetCareerProblem
except:

    # Add afccp to path
    import sys
    sys.path.insert(0,dir_path)

    # Import main problem class
    from afccp.core.problem_class import CadetCareerProblem


instance = CadetCareerProblem('2023', printing=True)
instance.import_default_value_parameters()
instance.generate_random_solution()
instance.export_to_excel()
# instance.set_instance_solution()
# instance.solve_vft_pyomo_model(max_time=10, provide_executable=False)
# instance.full_vft_model_solve(provide_executable=False, ga_printing=True, ga_max_time=120)
