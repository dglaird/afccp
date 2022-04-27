from afccp.core.problem_class import *
from afccp.research.laird.research_graphs import *


# Thesis: Section 4.2 (Solution Methodology: Random Data Solution Size Solver Checks)
def solution_analysis_solver_time_tests(sizes=None, model_dict=None, printing=True):
    """
    This procedure generates smaller sized problems and compares approximate/exact models with different solvers
    on time and objective value with different maximum times. Exports results to excel.
    :param sizes: list of tuples of N and M (number of cadets, number of AFSCs) to generate
    :param model_dict: dictionary of models (Approximate and Exact) and the solvers to use
    :param printing: whether the procedure should print something
    :return: None. (Exports to excel)
    """
    if printing:
        print('Conducting Model/Solver Max Time Analysis...')

    if sizes is None:
        sizes = [(100, 2), (200, 2), (400, 2), (800, 2), (1200, 2), (1600, 2), (1600, 4), (1600, 8), (1600, 12),
                 (1600, 16), (1600, 20), (1600, 24), (1600, 28), (1600, 32)]

    if model_dict is None:
        model_dict = {'Exact': ['mindtpy', 'ipopt'], 'Approximate': ['cbc', 'gurobi']}

    # Max times list
    max_times = [5, 10, 30, 60, 60 * 2, 60 * 4, 60 * 6, 60 * 8, 60 * 10, 60 * 15, 60 * 20, 60 * 30]

    # Define number of instances to run for each problem size
    instance_dict = {(100, 2): 5, (200, 2): 5, (400, 2): 5, (800, 2): 5, (1200, 2): 5,
                     (1600, 2): 5, (1600, 4): 5, (1600, 8): 5, (1600, 12): 5, (1600, 16): 5, (1600, 20): 5,
                     (1600, 24): 5, (1600, 28): 5, (1600, 32): 5}

    # Value parameter options
    options = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])

    # Loop through all problem sizes
    for size in sizes:

        # Get size parameters
        N = size[0]
        M = size[1]
        P = min(max(2, M - 1), 6)

        # Print size
        if printing:
            print('')
            print('')
            print_str = 32 * ' '
            print_str += '<' + str(N) + '_' + str(M) + '>'
            print_str += (72 - len(print_str)) * ' '
            print(print_str)
            print('')

        # Get lists of instance names and dictionary of instances themselves
        instance_names = []
        instances = {}
        for i_num in range(instance_dict[size]):
            # Choose random set of value parameters
            vp = np.random.choice(options, replace=False)

            # Create instance name
            instance_name = 'Instance_' + str(i_num + 1) + '_' + vp + '_' + str(N) + '_' + str(M)
            instance_names.append(instance_name)

            # Generate Random data
            instances[instance_name] = CadetCareerProblem(filepath='Random', N=N, M=M, P=P)

            # Get value parameters
            instances[instance_name].import_default_value_parameters('Value_Parameters_' + vp + '.xlsx',
                                                                     no_constraints=True)

        # Initialize results dictionary
        results_dict = {}
        success_times = {}
        for model in model_dict:
            results_dict[model] = {}
            success_times[model] = {}
            for solver in model_dict[model]:
                results_dict[model][solver] = {}
                success_times[model][solver] = 0
                for instance_name in instance_names:
                    results_dict[model][solver][instance_name] = {}

        # Loop through all models
        for model in model_dict:

            if model == 'Approximate':
                approximate = True
            else:
                approximate = False

            # Loop through all solvers for this model
            for solver in model_dict[model]:

                # Print to console
                if printing:
                    print('')
                    print_str = 27 * ' '
                    print_str += '<' + model + ' ' + solver.upper() + '>'
                    print_str += (72 - len(print_str)) * ' '
                    print(print_str)

                # Loop through all problem instances
                for i_num in range(instance_dict[size]):
                    instance_name = instance_names[i_num]
                    instance = instances[instance_name]

                    # Print to console
                    if printing:
                        print('')
                        print_str = 26 * '='
                        print_str += '<' + instance_name + '>'
                        print_str += (72 - len(print_str)) * '='
                        print(print_str)

                    # Necessary solver variables
                    best_z = 0
                    previous_solve_time = 0
                    integer_solution = False
                    warning = False
                    first_solve = True

                    # Initialize solver times
                    solver_max_times = []
                    for m, max_time in enumerate(max_times[:len(max_times) - 1]):
                        if max_times[m + 1] >= success_times[model][solver] or \
                                max_time >= success_times[model][solver]:
                            solver_max_times.append(max_time)

                    # Loop through all max time parameters (until stopping criteria)
                    for max_time in solver_max_times:

                        # Continue initialization of results dictionary
                        results_dict[model][solver][instance_name][max_time] = {}

                        # Print to console
                        if printing:
                            print_str = 30 * '-'
                            print_str += '<' + 'Max_Time_' + str(max_time) + '>'
                            print_str += (72 - len(print_str)) * '-'
                            print(print_str)
                            now = datetime.datetime.now()
                            if integer_solution:
                                print('Integer Solution Initialization. Solving for additional ' +
                                      str(max_time - current_max_time) + ' seconds.')
                                print('Solving ' + model.lower() + ' model using ' + solver + ' on ' +
                                      now.strftime('%A') + ' at time ' + now.strftime('%H:%M:%S') + '...')
                            else:
                                print('Solving ' + model.lower() + ' model using ' + solver + ' on ' +
                                      now.strftime('%A') + ' at time ' + now.strftime('%H:%M:%S') + '...')

                        # Don't bother if we know it won't work
                        if solver == 'gurobi' and max_time <= 60 * 4 and M >= 16:
                            success = False
                        else:
                            success = True

                        # Solve Model
                        if success:
                            try:

                                # If integer solution, we initialize with previous solution
                                if integer_solution:
                                    add_time = max_time - current_max_time
                                    solve_time = round(
                                        instance.solve_vft_pyomo_model(approximate=approximate, add_breakpoints=True,
                                                                       solve_name=solver, max_time=add_time, init_from_X=True,
                                                                       timing=True, report=True), 2)
                                    solve_time = previous_solve_time + solve_time
                                    previous_solve_time = solve_time

                                # If not integer solution, we must start from the beginning
                                else:
                                    solve_time = round(
                                        instance.solve_vft_pyomo_model(approximate=approximate, add_breakpoints=True,
                                                                       solve_name=solver, max_time=max_time,
                                                                       timing=True, report=True), 2)

                            except:

                                # If it didn't produce a solution, we keep trying until we get one
                                success = False

                        # Adjust current max time
                        current_max_time = max_time

                        # Basic dictionary additions
                        results_dict[model][solver][instance_name][max_time]['Model'] = model
                        results_dict[model][solver][instance_name][max_time]['Solver'] = solver
                        results_dict[model][solver][instance_name][max_time]['Instance'] = instance_name
                        results_dict[model][solver][instance_name][max_time]['Max Time'] = max_time
                        if success:

                            if first_solve:
                                success_times[model][solver] = max_time
                                first_solve = False

                            # If the model produced a solution, we load in the results
                            pyomo_z = round(instance.pyomo_z, 4)
                            approx_matrix_z = round(instance.measure_solution(approximate=True, matrix=True), 4)
                            approx_vector_z = round(instance.measure_solution(approximate=True, matrix=False), 4)
                            exact_matrix_z = round(instance.measure_solution(approximate=False, matrix=True), 4)
                            exact_vector_z = round(instance.measure_solution(approximate=False, matrix=False), 4)
                            solution = instance.solution
                            delta_z = pyomo_z - best_z
                            best_z = max(pyomo_z, best_z)
                            results_dict[model][solver][instance_name][max_time]['Solve Time'] = solve_time
                            results_dict[model][solver][instance_name][max_time]['Delta Z'] = delta_z
                            results_dict[model][solver][instance_name][max_time]['Pyomo Z'] = pyomo_z
                            results_dict[
                                model][solver][instance_name][max_time]['Approximate Matrix Z'] = approx_matrix_z
                            results_dict[
                                model][solver][instance_name][max_time]['Approximate Vector Z'] = approx_vector_z
                            results_dict[model][solver][instance_name][max_time]['Exact Matrix Z'] = exact_matrix_z
                            results_dict[model][solver][instance_name][max_time]['Exact Vector Z'] = exact_vector_z
                            results_dict[model][solver][instance_name][max_time]['Solution'] = solution

                            # Check integer conditions:
                            if approximate:
                                if approx_matrix_z == approx_vector_z:
                                    integer_solution = True
                            else:
                                if exact_matrix_z == exact_vector_z:
                                    integer_solution = True

                            # Print to console
                            if printing:
                                if solve_time > 120:
                                    time_str = str(round(solve_time / 60, 2)) + ' minutes'
                                else:
                                    time_str = str(solve_time) + ' seconds'

                                print_str = 'Solved in ' + time_str + '. Pyomo Z: ' + str(pyomo_z) + \
                                            ', Approx Matrix Z: ' + str(approx_matrix_z) + \
                                            ', \nApprox Vector Z: ' + str(approx_vector_z) + \
                                            ', Exact Matrix Z: ' + str(exact_matrix_z) + \
                                            ', Exact Vector Z: ' + str(exact_vector_z) + '.'
                                print(print_str)

                            # Check stopping criteria
                            if delta_z <= 0:
                                if warning:
                                    break
                                else:
                                    warning = True
                            else:
                                warning = False
                        else:

                            # If the model didn't produce a solution, we try again with the next max time
                            if printing:
                                print('Failed to generate ' + solver + ' solution.')

                            results_dict[model][solver][instance_name][max_time]['Solve Time'] = 0
                            results_dict[model][solver][instance_name][max_time]['Delta Z'] = 0
                            results_dict[model][solver][instance_name][max_time]['Pyomo Z'] = 0
                            results_dict[model][solver][instance_name][max_time]['Approximate Matrix Z'] = 0
                            results_dict[model][solver][instance_name][max_time]['Approximate Vector Z'] = 0
                            results_dict[model][solver][instance_name][max_time]['Exact Matrix Z'] = 0
                            results_dict[model][solver][instance_name][max_time]['Exact Vector Z'] = 0
                            results_dict[model][solver][instance_name][max_time]['Solution'] = np.zeros(N)

                            # If it's been 20 minutes and we haven't found a solution, we're done
                            if max_time >= 60 * 30:
                                break

                    # Print to console
                    if printing:
                        print(72 * '=')

        # Create Results Dataframe
        column_names = ['Model', 'Solver', 'Instance', 'Max Time', 'Solve Time', 'Pyomo Z', 'Delta Z',
                        'Approximate Matrix Z', 'Approximate Vector Z', 'Exact Matrix Z', 'Exact Vector Z']
        for column_name in column_names:
            column = []
            for model in results_dict:
                for solver in results_dict[model]:
                    for instance_name in results_dict[model][solver]:
                        for max_time in results_dict[model][solver][instance_name]:
                            column.append(results_dict[model][solver][instance_name][max_time][column_name])

            if column_name == 'Model':
                results_df = pd.DataFrame({column_name: column})
            else:
                results_df[column_name] = column

        # Construct Similarity Dataframes
        similarity_df_dict = {}
        for i_num, instance_name in enumerate(instance_names):

            # Get number of solutions for this instance
            num_solves = 0
            model_solver_names = []
            for model in results_dict:
                for solver in results_dict[model]:
                    for max_time in results_dict[model][solver][instance_name]:
                        num_solves += 1
                        model_solver_names.append(model + '_' + solver + '_' + str(max_time))

            # Construct Similarity Matrix
            similarity_matrix = np.zeros([num_solves, num_solves])
            row = 0
            for model_1 in results_dict:
                for solver_1 in results_dict[model_1]:
                    for max_time_1 in results_dict[model_1][solver_1][instance_name]:
                        col = 0
                        for model_2 in results_dict:
                            for solver_2 in results_dict[model_2]:
                                for max_time_2 in results_dict[model_2][solver_2][instance_name]:
                                    # Compare Solution Similarity
                                    solution_1 = results_dict[model_1][solver_1][instance_name][max_time_1]['Solution']
                                    solution_2 = results_dict[model_2][solver_2][instance_name][max_time_2]['Solution']

                                    similarity = np.sum((solution_1 == solution_2) * 1) / N
                                    similarity_matrix[row, col] = round(similarity, 2)

                                    # Next Column
                                    col += 1

                        # Next Row
                        row += 1

            # Construct Solution Similarity Matrix Dataframe
            similarity_df_dict[instance_name] = pd.DataFrame({'Solution Similarity': model_solver_names})
            for col, col_name in enumerate(model_solver_names):
                similarity_df_dict[instance_name][col_name] = similarity_matrix[:, col]

        # Export to excel
        filepath = str(N) + '_' + str(M) + '_Solver_Time_Results_2.xlsx'
        with pd.ExcelWriter(filepath) as writer:
            results_df.to_excel(writer, sheet_name="Results Table", index=False)
            for instance_name in instance_names:
                similarity_df_dict[instance_name].to_excel(writer, sheet_name=instance_name, index=False)


def compile_size_results(sizes=None, printing=True):
    """
    This procedure loads in all results tables from the specified instances/sizes, combines them, averages them
    and then exports the main results table along with the averaged information. It also produces the
    resulting graphs that visualizes the results.
    :param sizes: list
    :param printing: whether the procedure should print something
    :return: None (Exports/Saves everything
    """

    if printing:
        print('Compiling problem size results...')

    if sizes is None:
        sizes = [(100, 2), (200, 2), (400, 2), (800, 2), (1200, 2), (1600, 2), (1600, 4), (1600, 8), (1600, 12),
                 (1600, 24), (1600, 32)]

    for size in sizes:

        # Gather size information
        N = size[0]
        M = size[1]
        size_name = str(N) + '_' + str(M)

        # Import Data
        results_df = import_data(size_name + '_Solver_Time_Results.xlsx', sheet_name='Results Table')

        # Construct numpy arrays of columns
        instance_col = np.array(results_df.loc[:, 'Instance'])
        solver_col = np.array(results_df.loc[:, 'Solver'])
        model_col = np.array(results_df.loc[:, 'Model'])
        max_time_col = np.array(results_df.loc[:, 'Max Time'])
        metric_names = np.array(list(results_df.keys()))[4:]
        metric_cols = {}
        for metric_name in metric_names:
            metric_cols[metric_name] = np.array(results_df.loc[:, metric_name])

        # Instance names
        instance_names = np.unique(instance_col)

        # Solver-Model Combination names
        model_solvers = model_col + ' ' + solver_col
        models = np.unique(model_solvers)

        solver_df_dict = {}
        for model_solver in models:
            m_split = model_solver.split(' ')
            model, solver = m_split[0], m_split[1]

            instances, conv_time, conv_z, int_x, int_y = [], [], [], [], []
            for instance_num, instance in enumerate(instance_names):
                indices = (instance_col == instance) * (model_col == model) * \
                          (solver_col == solver) * (metric_cols['Solve Time'] > 0)
                indices = np.where(indices)[0]
                r_indices = np.flip(indices)
                if len(indices) > 0:
                    stop_i = indices[0]
                    for i in r_indices:
                        if metric_cols['Delta Z'][i] > 0.001:
                            stop_i = i
                            break

                    instances.append(instance_num)
                    conv_time.append(metric_cols['Solve Time'][stop_i])
                    conv_z.append(metric_cols['Exact Vector Z'][stop_i])

                    if metric_cols['Pyomo Z'][stop_i] == metric_cols[model + ' Matrix Z'][stop_i]:
                        int_y.append(1)
                    else:
                        int_y.append(0)

                    if metric_cols[model + ' Matrix Z'][stop_i] == metric_cols[model + ' Vector Z'][stop_i]:
                        int_x.append(1)
                    else:
                        int_x.append(0)

            solver_df_dict[model_solver] = pd.DataFrame({'Instances': instances, 'Conv. Time': conv_time,
                                                         'Conv. Z': conv_z, 'Int. X': int_x, 'Int. Y': int_y})

        filepath = size_name + '_Compiled_Results.xlsx'
        with pd.ExcelWriter(filepath) as writer:
            for model_solver in models:
                solver_df_dict[model_solver].to_excel(writer, sheet_name=model_solver, index=False)


def create_solver_size_table_graph(sizes=None, printing=True):
    """
    This procedure loads in all compiled results tables and exports a full table of solver/size
    results. Also creates the graph to show this information
    :param sizes: list of problem sizes
    :param printing: whether the procedure should print something
    :return: None (Exports/Saves everything
    """

    if printing:
        print('Creating problem size results graphics...')

    if sizes is None:
        sizes = [(100, 2), (200, 2), (400, 2), (800, 2), (1200, 2), (1600, 2), (1600, 4), (1600, 8), (1600, 12),
                 (1600, 24), (1600, 32)]

    column_names = ['Size', 'Instances', 'Cbc Time', 'Gurobi Time', 'Ipopt Time', 'Cbc Z', 'Gurobi Z',
                    'Ipopt Z']
    column_dict = {}
    for col in column_names:
        column_dict[col] = []

    for size in sizes:

        # Gather size information
        N = size[0]
        M = size[1]
        size_name = str(N) + '_' + str(M)

        # Import Data
        table_df = import_data('Solver_Size_Results.xlsx', sheet_name=size_name)

        # Hardcoded because it's easier in this case
        column_dict['Size'].append(size_name)
        column_dict['Instances'].append(table_df.loc[0, 'Instances'])
        column_dict['Cbc Time'].append(table_df.loc[0, 'Average Conv. Time'])
        column_dict['Gurobi Time'].append(table_df.loc[1, 'Average Conv. Time'])
        column_dict['Ipopt Time'].append(table_df.loc[2, 'Average Conv. Time'])
        column_dict['Cbc Z'].append(table_df.loc[0, 'Average Conv. Z'])
        column_dict['Gurobi Z'].append(table_df.loc[1, 'Average Conv. Z'])
        column_dict['Ipopt Z'].append(table_df.loc[2, 'Average Conv. Z'])

    # Full table
    full_table = pd.DataFrame({})
    for col in column_names:
        full_table[col] = column_dict[col]

    filepath = 'Solver_Size_Compiled_Table.xlsx'
    with pd.ExcelWriter(filepath) as writer:
        full_table.to_excel(writer, sheet_name='Results', index=False)


# Thesis: Section 4.2 (Solution Methodology: Realistic Data Solution Techniques)
def solution_analysis_simulated_data(instances=None, vp_size=4, vps_list=None, just_GA=False,
                                     models=None, printing=True):
    """
    This procedure simulates realistic cadet data, solves it using
    the Approximate VFT model and then again using the Exact VFT model and
    returns a table comparing the data. Also exports to excel
    :param models: list of models to use
    :param just_GA: if we already have approx/exact solutions and just want GA solutions
    :param vps_list: list of value parameters to use on each instance
    :param vp_size: number of value parameters to test on each data set
    :param instances: list of instances to evaluate
    :param printing: whether the procedure should print something
    :return: Results table
    """

    if printing:
        print('Conducting Simulated Data Solve Times Analysis...')

    if instances is None:
        instances = np.arange(1, 21).astype(int)

    if models is None:
        models = ['Approximate', 'Approximate_GA', 'Exact', 'Full_GA']

    # Initialize data name -> value parameter sets translation dictionary
    vps = {}
    data_names = []
    options = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    for instance in instances:
        data_name = 'CTGAN_Data_' + str(instance)
        data_names.append(data_name)
        if vps_list is None:
            vps[data_name] = np.random.choice(options, size=vp_size, replace=False)
        else:
            vps[data_name] = vps_list
            vp_size = len(vps_list)

    # Loop through all data instances
    for data_name in data_names:

        # Import Data
        if printing:
            print('')
            print('<-' + data_name + '->')
        instance = CadetCareerProblem(filepath=data_name + '.xlsx')
        cadet_column = instance.parameters['SS_encrypt']

        # Initialize solutions from excel
        if just_GA:
            init_solutions_df = import_data(data_name + '_Comparison_Results.xlsx', sheet_name="Solutions")
            init_results_df = import_data(data_name + '_Comparison_Results.xlsx', sheet_name="Results Table")

        # Initialize results dictionary
        results_dict = {}
        for vp in vps[data_name]:
            results_dict[vp] = {}
            for model in models:
                results_dict[vp][model] = {}

        # Loop through each set of value parameters
        for vp_index, vp in enumerate(vps[data_name]):

            # Load value parameters
            instance.import_default_value_parameters(filepath='Value_Parameters_' + vp + '.xlsx',
                                                     no_constraints=True)

            if printing:
                print('')
                print('*Value Parameters ' + vp + '*')
                print('')

            # Loop through each model
            for model in models:
                if printing:
                    now = datetime.datetime.now()
                    print('Acquiring Objective Value for ' + model + ' VFT Solution on ' + now.strftime('%A') +
                          ' at time ' + now.strftime('%H:%M:%S') + '...')

                success = True
                if model not in ['Approximate_GA', 'Full_GA']:
                    if just_GA:
                        afsc_solution = np.array(init_solutions_df[model + ' ' + vp])
                        instance.import_default_value_parameters('Value_Parameters_' + vp + '.xlsx',
                                                                 no_constraints=True)
                        instance.afsc_solution_to_solution(afsc_solution)
                        results_dict[vp][model]['z'] = round(instance.metrics['z'], 4)
                        results_dict[vp][model]['time'] = init_results_df.loc[vp_index, model + ' Model Time']
                        results_dict[vp][model]['solution'] = afsc_solution
                        solve_time = results_dict[vp][model]['time']
                    else:
                        if model == 'Approximate':
                            solve_time = round(instance.solve_vft_pyomo_model(approximate=True, timing=True,
                                                                              max_time=10, add_breakpoints=True), 2)
                        else:
                            try:
                                solve_time = round(instance.solve_vft_pyomo_model(approximate=False, add_breakpoints=True,
                                                                                  max_time=600, timing=True), 2)
                            except:
                                success = False

                    if success:
                        solution_1 = instance.stable_matching(set_solution=False)
                        solution_2 = instance.greedy_method(set_solution=False)

                        if model == 'Approximate':
                            approx_solution = instance.solution
                            initial_solutions = np.array([approx_solution, solution_1, solution_2])
                        else:
                            exact_solution = instance.solution
                            initial_solutions = np.array([approx_solution, exact_solution, solution_1, solution_2])
                else:
                    time_eval_df, solve_time = instance.genetic_algorithm(initialize=True, printing=False,
                                                                          initial_solutions=initial_solutions,
                                                                          num_time_points=100,
                                                                          constraints='Penalty',
                                                                          stopping_time=60 * 10, time_eval=True)
                    solve_time = round(solve_time, 2)
                    results_dict[vp][model]['time_table'] = time_eval_df

                if success:
                    if printing:
                        if solve_time > 120:
                            print(model + ' Objective Value of ' + str(round(instance.metrics['z'], 4)) +
                                  ' found in ' + str(round(solve_time / 60, 2)) + ' minutes.')
                        else:
                            print(model + ' Objective Value of ' + str(round(instance.metrics['z'], 4)) +
                                  ' found in ' + str(round(solve_time, 2)) + ' seconds.')

                    results_dict[vp][model]['z'] = round(instance.metrics['z'], 4)
                    results_dict[vp][model]['time'] = solve_time
                    results_dict[vp][model]['solution'] = instance.metrics['afsc_solution']
                else:
                    if printing:
                        print('Failed to generate ' + model + ' solution.')

                    results_dict[vp][model]['z'] = -1
                    results_dict[vp][model]['time'] = -1
                    results_dict[vp][model]['solution'] = np.zeros(instance.parameters['N']).astype(str)

        # Construct Results dataframe
        names = [data_name + ' ' + vp for vp in results_dict]
        results_df = pd.DataFrame({'Instances': names})
        for metric in ['Time', 'Z']:
            for model in models:
                column = []
                for vp in results_dict:
                    column.append(results_dict[vp][model][metric.lower()])
                results_df[model + ' Model ' + metric] = column

        # Construct Solution Similarity Matrix
        similarity_matrix = np.zeros([len(models) * vp_size, len(models) * vp_size])
        row = 0
        for vp_1 in results_dict:
            for model_1 in models:

                # Loop through all columns
                col = 0
                for vp_2 in results_dict:
                    for model_2 in models:
                        # Compare Solution Similarity
                        afsc_solution_1 = results_dict[vp_1][model_1]['solution']
                        afsc_solution_2 = results_dict[vp_2][model_2]['solution']
                        similarity = np.sum((afsc_solution_1 == afsc_solution_2) * 1) / len(afsc_solution_1)
                        similarity_matrix[row, col] = round(similarity, 2)

                        # Next Column
                        col += 1

                # Next Row
                row += 1

        # Create Solution dataframe and list of model/value parameter names
        solution_df = pd.DataFrame({'Encrypt_PII': cadet_column})
        model_vp_names = []
        for vp in results_dict:
            for model in models:
                col_name = model + ' ' + vp
                solution_df[col_name] = results_dict[vp][model]['solution']
                model_vp_names.append(col_name)

        # Similarity Matrix dataframe and solution dataframe
        similarity_df = pd.DataFrame({'Solution Similarity': model_vp_names})
        for col, col_name in enumerate(model_vp_names):
            similarity_df[col_name] = similarity_matrix[:, col]

        # Export to excel
        filepath = data_name + '_Comparison_Results.xlsx'
        with pd.ExcelWriter(filepath) as writer:
            results_df.to_excel(writer, sheet_name="Results Table", index=False)
            similarity_df.to_excel(writer, sheet_name="Solution Similarity", index=False)
            solution_df.to_excel(writer, sheet_name="Solutions", index=False)
            for vp in results_dict:
                for model in models:
                    if 'time_table' in results_dict[vp][model].keys():
                        results_dict[vp][model]['time_table'].to_excel(
                            writer, sheet_name=model + ' ' + vp + ' Time Table', index=False)


def compile_solution_technique_results(instances=None, printing=True):
    """
    This procedure loads in all results tables from the specified instances, combines them, averages them
    and then exports the main results table along with the averaged information. It also produces the
    resulting graphs that visualizes the results.
    :param instances: list of integers specifying which data to use
    :param printing: whether the procedure should print something
    :return: None (Exports/Saves everything)
    """

    if printing:
        print('Compiling Solution Technique results...')

    if instances is None:
        instances = np.arange(1, 16).astype(int)

    for index, instance_num in enumerate(instances):

        # Import Results
        data_name = 'CTGAN_Data_' + str(instance_num)
        if printing:
            print('')
            print('<-' + data_name + '->')

        results_df = import_data(paths['Analysis & Results'] + data_name + '_Comparison_Results.xlsx',
                                 sheet_name="Results Table")
        results_df['Approximate_GA Model Time'] = results_df['Approximate_GA Model Time'] + \
                                                  results_df['Approximate Model Time']
        results_df['Full_GA Model Time'] = results_df['Full_GA Model Time'] + \
                                           results_df['Exact Model Time']
        names = np.array(results_df['Instances'])
        vps = []
        for i_name in names:
            vps.append(i_name.split(' ')[1])

        time_tables_dict = {}
        for model in ['Approximate', 'Full']:
            time_tables_dict[model] = {}
            for i, vp in enumerate(vps):

                time_tables_dict[model][vp] = import_data(paths['Analysis & Results'] + data_name +
                                                            '_Comparison_Results.xlsx',
                                                          sheet_name=model + '_GA ' + vp + " Time Table")
                times = np.array(time_tables_dict[model][vp]['Time']) / 60
                if model == 'Full':
                    times = times + np.array(results_df.loc[i, 'Exact Model Time']) / 60
                else:
                    times = times + np.array(results_df.loc[i, 'Approximate Model Time']) / 60
                time_tables_dict[model][vp]['Time'] = times

                results_df.loc[i, model + '_GA Model Z'] = time_tables_dict[model][vp].loc[
                    len(times) - 1, 'Objective Value']

                if model == 'Approximate':
                    time_tables_dict[model][vp].loc[0, 'Objective Value'] = results_df.loc[i, 'Approximate Model Z']
                else:
                    time_tables_dict[model][vp].loc[0, 'Objective Value'] = results_df.loc[i, 'Exact Model Z']

        performance_chart = Solution_Analysis_Performance_Chart(results_df, time_tables_dict,
                                                                display_title=False, save=True)
        performance_chart.show()
        if index == 0:
            complete_results_df = results_df
        else:
            complete_results_df = complete_results_df.append(results_df, ignore_index=True)

    # if printing:
    #     print(complete_results_df)
    #
    # with pd.ExcelWriter(paths['Tables'] + 'Thesis_4.2.2_full_solution_technique.xlsx') as writer:
    #     complete_results_df.to_excel(writer, sheet_name="Results Table", index=False)


def random_solution_generation(num_solutions=5, instances=None, printing=True):
    """
    This procedure loads in the data for all instances and generates a set number of random solutions
    for each set of value parameters and measures them all
    :param num_solutions: number of solutions to generate for each instance
    :param instances: list of integers specifying which data to use
    :param printing: whether the procedure should print something
    :return: None (Exports/Saves everything)
    """

    if printing:
        print('Generating random solutions and results...')

    if instances is None:
        instances = np.arange(1, 16).astype(int)

    # Import results
    results_df = import_data("Thesis_Appendix_full_solution_technique.xlsx", sheet_name='Results Table')

    # Import value parameters
    vps = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    default_vp_dict = {}
    for vp in vps:
        default_vp_dict[vp] = default_value_parameters_from_excel(
            filepath='Value_Parameters_' + vp + '.xlsx')

    # Get variables
    full_instances = np.array(results_df.loc[:, 'Instances'])
    solution_z_dict = {}
    for sol in np.arange(num_solutions):
        solution_z_dict[sol] = np.zeros(60)

    # Loop through all instances
    i = 0
    for index, instance_num in enumerate(instances):

        # Import Results
        data_name = 'CTGAN_Data_' + str(instance_num)
        if printing:
            print('')
            print('<-' + data_name + '->')
        instance = CadetCareerProblem(data_name + '.xlsx')

        for _ in range(4):
            vp = full_instances[i].split(' ')[1]
            value_parameters = generate_value_parameters_from_defaults(
                instance.parameters, default_vp_dict[vp])
            value_parameters['constraint_type'] = np.zeros(
                [instance.parameters['M'], value_parameters['O']])
            value_parameters = model_value_parameters_set_additions(value_parameters)
            value_parameters = condense_value_functions(instance.parameters, value_parameters)
            instance.value_parameters = model_value_parameters_set_additions(value_parameters)
            for sol in np.arange(num_solutions):
                instance.generate_random_solution()
                solution_z_dict[sol][i] = round(instance.metrics['z'], 4)

            i += 1

    for sol in np.arange(num_solutions):
        results_df['Random Solution ' + str(sol + 1)] = solution_z_dict[sol]

    with pd.ExcelWriter('Thesis_4.2.2_full_solution_technique1.xlsx') as writer:
        results_df.to_excel(writer, sheet_name="Results Table", index=False)