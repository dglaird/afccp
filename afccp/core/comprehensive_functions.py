# Import libraries
from afccp.core.data_handling import *
from afccp.core.value_parameter_handling import *
from afccp.core.simulation_functions import *
from afccp.core.value_parameter_generator import *
from afccp.core.instance_graphs import *
from afccp.core.heuristic_solvers import *
import copy

if use_pyomo:
    from afccp.core.pyomo_models import *


# Sensitivity Analysis Functions
def least_squares_procedure(parameters, value_parameters, solution_1, solution_2, delta=0, printing=False):
    """
    Takes the parameters, value parameters, and solutions and then conducts the Least Squares Procedure as presented
    in the paper
    :param delta: how much the value of solution 2 should exceed solution 1
    :param parameters: fixed cadet AFSC parameters
    :param value_parameters: value parameters
    :param solution_1: optimal solution under these value parameters (vector)
    :param solution_2: some other solution (vector)
    :param printing: Whether the procedure should print something
    :return: weight parameters of solution 2 that make solution 2's value exceed solution 1 by amount delta
    """

    if use_pyomo:
        if printing:
            print("Conducting Least Squares Procedure...")

        M = parameters['M']
        O = value_parameters['O']
        metrics_1 = measure_solution_quality(solution_1, parameters, value_parameters, printing)
        metrics_2 = measure_solution_quality(solution_2, parameters, value_parameters, printing)

        model = lsp_model_build(printing)
        data = convert_parameters_to_lsp_model_inputs(parameters, value_parameters, metrics_1, metrics_2, delta,
                                                      printing)
        instance = model.create_instance(data)

        if printing:
            print('Solving LSP model...')

        solver = SolverFactory('ipopt', executable='../Main Directory/solvers/ipopt.exe')
        solver.solve(instance)

        afsc_objective_weight_2 = np.zeros([M, O])

        for j in instance.J.data():
            for k in instance.K_A[j].data():
                afsc_objective_weight_2[j, k] = instance.afsc_objective_weight[j, k].value

        value_parameters_2 = copy.deepcopy(value_parameters)
        value_parameters_2['objective_weight'] = afsc_objective_weight_2
    else:
        if printing:
            print("Pyomo not available")

        value_parameters_2 = value_parameters

    return value_parameters_2


def plot_solution_similarity(similarity_matrix, solution_names=None, instance_name=None, display_title=True,
                             save=False, printing=True):
    """
    This procedure plots the solution similarity matrix in 2 dimensional space
    :param similarity_matrix: similarity matrix
    :param solution_names: names of the solutions
    :param instance_name: name of the overall data instance
    :param display_title: if we should display the title for the chart
    :param save: if we should save the figure
    :param printing: if we should print something to the console
    :return: chart
    """

    if printing:
        print('Plotting solution similarity...')

    # Collect chart information
    if solution_names is None:
        solution_names = ['Data_' + str(i + 1) for i in range(len(similarity_matrix))]
    if instance_name is None:
        instance_name = 'Example Data'
    title = instance_name + ' Solution Similarity'

    # Get coordinates
    coords = solution_similarity_coordinates(similarity_matrix)

    # Plot similarity
    chart = solution_similarity_graph(coords, solution_names, title, display_title=display_title,
                                      use_models=True, save=save)

    if printing:
        chart.show()


# Value Function Visualizations
def exponential_value_function_example(rho=-0.3, target=0.5):
    """
    Very simple procedure that generates x and y points for some exponential value function with domain and range
    between 0 and 1
    :param rho: rho to use
    :param target: target to use
    :return: x, y
    """
    x_arr = (np.arange(11) / 10)
    y = []
    for x in x_arr:
        if x <= target:
            y_i = (1 - exp(-(x - 0) / rho)) / (1 - exp(-(target - 0) / rho))
        else:
            y_i = (1 - exp(-(1 - x) / rho)) / (1 - exp(-(1 - target) / rho))
        y.append(y_i)
    return x_arr, y


def piecewise_sanity_check(a=None, f_a=None, x=None, graph=True, printing=True):
    """
    Takes the value function parameters, and an optional x value and computes the f_x value using the linear
    piecewise methodology. If x is None, we find the optimal value. If x is specified, we find the f_x for that
    x.
    :param printing: Whether the procedure should print something
    :param graph: If we want to graph the Value Function and x value
    :param a: breakpoints
    :param f_a: value of the breakpoints
    :param x: optional x value
    :return: f_x (either maximum or at x)
    """
    if use_pyomo:
        if a is None:
            a = np.array([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1])

        if f_a is None:
            f_a = np.array([0, 0.2, 0.8, 1, 0.8, 0.2, 0])

        # Sets and Parameters
        r = len(a)
        L = np.arange(r)

        model = ConcreteModel()

        # Variables
        model.f_x = Var(within=PositiveReals)
        if x is None:
            model.x = Var(within=PositiveReals)
        model.y = Var((l for l in np.arange(0, r - 1)), within=Binary)
        model.lam = Var((l for l in L), within=PositiveReals)

        # Measure .. Value Constraints
        if x is None:
            model.x_constraint = Constraint(expr=model.x == np.sum(a[l] * model.lam[l] for l in L))
        else:
            model.x_constraint = Constraint(expr=x == np.sum(a[l] * model.lam[l] for l in L))
        model.f_x_constraint = Constraint(expr=model.f_x == np.sum(f_a[l] * model.lam[l] for l in L))

        # Lambda .. y constraints
        model.y1constraint = Constraint(expr=model.lam[0] <= model.y[0])
        model.y2constraint = ConstraintList()
        if r > 2:
            for l in range(1, r - 1):
                model.y2constraint.add(expr=model.lam[l] <= model.y[l - 1] + model.y[l])
        model.y3constraint = Constraint(expr=model.lam[r - 1] <= model.y[r - 2])

        # Y sum to 1 constraint
        model.y_sum_constraint = Constraint(expr=np.sum(model.y[l] for l in range(0, r - 1)) == 1)

        # Lambda sum to 1 constraint
        model.lambda_sum_constraint = Constraint(expr=sum(model.lam[l] for l in L) == 1)

        # Objective
        model.objective = Objective(expr=model.f_x, sense=maximize)
        if printing:
            print('solving')
        solver = SolverFactory("cbc", executable='../Main Directory/solvers/cbc.exe')
        solver.solve(model)
        obj = model.objective()
        if printing:
            if x is None:
                print('x: ' + str(model.x.value) + ', f(x): ' + str(obj))
            else:
                print('x: ' + str(x) + ', f(x): ' + str(obj))
            print("")

            for l in range(0, r - 1):
                print('l: ' + str(l) + ', lambda: ' + str(round(model.lam[l].value, 2)) + ', y: ' + str(
                    round(model.y[l].value, 2)))

            print('l: ' + str(r - 1) + ', lambda: ' + str(round(model.lam[r - 1].value, 2)))

        if x is None:
            x_point = model.x.value
        else:
            x_point = x
        f_x_point = obj

        if graph:
            x, y = value_function_points(a, f_a)

            chart = value_function_graph(x, y, x_point, f_x_point, breakpoints=[a, f_a])
            chart.show()


def value_function_points(a, fhat):
    """
    Takes the linear function parameters and returns the approximately non-linear coordinates
    :param a: function breakpoints
    :param fhat: function breakpoint values
    :return: x, y
    """
    x = (np.arange(1001) / 1000) * a[len(a) - 1]
    y = []
    r = len(a)
    for x_i in x:
        val = value_function(a, fhat, r, x_i)
        y.append(val)
    return x, y


def plot_value_function(afsc, objective, parameters, value_parameters, title=None, printing=False,
                        label_size=25, yaxis_tick_size=25, xaxis_tick_size=25, figsize=(12, 10),
                        facecolor='white', display_title=True, save=False):
    """
    This procedure takes a set of value parameters, as well as a specific afsc's objective and then plots
    the value function for that objective.
    :param save: if we should save the figure or not
    :param display_title: if we should display a title or not
    :param facecolor: background color of the figure
    :param figsize: size of the figure
    :param xaxis_tick_size: font size of the x axis tick marks
    :param yaxis_tick_size: font size of the y axis tick marks
    :param label_size: font size of the labels
    :param parameters: fixed cadet/afsc parameters
    :param afsc: selected afsc
    :param objective: selected objective
    :param value_parameters: value parameters
    :param title: title of the function
    :param printing: whether we should print something
    :return: chart
    """

    # Get correct indices for afsc and objective
    if type(afsc) == str or type(afsc) == np.str_:
        j = np.where(parameters['afsc_vector'] == afsc)[0][0]
    else:
        j = afsc
        afsc = parameters['afsc_vector'][j]
    if type(objective) == str or type(objective) == np.str_:
        k = np.where(value_parameters['objectives'] == objective)[0][0]
    else:
        k = objective
        objective = value_parameters['objectives'][k]

    if objective == 'Merit':
        x_label = 'Average Merit'
    elif objective == 'Combined Quota':
        x_label = 'Number of Cadets'
    elif objective == 'USAFA Quota':
        x_label = 'Number of USAFA Cadets'
    elif objective == 'ROTC Quota':
        x_label = 'Number of ROTC Cadets'
    elif objective == 'Utility':
        x_label = 'Average Utility'
    elif objective in ['Mandatory', 'Desired', 'Permitted', 'Male', 'Minority']:
        x_label = objective + ' Proportion'
    else:
        x_label = objective

    if title is None:
        title = afsc + ' ' + objective + ' Value Function'

    if printing:
        print('Creating value function chart for objective ' + objective + ' for AFSC ' + afsc + '...')

    a = value_parameters['a'][j][k]
    fhat = value_parameters['f^hat'][j][k]
    x, y = value_function_points(a, fhat)
    chart = value_function_graph(x, y, title=title, label_size=label_size, yaxis_tick_size=yaxis_tick_size,
                                 xaxis_tick_size=xaxis_tick_size, figsize=figsize, facecolor=facecolor,
                                 display_title=display_title, save=save, x_label=x_label)
    return chart


# Goal Programming Model Functions
def calculate_rewards_penalties(gp, solver_name='cbc', executable=None, provide_executable=False, printing=True):
    """
    This function takes a set of Rebecca's goal programming parameters and then returns the normalized
    penalties and rewards specific to this instance that are used in Rebecca's goal programming (GP) model
    :param solver_name: name of solver
    :param executable: path of the solver
    :param provide_executable: if we want to provide an executable directly
    :param printing: if we want to print status updates or not
    :param gp: Rebecca's goal programming parameters
    :return: gp norm penalties, gp norm rewards
    """
    num_constraints = len(gp['con']) + 1
    rewards = np.zeros(num_constraints)
    penalties = np.zeros(num_constraints)
    model = gp_model_build(gp, con_term=gp['con'][0], get_reward=True)
    for c, con in enumerate(gp['con']):

        # Get reward term
        def objective_function(m):
            return np.sum(m.Z[con, a] for a in gp['A^'][con])

        if printing:
            print('')
            print('Obtaining reward for constraint ' + con + '...')
        model.objective = Objective(rule=objective_function, sense=maximize)
        rewards[c] = gp_model_solve(model, gp, max_time=60 * 4, con_term=con, solver_name=solver_name,
                                    executable=executable, provide_executable=provide_executable)
        if printing:
            print('Reward:', rewards[c])

        # Get penalty term
        def objective_function(m):
            return np.sum(m.Y[con, a] for a in gp['A^'][con])

        if printing:
            print('')
            print('Obtaining penalty for constraint ' + con + '...')
        model.objective = Objective(rule=objective_function, sense=maximize)
        penalties[c] = gp_model_solve(model, gp, max_time=60 * 4, con_term=con, solver_name=solver_name,
                                      executable=executable, provide_executable=provide_executable)
        if printing:
            print('Penalty:', penalties[c])

    # S reward term
    def objective_function(m):
        return np.sum(np.sum(gp['utility'][c, a] * m.x[c, a] for a in gp['A^']['W^E'][c]) for c in gp['C'])

    if printing:
        print('')
        print('Obtaining reward for constraint S...')
    model.objective = Objective(rule=objective_function, sense=maximize)
    rewards[num_constraints - 1] = gp_model_solve(model, gp, max_time=60 * 4, con_term='S', solver_name=solver_name,
                                                  executable=executable, provide_executable=provide_executable)
    if printing:
        print('Reward:', rewards[num_constraints - 1])

    return rewards, penalties


# Export instance data functions
def data_to_excel(filepath, parameters, value_parameters=None, metrics=None, printing=False):
    """
    This procedures takes in an output filepath, as well as an array of parameters, then exports the parameters as a
    dataframe to that excel filepath
    :param printing: whether or not the procedure should print out something
    :param metrics: optional solution metrics
    :param value_parameters: optional user defined parameters, for if we want to print out those too
    :param filepath: The filepath we wish to write the dataframe to
    :param parameters: The array of fixed cadet/AFSC parameters we wish to write to excel (see above for structure of
    parameters)
    :return: None.
    """

    if printing:
        print("Exporting to excel...")

    # Construct fixed parameter dataframes
    cadets_fixed, afscs_fixed = model_data_frame_from_fixed_parameters(parameters)

    # Build value parameters dataframes if need be
    if value_parameters is not None:
        overall_weights_df, cadet_weights_df, afsc_weights_df = model_value_parameter_data_frame_from_parameters(
            parameters, value_parameters)

    # Build the solution metrics dataframes if need be
    if metrics is not None:

        cadet_solution_df = pd.DataFrame({'Cadet': parameters['SS_encrypt'], 'Matched': metrics['afsc_solution'],
                                          'Value': metrics['cadet_value'],
                                          'Weight': value_parameters['cadet_weight'],
                                          'Value Fail': metrics['cadet_constraint_fail']})

        objective_measures = pd.DataFrame({'AFSC': parameters['afsc_vector']})
        objective_values = pd.DataFrame({'AFSC': parameters['afsc_vector']})
        afsc_constraints_df = pd.DataFrame({'AFSC': parameters['afsc_vector']})
        for k in range(value_parameters['O']):
            objective_measures[value_parameters['objectives'][k]] = metrics['objective_measure'][:, k]
            objective_values[value_parameters['objectives'][k]] = metrics['objective_value'][:, k]
            afsc_constraints_df[value_parameters['objectives'][k]] = metrics['objective_constraint_fail'][:, k]

        objective_values['AFSC Value'] = metrics['afsc_value']
        afsc_constraints_df['AFSC Value Fail'] = metrics['afsc_constraint_fail']

        metric_names = ['Z', 'Cadet Value', 'AFSC Value', 'Num Ineligible', 'Failed Constraints']
        metric_results = [metrics['z'], metrics['cadets_overall_value'], metrics['afscs_overall_value'],
                          metrics['num_ineligible'], metrics['total_failed_constraints']]
        for k, objective in enumerate(value_parameters['objectives']):
            metric_names.append(objective + ' Score')
            metric_results.append(metrics['objective_score'][k])

        overall_solution = pd.DataFrame({'Solution Metric': metric_names, 'Result': metric_results})

    with pd.ExcelWriter(filepath) as writer:  # Export to excel
        cadets_fixed.to_excel(writer, sheet_name="Cadets Fixed", index=False)
        afscs_fixed.to_excel(writer, sheet_name="AFSCs Fixed", index=False)
        if value_parameters is not None:
            overall_weights_df.to_excel(writer, sheet_name="Overall Weights", index=False)
            cadet_weights_df.to_excel(writer, sheet_name="Cadet Weights", index=False)
            afsc_weights_df.to_excel(writer, sheet_name="AFSC Weights", index=False)
        if metrics is not None:
            cadet_solution_df.to_excel(writer, sheet_name="Cadet Solution Quality", index=False)
            objective_measures.to_excel(writer, sheet_name="AFSC Objective Measures", index=False)
            objective_values.to_excel(writer, sheet_name="AFSC Solution Quality", index=False)
            afsc_constraints_df.to_excel(writer, sheet_name="AFSC Constraint Fails", index=False)
            overall_solution.to_excel(writer, sheet_name="Overall Solution Quality", index=False)


def create_aggregate_instance_file(full_name, parameters, solution_dict=None, vp_dict=None, metrics_dict=None,
                                   gp_df=None, sensitive=False, printing=False):
    """
    This file takes all of the relevant data for a particular fixed instance and exports it to excel
    :param gp_df: dataframe of goal programming parameters
    :param full_name: name of the instance
    :param parameters: fixed cadet/afsc parameters
    :param solution_dict: dictionary of solutions
    :param vp_dict: dictionary of value parameters
    :param metrics_dict: dictionary of solution metrics
    :param sensitive: if we have sensitive data or not
    :param printing: whether to print status updates or not
    """
    if printing:
        print('Creating aggregate problem instance excel file...')

    # Get fixed dataframes
    cadets_fixed, afscs_fixed = model_data_frame_from_fixed_parameters(parameters)

    # Get other information
    afscs = parameters['afsc_vector']

    if vp_dict is not None:
        vp_names = list(vp_dict.keys())
        num_vps = len(vp_names)

        # Construct value parameter dataframes
        vp_afscs_df_dict = {}
        vp_weights = []
        for v, vp_name in enumerate(vp_names):
            overall_weights_df, cadet_weights_df, afsc_weights_df = model_value_parameter_data_frame_from_parameters(
                parameters, vp_dict[vp_name])
            if v == 0:
                vp_overall_df = overall_weights_df
            else:
                vp_overall_df = pd.concat([vp_overall_df, overall_weights_df], ignore_index=True)
            vp_afscs_df_dict[vp_name] = afsc_weights_df
            vp_weights.append(vp_dict[vp_name]['vp_weight'])

        # Add columns
        vp_overall_df.insert(loc=0, column='VP Name', value=vp_names)
        vp_overall_df['VP Weight'] = vp_weights

    if solution_dict is not None:
        solution_names = list(solution_dict.keys())
        num_solutions = len(solution_names)

        # Create solutions dataframe
        solutions_df = pd.DataFrame({})
        for solution_name in solution_names:
            # Translate AFSC indices into the AFSCs themselves
            solution = solution_dict[solution_name]
            solutions_df[solution_name] = [afscs[int(solution[i])] for i in parameters['I']]

    if metrics_dict is not None:
        metric_names = {'Z': 'z', 'Cadet Value': 'cadets_overall_value', 'AFSC Value': 'afscs_overall_value'}
        for k, objective in enumerate(vp_dict[vp_names[0]]['objectives']):
            metric_names[objective + ' Score'] = k
        num_metrics = len(metric_names)

        # Number of rows of metrics dataframe
        num_rows = num_solutions * num_metrics

        # Initialize columns
        column_dict = {'Solution': np.array([" " * 25 for _ in range(num_rows)]),
                       'Metric': np.array([" " * 25 for _ in range(num_rows)])}
        for vp_name in vp_names:
            column_dict[vp_name] = np.zeros(num_rows)
        for column_name in ['Avg.', 'WgtAvg.']:
            column_dict[column_name] = np.zeros(num_rows)

        # Input column data
        row = 0
        for metric_name in metric_names:
            for solution_name in solution_names:
                column_dict['Solution'][row] = solution_name
                column_dict['Metric'][row] = metric_name
                avg = 0
                w_avg = 0
                for vp_name in vp_names:
                    if metric_name in ['Z', 'Cadet Value', 'AFSC Value']:
                        m = round(metrics_dict[vp_name][solution_name][metric_names[metric_name]], 4)
                    else:
                        m = round(
                            metrics_dict[vp_name][solution_name]['objective_score'][metric_names[metric_name]], 4)
                    column_dict[vp_name][row] = m
                    avg += m / num_vps
                    w_avg += m * vp_dict[vp_name]['vp_local_weight']
                column_dict['Avg.'][row] = round(avg, 4)
                column_dict['WgtAvg.'][row] = round(w_avg, 4)
                row += 1

        # Construct metrics_df
        metrics_df = pd.DataFrame({})
        for column_name in column_dict:
            metrics_df[column_name] = column_dict[column_name]

    # Export to excel
    if sensitive:
        filepath = paths['s_instances'] + full_name + '.xlsx'
    else:
        filepath = paths['instances'] + full_name + '.xlsx'

    with pd.ExcelWriter(filepath) as writer:  # Export to excel
        cadets_fixed.to_excel(writer, sheet_name="Cadets Fixed", index=False)
        afscs_fixed.to_excel(writer, sheet_name="AFSCs Fixed", index=False)
        if gp_df is not None:
            gp_df.to_excel(writer, sheet_name="GP Parameters", index=False)
        if solutions_df is not None:
            solutions_df.to_excel(writer, sheet_name="Solutions", index=False)
        if metrics_df is not None:
            metrics_df.to_excel(writer, sheet_name="Results", index=False)
        if vp_overall_df is not None:
            vp_overall_df.to_excel(writer, sheet_name="VP Overall", index=False)
            for vp_name in vp_names:
                vp_afscs_df_dict[vp_name].to_excel(writer, sheet_name=vp_name, index=False)


def import_aggregate_instance_file(filepath, num_breakpoints=None, use_actual=True, printing=False):
    """
    This procedure imports all available information on a particular problem instance
    :param use_actual: if we want to use the sets of eligible cadets for each of the AFSCs
    in determining value functions
    :param num_breakpoints: number of breakpoints to use with value functions
    :param filepath: filepath of the problem instance excel aggregate file
    :param printing: if we should print status updates or not
    :return: parameters, solution_dict, vp_dict, metrics_dict
    """
    if printing:
        print('Importing problem instance data...')

    # Import fixed parameters
    cadets_fixed, afscs_fixed = import_fixed_cadet_afsc_data_from_excel(filepath)
    parameters = model_fixed_parameters_from_data_frame(cadets_fixed, afscs_fixed)
    parameters = model_fixed_parameters_set_additions(parameters)

    # Try to import GP parameter dataframe (may not exist)
    try:

        # Goal Programming parameter dataframe
        gp_df = import_data(filepath, sheet_name="GP Parameters")

    except:
        gp_df = None

    # Try to import solution information (may not exist)
    try:

        # Solution dictionary
        solutions_df = import_data(filepath, sheet_name="Solutions")
        solution_names = solutions_df.columns
        solution_dict = {}
        for solution_name in solution_names:
            afsc_solution = np.array(solutions_df[solution_name])
            solution = np.array([np.where(parameters['afsc_vector'] == afsc)[0][0] for afsc in afsc_solution])
            solution_dict[solution_name] = solution

    except:
        solution_dict = None

    # Try to import value parameter information (may not exist)
    try:

        # Value Parameter Dictionary
        overall_weights = import_data(filepath, sheet_name="VP Overall")
        vp_names = np.array(overall_weights['VP Name'])
        vp_weights = np.array(overall_weights['VP Weight'])
        vp_dict = {}
        for v, vp_name in enumerate(vp_names):

            # Load AFSC weight information
            afsc_weights = import_data(filepath, sheet_name=vp_name)

            # Load into the value_parameters dictionary
            M = parameters['M']
            O = int(len(afsc_weights) / M)

            value_parameters = {'O': O, "afscs_overall_weight": np.array(overall_weights['AFSCs Weight'])[v],
                                "cadets_overall_weight": np.array(overall_weights['Cadets Weight'])[v],
                                "cadet_weight_function": np.array(overall_weights['Cadet Weight Function'])[v],
                                "afsc_weight_function": np.array(overall_weights['AFSC Weight Function'])[v],
                                "cadets_overall_value_min": np.array(overall_weights['Cadets Min Value'])[v],
                                "afscs_overall_value_min": np.array(overall_weights['AFSCs Min Value'])[v], "M": M,
                                "afsc_value_min": np.zeros(M), 'cadet_value_min': np.zeros(parameters['N']),
                                "objective_value_min": np.array([[" " * 20 for _ in range(O)] for _ in range(M)]),
                                "value_functions": np.array([[" " * 200 for _ in range(O)] for _ in range(M)]),
                                "constraint_type": np.zeros([M, O]), 'a': [[[] for _ in range(O)] for _ in range(M)],
                                "objective_target": np.zeros([M, O]), 'f^hat': [[[] for _ in range(O)] for _ in range(M)],
                                "objective_weight": np.zeros([M, O]), "afsc_weight": np.zeros(M),
                                'objectives': np.array(afsc_weights.loc[:int(len(afsc_weights) / M - 1), 'Objective'])}

            # Set Cadet Weights
            if value_parameters['cadet_weight_function'] == 'Linear':
                value_parameters['cadet_weight'] = parameters['merit'] / np.sum(parameters['merit'])
            else:
                value_parameters['cadet_weight'] = np.repeat(1 / parameters['N'], parameters['N'])

            # Load in value parameter data for each AFSC
            for j in range(M):  # These are Os (Ohs) not 0s (zeros)
                value_parameters["objective_target"][j, :] = np.array(afsc_weights.loc[j * O:(j * O + O - 1),
                                                                      'Objective Target'])
                value_parameters["objective_weight"][j, :] = np.array(afsc_weights.loc[j * O:(j * O + O - 1),
                                                                      'Objective Weight'])
                value_parameters["objective_value_min"][j, :] = np.array(afsc_weights.loc[j * O:(j * O + O - 1),
                                                                         'Min Objective Value'])
                value_parameters["constraint_type"][j, :] = np.array(afsc_weights.loc[j * O:(j * O + O - 1),
                                                                     'Constraint Type'])
                value_parameters["value_functions"][j, :] = np.array(afsc_weights.loc[j * O:(j * O + O - 1),
                                                                     'Value Functions'])
                value_parameters["afsc_weight"][j] = afsc_weights.loc[j * O, "AFSC Weight"]
                value_parameters["afsc_value_min"][j] = afsc_weights.loc[j * O, "Min Value"]
                cadets = parameters['I^E'][j]

                # Loop through each objective for this AFSC
                for k, objective in enumerate(value_parameters['objectives']):

                    # We import the functions directly from the breakpoints
                    if num_breakpoints is None:
                        string = afsc_weights.loc[j * O + k, 'Function Breakpoints']
                        if type(string) == str:
                            value_parameters['a'][j][k] = [float(x) for x in string.split(",")]
                        string = afsc_weights.loc[j * O + k, 'Function Breakpoint Values']
                        if type(string) == str:
                            value_parameters['f^hat'][j][k] = [float(x) for x in string.split(",")]

                    # We recreate the functions from the vf strings
                    else:
                        vf_string = value_parameters["value_functions"][j, k]
                        if vf_string != 'None':
                            target = value_parameters['objective_target'][j, k]
                            actual = None
                            maximum = None

                            if use_actual:
                                if objective == 'Merit':
                                    actual = np.mean(parameters['merit'][cadets])
                                elif objective == 'USAFA Proportion':
                                    actual = np.mean(parameters['usafa'][cadets])

                            if objective == 'Combined Quota':
                                # Get bounds
                                split_str = value_parameters["objective_value_min"][j, k].split(',')

                                # Get constraint upper bound
                                maximum = float(split_str[1])

                            segment_dict = create_segment_dict_from_string(vf_string, target, actual=actual,
                                                                           maximum=maximum)
                            value_parameters['a'][j][k], value_parameters['f^hat'][j][k] = value_function_builder(
                                segment_dict, num_breakpoints=num_breakpoints)

            # Load value_parameter dictionary
            value_parameters = model_value_parameters_set_additions(value_parameters)
            value_parameters = condense_value_functions(parameters, value_parameters)
            value_parameters = model_value_parameters_set_additions(value_parameters)
            vp_dict[vp_name] = copy.deepcopy(value_parameters)
            vp_dict[vp_name]['vp_weight'] = vp_weights[v]
            vp_dict[vp_name]['vp_local_weight'] = vp_weights[v] / sum(vp_weights)

    except:
        vp_dict = None

    # Create metrics dictionary if we have solutions and value parameters
    if solution_dict is not None and vp_dict is not None:

        # Metrics Dictionary
        metrics_dict = {}
        for vp_name in vp_names:
            metrics_dict[vp_name] = {}
            for solution_name in solution_names:
                solution = solution_dict[solution_name]
                value_parameters = copy.deepcopy(vp_dict[vp_name])
                metrics_dict[vp_name][solution_name] = measure_solution_quality(solution, parameters, value_parameters)

    else:
        metrics_dict = None

    # return instance data
    return parameters, vp_dict, solution_dict, metrics_dict, gp_df
