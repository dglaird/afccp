# Import libraries
from data_handling import *
from value_parameter_handling import *
from simulation_functions import *
from instance_graphs import *
from heuristic_solvers import *
import copy
if use_pyomo:
    from pyomo_models import *


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
        data = convert_parameters_to_lsp_model_inputs(parameters, value_parameters, metrics_1, metrics_2, delta, printing)
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


def value_function_points(F_bp, F_v):
    """
    Takes the linear function parameters and returns the approximately non-linear coordinates
    :param F_bp: function breakpoints
    :param F_v: function breakpoint values
    :return: x, y
    """
    x = (np.arange(1001) / 1000) * F_bp[len(F_bp) - 1]
    y = []
    r = len(F_bp)
    for x_i in x:
        val = value_function(F_bp, F_v, r, x_i)
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

    F_bp = value_parameters['F_bp'][j][k]
    F_v = value_parameters['F_v'][j][k]
    x, y = value_function_points(F_bp, F_v)
    chart = value_function_graph(x, y, title=title, label_size=label_size, yaxis_tick_size=yaxis_tick_size,
                                 xaxis_tick_size=xaxis_tick_size, figsize=figsize, facecolor=facecolor,
                                 display_title=display_title, save=save, x_label=x_label)
    return chart


# Value Parameter Generator
def value_parameter_realistic_generator(parameters, default_value_parameters, constraint_options=None,
                                        num_breakpoints=20, deterministic=False, constrain_merit=False,
                                        printing=False, data_type="B"):
    """
    Generates realistic value parameters based on cadet data
    :param data_type: type of data instance
    :param constrain_merit: if we want to constrain average merit
    :param deterministic: if we're generating random parameters or not
    :param num_breakpoints: number of breakpoints to use on value functions
    :param printing: if the procedure should print something
    :param parameters: fixed cadet data
    :param default_value_parameters: default value parameters
    :param constraint_options: df for constraint options
    :return: value parameters
    """

    if printing:
        print('Generating realistic value parameters...')

    # Load parameter data
    N = parameters['N']
    M = parameters['M']
    afscs = parameters['afsc_vector']
    small_afscs = np.where(parameters['quota'] < 40)[0]
    large_afscs = np.where(parameters['quota'] >= 40)[0]
    usafa_max_col = np.array(constraint_options.loc[:, 'USAFA Max']).astype(str)
    u_con_indices = np.where(usafa_max_col != '1')[0]
    usafa_con_afscs = afscs[u_con_indices]

    # Add the AFSC objectives that are included in this instance
    objective_lookups = {'Merit': 'merit', 'USAFA Proportion': 'usafa', 'Combined Quota': 'quota',
                         'USAFA Quota': 'usafa_quota', 'ROTC Quota': 'rotc_quota', 'Mandatory': 'mandatory',
                         'Desired': 'desired', 'Permitted': 'permitted', 'Utility': 'utility', 'Male': 'male',
                         'Minority': 'minority'}
    objectives = []
    for objective in list(objective_lookups.keys()):
        if objective_lookups[objective] in parameters:
            objectives.append(objective)
    objectives = np.array(objectives)
    O = len(objectives)

    # Initialize set of value parameters
    value_parameters = {'objectives': objectives, 'O': O, 'M': M, 'objective_weight': np.zeros([M, O]),
                        'objective_target': np.zeros([M, O]), 'num_breakpoints': num_breakpoints,
                        'value_functions': np.array([[" " * 100 for _ in range(O)] for _ in range(M)]),
                        'F_bp': [[[] for _ in range(O)] for _ in range(M)],
                        'F_v': [[[] for _ in range(O)] for _ in range(M)],
                        'constraint_type': np.zeros([M, O]).astype(int),
                        'objective_value_min': np.array([[" " * 20 for _ in range(O)] for _ in range(M)]),
                        'afsc_value_min': np.zeros(M), 'cadet_value_min': np.zeros(N),
                        'cadets_overall_value_min': 0, 'afscs_overall_value_min': 0}

    # Parameters that we loop through
    params = ['cadets_overall_weight', 'cadet_weight_function', 'afsc_weight_function', 'objective_weight',
              'objective_target', 'value_functions']

    # Loop through all value parameters
    for param in params:

        # Overall weights
        if param == 'cadets_overall_weight':

            # Load distribution parameters
            l, u, mu, sigma = 0.1, 0.6, 0.35, 0.2

            # Generate parameter
            if deterministic:
                v = 0.3
            else:
                v = l - 0.1
                while v < l or v > u:
                    v = np.random.normal(mu, sigma)

            # Load parameter
            value_parameters[param] = round(v, 2)
            value_parameters['afscs_overall_weight'] = round(1 - v, 2)

        # Cadet Weights
        elif param == 'cadet_weight_function':

            # Define function choices
            options = ['Equal', 'Linear', 'Exponential']
            probabilities = [0.05, 0.7, 0.25]

            # Choose Function
            if deterministic:
                func = 'Linear'
            else:
                func = random.choices(options, probabilities)[0]
            value_parameters['cadet_weight_function'] = func

            if func == 'Equal':
                value_parameters['cadet_weight'] = np.repeat(1 / N, N)
            elif func == 'Linear':
                value_parameters['cadet_weight'] = parameters['merit'] / parameters['sum_merit']
            else:

                # Generate rho parameter
                l, u, mu, sigma = 0.1, 0.5, 0.3, 0.1
                rho = l - 0.1
                while rho < l or rho > u:
                    rho = np.random.normal(mu, sigma)

                if random.uniform(0, 1) < 0.5:
                    rho = -rho

                # Generate cadet weights
                swing_weights = np.array([
                    (1 - exp(-i / rho)) / (1 - exp(-1 / rho)) for i in parameters['merit']])
                value_parameters['cadet_weight'] = swing_weights / sum(swing_weights)

        # AFSC Weights
        elif param == 'afsc_weight_function':

            # Define function choices
            options = ['Equal', 'Linear', 'Piece', 'Norm']
            probabilities = [0.1, 0.3, 0.3, 0.3]

            # Choose Function
            if deterministic:
                func = 'Piece'
            else:
                func = random.choices(options, probabilities)[0]
            value_parameters['afsc_weight_function'] = func

            if func == 'Equal':
                value_parameters['afsc_weight'] = np.repeat(1 / M, M)
            elif func == 'Linear':
                value_parameters['afsc_weight'] = parameters['quota'] / sum(parameters['quota'])
            elif func == 'Piece':

                # Generate AFSC weights
                swing_weights = np.zeros(M)
                for j, quota in enumerate(parameters['quota']):
                    if quota >= 200:
                        swing_weights[j] = 1
                    elif 150 <= quota < 200:
                        swing_weights[j] = 0.9
                    elif 100 <= quota < 150:
                        swing_weights[j] = 0.8
                    elif 50 <= quota < 100:
                        swing_weights[j] = 0.7
                    elif 25 <= quota < 50:
                        swing_weights[j] = 0.6
                    else:
                        swing_weights[j] = 0.5

                # Load weights
                value_parameters['afsc_weight'] = np.around(swing_weights / sum(swing_weights), 4)
            else:

                # Load distribution parameters
                l, u, mu, sigma = 0.5, 1.5, 1, 0.15

                # Generate AFSC weights
                swing_weights = np.zeros(M)
                for j, quota in enumerate(parameters['quota']):
                    v = l - 0.1
                    while v < l or v > u:
                        v = np.random.normal(mu, sigma)
                    swing_weights[j] = quota * v

                # Load weights
                value_parameters['afsc_weight'] = np.around(swing_weights / sum(swing_weights), 2)

        # AFSC Objective Weights
        elif param == 'objective_weight':

            swing_weights = np.zeros([M, O])

            # Choose balancing function
            options = ['Method 1', 'Method 2', 'Method 3']
            probabilities = [0.3, 0.3, 0.4]
            if deterministic:
                balancing_func = 'Method 3'
            else:
                balancing_func = random.choices(options, probabilities)[0]

            # Loop through all objectives
            loc_k = np.zeros(O).astype(int)
            for k, objective in enumerate(objectives):

                loc_k[k] = np.where(default_value_parameters['objectives'] == objective)[0][0]
                if objective == 'Merit':
                    value_parameters['merit_weight_function'] = balancing_func

                    if balancing_func == 'Method 1':
                        swing_weights[:, k] = np.array([
                            max(min(np.random.normal(30, 5), 50), 10) for _ in range(M)])

                    elif balancing_func == 'Method 2':
                        swing_weights[large_afscs, k] = np.array([
                            max(min(np.random.normal(40, 10), 60), 20) for _ in range(len(large_afscs))])

                    else:

                        if deterministic:
                            swing_weights[large_afscs, k] = np.repeat(40, len(large_afscs))
                            swing_weights[small_afscs, k] = np.repeat(10, len(small_afscs))
                        else:
                            swing_weights[large_afscs, k] = np.array([
                                max(min(np.random.normal(40, 10), 60), 20) for _ in range(len(large_afscs))])
                            swing_weights[small_afscs, k] = np.array([
                                max(min(np.random.normal(10, 2), 15), 5) for _ in range(len(small_afscs))])

                elif objective == 'USAFA Proportion':
                    value_parameters['usafa_proportion_weight_function'] = balancing_func

                    if balancing_func == 'Method 1':
                        swing_weights[:, k] = np.array([
                            max(min(np.random.normal(20, 5), 35), 5) for _ in range(M)])

                    elif balancing_func == 'Method 2':
                        swing_weights[large_afscs, k] = np.array([
                            max(min(np.random.normal(30, 10), 50), 10) for _ in range(len(large_afscs))])
                        swing_weights[u_con_indices, k] = np.array([
                            max(min(np.random.normal(30, 10), 50), 10) for _ in range(len(u_con_indices))])

                    else:

                        if deterministic:
                            swing_weights[large_afscs, k] = np.repeat(40, len(large_afscs))
                            swing_weights[small_afscs, k] = np.repeat(5, len(small_afscs))
                        else:
                            swing_weights[large_afscs, k] = np.array([
                                max(min(np.random.normal(40, 10), 60), 20) for _ in range(len(large_afscs))])
                            swing_weights[small_afscs, k] = np.array([
                                max(min(np.random.normal(5, 1), 10), 0) for _ in range(len(small_afscs))])

                elif objective == 'Combined Quota':
                    if deterministic:
                        swing_weights[:, k] = np.repeat(100, M)
                    else:
                        swing_weights[:, k] = np.array([
                            max(min(np.random.normal(100, 5), 110), 90) for _ in range(M)])

                elif objective == 'Utility':
                    if deterministic:
                        swing_weights[:, k] = np.repeat(35, M)
                    else:
                        swing_weights[:, k] = np.array([
                            max(min(np.random.normal(35, 20), 60), 10) for _ in range(M)])

            # Determine AFOCD tier makeup
            afocd_strs = np.array(['   ' for _ in range(M)])
            loc_j = np.zeros(M).astype(int)
            for j, afsc in enumerate(afscs):

                loc_j[j] = np.where(default_value_parameters['complete_afsc_vector'] == afsc)[0][0]
                a_str = ''
                for objective in ['Mandatory', 'Desired', 'Permitted']:

                    k = np.where(default_value_parameters['objectives'] == objective)[0][0]
                    if default_value_parameters['objective_target'][loc_j[j], k] != 0:
                        a_str += objective[:1]
                afocd_strs[j] = a_str

            # AFOCD Weights
            k_m = np.where(objectives == 'Mandatory')[0][0]
            k_d = np.where(objectives == 'Desired')[0][0]
            k_p = np.where(objectives == 'Permitted')[0][0]
            for j in range(M):

                if deterministic:
                    if afocd_strs[j] == 'M':
                        swing_weights[j, k_m] = 90
                    elif afocd_strs[j] == 'MD':
                        swing_weights[j, k_m] = 80
                        swing_weights[j, k_d] = 40
                    elif afocd_strs[j] == 'MP':
                        swing_weights[j, k_m] = 80
                        swing_weights[j, k_p] = 20
                    elif afocd_strs[j] == 'MDP':
                        swing_weights[j, k_m] = 80
                        swing_weights[j, k_d] = 50
                        swing_weights[j, k_p] = 30
                    elif afocd_strs[j] == 'DP':
                        swing_weights[j, k_d] = 60
                        swing_weights[j, k_p] = 30
                else:
                    if afocd_strs[j] == 'M':
                        swing_weights[j, k_m] = max(min(np.random.normal(90, 5), 100), 80)
                    elif afocd_strs[j] == 'MD':
                        swing_weights[j, k_m] = max(min(np.random.normal(80, 5), 90), 70)
                        swing_weights[j, k_d] = max(min(np.random.normal(40, 5), 50), 30)
                    elif afocd_strs[j] == 'MP':
                        swing_weights[j, k_m] = max(min(np.random.normal(80, 5), 85), 75)
                        swing_weights[j, k_p] = max(min(np.random.normal(20, 5), 30), 10)
                    elif afocd_strs[j] == 'MDP':
                        swing_weights[j, k_m] = max(min(np.random.normal(80, 10), 100), 60)
                        swing_weights[j, k_d] = max(min(np.random.normal(50, 5), 60), 40)
                        swing_weights[j, k_p] = max(min(np.random.normal(30, 5), 40), 20)
                    elif afocd_strs[j] == 'DP':
                        swing_weights[j, k_d] = max(min(np.random.normal(60, 5), 70), 50)
                        swing_weights[j, k_p] = max(min(np.random.normal(30, 5), 40), 20)

                # Load objective weights for this AFSC
                value_parameters[param][j, :] = swing_weights[j, :] / np.sum(swing_weights[j, :])

        # AFSC Objective Targets
        elif param == 'objective_target':

            for j, afsc in enumerate(afscs):

                usafa_max = constraint_options.loc[j, 'USAFA Max']
                for k, objective in enumerate(objectives):

                    if objective == 'Merit':
                        value_parameters['objective_target'][j, k] = parameters['sum_merit'] / N
                        if constrain_merit:
                            value_parameters['constraint_type'][j, k] = 4
                            value_parameters['objective_value_min'][j][k] = "0.35, 2"
                            if data_type in ['E', "2018"] and afsc in ['E31', 'E32', "62EXI", "32EXA"]:
                                value_parameters['objective_value_min'][j][k] = "0.16, 2"
                            elif data_type in ['F', "2016"] and afsc in ["32EXA", "F31"]:
                                value_parameters['objective_value_min'][j][k] = "0.29, 2"
                            elif data_type in ['F', "2016"] and afsc in ["65F", "F13"]:
                                value_parameters['objective_value_min'][j][k] = "0.34, 2"

                    elif objective == 'USAFA Proportion':
                        if parameters['usafa_quota'][j] == 0 and data_type not in ["D", "F", "2016", "2017"]:
                            value_parameters['objective_target'][j, k] = 0

                        elif parameters['usafa_quota'][j] == parameters['quota'][j]:
                            value_parameters['objective_target'][j, k] = 1

                        else:
                            value_parameters['objective_target'][j, k] = parameters['usafa_proportion']

                        # AFSCs with maximum USAFA constraint aren't trying to balance the proportion
                        if afsc in usafa_con_afscs:
                            u_target = float(usafa_max.split(',')[0])
                            u_max = float(usafa_max.split(',')[1])
                            value_parameters['objective_target'][j, k] = u_target

                            # Constraints
                            value_parameters['constraint_type'][j, k] = 4
                            value_parameters['objective_value_min'][j][k] = str(0) + ", " + \
                                                                            str(u_max)

                    elif objective == 'Combined Quota':
                        value_parameters['objective_target'][j, k] = parameters['quota'][j]

                    elif objective == 'USAFA Quota':
                        value_parameters['objective_target'][j, k] = parameters['usafa_quota'][j]

                    elif objective == 'ROTC Quota':
                        value_parameters['objective_target'][j, k] = parameters['rotc_quota'][j]

                    elif objective == 'Male':
                        value_parameters['objective_target'][j, k] = parameters['male_proportion']

                    elif objective == 'Minority':
                        value_parameters['objective_target'][j, k] = parameters['minority_proportion']

                    elif objective == 'Utility':
                        value_parameters['objective_target'][j, k] = 1

                    else:  # AFOCD Objectives
                        value_parameters['objective_target'][j, k] = default_value_parameters[
                            'objective_target'][loc_j[j], loc_k[k]]

        # Value Functions
        elif param == 'value_functions':

            for j, afsc in enumerate(afscs):

                # AFSC variables
                cadets = parameters['I_E'][j]
                quota_str = constraint_options.loc[j, 'Combined Quota']
                mand_str = constraint_options.loc[j, 'Mandatory']

                # Loop through all objectives to create value functions
                for k, objective in enumerate(objectives):

                    # Initialize function parameters
                    target = value_parameters['objective_target'][j, k]
                    actual = None
                    maximum = None
                    q_minimum = 1

                    if value_parameters['objective_weight'][j, k] != 0:

                        if objective in ['Merit', 'USAFA Proportion']:

                            # Generate buffer y parameter
                            if deterministic:
                                buffer_y = 0.7
                            else:
                                buffer_y = 0
                                while buffer_y < 0.65 or buffer_y > 0.85:
                                    buffer_y = round(np.random.normal(0.7, 0.05), 3)

                            if objective == 'Merit':

                                # Function parameters
                                left_bm, right_bm = 0.1, 0.14
                                actual = np.mean(parameters['merit'][cadets])

                                # Rho distribution parameters
                                rho_params = {0: {'l': 0.04, 'u': 0.1, 'mu': 0.07, 'sigma': 0.01},
                                              1: {'l': 0.06, 'u': 0.1, 'mu': 0.08, 'sigma': 0.005},
                                              2: {'l': 0.06, 'u': 0.1, 'mu': 0.08, 'sigma': 0.005},
                                              3: {'l': 0.1, 'u': 0.15, 'mu': 0.125, 'sigma': 0.1}}

                                # Generate rho parameters
                                if deterministic:
                                    rhos = [0.07, 0.08, 0.08, 0.125]
                                else:
                                    rhos = []
                                    for i in range(4):
                                        r = rho_params[i]
                                        rho = r['l'] - 0.1
                                        while rho < r['l'] or rho > r['u']:
                                            rho = round(np.random.normal(r['mu'], r['sigma']), 3)
                                        rhos.append(rho)

                                # Create function string
                                vf_string = "Balance|" + str(left_bm) + ", " + str(right_bm) + ", " + \
                                            str(rhos[0]) + ", " + str(rhos[1]) + ", " + str(rhos[2]) + ", " + \
                                            str(rhos[3]) + ", " + str(buffer_y)

                            else:

                                # Function parameters
                                left_bm, right_bm = 0.12, 0.12
                                actual = np.mean(parameters['usafa'][cadets])

                                # Generate rho parameters
                                if deterministic:
                                    rhos = [0.1, 0.1, 0.1, 0.1]
                                else:
                                    rho = 0
                                    while rho < 0.08 or rho > 0.12:
                                        rho = round(np.random.normal(0.1, 0.01), 3)
                                    rhos = [rho, rho, rho, rho]

                                # Create function string
                                if target == 0 or afsc in usafa_con_afscs:
                                    vf_string = "Min Decreasing|" + str(rhos[0])
                                elif target == 1:
                                    vf_string = "Min Increasing|" + str(rhos[0])
                                else:
                                    vf_string = "Balance|" + str(left_bm) + ", " + str(right_bm) + ", " + \
                                                str(rhos[0]) + ", " + str(rhos[1]) + ", " + str(rhos[2]) + ", " + \
                                                str(rhos[3]) + ", " + str(buffer_y)

                        elif objective == 'Combined Quota':

                            # Function parameters
                            domain_max = 0.2
                            if '*' in quota_str:

                                # split string
                                split_str = quota_str.split('|')

                                # Get target upper bound
                                target_u = split_str[0].split(',')[1]
                                target_u = float(target_u[:len(target_u) - 1])  # remove *

                                # Get constraint upper bound
                                con_u = float(split_str[1].split(',')[1])

                                # Get constraint lower bound
                                q_minimum = float(split_str[1].split(',')[0])

                                # Choose Method
                                methods = ['Method 1', 'Method 2']
                                weights = [0.2, 0.8]
                                if deterministic:
                                    method = 'Method 2'
                                else:
                                    method = random.choices(methods, weights, k=1)[0]

                                if afsc in ["A5", "B3", "C3", "D5", "E6", "F5", "G5", "13N"]:
                                    method = 'Method 1'

                                actual = con_u
                                if method == 'Method 1':
                                    maximum = con_u
                                else:
                                    maximum = target_u
                            else:

                                # Pick Method 1
                                method = 'Method 1'

                                # if both ranges are valid
                                if '|' in quota_str:

                                    # split string
                                    split_str = quota_str.split('|')

                                    # Choose one of the upper bounds to use
                                    weights = [0.4, 0.6]
                                    if deterministic:
                                        index = 1
                                    else:
                                        index = random.choices([0, 1], weights, k=1)[0]

                                    maximum = float(split_str[index].split(',')[1])

                                else:
                                    maximum = float(quota_str.split(',')[1])

                                actual = maximum

                            # Rho distribution parameters
                            rho_params = {0: {'l': 0.2, 'u': 0.3, 'mu': 0.25, 'sigma': 0.03},
                                          1: {'l': 0.05, 'u': 0.1, 'mu': 0.075, 'sigma': 0.01},
                                          2: {'l': 0.03, 'u': 0.7, 'mu': 0.05, 'sigma': 0.005}}

                            if method == 'Method 1':

                                # Generate rho parameters
                                if deterministic:
                                    rhos = [0.25, 0.075]
                                else:
                                    rhos = []
                                    for i in range(2):
                                        r = rho_params[i]
                                        rho = r['l'] - 0.1
                                        while rho < r['l'] or rho > r['u']:
                                            rho = round(np.random.normal(r['mu'], r['sigma']), 3)
                                        rhos.append(rho)

                                # Create function string
                                vf_string = "Quota_Normal|" + str(domain_max) + ", " + str(rhos[0]) + ", " + \
                                            str(rhos[1])

                            else:

                                # Generate rho parameters
                                if deterministic:
                                    rhos = [0.25, 0.075, 0.05]
                                else:
                                    rhos = []
                                    for i in range(3):
                                        r = rho_params[i]
                                        rho = r['l'] - 0.1
                                        while rho < r['l'] or rho > r['u']:
                                            rho = round(np.random.normal(r['mu'], r['sigma']), 3)
                                        rhos.append(rho)

                                # Generate buffer y parameter
                                if deterministic:
                                    buffer_y = 0.6
                                else:
                                    buffer_y = 0
                                    while buffer_y < 0.5 or buffer_y > 0.7:
                                        buffer_y = round(np.random.normal(0.6, 0.05), 3)

                                # Create function string
                                vf_string = "Quota_Over|" + str(domain_max) + ", " + str(rhos[0]) + ", " + \
                                            str(rhos[1]) + ", " + str(rhos[2]) + ", " + str(buffer_y)

                            # Constraints
                            value_parameters['constraint_type'][j, k] = 4
                            value_parameters['objective_value_min'][j][k] = str(int(q_minimum * target)) + ", " + \
                                                                            str(int(actual * target))

                        elif objective in ['Mandatory', 'Desired', 'Permitted']:

                            # Generate rho parameter
                            if deterministic:
                                rho = 0.1
                            else:
                                rho = 0
                                while rho < 0.08 or rho > 0.12:
                                    rho = round(np.random.normal(0.1, 0.01), 3)

                            if objective == 'Mandatory':
                                function = 'Min Increasing'

                                if '*' in mand_str:

                                    # split string
                                    split_str = mand_str.split('|')

                                    # Get constraint bounds
                                    lower = float(split_str[1].split(',')[0])
                                    upper = float(split_str[1].split(',')[1])
                                else:
                                    lower = float(mand_str.split(',')[0])
                                    upper = float(mand_str.split(',')[1])

                                # Constraints
                                value_parameters['constraint_type'][j, k] = 3
                                value_parameters['objective_value_min'][j][k] = str(lower) + ", " + str(upper)

                                if data_type in ["D", "E", "F", "2016", "2017", "2018"] and \
                                        afsc in ["D18", "E18", "F19", "35P"]:
                                    value_parameters['constraint_type'][j, k] = 0

                            elif objective == 'Permitted':
                                function = 'Min Decreasing'
                            else:
                                if afsc in ["A26", "B33", "C26", "D27", "E26", "F26", "A13", "B15", "C14", "D16",
                                            "E12", "F15", "A1", "B2", "B11", "B34", "B36", "C1", "D2", "E3", "F2",
                                            "G1", "G13", "G27", "14F", "61A", "15A", "17D", "17S", "17DXS", "17SXS",
                                            "17X"]:
                                    function = 'Min Decreasing'
                                else:
                                    function = 'Min Increasing'

                            vf_string = function + '|' + str(rho)

                        elif objective == 'Utility':

                            # Generate rho parameter
                            if deterministic:
                                rho = 0.25
                            else:
                                rho = 0
                                while rho < 0.15 or rho > 0.35:
                                    rho = round(np.random.normal(0.25, 0.05), 3)

                            vf_string = 'Min Increasing|' + str(rho)
                        else:
                            vf_string = 'None'
                    else:
                        vf_string = 'None'

                    # Create function
                    value_parameters['value_functions'][j][k] = vf_string
                    if vf_string != 'None':
                        segment_dict = create_segment_dict_from_string(vf_string, target, actual=actual,
                                                                       maximum=maximum)
                        # print(afsc, objective, segment_dict)
                        value_parameters['F_bp'][j][k], value_parameters['F_v'][j][k] = value_function_builder(
                            segment_dict, num_breakpoints=num_breakpoints)

    return value_parameters





