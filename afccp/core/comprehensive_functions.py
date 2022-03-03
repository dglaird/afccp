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








