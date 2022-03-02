# Import libraries
import time
import numpy as np
import logging
import warnings
from globals import *

# Ignore warnings
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


def original_pyomo_model_build(printing=False):
    """
    Builds the original AFPC model
    :param printing: Whether the procedure should print something
    :return: pyomo model as object
    """
    if printing:
        print('Building Original Pyomo Model...')

    # Build Model
    model = AbstractModel()

    # Define sets
    model.I = Set(doc='Cadets')  # range of N
    model.J = Set(doc='AFSCs')  # range of M
    model.L = Set(doc='Large AFSCs')  # range of G
    model.R = Set(doc='AFSCs with Mandatory Requirements')  # Range of M_R

    # Define Parameters
    model.N = Param(doc='Number of Cadets')
    model.M = Param(doc='Number of AFSCs')
    model.M_R = Param(doc='Number of AFSCs with Mandatory Requirements')
    model.G = Param(doc='Number of Large AFSCs')
    model.merit = Param(model.I, doc='percentile for cadet i')
    model.usafa = Param(model.I, doc='1 if cadet i is a usafa graduate, 0 o/w')
    model.target = Param(model.J, doc='target for AFSC j')
    model.over = Param(model.J, doc='amount by which AFSC j can be over-classified')
    model.eduM = Param(model.J, doc='target accession rate for mandatory degrees for AFSC j')
    model.tierM = Param(model.I, model.J, doc='1 if cadet i has a mandatory degree for AFSC j, 0 o/w')
    model.C = Param(model.I, model.J, doc='utility of assigning cadet i to AFSC j')

    # Define variables
    model.x = Var(model.I, model.J, within=Binary, doc='1 if cadet i is assigned to AFSC j, 0 o/w')

    def one_afsc_rule(model, i):
        return sum(model.x[i, j] for j in model.J) == 1

    model.con_one_afsc = Constraint(model.I, rule=one_afsc_rule)

    def min_target_rule(model, j):
        return sum(model.x[i, j] for i in model.I) >= model.target[j]

    model.con_min_target = Constraint(model.J, rule=min_target_rule)

    def max_target_rule(model, j):
        return sum(model.x[i, j] for i in model.I) <= model.over[j]

    model.con_max_target = Constraint(model.J, rule=max_target_rule)

    def min_mandatory_rule(model, j):  # arbitrary constraint bound extension
        return sum(model.tierM[i, j] * model.x[i, j] for i in model.I) >= (model.eduM[j] * model.target[j]) - 30

    model.con_min_mandatory = Constraint(model.R, rule=min_mandatory_rule)

    def min_usafa_rule(model, j):  # arbitrary constraint bound extension
        return sum(model.usafa[i] * model.x[i, j] for i in model.I) >= (0.2 * model.target[j]) - 20

    model.con_min_usafa = Constraint(model.L, rule=min_usafa_rule)

    def max_usafa_rule(model, j):  # arbitrary constraint bound extension
        return sum(model.usafa[i] * model.x[i, j] for i in model.I) <= (0.4 * model.target[j]) + 20

    model.con_min_usafa = Constraint(model.L, rule=max_usafa_rule)

    def min_percentile_rule(model, j):  # arbitrary constraint bound extension
        return sum(model.merit[i] * model.x[i, j] for i in model.I) >= (0.35 * model.target[j]) - 20

    model.con_min_percentile = Constraint(model.L, rule=min_percentile_rule)

    def max_percentile_rule(model, j):  # arbitrary constraint bound extension
        return sum(model.merit[i] * model.x[i, j] for i in model.I) <= (0.65 * model.target[j]) + 20

    model.con_max_percentile = Constraint(model.L, rule=max_percentile_rule)

    def objective_rule(model):
        return sum(sum(model.C[i, j] * model.x[i, j] for j in model.J) for i in model.I)

    model.objective = Objective(rule=objective_rule, sense=maximize, doc='Objective Function')

    return model


def convert_parameters_to_original_model_inputs(parameters, value_parameters, printing=False):
    """
    Converts the parameters and value parameters to the pyomo data structure
    :param parameters: fixed cadet/AFSC parameters
    :param value_parameters: user defined parameters
    :param printing: Whether the procedure should print something
    :return: pyomo data
    """
    if printing:
        print("Converting parameters to Original Model Pyomo Inputs...")

    N = parameters['N']
    M = parameters['M']

    # New "Large" AFSCs set
    large_afscs = np.where(parameters['quota'] >= 40)[0]
    G = len(large_afscs)

    # Objective indices
    mand_k = np.where(value_parameters['objectives'] == 'Mandatory')[0][0]
    quota_k = np.where(value_parameters['objectives'] == 'Combined Quota')[0][0]

    # Utility Matrix
    C = np.zeros([N, M])
    for i in range(N):
        for j in range(M):

            # If AFSC j is a preference for cadet i
            if parameters['utility'][i, j] != 0:

                if parameters['mandatory'][i, j] == 1:
                    C[i, j] = 10 * parameters['merit'][i] * parameters['utility'][i, j] + 250
                elif parameters['desired'][i, j] == 1:
                    C[i, j] = 10 * parameters['merit'][i] * parameters['utility'][i, j] + 150
                elif parameters['permitted'][i, j] == 1:
                    C[i, j] = 10 * parameters['merit'][i] * parameters['utility'][i, j]
                else:
                    C[i, j] = -50000

            # If it is not a preference for cadet i
            else:

                if parameters['mandatory'][i, j] == 1:
                    C[i, j] = 100 * parameters['merit'][i]
                elif parameters['desired'][i, j] == 1:
                    C[i, j] = 50 * parameters['merit'][i]
                elif parameters['permitted'][i, j] == 1:
                    C[i, j] = 0
                else:
                    C[i, j] = -50000

    # Construct dictionary
    data = {None: {
        'I': {None: np.arange(N)},
        'J': {None: np.arange(M)},
        'L': {None: large_afscs},
        'R': {None: value_parameters['J_A'][mand_k]},
        'N': {None: N},
        'M': {None: M},
        'G': {None: G},
        'M_R': {None: len(value_parameters['J_A'][mand_k])},
        'merit': {i: parameters['merit'][i] for i in range(N)},
        'usafa': {i: parameters['usafa'][i] for i in range(N)},
        'target': {j: value_parameters['objective_target'][j, quota_k] for j in range(M)},
        'over': {j: parameters['quota_max'][j] for j in range(M)},
        'eduM': {j: value_parameters['objective_target'][j, mand_k] for j in range(M)},
        'tierM': {(i, j): parameters['mandatory'][i][j] for i in range(N) for j in range(M)},
        'C': {(i, j): C[i][j] for i in range(N) for j in range(M)}
    }}

    return data


def solve_original_pyomo_model(data, model, model_name='Original Model', solve_name="cbc", printing=False):
    """
    Solves the pyomo model and returns the solution
    :param solve_name: which solver to use
    :param model_name: kind of model we're solving
    :param data: pyomo model parameters
    :param model: abstract model
    :param printing: Whether the procedure should print something
    :return: solution (vector), X (matrix)
    """
    if printing:
        print('Creating ' + model_name + ' instance...')

    instance = model.create_instance(data)

    if printing:
        print('Solving ' + model_name + ' instance with solver ' + solve_name + '...')

    if solve_name == "baron":
        solver = SolverFactory(solve_name, executable='../Main Directory/Solvers/baron/' + solve_name + '.exe')
    elif solve_name == 'cplex':
        solver = SolverFactory(solve_name)
    else:
        solver = SolverFactory(solve_name, executable=paths['Solvers'] + solve_name + '.exe')

    solver.solve(instance)
    solution = np.zeros(instance.N.value)
    for i in range(instance.N.value):
        for j in range(instance.M.value):
            if round(instance.x[i, j].value):
                solution[i] = int(j)

    return solution


def vft_model_build(parameters, value_parameters, initial=None, convex=True, add_breakpoints=True,
                    printing=False):
    """
    Builds the VFT optimization model using pyomo
    :param add_breakpoints: if we should add breakpoints to adjust the approximate model
    :param initial: if this model has a warm start or not
    :param convex: if we use the target quota instead of summation of cadets to calculate proportions/averages
    :param value_parameters: weight and value parameters
    :param parameters: fixed cadet/afsc data
    :param printing: Whether the procedure should print something
    :return: pyomo model as object
    """
    if printing:
        print('Building VFT Model...')

    # Build Model
    model = ConcreteModel()

    # Set Definitions
    M = parameters['M']  # number of afscs
    O = value_parameters['O']  # number of objectives
    I = parameters['I']  # set of cadets
    J = parameters['J']  # set of afscs
    K = range(O)  # set of objectives
    J_E = parameters['J_E']  # set of AFSCs for which cadet i is eligible
    I_E = parameters['I_E']  # set of cadets that are eligible for AFSC j
    I_D = parameters['I_D']  # contains sets of usafa, mandatory, desired, permitted, male,
    # minority cadets that are eligible for AFSC j
    K_A = value_parameters['K_A']  # set of objectives for AFSC j
    K_C = value_parameters['K_C']  # set of constrained objectives for AFSC j
    K_D = value_parameters['K_D']  # set of objectives that seek some demographic of cadets

    # Value Parameters (Take out parameters from dictionaries so its easier to read everything)
    target = value_parameters['objective_target']
    objectives = value_parameters['objectives']
    objective_weight = value_parameters['objective_weight']
    afsc_weight = value_parameters['afsc_weight']
    afsc_value_min = value_parameters['afsc_value_min']
    afscs_overall_weight = value_parameters['afscs_overall_weight']
    afscs_overall_value_min = value_parameters['afscs_overall_value_min']
    cadet_weight = value_parameters['cadet_weight']
    cadet_value_min = value_parameters['cadet_value_min']
    cadets_overall_weight = value_parameters['cadets_overall_weight']
    cadets_overall_value_min = value_parameters['cadets_overall_value_min']
    objective_constraint_type = value_parameters['constraint_type']

    # Value Function Parameters
    r = [[len(value_parameters['F_bp'][j][k]) for k in K] for j in J]  # number of breakpoints (bps)
    L = [[list(range(r[j][k])) for k in K] for j in J]  # set of bps
    a = [[[value_parameters['F_bp'][j][k][l] for l in L[j][k]] for k in K] for j in J]  # measures of bps
    f = [[[value_parameters['F_v'][j][k][l] for l in L[j][k]] for k in K] for j in J]  # values of bps

    # Initialize AFSC objective measure constraint ranges
    objective_min_value = np.zeros([M, O])
    objective_max_value = np.zeros([M, O])

    # Loop through each AFSC
    for j in J:

        # Loop through each objective for each AFSC
        for k in K_A[j]:

            # We need to add an extra breakpoint to effectively extend the range
            if add_breakpoints:

                # We add an extra breakpoint far along the x-axis with the same y value as the previous one
                last_a = a[j][k][r[j][k] - 1]
                last_f = f[j][k][r[j][k] - 1]
                a[j][k].append(last_a * 2000)  # arbitrarily large number
                f[j][k].append(last_f)  # same y-value as previous one
                L[j][k].append(r[j][k])  # add the new breakpoint index
                r[j][k] += 1  # add a breakpoint to the number of breakpoints

            # Retrieve minimum values based on constraint type (approximate/exact and value/measure)
            if objective_constraint_type[j, k] == 1 or objective_constraint_type[j, k] == 2:

                # These are "value" constraints and so only a minimum is needed
                objective_min_value[j, k] = float(value_parameters['objective_value_min'][j, k])
            elif objective_constraint_type[j, k] == 3 or objective_constraint_type[j, k] == 4:

                # These are "measure" constraints and so a range is needed
                value_list = value_parameters['objective_value_min'][j, k].split(",")
                objective_min_value[j, k] = float(value_list[0].strip())
                objective_max_value[j, k] = float(value_list[1].strip())

    # Convert to numpy arrays of lists
    r = np.array(r)
    L = np.array(L)
    a = np.array(a)
    f = np.array(f)

    # Variables
    if initial is None:

        # If we don't use a warm-start, we don't initialize starting values for the variables
        model.x = Var(((i, j) for i in I for j in J_E[i]), within=Binary)  # main decision variable (x)
        model.f_value = Var(((j, k) for j in J for k in K_A[j]), within=NonNegativeReals)  # objective value

        # Lambda and y variables for value functions
        model.lam = Var(((j, k, l) for j in J for k in K_A[j] for l in L[j, k]), within=NonNegativeReals, bounds=(0, 1))
        model.y = Var(((j, k, l) for j in J for k in K_A[j] for l in range(0, r[j, k] - 1)), within=Binary)

    # Variable Initialization
    else:

        # If we do use a warm-start, we initialize the variables
        model.x = Var(((i, j) for i in I for j in J_E[i]), within=Binary)
        for i in I:
            for j in J_E[i]:
                model.x[i, j] = round(initial['X'][i, j])
        model.f_value = Var(((j, k) for j in J for k in K_A[j]), within=NonNegativeReals)
        for j in J:
            for k in K_A[j]:
                model.f_value[j, k] = initial['F_X'][j, k]
        model.lam = Var(((j, k, l) for j in J for k in K_A[j] for l in L[j, k]), within=NonNegativeReals,
                        bounds=(0, 1))
        for j in J:
            for k in K_A[j]:
                for l in L[j, k]:
                    model.lam[j, k, l] = initial['lam'][j, k, l]
        model.y = Var(((j, k, l) for j in J for k in K_A[j] for l in range(0, r[j, k] - 1)), within=Binary)
        for j in J:
            for k in K_A[j]:
                for l in range(r[j, k] - 1):
                    model.y[j, k, l] = initial['y'][j, k, l]

    # Cadets receive one and only one AFSC (Ineligibility constraint is always met as a result of the indexed sets)
    model.one_afsc_constraints = ConstraintList()
    for i in I:
        model.one_afsc_constraints.add(expr=np.sum(model.x[i, j] for j in J_E[i]) == 1)

    # Objective Constraint Lists
    model.measure_constraints = ConstraintList()
    model.value_constraints = ConstraintList()

    # Value Function Constraints (Functional constraints)
    model.measure_vf_constraints = ConstraintList()
    model.value_vf_constraints = ConstraintList()
    model.lambda_y_constraint1 = ConstraintList()
    model.lambda_y_constraint2 = ConstraintList()
    model.lambda_y_constraint3 = ConstraintList()
    model.y_constraint = ConstraintList()
    model.lambda_sum_constraint = ConstraintList()
    model.lambda_positive = ConstraintList()
    model.f_value_positive = ConstraintList()

    # Min Value Constraints (AFSCs and Cadets)
    model.min_afsc_value_constraints = ConstraintList()
    model.min_cadet_value_constraints = ConstraintList()

    # Find quota objective index
    quota_k = np.where(objectives == 'Combined Quota')[0][0]

    # Loop through all AFSCs
    for j in J:

        # Get count variables for this AFSC
        count = np.sum(model.x[i, j] for i in I_E[j])
        if 'usafa' in parameters:

            # Number of USAFA cadets assigned to the AFSC
            usafa_count = np.sum(model.x[i, j] for i in I_D['USAFA Proportion'][j])

        # Quota for the AFSC
        target_quota = target[j, quota_k]

        # Are we using approximate measures or not
        if convex:
            num_cadets = target_quota  # Approximate
        else:
            num_cadets = count  # Exact

        # Loop through all objectives for this AFSC
        for k in K_A[j]:

            # Get the right objective measure calculation
            objective = objectives[k]

            # If it's a demographic objective, we sum over the cadets with that demographic
            if objective in K_D:
                numerator = np.sum(model.x[i, j] for i in I_D[objective][j])
                measure_jk = numerator / num_cadets
            elif objective == "Merit":
                numerator = np.sum(parameters['merit'][i] * model.x[i, j] for i in I_E[j])
                measure_jk = numerator / num_cadets
            elif objective == "Combined Quota":
                measure_jk = count
            elif objective == "USAFA Quota":
                measure_jk = usafa_count
            elif objective == "ROTC Quota":
                measure_jk = count - usafa_count
            else:  # Utility
                numerator = np.sum(parameters['utility'][i, j] * model.x[i, j] for i in I_E[j])
                measure_jk = numerator / num_cadets

            # Add Linear Value Function Constraints
            model.measure_vf_constraints.add(expr=measure_jk == np.sum(  # Measure Constraint for Value Function
                a[j, k][l] * model.lam[j, k, l] for l in L[j, k]))
            model.value_vf_constraints.add(expr=model.f_value[j, k] == np.sum(  # Value Constraint for Value Function
                f[j, k][l] * model.lam[j, k, l] for l in L[j, k]))

            # Lambda .. y constraints
            model.lambda_y_constraint1.add(expr=model.lam[j, k, 0] <= model.y[j, k, 0])
            if r[j, k] > 2:
                for l in range(1, r[j, k] - 1):
                    model.lambda_y_constraint2.add(expr=model.lam[j, k, l] <= model.y[j, k, l - 1] + model.y[j, k, l])
            model.lambda_y_constraint3.add(expr=model.lam[j, k, r[j, k] - 1] <= model.y[j, k, r[j, k] - 2])

            # Y sum to 1 constraint
            model.y_constraint.add(expr=np.sum(model.y[j, k, l] for l in range(0, r[j, k] - 1)) == 1)

            # Lambda sum to 1 constraint
            model.lambda_sum_constraint.add(expr=np.sum(model.lam[j, k, l] for l in L[j, k]) == 1)

            # Lambda .. value positive constraint
            for l in L[j, k]:
                model.lambda_positive.add(expr=model.lam[j, k, l] >= 0)
            model.f_value_positive.add(expr=model.f_value[j, k] >= 0)

            # Add Min Value/Measure Constraints
            if k in K_C[j]:  # (1/2 constrain value, 3 constrains approximate measure, 4 constrains exact measure)

                # Constrained Value (I decided against this for AFSC objectives and just went with measure constraints)
                if objective_constraint_type[j, k] == 1 or objective_constraint_type[j, k] == 2:

                    # The formulation only lists "objective_min, objective_max" since I no longer want value constraints
                    model.value_constraints.add(expr=model.f_value[j, k] >= objective_min_value[j, k])

                # Constrained Approximate Measure
                elif objective_constraint_type[j, k] == 3:
                    if objectives[k] in ['Combined Quota', 'USAFA Quota', 'ROTC Quota']:
                        model.measure_constraints.add(expr=measure_jk >= objective_min_value[j, k])
                        model.measure_constraints.add(expr=measure_jk <= objective_max_value[j, k])
                    else:
                        model.measure_constraints.add(expr=numerator - objective_min_value[j, k] * target_quota >= 0)
                        model.measure_constraints.add(expr=numerator - objective_max_value[j, k] * target_quota <= 0)

                # Constrained Exact Measure
                else:
                    if objectives[k] in ['Combined Quota', 'USAFA Quota', 'ROTC Quota']:
                        model.measure_constraints.add(expr=measure_jk >= objective_min_value[j, k])
                        model.measure_constraints.add(expr=measure_jk <= objective_max_value[j, k])
                    else:
                        model.measure_constraints.add(expr=numerator - objective_min_value[j, k] * count >= 0)
                        model.measure_constraints.add(expr=numerator - objective_max_value[j, k] * count <= 0)

        # AFSC value constraint
        if afsc_value_min[j] != 0:
            model.min_afsc_value_constraints.add(expr=np.sum(
                objective_weight[j, k] * model.f_value[j, k] for k in K_A[j]) >= afsc_value_min[j])

    # Cadet value constraint
    for i in I:
        if cadet_value_min[i] != 0:
            model.min_cadet_value_constraints.add(expr=np.sum(
                parameters['utility'][i, j] * model.x[i, j] for j in J_E[i]) >= cadet_value_min[i])

    # AFSC Overall Min Value
    if afscs_overall_value_min != 0:
        pass  # I just haven't put that here yet because I doubt I'll ever constrain this

    # Cadet Overall Min Value
    if cadets_overall_value_min != 0:
        pass  # I just haven't put that here yet because I doubt I'll ever constrain this

    # Max Z!
    def objective_function(model):
        return afscs_overall_weight * np.sum(afsc_weight[j] * np.sum(
            objective_weight[j, k] * model.f_value[j, k] for k in K_A[j]) for j in J) + \
               cadets_overall_weight * np.sum(cadet_weight[i] * np.sum(
            parameters['utility'][i, j] * model.x[i, j] for j in J_E[i]) for i in I)

    model.objective = Objective(rule=objective_function, sense=maximize)

    return model


def vft_model_solve(model, parameters, value_parameters, solve_name="cbc", approximate=True, report=False,
                    max_time=None, timing=False, printing=False):
    """
    Solve VFT Model
    :param timing: If we want to time the model
    :param max_time: max time in seconds the solver is allowed to solve
    :param approximate: if the model is convex or not
    :param value_parameters: model value parameters
    :param report: if we want to grab all the information to sanity check the solution
    :param parameters: fixed parameters
    :param model: pyomo model
    :param solve_name: name of solver
    :param printing: if we should print something
    :return: solution
    """

    # Get right solver
    if solve_name == "baron":
        solver = SolverFactory(solve_name, executable=paths['Solvers'] + 'baron\\baron.exe')
    elif solve_name in ['cplex', 'mindtpy']:
        solver = SolverFactory(solve_name)
    elif solve_name == 'gurobi':
        solver = SolverFactory(solve_name, solver_io='python')
    else:
        # solver = SolverFactory(solve_name, executable=paths['Solvers'] + solve_name + '.exe')
        solver = SolverFactory(solve_name)

    # Print what we're solving
    if printing:
        if approximate:
            model_str = 'Approximate'
        else:
            model_str = 'Exact'
        print('Solving ' + model_str + ' VFT Model instance with solver ' + solve_name + '...')

    # Start Time
    if timing:
        start_time = time.perf_counter()

    # Solve Model
    if max_time is not None:
        if solve_name == 'mindtpy':
            solver.solve(model, time_limit=max_time,
                         mip_solver='cplex_persistent', nlp_solver='ipopt')
        elif solve_name == 'gurobi':
            solver.solve(model, options={'TimeLimit': max_time, 'IntFeasTol': 0.05})
        elif solve_name == 'ipopt':
            solver.options['max_cpu_time'] = max_time
            solver.solve(model)
        elif solve_name == 'cbc':
            solver.options['seconds'] = max_time
            solver.solve(model)
        elif solve_name == 'baron':
            # solver.options['maxTimeLimit'] = max_time
            solver.solve(model, options={'MaxTime': max_time})
        else:
            solver.solve(model)
    else:
        if solve_name == 'mindtpy':
            solver.solve(model, mip_solver='cplex_persistent', nlp_solver='ipopt')
        else:
            solver.solve(model)

    # Stop Time
    if timing:
        solve_time = round(time.perf_counter() - start_time, 2)

    # Grab parameters
    N = parameters['N']  # number of cadets
    M = parameters['M']  # number of afscs
    I = parameters['I']  # set of cadets
    J_E = parameters['J_E']  # set of AFSCs for which cadet i is eligible
    J = parameters['J']  # set of afscs
    I_E = parameters['I_E']  # set of cadets that are eligible for AFSC j
    I_D = parameters['I_D']  # contains sets of cadets with certain demographics for AFSC j
    K_A = value_parameters['K_A']  # set of objectives for AFSC j
    objectives = value_parameters['objectives']
    target = value_parameters['objective_target']
    quota_k = np.where(objectives == "Combined Quota")[0]
    solution = np.zeros(N)

    # Create solution X Matrix
    X = np.zeros([N, M])
    for i in I:
        for j in J_E[i]:
            X[i, j] = model.x[i, j].value
            if round(X[i, j]):
                solution[i] = int(j)
    if report:
        obj = model.objective()
        if printing:
            if approximate:
                print("Approximate Pyomo Model Objective Value: " + str(round(obj, 4)))
            else:
                print("Exact Pyomo Model Objective Value: " + str(round(obj, 4)))

        # Initialize measure/value matrices
        measure = np.zeros([parameters['M'], value_parameters['O']])
        value = np.zeros([parameters['M'], value_parameters['O']])

        # Loop through all AFSCs to get their values
        for j in J:

            # Get variables for this AFSC
            count = np.sum(X[i, j] for i in I_E[j])
            if 'usafa' in parameters.keys():
                usafa_count = np.sum(X[i, j] for i in I_D['USAFA Proportion'][j])
            target_amount = target[j, quota_k]

            # Are we using approximate measures or not
            if approximate:
                num_cadets = target_amount
            else:
                num_cadets = count

            # Loop through all objectives for this AFSC
            for k in K_A[j]:
                objective = objectives[k]

                # Get the right measure calculation
                if objective in I_D:
                    numerator = np.sum(X[i, j] for i in I_D[objective][j])
                    measure[j, k] = numerator / num_cadets
                elif objective == "Merit":
                    numerator = np.sum(parameters['merit'][i] * X[i, j] for i in I_E[j])
                    measure[j, k] = numerator / num_cadets
                elif objective == "Combined Quota":
                    measure[j, k] = count
                elif objective == "USAFA Quota":
                    measure[j, k] = usafa_count
                elif objective == "ROTC Quota":
                    measure[j, k] = count - usafa_count
                else:  # Utility
                    numerator = np.sum(parameters['utility'][i, j] * X[i, j] for i in I_E[j])
                    measure[j, k] = numerator / num_cadets

                # Value of measure
                value[j, k] = model.f_value[j, k].value

        if timing:
            return solution, X, measure, value, obj, solve_time
        else:
            return solution, X, measure, value, obj
    else:
        if timing:
            return solution, solve_time
        else:
            return solution


def gp_model_build(gp, printing=False):
    """
    This is Rebecca's model. We've incorporated her parameters and are building that model
    :param gp: goal programming model parameters
    :param printing: Whether or not to print something
    :return: pyomo model
    """
    if printing:
        print('Building R Model...')

    # Create model
    m = ConcreteModel()

    # ______________VARIABLE DEFINITIONS______________
    m.x = Var(((c, a) for c in gp['C'] for a in gp['A^']['E'][c]), within=Binary)

    # Amount by which the constraint is not met
    m.Y = Var(((con, a) for con in gp['con'] for a in gp['A^'][con]), within=NonNegativeReals)

    # Amount by which the constraint is exceeded
    m.Z = Var(((con, a) for con in gp['con'] for a in gp['A^'][con]), within=NonNegativeReals)

    # Binary variable indicating if Y is used (1) or if Z is used (0)
    m.alpha = Var(((con, a) for con in gp['con'] for a in gp['A^'][con]), within=Binary)

    # ______________FORMULATION______________
    def objective_function(m):
        return np.sum(  # Sum across each constraint
            np.sum(  # Sum across each AFSC with that constraint

            # Calculate penalties and rewards (for each necessary AFSC for each constraint)
            gp['lam^'][con] * m.Z[con, a] - gp['mu^'][con] * m.Y[con, a] for a in gp['A^'][con]) for con in gp['con']) \
               + gp['lam^']['S'] * np.sum(  # Sum across every cadet
            np.sum(  # Sum across each AFSC that the cadet is both eligible for and has placed a preference on

                # Calculate utility that the cadet received  (for each preferred AFSC for each constraint)
                gp['utility'][c, a] * m.x[c, a] for a in gp['A^']['W^E'][c]) for c in gp['C'])

    # Define model objective function
    m.objective = Objective(rule=objective_function, sense=maximize)

    # Each Cadet gets one AFSC for which they're eligible
    m.one_afsc_constraints = ConstraintList()
    for c in gp['C']:
        m.one_afsc_constraints.add(expr=np.sum(m.x[c, a] for a in gp['A^']['E'][c]) == 1)

    m.con_constraints = ConstraintList()  # List of goal constraints (Each goal constraint for each AFSC)
    m.Y_constraints = ConstraintList()  # List of Y/alpha constraints
    m.Z_constraints = ConstraintList()  # List of Z/alpha constraints

    # Loop through each "goal" constraint
    for con in gp['con']:

        # Loop through all AFSCs for this constraint
        for a in gp['A^'][con]:

            # Number of cadets assigned to this AFSC
            count = np.sum(m.x[c, a] for c in gp['C^']['E'][a])

            if con in ['R_under', 'R_over']:

                # Sum of percentiles of cadets assigned to this AFSC
                con_count = np.sum(gp['merit'][c] * m.x[c, a] for c in gp['C^'][con][a])
            else:

                # Number of cadets assigned to this AFSC that pertain to this constraint
                con_count = np.sum(m.x[c, a] for c in gp['C^'][con][a])

            # Parameter for this constraint for this AFSC
            parameter = gp['param'][con][a]

            if con == 'T':
                m.con_constraints.add(expr=count == parameter - m.Y[con, a] + m.Z[con, a])
            elif con == 'F':
                m.con_constraints.add(expr=count == parameter + m.Y[con, a] - m.Z[con, a])
            elif con in ['M', 'D_under', 'W', 'U_under', 'R_under']:
                m.con_constraints.add(expr=con_count == parameter * count - m.Y[con, a] + m.Z[con, a])
            else:  # ('D_over', 'P', 'U_over', 'R_over')
                m.con_constraints.add(expr=con_count == parameter * count + m.Y[con, a] - m.Z[con, a])

            # Y/alpha constraint
            m.Y_constraints.add(expr=m.Y[con, a] <= gp['Big_M'] * m.alpha[con, a])

            # Z/alpha constraint
            m.Z_constraints.add(expr=m.Z[con, a] <= gp['Big_M'] * (1 - m.alpha[con, a]))

    # If we have AFSCs that have specified a limit on the number of USAFA cadets
    if len(gp['A^']['U_lim']) > 0:

        # Number of USAFA cadets assigned to AFSCs that have an upper limit on USAFA cadets
        usafa_cadet_lim_afsc_count = np.sum(np.sum(m.x[c, a] for c in gp['C^']['U'][a]) for a in gp['A^']['U_lim'])

        # Overall number of cadets assigned to AFSCs that have an upper limit on USAFA cadets
        cadet_lim_afsc_count = np.sum(np.sum(m.x[c, a] for c in gp['C^']['E'][a]) for a in gp['A^']['U_lim'])

        # USAFA upper limit constraint
        def USAFA_Limit(model):
            return usafa_cadet_lim_afsc_count <= gp['u_limit'] * cadet_lim_afsc_count
        m.usafa_limit_constraint = Constraint(rule=USAFA_Limit)

    if printing:
        print('Model built.')
    return m


def gp_model_solve(model, gp, solve_name="gurobi", max_time=None, printing=False):
    """
    This procedure solves Rebecca's model.
    :param gp: goal programming model parameters
    :param max_time: maximum time to solve
    :param model: the instantiated model
    :param solve_name: solver name
    :param printing: Whether or not to print something
    :return: solution vector
    """
    if printing:
        print('Solving R Model...')

    if solve_name == 'gurobi':
        solver = SolverFactory(solve_name, solver_io='python')
    else:  # assume cbc
        # solver = SolverFactory(solve_name, executable=paths['Solvers'] + solve_name + '.exe')
        solver = SolverFactory(solve_name)

    if max_time is not None:
        if solve_name == 'gurobi':
            # solver.solve(model, options={'TimeLimit': max_time, 'IntFeasTol': 0.05})
            solver.solve(model, options={'TimeLimit': max_time})
        else:  # assume cbc
            solver.options['seconds'] = max_time
            solver.solve(model)

    else:
        solver.solve(model)

    # Get solution
    N, M = len(gp['C']), len(gp['A'])
    solution = np.zeros(N)
    X = np.zeros((N, M))
    for c in gp['C']:
        for a in gp['A^']['E'][c]:
            X[c, a] = model.x[c, a].value
            if round(X[c, a]):
                solution[c] = int(a)

    if printing:
        print('Model solved.')

    return solution, X


def convert_parameters_to_lsp_model_inputs(parameters, value_parameters, metrics_1, metrics_2, delta, printing=False):
    """
    Takes the parameters, value parameters, and solutions metrics and then converts them to data used by the pyomo model
    :param delta: how much the value of solution 2 should exceed solution 1
    :param parameters: fixed cadet AFSC parameters
    :param value_parameters: value parameters
    :param metrics_1: optimal solution metrics under these value parameters (vector)
    :param metrics_2: some other solution metrics (vector)
    :param printing: Whether the procedure should print something
    :return: pyomo model data
    """
    if printing:
        print("Converting LSP information into pyomo model parameters...")

    N = parameters['N']
    M = parameters['M']
    O = value_parameters['O']

    data = {None: {
        'I': {None: np.arange(N)},
        'J': {None: np.arange(M)},
        'K': {None: np.arange(O)},
        'K_A': {j: value_parameters['K_A'][j] for j in range(M)},
        'afsc_objective_value_t': {(j, k): metrics_2['objective_value'][j][k]
                                   for j in range(M) for k in range(O)},
        'cadet_value_t': {i: metrics_2['cadet_value'][i] for i in range(N)},
        'afsc_objective_value_b': {(j, k): metrics_1['objective_value'][j][k]
                                   for j in range(M) for k in range(O)},
        'cadet_value_b': {i: metrics_1['cadet_value'][i] for i in range(N)},
        'afsc_objective_weight_b': {(j, k): value_parameters['objective_weight'][j][k]
                                    for j in range(M) for k in range(O)},
        'afsc_weight': {j: value_parameters['afsc_weight'][j] for j in range(M)},
        'afscs_overall_weight': {None: value_parameters['afscs_overall_weight']},
        'cadet_weight': {i: value_parameters['cadet_weight'][i] for i in range(N)},
        'cadets_overall_weight': {None: value_parameters['cadets_overall_weight']},
        'delta': {None: delta}
    }}

    return data


def lsp_model_build(printing=False):
    """
    This is the model for the Least Squares Procedure to conduct sensitivity analysis on the weights for two solutions
    :param printing: Whether the procedure should print something
    :return: pyomo model
    """
    if printing:
        print('Building Least Squares Procedure Model...')

    # Build Model
    model = AbstractModel()

    # Define sets
    model.I = Set(doc='Cadets')  # range of N
    model.J = Set(doc='AFSCs')  # range of M
    model.K = Set(doc='AFSC Objectives')  # range of M
    model.K_A = Set(model.J, doc="set of objectives specific to each AFSC")  # range of O

    # value parameters
    model.afsc_objective_value_t = Param(model.J, model.K, doc="solution t's value of objective k for afsc j")
    model.cadet_value_t = Param(model.I, doc="solution t's value for cadet i")
    model.afsc_objective_value_b = Param(model.J, model.K, doc="solution b's value of objective k for afsc j")
    model.cadet_value_b = Param(model.I, doc="solution b's value for cadet i")

    # weight parameters
    model.afsc_objective_weight_b = Param(model.J, model.K, doc="solution b's weight on objective k for afsc j")
    model.afsc_weight = Param(model.J, doc="weight on afsc j")
    model.afscs_overall_weight = Param(doc="weight on all of the afscs")
    model.cadet_weight = Param(model.I, doc="weight on cadet i")
    model.cadets_overall_weight = Param(doc="weight on all of the cadets")

    # delta
    model.delta = Param(doc="the amount by which the objective value for solution t should exceed that of solution b")

    # Variables
    model.afsc_objective_weight = Var(model.J, model.K, doc="new weight on objective k for afsc j",
                                      domain=PositiveReals)

    def afsc_objective_weights_rule(model, j):
        return sum(model.afsc_objective_weight[j, k] for k in model.K_A[j]) == 1

    model.con_afsc_objective_weights = Constraint(model.J, rule=afsc_objective_weights_rule)

    def delta_rule(model):
        return (model.afscs_overall_weight * sum(
            model.afsc_weight[j] * sum(
                model.afsc_objective_weight[j, k] * model.afsc_objective_value_t[j, k] for k in model.K_A[j])
            for j in model.J) + model.cadets_overall_weight * sum(
            model.cadet_weight[i] * model.cadet_value_t[i] for i in model.I)) - (model.afscs_overall_weight * sum(
            model.afsc_weight[j] * sum(
                model.afsc_objective_weight[j, k] * model.afsc_objective_value_b[j, k] for k in model.K_A[j])
            for j in model.J) + model.cadets_overall_weight * sum(
            model.cadet_weight[i] * model.cadet_value_b[i] for i in model.I)) == model.delta

    model.con_delta = Constraint(rule=delta_rule)

    # Objective Function
    def objective_rule(model):
        return sum(sum((model.afsc_objective_weight[j, k] -
                        model.afsc_objective_weight_b[j, k]) ** 2 for k in model.K_A[j]) for j in model.J)

    model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')

    return model


def x_to_solution_initialization(parameters, value_parameters, measures, values):
    """
    This procedure takes the values and measures of a solution, along with other model parameters, and then returns
    the value function variables used to initialize a VFT pyomo model. This is meant to
    initialize the exact VFT model with an approximate solution
    :param measures: AFSC objective measures
    :param values: AFSC objective values
    :param parameters: cadet/AFSC parameters
    :param value_parameters: value parameters
    :return: lam, y
    """

    # Set Definitions
    M = parameters['M']  # number of afscs
    O = value_parameters['O']  # number of objectives
    J = parameters['J']  # set of afscs
    K = range(O)  # set of objectives
    K_A = value_parameters['K_A']  # set of objectives for AFSC j

    # Value Function Parameters
    r = [[len(value_parameters['F_bp'][j][k]) for k in K] for j in J]  # number of breakpoints
    L = [[list(range(r[j][k])) for k in K] for j in J]  # set of breakpoints
    a = [[[value_parameters['F_bp'][j][k][l] for l in L[j][k]] for k in K] for j in J]  # breakpoints
    f = [[[value_parameters['F_v'][j][k][l] for l in L[j][k]] for k in K] for j in J]  # values of bps
    r = np.array(r)
    L = np.array(L)
    a = np.array(a)
    f = np.array(f)

    max_L = 0
    for j in J:
        for k in K_A[j]:
            if len(L[j][k]) > max_L:
                max_L = int(len(L[j][k]))
            # Add an extra breakpoint so we can capture more possible values
            last_a = a[j][k][r[j][k] - 1]
            last_f = f[j][k][r[j][k] - 1]
            a[j][k].append(last_a * 1000)  # arbitrarily large number
            f[j][k].append(last_f)
            L[j][k].append(r[j][k])
            r[j][k] += 1

    model = ConcreteModel()

    # Variables
    model.lam = Var(((j, k, l) for j in J for k in K_A[j] for l in L[j, k]))
    model.y = Var(((j, k, l) for j in J for k in K_A[j] for l in range(0, r[j, k] - 1)), within=Binary)

    # Objective Constraints
    model.measure_vf_constraints = ConstraintList()
    model.value_vf_constraints = ConstraintList()
    model.lambda_y_constraint1 = ConstraintList()
    model.lambda_y_constraint2 = ConstraintList()
    model.lambda_y_constraint3 = ConstraintList()
    model.y_constraint = ConstraintList()
    model.lambda_sum_constraint = ConstraintList()
    model.lambda_positive = ConstraintList()

    for j in J:
        for k in K_A[j]:

            model.measure_vf_constraints.add(expr=measures[j, k] == np.sum(  # Measure Constraint for Value Function
                a[j, k][l] * model.lam[j, k, l] for l in L[j, k]))
            model.value_vf_constraints.add(expr=values[j, k] == np.sum(  # Value Constraint for Value Function
                f[j, k][l] * model.lam[j, k, l] for l in L[j, k]))
            model.lambda_y_constraint1.add(expr=model.lam[j, k, 0] <= model.y[j, k, 0])
            model.lambda_y_constraint3.add(expr=model.lam[j, k, r[j, k] - 1] <= model.y[j, k, r[j, k] - 2])
            if r[j, k] > 2:
                for l in range(1, r[j, k] - 1, 1):
                    model.lambda_y_constraint2.add(expr=model.lam[j, k, l] <= model.y[j, k, l - 1] + model.y[j, k, l])
            model.y_constraint.add(expr=np.sum(model.y[j, k, l] for l in range(0, r[j, k] - 1, 1)) == 1)
            model.lambda_sum_constraint.add(expr=np.sum(model.lam[j, k, l] for l in L[j, k]) == 1)
            for l in L[j, k]:
                model.lambda_positive.add(expr=model.lam[j, k, l] >= 0)

    def objective_function(model):
        return 5  # arbitrary objective function just to get solution that meets the constraints

    model.objective = Objective(rule=objective_function, sense=maximize)
    solver = SolverFactory('cbc', executable='../Main Directory/Solvers/cbc.exe')
    solver.solve(model)

    # Load model variables
    lam = np.zeros([M, O, max_L + 1])
    y = np.zeros([M, O, max_L + 1]).astype(int)
    for j in J:
        for k in K_A[j]:
            for l in L[j][k]:
                lam[j, k, l] = model.lam[j, k, l].value
                if l < r[j][k] - 1:
                    y[j, k, l] = round(model.y[j, k, l].value)
                    # if y[j, k, l] != 1 and y[j, k, l] != 0:
                    #     print(y[j, k, l])

    return lam, y
