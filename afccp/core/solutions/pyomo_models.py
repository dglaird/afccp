# Import libraries
import time
import numpy as np
import logging
import warnings

from afccp.core.globals import *

# Ignore warnings
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


def solve_original_pyomo_model(instance, printing=False):
    """
    Converts the parameters and value parameters to the pyomo data structure
    :param instance: problem instance object
    :param printing: Whether the procedure should print something
    :return: pyomo data
    """
    if printing:
        print("Building original model...")

    # Shorthand
    p = instance.parameters
    vp = instance.value_parameters
    mdl_p = instance.mdl_p

    # Utility Matrix
    c = np.zeros([p['N'], p['M']])
    for i in range(p['N']):
        for j in range(p['M']):

            # If AFSC j is a preference for cadet i
            if p['utility'][i, j] > 0:

                if p['mandatory'][i, j] == 1:
                    c[i, j] = 10 * p['merit'][i] * p['utility'][i, j] + 250
                elif p['desired'][i, j] == 1:
                    c[i, j] = 10 * p['merit'][i] * p['utility'][i, j] + 150
                elif p['permitted'][i, j] == 1:
                    c[i, j] = 10 * p['merit'][i] * p['utility'][i, j]
                else:
                    c[i, j] = -50000

            # If it is not a preference for cadet i
            else:

                if p['mandatory'][i, j] == 1:
                    c[i, j] = 100 * p['merit'][i]
                elif p['desired'][i, j] == 1:
                    c[i, j] = 50 * p['merit'][i]
                elif p['permitted'][i, j] == 1:
                    c[i, j] = 0
                else:
                    c[i, j] = -50000

    # Build Model
    m = ConcreteModel()

    # ___________________________________VARIABLE DEFINITION_________________________________
    m.x = Var(((i, j) for i in p['I'] for j in p['J^E'][i]), within=Binary)

    # Fixing variables if necessary
    for i, afsc in enumerate(p["assigned"]):
        j = np.where(p["afsc_vector"] == afsc)[0]  # AFSC index

        # Check if the cadet is actually assigned an AFSC already (it's not blank)
        if len(j) != 0:
            j = j[0]  # Actual index

            # Check if the cadet is assigned to an AFSC they're not eligible for
            if j not in p["J^E"][i]:
                raise ValueError("Cadet " + str(i) + " assigned to '" + afsc + "' but is not eligible for it. "
                                                                               "Adjust the qualification matrix!")

            # Fix the variable
            m.x[i, j].fix(1)

    # ___________________________________OBJECTIVE FUNCTION__________________________________
    def objective_function(m):
        return np.sum(np.sum(c[i, j] * m.x[i, j] for j in p["J^E"][i]) for i in p["I"])

    m.objective = Objective(rule=objective_function, sense=maximize)

    # ________________________________________CONSTRAINTS_____________________________________
    pass

    # Cadets receive one and only one AFSC (Ineligibility constraint is always met as a result of the indexed sets)
    m.one_afsc_constraints = ConstraintList()
    for i in p['I']:
        m.one_afsc_constraints.add(expr=np.sum(m.x[i, j] for j in p['J^E'][i]) == 1)

    # 5% cap on total percentage of USAFA cadets allowed into certain AFSCs
    if vp["J^USAFA"] is not None:
        # This is a pretty arbitrary constraint and will only be used for real class years
        real_n = 960  # Total number of USAFA cadets (NonRated, Rated, and SF)
        cap = 0.05 * real_n

        # USAFA 5% Cap Constraint
        def usafa_afscs_rule(m):
            return np.sum(np.sum(m.x[i, j] for i in p['I^D']['USAFA Proportion'][j]) for j in vp["J^USAFA"]) <= cap

        m.usafa_afscs_constraint = Constraint(rule=usafa_afscs_rule)

    # Loop through each AFSC and add constraints if they need it
    m.measure_constraints = ConstraintList()
    for j in p["J"]:

        # Get count variables for this AFSC
        count = np.sum(m.x[i, j] for i in p['I^E'][j])

        # Loop through the objectives/constraints we care about
        for objective in ["Combined Quota", "Merit", "USAFA Proportion", "Mandatory"]:
            k = np.where(vp["objectives"] == objective)[0]

            # If the constraint is turned on
            if vp["constraint_type"][j, k] == 3 or vp['constraint_type'][j, k] == 4:

                # Get the lower and upper bounds
                try:
                    value_list = vp['objective_value_min'][j, k].split(",")
                except:
                    value_list = vp['objective_value_min'][j, k][0].split(",")
                min_value = float(value_list[0].strip())
                max_value = float(value_list[1].strip())

                # Calculate Objective Measure
                if objective in vp['K^D']:
                    numerator = np.sum(m.x[i, j] for i in p['I^D'][objective][j])
                elif objective == "Merit":
                    numerator = np.sum(p['merit'][i] * m.x[i, j] for i in p['I^E'][j])
                else:  # Combined Quota
                    measure_jk = count

                # Add the right kind of constraint
                if objective == 'Combined Quota':
                    m.measure_constraints.add(expr=measure_jk >= min_value)
                    m.measure_constraints.add(expr=measure_jk <= max_value)
                else:

                    # Constrained Approximate Measure
                    if vp['constraint_type'][j, k] == 3:
                        m.measure_constraints.add(expr=numerator - min_value * p['quota'][j] >= 0)
                        m.measure_constraints.add(expr=numerator - max_value * p['quota'][j] <= 0)

                    # Constrained Exact Measure  (type = 4)
                    else:
                        m.measure_constraints.add(expr=numerator - min_value * count >= 0)
                        m.measure_constraints.add(expr=numerator - max_value * count <= 0)

    if printing:
        print("Done. Solving original model...")

    # Solve the model
    model = solve_pyomo_model(instance, model)

    # Initialize solution
    solution = np.zeros(p['N'])

    # Create solution X Matrix
    x = np.zeros([p['N'], p['M']])
    for i in p['I']:
        found = False
        for j in p['J^E'][i]:
            x[i, j] = model.x[i, j].value
            try:
                if round(x[i, j]):
                    solution[i] = int(j)
                    found = True
            except:
                raise ValueError("Solution didn't come out right, likely model is infeasible.")

        # For some reason we may not have assigned a cadet to an AFSC in which case we just give them to one
        # they're eligible for and want
        if not found:
            if len(p["J^P"][i]) != 0:
                solution[i] = int(p["J^P"][i][0])  # Try to give them an AFSC they wanted
            else:
                solution[i] = int(p["J^E"][i][0])  # Just give them an AFSC they're eligible for

            afsc = p["afsc_vector"][int(solution[i])]

            if printing:
                print("Cadet " + str(i) + " was not assigned by the model for some reason. We assigned them to", afsc)
    return solution


def vft_model_build(instance, printing=False):
    """
    Builds the VFT optimization model using pyomo
    :param instance: problem instance object
    :param printing: Whether the procedure should print something
    :return: pyomo model as object
    """
    if printing:
        print('Building VFT Model...')

    # Build Model
    m = ConcreteModel()

    # Shorthand
    p = instance.parameters
    vp = instance.value_parameters

    # _________________________________PARAMETER ADJUSTMENTS_________________________________
    pass

    # Value Function Parameters
    r = [[len(vp['a'][j][k]) for k in vp['K']] for j in p['J']]  # number of breakpoints (bps)
    L = [[list(range(r[j][k])) for k in vp['K']] for j in p['J']]  # set of bps
    a = [[[vp['a'][j][k][l] for l in vp['L'][j][k]] for k in vp['K']] for j in p['J']]  # measures of bps
    f = [[[vp['f^hat'][j][k][l] for l in vp['L'][j][k]] for k in vp['K']] for j in p['J']]  # values of bps

    # Initialize AFSC objective measure constraint ranges
    objective_min_value = np.zeros([p['M'], vp['O']])
    objective_max_value = np.zeros([p['M'], vp['O']])

    # Loop through each AFSC
    for j in p['J']:

        # Loop through each objective for each AFSC
        for k in vp['K^A'][j]:

            # We need to add an extra breakpoint to effectively extend the domain
            if instance.mdl_p["add_breakpoints"]:

                # We add an extra breakpoint far along the x-axis with the same y value as the previous one
                last_a = a[j][k][r[j][k] - 1]
                last_f = f[j][k][r[j][k] - 1]
                a[j][k].append(last_a * 2000)  # arbitrarily large number
                f[j][k].append(last_f)  # same y-value as previous one
                L[j][k].append(r[j][k])  # add the new breakpoint index
                r[j][k] += 1  # add a breakpoint to the number of breakpoints

            # Retrieve minimum values based on constraint type (approximate/exact and value/measure)
            if vp['constraint_type'][j, k] == 1 or vp['constraint_type'][j, k] == 2:

                # These are "value" constraints and so only a minimum value is needed
                objective_min_value[j, k] = float(vp['objective_value_min'][j, k])
            elif vp['constraint_type'][j, k] == 3 or vp['constraint_type'][j, k] == 4:

                # These are "measure" constraints and so a range is needed
                value_list = vp['objective_value_min'][j, k].split(",")
                objective_min_value[j, k] = float(value_list[0].strip())
                objective_max_value[j, k] = float(value_list[1].strip())

    # Convert to numpy arrays of lists
    r = np.array(r)
    L = np.array(L)
    a = np.array(a)
    f = np.array(f)

    # _________________________________VARIABLE DEFINITIONS_________________________________
    pass

    # If we don't initialize variables
    if instance.mdl_p["warm_start"] is None:

        # If we don't use a warm-start, we don't initialize starting values for the variables
        m.x = Var(((i, j) for i in p['I'] for j in p['J^E'][i]), within=Binary)  # main decision variable (x)
        m.f_value = Var(((j, k) for j in p['J'] for k in vp['K^A'][j]), within=NonNegativeReals)  # objective value
        m.lam = Var(((j, k, l) for j in p['J'] for k in vp['K^A'][j] for l in L[j, k]),
                    within=NonNegativeReals, bounds=(0, 1))  # Lambda and y variables for value functions
        m.y = Var(((j, k, l) for j in p['J'] for k in vp['K^A'][j] for l in range(0, r[j, k] - 1)), within=Binary)

    # If we do want to initialize the variables. Probably initializing the exact model using the approximate solution
    else:

        # x: 1 if assign cadet i to AFSC j; 0 otherwise
        m.x = Var(((i, j) for i in p['I'] for j in p['J^E'][i]), within=Binary)
        for i in p['I']:
            for j in p['J^E'][i]:
                m.x[i, j] = round(instance.mdl_p["warm_start"]['X'][i, j])

        # f^hat: value for AFSC j objective k
        m.f_value = Var(((j, k) for j in p['J'] for k in vp['K^A'][j]), within=NonNegativeReals)
        for j in p['J']:
            for k in vp['K^A'][j]:
                m.f_value[j, k] = instance.mdl_p["warm_start"]['F_X'][j, k]

        # lambda: % between breakpoint l and l + 1 that the measure for AFSC j objective k "has yet to travel"
        m.lam = Var(((j, k, l) for j in p['J'] for k in vp['K^A'][j] for l in L[j, k]), within=NonNegativeReals,
                    bounds=(0, 1))
        for j in p['J']:
            for k in vp['K^A'][j]:
                for l in L[j, k]:
                    m.lam[j, k, l] = instance.mdl_p["warm_start"]['lam'][j, k, l]

        # y: 1 if the objective measure for AFSC j objective k is along line segment between breakpoints l and l + 1
        m.y = Var(((j, k, l) for j in p['J'] for k in vp['K^A'][j] for l in range(0, r[j, k] - 1)), within=Binary)
        for j in p['J']:
            for k in vp['K^A'][j]:
                for l in range(r[j, k] - 1):
                    m.y[j, k, l] = instance.mdl_p["warm_start"]['y'][j, k, l]

    # Fixing variables if necessary
    for i, afsc in enumerate(p["assigned"]):
        j = np.where(p["afsc_vector"] == afsc)[0]  # AFSC index

        # Check if the cadet is actually assigned an AFSC already (it's not blank)
        if len(j) != 0:
            j = j[0]  # Actual index

            # Check if the cadet is assigned to an AFSC they're not eligible for
            if j not in p["J^E"][i]:
                raise ValueError("Cadet " + str(i) + " assigned to '" + afsc + "' but is not eligible for it. "
                                                                               "Adjust the qualification matrix!")

            # Fix the variable
            m.x[i, j].fix(1)

    # _________________________________OBJECTIVE FUNCTION_________________________________
    pass

    # Max Z!
    def objective_function(m):
        return vp['afscs_overall_weight'] * np.sum(vp['afsc_weight'][j] * np.sum(
            vp['objective_weight'][j, k] * m.f_value[j, k] for k in vp['K^A'][j]) for j in p['J']) + \
               vp['cadets_overall_weight'] * np.sum(vp['cadet_weight'][i] * np.sum(
            p['utility'][i, j] * m.x[i, j] for j in p['J^E'][i]) for i in p['I'])

    m.objective = Objective(rule=objective_function, sense=maximize)

    # ____________________________________CONSTRAINTS_____________________________________
    pass

    # Cadets receive one and only one AFSC (Ineligibility constraint is always met as a result of the indexed sets)
    m.one_afsc_constraints = ConstraintList()
    for i in p['I']:
        m.one_afsc_constraints.add(expr=np.sum(m.x[i, j] for j in p['J^E'][i]) == 1)

    # 5% cap on total percentage of USAFA cadets allowed into certain AFSCs
    if vp["J^USAFA"] is not None:
        cap = 0.05 * instance.mdl_p["real_usafa_n"]  # Total number of USAFA cadets (Rated, SF, and NonRated)

        # USAFA 5% Cap Constraint
        def usafa_afscs_rule(m):
            return np.sum(np.sum(m.x[i, j] for i in p['I^D']['USAFA Proportion'][j]) for j in vp["J^USAFA"]) <= cap
        m.usafa_afscs_constraint = Constraint(rule=usafa_afscs_rule)

    # Value Function Constraints: Linking main methodology with value function methodology
    m.measure_vf_constraints = ConstraintList()  # 20a in Thesis
    m.value_vf_constraints = ConstraintList()  # 20b in Thesis

    # Value Function Constraints: Functional constraints enforcing the relationship above
    m.lambda_y_constraint1 = ConstraintList()  # 20c in Thesis
    m.lambda_y_constraint2 = ConstraintList()  # 20d in Thesis
    m.lambda_y_constraint3 = ConstraintList()  # 20e in Thesis
    m.y_sum_constraint = ConstraintList()  # 20f in Thesis
    m.lambda_sum_constraint = ConstraintList()  # 20g in Thesis
    m.lambda_positive = ConstraintList()  # Lambda domain (20h)
    m.f_value_positive = ConstraintList()  # AFSC objective value domain

    # Cadet/AFSC Value Constraints (Optional decision-maker constraints)
    m.min_afsc_value_constraints = ConstraintList()
    m.min_cadet_value_constraints = ConstraintList()

    # AFSC Objective Measure/Value Constraints (Optional decision-maker constraints)
    m.measure_constraints = ConstraintList()
    m.value_constraints = ConstraintList()

    # Loop through all AFSCs
    for j in p['J']:

        # Get count variables for this AFSC
        count = np.sum(m.x[i, j] for i in p['I^E'][j])
        if 'usafa' in p:

            # Number of USAFA cadets assigned to the AFSC
            usafa_count = np.sum(m.x[i, j] for i in p['I^D']['USAFA Proportion'][j])

        # Are we using approximate measures or not
        if instance.mdl_p["approximate"]:
            num_cadets = p['quota'][j]  # Approximate
        else:
            num_cadets = count  # Exact

        # Loop through all objectives for this AFSC
        for k in vp['K^A'][j]:

            # Get the right objective measure calculation
            objective = vp['objectives'][k]

            # If it's a demographic objective, we sum over the cadets with that demographic
            if objective in vp['K^D']:
                numerator = np.sum(m.x[i, j] for i in p['I^D'][objective][j])
                measure_jk = numerator / num_cadets
            elif objective == "Merit":
                numerator = np.sum(p['merit'][i] * m.x[i, j] for i in p['I^E'][j])
                measure_jk = numerator / num_cadets
            elif objective == "Combined Quota":
                measure_jk = count
            elif objective == "USAFA Quota":
                measure_jk = usafa_count
            elif objective == "ROTC Quota":
                measure_jk = count - usafa_count
            elif objective == "Norm Score":

                if instance.mdl_p["approximate"]:
                    best_range = range(num_cadets)
                    best_sum = np.sum(c for c in best_range)
                    worst_range = range(p["num_eligible"][j] - num_cadets, p["num_eligible"][j])
                    worst_sum = np.sum(c for c in worst_range)
                    achieved_sum = np.sum(p["a_pref_matrix"][i, j] * m.x[i, j] for i in p["I^E"][j])
                    measure_jk = 1 - (achieved_sum - best_sum) / (worst_sum - best_sum)
                else:
                    numerator = np.sum(p['afsc_utility'][i, j] * m.x[i, j] for i in p['I^E'][j])
                    measure_jk = numerator / num_cadets

            else:  # Utility
                numerator = np.sum(p['utility'][i, j] * m.x[i, j] for i in p['I^E'][j])
                measure_jk = numerator / num_cadets

            # Add Linear Value Function Constraints
            m.measure_vf_constraints.add(expr=measure_jk == np.sum(  # Measure Constraint for Value Function
                a[j, k][l] * m.lam[j, k, l] for l in L[j, k]))
            m.value_vf_constraints.add(expr=m.f_value[j, k] == np.sum(  # Value Constraint for Value Function
                f[j, k][l] * m.lam[j, k, l] for l in L[j, k]))

            # Lambda .. y constraints
            m.lambda_y_constraint1.add(expr=m.lam[j, k, 0] <= m.y[j, k, 0])
            if r[j, k] > 2:
                for l in range(1, r[j, k] - 1):
                    m.lambda_y_constraint2.add(expr=m.lam[j, k, l] <= m.y[j, k, l - 1] + m.y[j, k, l])
            m.lambda_y_constraint3.add(expr=m.lam[j, k, r[j, k] - 1] <= m.y[j, k, r[j, k] - 2])

            # Y sum to 1 constraint
            m.y_sum_constraint.add(expr=np.sum(m.y[j, k, l] for l in range(0, r[j, k] - 1)) == 1)

            # Lambda sum to 1 constraint
            m.lambda_sum_constraint.add(expr=np.sum(m.lam[j, k, l] for l in L[j, k]) == 1)

            # Lambda .. value positive constraint
            for l in L[j, k]:
                m.lambda_positive.add(expr=m.lam[j, k, l] >= 0)
            m.f_value_positive.add(expr=m.f_value[j, k] >= 0)

            # Add Min Value/Measure Constraints
            if k in vp['K^C'][j]:  # (1/2 constrain value, 3 constrains approximate measure, 4 constrains exact measure)

                # Constrained Value (I decided against this for AFSC objectives and just went with measure constraints)
                if vp['constraint_type'][j, k] == 1 or vp['constraint_type'][j, k] == 2:

                    # The formulation only lists "objective_min, objective_max" since I no longer want value constraints
                    m.value_constraints.add(expr=m.f_value[j, k] >= objective_min_value[j, k])

                # Constrained PGL/Approximate Measure
                elif vp['constraint_type'][j, k] == 3:
                    if objective in ['Combined Quota', 'USAFA Quota', 'ROTC Quota']:
                        m.measure_constraints.add(expr=measure_jk >= objective_min_value[j, k])
                        m.measure_constraints.add(expr=measure_jk <= objective_max_value[j, k])
                    else:

                        if "pgl" in p:  # PGL should be more "forgiving" as a constraint
                            m.measure_constraints.add(expr=numerator - objective_min_value[j, k] * p['pgl'][j] >= 0)
                            m.measure_constraints.add(expr=numerator - objective_max_value[j, k] * p['pgl'][j] <= 0)
                        else:
                            m.measure_constraints.add(expr=numerator - objective_min_value[j, k] * p['quota'][j] >= 0)
                            m.measure_constraints.add(expr=numerator - objective_max_value[j, k] * p['quota'][j] <= 0)

                # Constrained Exact Measure  (type = 4)
                else:
                    if objective in ['Combined Quota', 'USAFA Quota', 'ROTC Quota']:
                        m.measure_constraints.add(expr=measure_jk >= objective_min_value[j, k])
                        m.measure_constraints.add(expr=measure_jk <= objective_max_value[j, k])
                    else:
                        m.measure_constraints.add(expr=numerator - objective_min_value[j, k] * count >= 0)
                        m.measure_constraints.add(expr=numerator - objective_max_value[j, k] * count <= 0)

        # AFSC value constraint
        if vp['afsc_value_min'][j] != 0:
            m.min_afsc_value_constraints.add(expr=np.sum(
                vp['objective_weight'][j, k] * m.f_value[j, k] for k in vp['K^A'][j]) >= vp['afsc_value_min'][j])

    # Cadet value constraint
    for i in p['I']:
        if vp['cadet_value_min'][i] != 0:
            m.min_cadet_value_constraints.add(expr=np.sum(
                p['utility'][i, j] * m.x[i, j] for j in p['J^E'][i]) >= vp['cadet_value_min'][i])

    # AFSCs Overall Min Value
    def afsc_min_value_constraint(m):
        return vp['afscs_overall_value_min'] <= np.sum(vp['afsc_weight'][j] * np.sum(
            vp['objective_weight'][j, k] * m.f_value[j, k] for k in vp['K^A'][j]) for j in p['J'])

    if vp['afscs_overall_value_min'] != 0:
        m.afsc_min_value_constraint = Constraint(rule=afsc_min_value_constraint)

    # Cadets Overall Min Value
    def cadet_min_value_constraint(m):
        return vp['cadets_overall_value_min'] <= np.sum(vp['cadet_weight'][i] * np.sum(
            p['utility'][i, j] * m.x[i, j] for j in p['J^E'][i]) for i in p['I'])

    if vp['cadets_overall_value_min'] != 0:
        m.cadet_min_value_constraint = Constraint(rule=cadet_min_value_constraint)

    return m


def vft_model_solve(instance, model, printing=False):
    """
    Solve VFT Model
    :param instance: problem instance to solve
    :param model: pyomo model
    :param printing: if we should print something
    :return: solution
    """

    # Shorthand
    p = instance.parameters
    vp = instance.value_parameters
    mdl_p = instance.mdl_p

    # Print what we're solving
    if printing:
        if mdl_p["approximate"]:
            model_str = 'Approximate'
        else:
            model_str = 'Exact'
        print('Solving ' + model_str + ' VFT Model instance with solver ' + mdl_p["solver_name"] + '...')

    # Start Time
    if mdl_p["time_eval"]:
        start_time = time.perf_counter()

    # Solve the model
    model = solve_pyomo_model(instance, model)

    # Stop Time
    if mdl_p["time_eval"]:
        solve_time = round(time.perf_counter() - start_time, 2)

    # Initialize solution
    solution = np.zeros(p['N'])

    # Create solution X Matrix
    x = np.zeros([p['N'], p['M']])
    for i in p['I']:
        found = False
        for j in p['J^E'][i]:
            x[i, j] = model.x[i, j].value
            try:
                if round(x[i, j]):
                    solution[i] = int(j)
                    found = True
            except:
                raise ValueError("Solution didn't come out right, likely model is infeasible.")

        # For some reason we may not have assigned a cadet to an AFSC in which case we just give them to one
        # they're eligible for and want  (happens usually to only 1-3 people through VFT model)
        if not found:

            # Try to give the cadet their top choice AFSC for which they're eligible
            if len(p["J^P"][i]) != 0:
                max_util = 0
                max_j = 0
                for j in p["J^P"][i]:
                    if p["utility"][i, j] >= max_util:
                        max_j = j
                        max_util = max_util
                solution[i] = int(max_j)

            # If we don't have any eligible preferences from the cadet, they get Needs of the Air Force
            else:

                if len(p["J^E"][i]) >= 2:
                    solution[i] = int(p["J^E"][i][1])
                else:
                    solution[i] = int(p["J^E"][i][0])

            afsc = p["afsc_vector"][int(solution[i])]

            if printing:
                print("Cadet " + str(i) + " was not assigned by the model for some reason. We assigned them to", afsc)
    if mdl_p["report"]:
        obj = model.objective()
        if printing:
            if mdl_p["approximate"]:
                print("Approximate Pyomo Model Objective Value: " + str(round(obj, 4)))
            else:
                print("Exact Pyomo Model Objective Value: " + str(round(obj, 4)))

        # Initialize measure/value matrices
        measure = np.zeros([p['M'], vp['O']])
        value = np.zeros([p['M'], vp['O']])

        # Loop through all AFSCs to get their values
        for j in p['J']:

            # Get variables for this AFSC
            count = np.sum(x[i, j] for i in p['I^E'][j])
            if 'usafa' in p:
                usafa_count = np.sum(x[i, j] for i in p['I^D']['USAFA Proportion'][j])

            # Are we using approximate measures or not
            if mdl_p["approximate"]:
                num_cadets = p['quota'][j]
            else:
                num_cadets = count

            # Loop through all objectives for this AFSC
            for k in vp['K^A'][j]:
                objective = vp['objectives'][k]

                # Get the right measure calculation
                if objective in vp['K^D']:
                    numerator = np.sum(x[i, j] for i in p['I^D'][objective][j])
                    measure[j, k] = numerator / num_cadets
                elif objective == "Merit":
                    numerator = np.sum(p['merit'][i] * x[i, j] for i in p['I^E'][j])
                    measure[j, k] = numerator / num_cadets
                elif objective == "Combined Quota":
                    measure[j, k] = count
                elif objective == "USAFA Quota":
                    measure[j, k] = usafa_count
                elif objective == "ROTC Quota":
                    measure[j, k] = count - usafa_count
                else:  # Utility
                    numerator = np.sum(p['utility'][i, j] * x[i, j] for i in p['I^E'][j])
                    measure[j, k] = numerator / num_cadets

                # Value of measure
                value[j, k] = model.f_value[j, k].value

        if mdl_p["time_eval"]:
            return solution, x, measure, value, obj, solve_time
        else:
            return solution, x, measure, value, obj
    else:
        if mdl_p["time_eval"]:
            return solution, solve_time
        else:
            return solution


def solve_pyomo_model(instance, model):
    """
    Simple function that calls the pyomo solver using the specified model and max_time
    """

    # Shorthand
    mdl_p = instance.mdl_p

    # Determine how the solver is called
    if mdl_p["executable"] is None:
        if mdl_p["provide_executable"]:
            if mdl_p["exe_extension"]:
                mdl_p["executable"] = paths['solvers'] + mdl_p["solver_name"] + '.exe'
            else:
                mdl_p["executable"] = paths['solvers'] + mdl_p["solver_name"]
    else:
        mdl_p["provide_executable"] = True

    # Get correct solver
    if mdl_p["provide_executable"]:
        if mdl_p["solver_name"] == 'gurobi':
            solver = SolverFactory(mdl_p["solver_name"], solver_io='python', executable=mdl_p["executable"])
        else:
            solver = SolverFactory(mdl_p["solver_name"], executable=mdl_p["executable"])
    else:
        if mdl_p["solver_name"] == 'gurobi':
            solver = SolverFactory(mdl_p["solver_name"], solver_io='python')
        else:
            solver = SolverFactory(mdl_p["solver_name"])

    # Solve Model
    if mdl_p["max_time"] is not None:
        if mdl_p["solver_name"] == 'mindtpy':
            solver.solve(model, time_limit=mdl_p["max_time"],
                         mip_solver='cplex_persistent', nlp_solver='ipopt')
        elif mdl_p["solver_name"] == 'gurobi':
            solver.solve(model, options={'TimeLimit': mdl_p["max_time"], 'IntFeasTol': 0.05})
        elif mdl_p["solver_name"] == 'ipopt':
            solver.options['max_cpu_time'] = mdl_p["max_time"]
            solver.solve(model)
        elif mdl_p["solver_name"] == 'cbc':
            solver.options['seconds'] = mdl_p["max_time"]
            solver.solve(model)
        elif mdl_p["solver_name"] == 'baron':
            solver.solve(model, options={'MaxTime': mdl_p["max_time"]})
        else:
            solver.solve(model)
    else:
        if mdl_p["solver_name"] == 'mindtpy':
            solver.solve(model, mip_solver='cplex_persistent', nlp_solver='ipopt')
        else:
            solver.solve(model)

    return model


def gp_model_build(instance, printing=False):
    """
    This is Rebecca's model. We've incorporated her parameters and are building that model
    :param instance: problem instance to solve
    :param printing: Whether to print something
    :return: pyomo model
    """
    if printing:
        print('Building GP Model...')

    # Shorthand
    gp = instance.gp_parameters
    mdl_p = instance.mdl_p

    # Create model
    m = ConcreteModel()

    # ___________________________________VARIABLE DEFINITIONS_________________________________
    m.x = Var(((c, a) for c in gp['C'] for a in gp['A^']['E'][c]), within=Binary)

    # Amount by which the constraint is not met
    m.Y = Var(((con, a) for con in gp['con'] for a in gp['A^'][con]), within=NonNegativeReals)

    # Amount by which the constraint is exceeded
    m.Z = Var(((con, a) for con in gp['con'] for a in gp['A^'][con]), within=NonNegativeReals)

    # Binary variable indicating if Y is used (1) or if Z is used (0)
    m.alpha = Var(((con, a) for con in gp['con'] for a in gp['A^'][con]), within=Binary)

    # ___________________________________OBJECTIVE FUNCTION___________________________________
    def main_objective_function(m):
        return np.sum(  # Sum across each constraint
            np.sum(  # Sum across each AFSC with that constraint

                # Calculate penalties and rewards for each necessary AFSC
                gp['lam^'][con] * m.Z[con, a] - gp['mu^'][con] * m.Y[con, a] for a in gp['A^'][con]) for con in
            gp['con']) + gp['lam^']['S'] * np.sum(  # Sum across every cadet
            np.sum(  # Sum across each AFSC that the cadet is both eligible for and has placed a preference on

                # Calculate utility that the cadet received  (for each preferred AFSC for each constraint)
                gp['utility'][c, a] * m.x[c, a] for a in gp['A^']['W^E'][c]) for c in gp['C'])

    def penalty_objective_function(m):
        return np.sum(m.Y[mdl_p["con_term"], a] for a in gp['A^'][mdl_p["con_term"]])

    def reward_objective_function(m):
        if mdl_p["con_term"] == 'S':
            return np.sum(np.sum(gp['utility'][c, a] * m.x[c, a] for a in gp['A^']['W^E'][c]) for c in gp['C'])
        else:
            return np.sum(m.Z[mdl_p["con_term"], a] for a in gp['A^'][mdl_p["con_term"]])

    # Define model objective function
    if mdl_p["con_term"] is not None:  # Reward/Penalty specific objective function to get raw rewards/penalties
        if mdl_p["get_reward"]:
            m.objective = Objective(rule=reward_objective_function, sense=maximize)
        else:
            m.objective = Objective(rule=penalty_objective_function, sense=maximize)
    else:  # Regular objective function
        m.objective = Objective(rule=main_objective_function, sense=maximize)

    # ___________________________________CONSTRAINTS______________________________________
    pass

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


def gp_model_solve(instance, model, printing=False):
    """
    This procedure solves Rebecca's model.
    :param instance: problem instance to solve
    :param model: the instantiated model
    :param printing: Whether to print something
    :return: solution vector
    """
    if printing:
        print('Solving GP Model...')

    # Shorthand
    mdl_p = instance.mdl_p

    # Solve the model
    model = solve_pyomo_model(instance, model)

    if mdl_p["con_term"] is not None:
        return model.objective()
    else:

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
        'K_A': {j: value_parameters['K^A'][j] for j in range(M)},
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


def x_to_solution_initialization(parameters, value_parameters, measures, values, solver_name="cbc"):
    """
    This procedure takes the values and measures of a solution, along with other model parameters, and then returns
    the value function variables used to initialize a VFT pyomo model. This is meant to
    initialize the exact VFT model with an approximate solution
    :param solver_name: name of solver
    :param executable: optional executable path
    :param provide_executable: if we want to use the "solver" folder for an executable
    :param measures: AFSC objective measures
    :param values: AFSC objective values
    :param parameters: cadet/AFSC parameters
    :param value_parameters: value parameters
    :return: lam, y
    """

    # Shorthand
    p = parameters
    vp = value_parameters

    # Set Definitions
    M = parameters['M']  # number of afscs
    O = value_parameters['O']  # number of objectives
    J = parameters['J']  # set of afscs
    K = range(O)  # set of objectives

    # Value Function Parameters
    r = [[len(value_parameters['a'][j][k]) for k in vp['K']] for j in J]  # number of breakpoints
    L = [[list(range(r[j][k])) for k in K] for j in J]  # set of breakpoints
    a = [[[value_parameters['a'][j][k][l] for l in L[j][k]] for k in K] for j in J]  # breakpoints
    f = [[[value_parameters['f^hat'][j][k][l] for l in L[j][k]] for k in K] for j in J]  # values of bps
    r = np.array(r)
    L = np.array(L)
    a = np.array(a)
    f = np.array(f)

    max_L = 0
    for j in J:
        for k in vp['K^A'][j]:
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
    model.lam = Var(((j, k, l) for j in J for k in vp['K^A'][j] for l in L[j, k]))
    model.y = Var(((j, k, l) for j in J for k in vp['K^A'][j] for l in range(0, r[j, k] - 1)), within=Binary)

    # Objective Constraints
    model.measure_vf_constraints = ConstraintList()
    model.value_vf_constraints = ConstraintList()
    model.lambda_y_constraint1 = ConstraintList()
    model.lambda_y_constraint2 = ConstraintList()
    model.lambda_y_constraint3 = ConstraintList()
    model.y_sum_constraint = ConstraintList()
    model.lambda_sum_constraint = ConstraintList()
    model.lambda_positive = ConstraintList()

    for j in J:
        for k in vp['K^A'][j]:

            model.measure_vf_constraints.add(expr=measures[j, k] == np.sum(  # Measure Constraint for Value Function
                a[j, k][l] * model.lam[j, k, l] for l in L[j, k]))
            model.value_vf_constraints.add(expr=values[j, k] == np.sum(  # Value Constraint for Value Function
                f[j, k][l] * model.lam[j, k, l] for l in L[j, k]))
            model.lambda_y_constraint1.add(expr=model.lam[j, k, 0] <= model.y[j, k, 0])
            model.lambda_y_constraint3.add(expr=model.lam[j, k, r[j, k] - 1] <= model.y[j, k, r[j, k] - 2])
            if r[j, k] > 2:
                for l in range(1, r[j, k] - 1, 1):
                    model.lambda_y_constraint2.add(expr=model.lam[j, k, l] <= model.y[j, k, l - 1] + model.y[j, k, l])
            model.y_sum_constraint.add(expr=np.sum(model.y[j, k, l] for l in range(0, r[j, k] - 1, 1)) == 1)
            model.lambda_sum_constraint.add(expr=np.sum(model.lam[j, k, l] for l in L[j, k]) == 1)
            for l in L[j, k]:
                model.lambda_positive.add(expr=model.lam[j, k, l] >= 0)

    def objective_function(model):
        return 5  # arbitrary objective function just to get solution that meets the constraints

    model.objective = Objective(rule=objective_function, sense=maximize)
    model = solve_pyomo_model(model, solver_name)

    # Load model variables
    lam = np.zeros([M, O, max_L + 1])
    y = np.zeros([M, O, max_L + 1]).astype(int)
    for j in p['J']:
        for k in vp['K^A'][j]:
            for l in L[j][k]:
                lam[j, k, l] = model.lam[j, k, l].value
                if l < r[j][k] - 1:
                    y[j, k, l] = round(model.y[j, k, l].value)
    return lam, y
