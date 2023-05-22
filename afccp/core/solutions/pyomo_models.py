# Import libraries
import time
import numpy as np
import logging
import warnings

import afccp.core.globals
from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.util.infeasible import log_active_constraints

# Ignore warnings
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


def original_model_build(instance, printing=False):
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

    # Utility/Cost Matrix
    c = np.zeros([p['N'], p['M']])
    for i in p['I']:
        for j in p['J^E'][i]:  # Only looping through AFSCs that the cadet is eligible for

            # If AFSC j is a preference for cadet i
            if p['utility'][i, j] > 0:

                if p['mandatory'][i, j] == 1:
                    c[i, j] = 10 * p['merit'][i] * p['utility'][i, j] + 250
                elif p['desired'][i, j] == 1:
                    c[i, j] = 10 * p['merit'][i] * p['utility'][i, j] + 150
                else:  # Permitted, though it could also be an "exception"
                    c[i, j] = 10 * p['merit'][i] * p['utility'][i, j]

            # If it is not a preference for cadet i
            else:

                if p['mandatory'][i, j] == 1:
                    c[i, j] = 100 * p['merit'][i]
                elif p['desired'][i, j] == 1:
                    c[i, j] = 50 * p['merit'][i]
                else:  # Permitted, though it could also be an "exception"
                    c[i, j] = 0

    # Build Model
    m = ConcreteModel()

    # _________________________________PARAMETER ADJUSTMENTS_________________________________
    def adjust_parameters():
        """
        Mirrors the function in the VFT model, but this time we just need the objective mins and maxes
        """
        # New dictionary of parameters used in this main function ("vft_model_build") and in "vft_model_solve"
        q = {"objective_min": np.zeros([p['M'], vp['O']]),  # Min AFSC objective value
             "objective_max": np.zeros([p['M'], vp['O']])}  # Max AFSC objective value

        # Loop through each AFSC
        for j in p['J']:

            # Loop through each objective for each AFSC
            for k in vp['K^A'][j]:

                # Retrieve minimum/maximum AFSC objective measures based on constraint type. 1: Approximate, 2: Exact
                if vp['constraint_type'][j, k] in [1, 2]:  # (NOT Zero)
                    value_list = vp['objective_value_min'][j, k].split(",")
                    q["objective_min"][j, k] = float(value_list[0].strip())
                    q["objective_max"][j, k] = float(value_list[1].strip())

        return q  # Return the new dictionary

    q = adjust_parameters()  # Call the function

    # ___________________________________VARIABLE DEFINITION_________________________________
    m.x = Var(((i, j) for i in p['I'] for j in p['J^E'][i]), within=Binary)

    # Fixing variables if necessary
    for i, afsc in enumerate(p["assigned"]):
        j = np.where(p["afscs"] == afsc)[0]  # AFSC index

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
        real_n = instance.mdl_p['real_usafa_n']  # Total number of USAFA cadets (Line/Non-Line)
        cap = 0.05 * real_n

        # USAFA 5% Cap Constraint
        def usafa_afscs_rule(m):
            return np.sum(np.sum(m.x[i, j] for i in p['I^D']['USAFA Proportion'][j]) for j in vp["J^USAFA"]) <= cap

        m.usafa_afscs_constraint = Constraint(rule=usafa_afscs_rule)

    # AFSC Objective Measure Constraints (Optional decision-maker constraints)
    m.measure_constraints = ConstraintList()

    # Loop through all AFSCs to add constraints
    for j in p["J"]:

        # Get count variables for this AFSC
        count = np.sum(m.x[i, j] for i in p['I^E'][j])
        if 'usafa' in p:

            # Number of USAFA cadets assigned to the AFSC
            usafa_count = np.sum(m.x[i, j] for i in p['I^D']['USAFA Proportion'][j])

        # Loop through all objectives for this AFSC
        for k in vp['K^A'][j]:
            objective = vp['objectives'][k]

            # If it's a demographic objective, we sum over the cadets with that demographic
            if objective in vp['K^D']:
                numerator = np.sum(m.x[i, j] for i in p['I^D'][objective][j])
                measure_jk = numerator / count
            elif objective == "Merit":
                numerator = np.sum(p['merit'][i] * m.x[i, j] for i in p['I^E'][j])
                measure_jk = numerator / count
            elif objective == "Combined Quota":
                measure_jk = count
            elif objective == "USAFA Quota":
                measure_jk = usafa_count
            elif objective == "ROTC Quota":
                measure_jk = count - usafa_count
            elif objective == "Utility":
                numerator = np.sum(p['utility'][i, j] * m.x[i, j] for i in p['I^E'][j])
                measure_jk = numerator / count
            else:
                continue  # Skip this objective, whatever it is

            # Add Min Measure Constraints
            if k in vp['K^C'][j]:  # (1 constrains "approximate" measure, 2 constrains "exact" measure)

                # Constrained PGL/Approximate Measure
                if vp['constraint_type'][j, k] == 1:
                    if objective in ['Combined Quota', 'USAFA Quota', 'ROTC Quota']:
                        m.measure_constraints.add(expr=measure_jk >= q["objective_min"][j, k])
                        m.measure_constraints.add(expr=measure_jk <= q["objective_max"][j, k])
                    else:

                        if p["pgl"][j] > p['quota_min'][j]:
                            m.measure_constraints.add(expr=numerator - q["objective_min"][j, k] * p['quota_min'][j] >= 0)
                            m.measure_constraints.add(expr=numerator - q["objective_max"][j, k] * p['quota_min'][j] <= 0)
                        else:
                            m.measure_constraints.add(expr=numerator - q["objective_min"][j, k] * p['pgl'][j] >= 0)
                            m.measure_constraints.add(expr=numerator - q["objective_max"][j, k] * p['pgl'][j] <= 0)

                # Constrained Exact Measure
                elif vp['constraint_type'][j, k] == 2:
                    if objective in ['Combined Quota', 'USAFA Quota', 'ROTC Quota']:
                        m.measure_constraints.add(expr=measure_jk >= q["objective_min"][j, k])
                        m.measure_constraints.add(expr=measure_jk <= q["objective_max"][j, k])
                    else:
                        m.measure_constraints.add(expr=numerator - q["objective_min"][j, k] * count >= 0)
                        m.measure_constraints.add(expr=numerator - q["objective_max"][j, k] * count <= 0)

    if printing:
        print("Done. Solving original model...")

    return m, q  # Return model and additional component dictionary


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
    def adjust_parameters():
        """
        Function defined here to adjust certain parameters. The parameters adjusted here are the value function
        breakpoint parameters (r, a, f^hat) and set (L) as well as the AFSC objective constraint min, max values since
        they've been stored as strings (3, 6, for example) up until this point. These parameters are saved into a new
        dictionary "q" for use in the model. This is done because the value function breakpoints need to be adjusted
        due to the approximate model's capability of exceeding the normal domain, and I don't want it saved to "vp"
        """
        # Written here, so I can reference it below (Keys "r" and "L" both use this)
        r = [[len(vp['a'][j][k]) for k in vp['K']] for j in p['J']]

        # New dictionary of parameters used in this main function ("vft_model_build") and in "vft_model_solve"
        q = {"r": r,  # Number of breakpoints (bps) for objective k's function for AFSC j
             "L": [[list(range(r[j][k])) for k in vp['K']] for j in p['J']],  # Set of breakpoints
             "a": [[[vp['a'][j][k][l] for l in vp['L'][j][k]] for k in vp['K']] for j in p['J']],  # Measures of bps
             "f^hat": [[[vp['f^hat'][j][k][l] for l in vp['L'][j][k]] for k in vp['K']] for j in p['J']],  # Values of bps
             "objective_min": np.zeros([p['M'], vp['O']]),  # Min AFSC objective value
             "objective_max": np.zeros([p['M'], vp['O']])  # Max AFSC objective value
             }

        # Loop through each AFSC
        for j in p['J']:

            # Loop through each objective for each AFSC
            for k in vp['K^A'][j]:

                # We need to add an extra breakpoint to effectively extend the domain
                if instance.mdl_p["add_breakpoints"]:

                    # We add an extra breakpoint far along the x-axis with the same y value as the previous one
                    last_a = q["a"][j][k][q['r'][j][k] - 1]
                    last_f = q["f^hat"][j][k][q['r'][j][k] - 1]
                    q["a"][j][k].append(last_a * 2000)  # arbitrarily large number in the domain (x-space)
                    q["f^hat"][j][k].append(last_f)  # same AFSC objective "y-value" as previous one
                    q["L"][j][k].append(q['r'][j][k])  # add the new breakpoint index
                    q['r'][j][k] += 1  # increase number of breakpoints by 1

                # Retrieve minimum/maximum AFSC objective measures based on constraint type. 1: Approximate, 2: Exact
                if vp['constraint_type'][j, k] in [1, 2]:  # (NOT Zero)
                    value_list = vp['objective_value_min'][j, k].split(",")
                    q["objective_min"][j, k] = float(value_list[0].strip())
                    q["objective_max"][j, k] = float(value_list[1].strip())

        # Convert to numpy arrays of lists
        for key in ["L", "r", "a", "f^hat"]:
            q[key] = np.array(q[key])

        return q  # Return the new dictionary

    q = adjust_parameters()  # Call the function

    # _________________________________VARIABLE DEFINITIONS_________________________________
    m.x = Var(((i, j) for i in p['I'] for j in p['J^E'][i]), within=Binary)  # main decision variable (x)
    m.f_value = Var(((j, k) for j in p['J'] for k in vp['K^A'][j]), within=NonNegativeReals)  # AFSC objective value
    m.lam = Var(((j, k, l) for j in p['J'] for k in vp['K^A'][j] for l in q['L'][j, k]),
                within=NonNegativeReals, bounds=(0, 1))  # Lambda and y variables for value functions
    m.y = Var(((j, k, l) for j in p['J'] for k in vp['K^A'][j] for l in range(q['r'][j, k] - 1)), within=Binary)

    def variable_adjustments(m):
        """
        This function initializes the 4 variables defined above if applicable (warm start has been determined) and also
        fixes certain x variables if necessary/applicable as well.
        """

        # If we initialize variables
        if instance.mdl_p["warm_start"] is not None:

            # For each cadet, for each AFSC that the cadet is eligible
            for i in p['I']:
                for j in p['J^E'][i]:

                    # x: 1 if we assign cadet i to AFSC j; 0 otherwise
                    m.x[i, j] = round(instance.mdl_p["warm_start"]['x'][i, j])

            # Loop through each AFSC objective for each AFSC
            for j in p['J']:
                for k in vp['K^A'][j]:

                    # Value for AFSC j objective k  (Used in Constraint 20b in VFT thesis)
                    m.f_value[j, k] = instance.mdl_p["warm_start"]['f(measure)'][j, k]

                    # Loop through each breakpoint for this AFSC objective value function
                    for l in q['L'][j, k]:

                        # % between breakpoint l and l + 1 that the measure for AFSC j objective k "has yet to travel"
                        m.lam[j, k, l] = instance.mdl_p["warm_start"]['lambda'][j, k, l]

                        # There is one less "y" variable than lambda because this is for the line segments between bps
                        if l < q['r'][j, k] - 1:

                            # 1 if AFSC j objective measure k is on line segment between breakpoints l and l + 1; 0 o/w
                            m.y[j, k, l] = instance.mdl_p["warm_start"]['y'][j, k, l]

        # Fixing variables if necessary
        if "assigned" in p:
            for i, afsc in enumerate(p["assigned"]):

                if afsc in p["afscs"]:
                    j = np.where(p["afscs"] == afsc)[0][0]  # AFSC index
                else:
                    continue  # Skip the cadet if the assigned AFSC is not "valid"

                # Check if the cadet is assigned to an AFSC they're not eligible for
                if j not in p["J^E"][i]:
                    raise ValueError("Cadet " + str(i) + " assigned to '" + afsc + "' but is not eligible for it. "
                                                                                   "Adjust the qualification matrix!")

                # Fix the variable
                m.x[i, j].fix(1)

        # Return model (m)
        return m

    m = variable_adjustments(m)  # Call the function

    # _________________________________OBJECTIVE FUNCTION_________________________________
    def objective_function(m):
        """
        The objective function is to maximize "Z", the overall weighted sum of all VFT objectives
        """
        return vp['afscs_overall_weight'] * np.sum(vp['afsc_weight'][j] * np.sum(
            vp['objective_weight'][j, k] * m.f_value[j, k] for k in vp['K^A'][j]) for j in p['J']) + \
               vp['cadets_overall_weight'] * np.sum(vp['cadet_weight'][i] * np.sum(
            p['utility'][i, j] * m.x[i, j] for j in p['J^E'][i]) for i in p['I'])

    m.objective = Objective(rule=objective_function, sense=maximize)

    # ____________________________________CONSTRAINTS_____________________________________
    pass  # Here so pycharm doesn't yell at me for the constraint line above

    # Cadets receive one and only one AFSC (Ineligibility constraint is always met as a result of the indexed sets)
    m.one_afsc_constraints = ConstraintList()
    for i in p['I']:
        m.one_afsc_constraints.add(expr=np.sum(m.x[i, j] for j in p['J^E'][i]) == 1)

    # 5% cap on total percentage of USAFA cadets allowed into certain AFSCs
    if vp["J^USAFA"] is not None:
        cap = 0.05 * instance.mdl_p["real_usafa_n"]  # Total number of graduating USAFA cadets (Line & Non-line cadets)

        # USAFA 5% Cap Constraint
        def usafa_afscs_rule(m):
            """
            This is the 5% USAFA graduating class cap constraint for certain AFSCs (support AFSCs). I will note that
            as of Mar '23 this constraint is effectively null and void! Still here for documentation however and for any
            potential future experiment
            """
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

    # AFSC Objective Measure Constraints (Optional decision-maker constraints)
    m.measure_constraints = ConstraintList()

    # Loop through all AFSCs to add constraints
    for j in p['J']:

        # Get count variables for this AFSC
        count = np.sum(m.x[i, j] for i in p['I^E'][j])
        if 'usafa' in p:

            # Number of USAFA cadets assigned to the AFSC
            usafa_count = np.sum(m.x[i, j] for i in p['I^D']['USAFA Proportion'][j])

        # Are we using approximate measures or not
        if instance.mdl_p["approximate"]:
            num_cadets = p['quota_e'][j]  # Approximate
        else:
            num_cadets = count  # Exact

        # Loop through all objectives for this AFSC
        for k in vp['K^A'][j]:
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

            elif objective == "Utility":
                numerator = np.sum(p['utility'][i, j] * m.x[i, j] for i in p['I^E'][j])
                measure_jk = numerator / num_cadets

            else:
                raise ValueError("Error. Objective '" + objective + "' does not have a means of calculation in the"
                                                                    " VFT model. Please adjust.")

            # Add Linear Value Function Constraints
            m.measure_vf_constraints.add(expr=measure_jk == np.sum(  # Measure Constraint for Value Function (20a)
                q['a'][j, k][l] * m.lam[j, k, l] for l in q['L'][j, k]))
            m.value_vf_constraints.add(expr=m.f_value[j, k] == np.sum(  # Value Constraint for Value Function (20b)
                q['f^hat'][j, k][l] * m.lam[j, k, l] for l in q['L'][j, k]))

            # Lambda .. y constraints (20c, 20d, 20e)
            m.lambda_y_constraint1.add(expr=m.lam[j, k, 0] <= m.y[j, k, 0])  # (20c)
            if q['r'][j, k] > 2:
                for l in range(1, q['r'][j, k] - 1):
                    m.lambda_y_constraint2.add(expr=m.lam[j, k, l] <= m.y[j, k, l - 1] + m.y[j, k, l])  # (20d)
            m.lambda_y_constraint3.add(expr=m.lam[j, k, q['r'][j, k] - 1] <= m.y[j, k, q['r'][j, k] - 2])  # (20e)

            # Y sum to 1 constraint (20f)
            m.y_sum_constraint.add(expr=np.sum(m.y[j, k, l] for l in range(0, q['r'][j, k] - 1)) == 1)

            # Lambda sum to 1 constraint (20g)
            m.lambda_sum_constraint.add(expr=np.sum(m.lam[j, k, l] for l in q['L'][j, k]) == 1)

            # Lambda .. value positive constraint (20h) although the "f_value" constraint is implied in the thesis
            for l in q['L'][j, k]:
                m.lambda_positive.add(expr=m.lam[j, k, l] >= 0)
            m.f_value_positive.add(expr=m.f_value[j, k] >= 0)

            # Add Min Value/Measure Constraints
            if k in vp['K^C'][j]:  # (1 constrains "approximate" measure, 2 constrains "exact" measure)

                # Constrained PGL/Approximate Measure
                if vp['constraint_type'][j, k] == 1:
                    if objective in ['Combined Quota', 'USAFA Quota', 'ROTC Quota']:
                        m.measure_constraints.add(expr=measure_jk >= q["objective_min"][j, k])
                        m.measure_constraints.add(expr=measure_jk <= q["objective_max"][j, k])
                    else:

                        if "pgl" in p:  # Check which "reference minimum" we should use
                            if p["pgl"][j] > p['quota_min'][j]:
                                m.measure_constraints.add(
                                    expr=numerator - q["objective_min"][j, k] * p['quota_min'][j] >= 0)
                                m.measure_constraints.add(
                                    expr=numerator - q["objective_max"][j, k] * p['quota_min'][j] <= 0)
                            else:
                                m.measure_constraints.add(expr=numerator - q["objective_min"][j, k] * p['pgl'][j] >= 0)
                                m.measure_constraints.add(expr=numerator - q["objective_max"][j, k] * p['pgl'][j] <= 0)
                        else:
                            m.measure_constraints.add(
                                expr=numerator - q["objective_min"][j, k] * p['quota_min'][j] >= 0)
                            m.measure_constraints.add(
                                expr=numerator - q["objective_min"][j, k] * p['quota_min'][j] <= 0)

                # Constrained Exact Measure
                elif vp['constraint_type'][j, k] == 2:
                    if objective in ['Combined Quota', 'USAFA Quota', 'ROTC Quota']:
                        m.measure_constraints.add(expr=measure_jk >= q["objective_min"][j, k])
                        m.measure_constraints.add(expr=measure_jk <= q["objective_max"][j, k])
                    else:
                        m.measure_constraints.add(expr=numerator - q["objective_min"][j, k] * count >= 0)
                        m.measure_constraints.add(expr=numerator - q["objective_max"][j, k] * count <= 0)

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

    return m, q  # Return model and additional component dictionary


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


def solve_pyomo_model(instance, model, model_name, q=None, printing=False):
    """
    Simple function that calls the pyomo solver using the specified model and max_time
    """

    # Shorthand
    p, vp, gp, mdl_p = instance.parameters, instance.value_parameters, instance.gp_parameters, instance.mdl_p

    # Adjust solver if necessary
    if not mdl_p["approximate"] and model_name == "VFT":
        if mdl_p["solver_name"] == 'cbc':
            mdl_p["solver_name"] = 'ipopt'

    # Determine how the solver is called here
    if mdl_p["executable"] is None:
        if mdl_p["provide_executable"]:
            if mdl_p["exe_extension"]:
                mdl_p["executable"] = afccp.core.globals.paths['solvers'] + mdl_p["solver_name"] + '.exe'
            else:
                mdl_p["executable"] = afccp.core.globals.paths['solvers'] + mdl_p["solver_name"]
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

    # Print Statement
    if printing:
        if model_name == "VFT":
            if mdl_p["approximate"]:
                print('Solving Approximate VFT Model instance with solver ' + mdl_p["solver_name"] + '...')
            else:
                print('Solving Exact VFT Model instance with solver ' + mdl_p["solver_name"] + '...')
        else:
            print('Solving ' + model_name + ' Model instance with solver ' + mdl_p["solver_name"] + '...')


    # Solve Model
    start_time = time.perf_counter()
    if mdl_p["pyomo_max_time"] is not None:
        if mdl_p["solver_name"] == 'mindtpy':
            solver.solve(model, time_limit=mdl_p["pyomo_max_time"],
                         mip_solver='cplex_persistent', nlp_solver='ipopt')
        elif mdl_p["solver_name"] == 'gurobi':
            solver.solve(model, options={'TimeLimit': mdl_p["pyomo_max_time"], 'IntFeasTol': 0.05})
        elif mdl_p["solver_name"] == 'ipopt':
            solver.options['max_cpu_time'] = mdl_p["pyomo_max_time"]
            solver.solve(model)
        elif mdl_p["solver_name"] == 'cbc':
            solver.options['seconds'] = mdl_p["pyomo_max_time"]
            solver.solve(model)
        elif mdl_p["solver_name"] == 'baron':
            solver.solve(model, options={'MaxTime': mdl_p["pyomo_max_time"]})
        else:
            solver.solve(model)
    else:
        if mdl_p["solver_name"] == 'mindtpy':
            solver.solve(model, mip_solver='cplex_persistent', nlp_solver='ipopt')
        else:
            solver.solve(model)

    if printing:
        print("Model solved in", round(time.perf_counter() - start_time, 2), "seconds.")

    # Goal Programming Model specific actions
    if model_name == "GP":

        # We're "pre-process" solving the model for a specific GP constraint
        if mdl_p["con_term"] is not None:
            return model.objective()

        # We're actually solving the model for a solution
        else:

            # Get solution
            solution = np.zeros(gp['N'])
            x = np.zeros((gp['N'], gp['M']))
            for c in gp['C']:
                for a in gp['A^']['E'][c]:
                    x[c, a] = model.x[c, a].value
                    if round(x[c, a]):
                        solution[c] = int(a)

            if printing:
                print('Model solved.')

            return solution.astype(int), x

    # VFT/Original Model specific actions
    else:

        # Obtain solution from the model
        def obtain_solution():
            """
            This nested function obtains the X matrix and the solution vector from the pyomo model
            """

            # Initialize solution and X matrix
            solution = np.zeros(p['N']).astype(int)
            x = np.zeros([p['N'], p['M']])

            # Loop through each cadet to determine what AFSC they're assigned
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
                # they're eligible for and want (happens usually to only 1-3 people through VFT model)
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

                    afsc = p["afscs"][int(solution[i])]

                    if printing:
                        print("Cadet " + str(i) + " was not assigned by the model for some reason. "
                                                  "We assigned them to", afsc)

            return solution, x

        solution, x = obtain_solution()

        # Obtain "warm start" variables used to initialize the VFT pyomo model
        def obtain_warm_start_variables():
            """
            This nested function obtains the variables used for the warm start (variable initialization) of the pyomo model
            """

            # Determine maximum number of breakpoints for any particular AFSC
            max_r = 0
            for j in p["J"]:
                for k in vp["K^A"][j]:
                    if q["r"][j][k] > max_r:
                        max_r = q["r"][j][k]

            # Initialize dictionary
            warm_start = {'f(measure)': np.zeros([p['M'], vp['O']]), 'x': x,
                          'lambda': np.zeros([p['M'], p['O'], max_r + 1]),
                          'y': np.zeros([p['M'], p['O'], max_r + 1]).astype(int), 'obj': model.objective()}

            # Load warm start variables
            for j in p['J']:
                for k in vp['K^A'][j]:
                    warm_start['f(measure)'][j, k] = model.f_value[j, k].value
                    for l in range(q['r'][j, k]):
                        warm_start['lambda'][j, k, l] = model.lam[j, k, l].value
                        if l < q['r'][j, k] - 1:
                            warm_start['y'][j, k, l] = round(model.y[j, k, l].value)

            # Return the "warm start" dictionary
            return warm_start

        warm_start = None  # Empty dictionary
        if mdl_p["obtain_warm_start_variables"]:
            warm_start = obtain_warm_start_variables()

        # Return solution, X matrix, and "warm start" dictionary
        return solution.astype(int), x, warm_start

