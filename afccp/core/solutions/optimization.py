import time
import copy
import numpy as np
import logging
import warnings
import pandas as pd
from pyomo.environ import *

# afccp modules
import afccp.core.globals
import afccp.core.solutions.handling

# Ignore warnings
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# Main Model Building
def assignment_model_build(instance, printing=False):
    """
    Converts the parameters and value parameters to the pyomo data structure

    Parameters:
        instance (object): Problem instance object
        printing (bool, optional): Whether the procedure should print something. Default is False.

    Returns:
        pyomo data: Pyomo data representing the converted model

    Description:
        This function builds a Pyomo model based on the provided problem instance. It converts the parameters and value
        parameters into the Pyomo data structure and constructs the objective function and constraints of the model.

        The function takes a problem instance object as input, which contains the necessary parameters and value parameters
        for building the model. The `printing` parameter controls whether the procedure should print progress information
        during model construction.

        The utility/cost matrix is computed based on the parameters and value parameters. The AFSC preferences, merit,
        and eligibility information are used to calculate the cost values for each cadet-AFSC pair in the matrix.

        The model is built using the Pyomo `ConcreteModel` class. The variables, objective function, and constraints
        are defined within the model.

        The objective function is defined as the sum of the cost values multiplied by the corresponding decision variable
        for each cadet-AFSC pair.

        The constraints include ensuring that each cadet is assigned to exactly one AFSC, limiting the percentage of
        USAFA cadets in certain AFSCs, and applying AFSC objective measure constraints if specified.

        If `printing` is set to True, the function prints progress information during the model construction.

        Finally, the constructed model is returned.

    Example:
        instance = ProblemInstance()
        model = assignment_model_build(instance, printing=True)
        ...
    """

    # Shorthand
    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p

    # *New* Utility/Cost Matrix based on CFM preferences and cadet preferences
    if mdl_p['assignment_model_obj'] == 'Global Utility':
        c = vp['global_utility']

        if printing:
            print("Building assignment problem model...")

    # Original Model Utility/Cost Matrix
    else:

        if printing:
            print("Building original model...")

        c = np.zeros([p['N'], p['M']])
        for i in p['I']:
            for j in p['J^E'][i]:  # Only looping through AFSCs that the cadet is eligible for

                # If AFSC j is a preference for cadet i
                if p['cadet_utility'][i, j] > 0:

                    if p['mandatory'][i, j] == 1:
                        c[i, j] = 10 * p['merit'][i] * p['cadet_utility'][i, j] + 250
                    elif p['desired'][i, j] == 1:
                        c[i, j] = 10 * p['merit'][i] * p['cadet_utility'][i, j] + 150
                    else:  # Permitted, though it could also be an "exception"
                        c[i, j] = 10 * p['merit'][i] * p['cadet_utility'][i, j]

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

    # ___________________________________VARIABLE DEFINITION_________________________________
    m = common_optimization_handling(m, p, vp, mdl_p)  # Define x along with additional functional constraints

    # ___________________________________OBJECTIVE FUNCTION__________________________________
    def objective_function(m):
        return np.sum(np.sum(c[i, j] * m.x[i, j] for j in p["J^E"][i]) for i in p["I"])

    m.objective = Objective(rule=objective_function, sense=maximize)

    # ________________________________________CONSTRAINTS_____________________________________
    pass

    # AFSC Objective Measure Constraints (Optional decision-maker constraints)
    m.measure_constraints = ConstraintList()

    # Loop through all AFSCs to add AFSC objective measure constraints
    for j in p['J']:

        # Loop through all constrained AFSC objectives
        for k in vp['K^C'][j]:

            # Calculate AFSC objective measure components
            measure, numerator = afccp.core.solutions.handling.calculate_objective_measure_matrix(
                m.x, j, vp['objectives'][k], p, vp, approximate=True)

            # Add AFSC objective measure constraint
            m = add_objective_measure_constraint(m, j, k, measure, numerator, p, vp)

    if printing:
        print("Done. Solving model...")

    return m  # Return model


def vft_model_build(instance, printing=False):
    """
    Builds the VFT optimization model using pyomo.

    Parameters:
        instance (object): Problem instance object.
        printing (bool): Whether the procedure should print something. Default is False.

    Returns:
        object: Pyomo model object.

    This function builds the VFT (Value Focused Thinking) optimization model using the Pyomo library. It takes a
    problem instance as input and returns the constructed model.

    The function performs the following steps:
    1. Initializes the Pyomo model.
    2. Adjusts certain parameters used in the model.
    3. Defines and initializes the decision variables of the model.
    4. Defines the objective function of the model.
    5. Defines the constraints of the model.

    Parameter Adjustments:
    The function adjusts certain parameters related to the value function breakpoints and sets. These adjustments are
    necessary to account for the approximate model's capability of exceeding the normal domain. The adjusted parameters
    are stored in a new dictionary called 'q' for use in the model.

    Variable Definitions:
    The function defines the decision variables used in the model, including 'x' (main decision variable), 'f_value'
    (AFSC objective value), 'lam' (lambda and y variables for value functions), and 'y' (binary variable for line
    segments between breakpoints).

    Variable Adjustments:
    This function initializes the variables defined above if applicable (warm start has been determined) and fixes
    certain 'x' variables if necessary/applicable.

    Objective Function:
    The objective function of the model is to maximize the overall weighted sum of all VFT objectives. It combines the
    AFSC objective values and the cadet utility values based on their respective weights.

    Constraints:
    The function defines various constraints for the model, including the constraint that each cadet receives one and
    only one AFSC, the 5% cap on the total percentage of USAFA cadets allowed into certain AFSCs, the value function
    constraints linking the main methodology with the value function methodology, and optional decision-maker
    constraints.

    AFSC Objective Measure Constraints:
    The function adds AFSC objective measure constraints for each AFSC and objective. It calculates the objective measure
    components and adds linear value function constraints based on the measure and value functions.

    AFSC Value Constraints:
    Optional decision-maker constraints can be added to enforce minimum AFSC objective values. The function adds
    constraints to ensure that the weighted sum of AFSC objective values meets the specified minimum value for each AFSC.

    Cadet Value Constraints:
    Optional decision-maker constraints can be added to enforce minimum cadet utility values. The function adds
    constraints to ensure that the weighted sum of cadet utility values meets the specified minimum value for each cadet.

    AFSCs Overall Min Value Constraint:
    If a minimum overall value for AFSCs is specified, the function adds a constraint to ensure that the weighted sum of
    AFSC objective values meets the specified minimum value for all AFSCs.

    Note: The function assumes the availability of additional helper functions, such as 'add_objective_measure_constraint',
    which are used to add specific types of constraints to the model.

    """

    if printing:
        print('Building VFT Model...')

    # Build Model
    m = ConcreteModel()

    # Shorthand
    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p

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
             "f^hat": [[[vp['f^hat'][j][k][l] for l in vp['L'][j][k]] for k in vp['K']] for j in p['J']]}  # Values of bps

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

        # Convert to numpy arrays of lists
        for key in ["L", "r", "a", "f^hat"]:
            q[key] = np.array(q[key])

        return q  # Return the new dictionary

    q = adjust_parameters()  # Call the function

    # _________________________________VARIABLE DEFINITIONS_________________________________
    m = common_optimization_handling(m, p, vp, mdl_p)  # Define x along with additional functional constraints
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
            p['cadet_utility'][i, j] * m.x[i, j] for j in p['J^E'][i]) for i in p['I'])

    m.objective = Objective(rule=objective_function, sense=maximize)

    # ____________________________________CONSTRAINTS_____________________________________
    pass  # Here so pycharm doesn't yell at me for the constraint line above

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

    # AFSC Value Constraints (Optional decision-maker constraints)
    m.min_afsc_value_constraints = ConstraintList()

    # AFSC Objective Measure Constraints (Optional decision-maker constraints)
    m.measure_constraints = ConstraintList()

    # Loop through all AFSCs to add AFSC objective measure constraints
    for j in p['J']:

        # Loop through all AFSC objectives
        for k, objective in enumerate(vp['objectives']):

            # Add AFSC objective measure value function "functional" constraints
            if k in vp['K^A'][j]:

                # Calculate AFSC objective measure components
                measure, numerator = afccp.core.solutions.handling.calculate_objective_measure_matrix(
                    m.x, j, objective, p, vp, approximate=instance.mdl_p['approximate'])

                # Add Linear Value Function Constraints
                m.measure_vf_constraints.add(expr=measure == np.sum(  # Measure Constraint for Value Function (20a)
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

                # Add AFSC objective measure constraint
                if k in vp['K^C'][j]:
                    m = add_objective_measure_constraint(m, j, k, measure, numerator, p, vp)

        # AFSC value constraint
        if vp['afsc_value_min'][j] != 0:
            m.min_afsc_value_constraints.add(expr=np.sum(
                vp['objective_weight'][j, k] * m.f_value[j, k] for k in vp['K^A'][j]) >= vp['afsc_value_min'][j])

    # AFSCs Overall Min Value
    def afsc_min_value_constraint(m):
        return vp['afscs_overall_value_min'] <= np.sum(vp['afsc_weight'][j] * np.sum(
            vp['objective_weight'][j, k] * m.f_value[j, k] for k in vp['K^A'][j]) for j in p['J'])

    if vp['afscs_overall_value_min'] != 0:
        m.afsc_min_value_constraint = Constraint(rule=afsc_min_value_constraint)

    # Cadets Overall Min Value
    def cadet_min_value_constraint(m):
        return vp['cadets_overall_value_min'] <= np.sum(vp['cadet_weight'][i] * np.sum(
            p['cadet_utility'][i, j] * m.x[i, j] for j in p['J^E'][i]) for i in p['I'])

    if vp['cadets_overall_value_min'] != 0:
        m.cadet_min_value_constraint = Constraint(rule=cadet_min_value_constraint)

    return m, q  # Return model and additional component dictionary


def gp_model_build(instance, printing=False):
    """
    Builds Rebecca's goal programming (GP) model using the provided problem instance.

    Args:
        instance (object): The problem instance to solve.
        printing (bool, optional): Specifies whether to print status updates during model building. Default is False.

    Returns:
        pyomo.core.base.PyomoModel.ConcreteModel: The constructed Pyomo model.

    Raises:
        None

    Detailed Description:
        This function builds the GP model according to Rebecca's goal programming formulation using the provided problem instance.
        The model incorporates Rebecca's parameters and constructs the necessary variables, objective function, and constraints.

        The function iterates over each constraint and AFSC to create the decision variables, penalty variables, and reward variables.
        It defines the main objective function that represents the overall goal programming problem.
        Additionally, it defines penalty and reward specific objective functions to obtain raw penalties and rewards.

        The function also constructs various constraints related to AFSC assignments, penalty terms, and reward terms.

    Parameter Details:
        - instance (object): The problem instance to solve. It should contain the following attributes:
            - gp_parameters (dict): The GP parameters, including the constraint terms, utility values, and sets.
            - mdl_p (dict): Additional model parameters, including the current constraint term and the reward/penalty flag.
        - printing (bool, optional): Specifies whether to print status updates during model building. Default is False.

    Returns:
        - model (pyomo.core.base.PyomoModel.ConcreteModel): The constructed Pyomo model representing the GP problem.

    Note:
        The function assumes that the necessary libraries and packages (such as NumPy) are imported.
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


# Pyomo Model Solving Function
def solve_pyomo_model(instance, model, model_name, q=None, printing=False):
    """
    Solve a Pyomo model using a specified solver.

    This function takes an instance, a Pyomo model, the model name, optional parameters (q), and a flag for printing
    intermediate information. It adjusts the solver settings based on the provided instance parameters, solves the
    model, and returns the solution.

    Args:
        instance: The Pyomo instance.
        model: The Pyomo model to solve.
        model_name (str): The name of the model.
        q (dict, optional): Optional parameters.
        printing (bool, optional): Flag for printing intermediate information.

    Returns:
        solution (int or tuple): The solution of the model.
            - If the model name is "GP", returns a tuple (solution, x) where solution is an array of integers representing
              the AFSCs assigned to cadets, and x is a 2D array representing the assignment matrix.
            - Otherwise, returns a tuple (solution, x, warm_start), where solution is an array of integers representing
              the AFSCs assigned to cadets, x is a 2D array representing the assignment matrix, and warm_start is a
              dictionary containing warm start variables used for initializing the VFT Pyomo model.
    """

    # Different parameters are needed based on the model
    if model_name == 'CadetBoard':
        b, mdl_p = instance.b, instance.b  # It's weird, I know, but this works
        mdl_p["solver_name"] = b['b_solver_name']  # Change the solver
        mdl_p["pyomo_max_time"] = b['b_pyomo_max_time']  # Set the max time
    else:
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
            solver.solve(model, time_limit=mdl_p["pyomo_max_time"]),
                         #mip_solver='cplex_persistent', nlp_solver='ipopt')
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
            model.pprint()
            solver.solve(model, mip_solver='cplex_persistent', nlp_solver='ipopt')
        else:
            solver.solve(model) #, tee=True)

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
            solution = {"method": "GP", "j_array": np.zeros(gp['N']).astype(int), "x": np.zeros((gp['N'], gp['M']))}
            for c in gp['C']:
                for a in gp['A^']['E'][c]:
                    solution['x'][c, a] = model.x[c, a].value
                    if round(solution['x'][c, a]):
                        solution['j_array'][c] = int(a)

            if printing:
                print('Model solved.')

            return solution

    # "Cadet Board Figure" optimization model
    elif model_name == 'CadetBoard':

        # Get the values from the model and return them
        x, y, s = {}, {}, model.s.value

        for j in b['J^translated']:
            idx = b['J^translated'][j]
            x[j], y[j] = model.x[idx].value, model.y[idx].value
        return s, x, y

    # VFT/Assignment Model specific actions
    else:

        # Obtain solution from the model
        def obtain_solution():
            """
            This nested function obtains the X matrix and the solution vector from the pyomo model
            """

            # Get solution
            solution = {"method": model_name, "j_array": np.zeros(p['N']).astype(int), "x": np.zeros((p['N'], p['M']))}

            # Loop through each cadet to determine what AFSC they're assigned
            for i in p['I']:
                found = False
                for j in p['J^E'][i]:
                    solution['x'][i, j] = model.x[i, j].value
                    try:
                        if round(solution['x'][i, j]):
                            solution['j_array'][i] = int(j)
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
                            if p['cadet_utility'][i, j] >= max_util:
                                max_j = j
                                max_util = max_util
                        solution['j_array'][i] = int(max_j)

                    # If we don't have any eligible preferences from the cadet, they get Needs of the Air Force
                    else:

                        if len(p["J^E"][i]) >= 2:
                            solution['j_array'][i] = int(p["J^E"][i][1])
                        else:
                            solution['j_array'][i] = int(p["J^E"][i][0])

                    afsc = p["afscs"][int(solution['j_array'][i])]

                    if printing:
                        print("Cadet " + str(i) + " was not assigned by the model for some reason. "
                                                  "We assigned them to", afsc)

            return solution

        solution = obtain_solution()

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
            warm_start = {'f(measure)': np.zeros([p['M'], vp['O']]),
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

        # Add additional components to solution dictionary
        if mdl_p["obtain_warm_start_variables"] and 'VFT' in model_name:
            warm_start = obtain_warm_start_variables()
            for key in warm_start:
                solution[key] = warm_start[key]

        # Return solution dictionary
        return solution


# Goal Programming Model Pre-Processing
def calculate_rewards_penalties(instance, printing=True):
    """
    This function calculates the normalized penalties and rewards specific to an instance of
    Rebecca's goal programming (GP) model.

    Args:
        instance (object): The problem instance to solve, which contains the GP parameters.
        printing (bool, optional): Specifies whether to print status updates during the calculation. Default is True.

    Returns:
        tuple: A tuple containing the normalized penalties and rewards as NumPy arrays.

    Detailed Description:
        This function takes a set of Rebecca's goal programming parameters and returns the normalized penalties and
        rewards specific to the given instance. The function iterates over each constraint in the GP parameters and
        calculates the penalties and rewards using a GP model.

        The GP model is built by initializing the necessary parameters and solving the model for each constraint.
        The rewards are calculated by maximizing the objective function that represents the reward term, while the
        penalties are calculated by maximizing the objective function that represents the penalty term.

        The function also calculates the reward term for the special constraint 'S' separately.

    Parameter Details:
        - instance (object): The problem instance to solve. It should contain the following attributes:
            - gp_parameters (dict): The GP parameters, including the constraint terms, utility values, and sets.
        - printing (bool, optional): Specifies whether to print status updates during the calculation. Default is True.

    Returns:
        - rewards (numpy.ndarray): An array containing the normalized rewards for each constraint, including the reward
        for constraint 'S'.
        - penalties (numpy.ndarray): An array containing the normalized penalties for each constraint.

    Note:
        The function assumes that the necessary functions 'gp_model_build' and 'solve_pyomo_model'
        are defined and accessible.
    """

    # Shorthand
    gp = instance.gp_parameters

    # Initialize gp arrays
    num_constraints = len(gp['con']) + 1
    rewards = np.zeros(num_constraints)
    penalties = np.zeros(num_constraints)

    # Initialize model
    instance.mdl_p["con_term"] = gp['con'][0]  # Initialize constraint term
    instance.mdl_p["get_reward"] = True  # We want the reward term
    instance.mdl_p["solve_time"] = 60 * 4  # Don't want to be solving this thing for too long
    model = gp_model_build(instance, printing=False)  # Build model

    # Loop through each constraint
    for c, con in enumerate(gp['con']):

        # Set the constraint term
        instance.mdl_p["con_term"] = con

        # Get reward term
        def objective_function(m):
            return np.sum(m.Z[con, a] for a in gp['A^'][con])

        if printing:
            print('')
            print('Obtaining reward for constraint ' + con + '...')
        model.objective = Objective(rule=objective_function, sense=maximize)
        rewards[c] = solve_pyomo_model(instance, model, "GP")
        if printing:
            print('Reward:', rewards[c])

        # Get penalty term
        def objective_function(m):
            return np.sum(m.Y[con, a] for a in gp['A^'][con])

        if printing:
            print('')
            print('Obtaining penalty for constraint ' + con + '...')
        model.objective = Objective(rule=objective_function, sense=maximize)
        penalties[c] = solve_pyomo_model(instance, model, "GP")
        if printing:
            print('Penalty:', penalties[c])

    # S reward term
    def objective_function(m):
        return np.sum(np.sum(gp['utility'][c, a] * m.x[c, a] for a in gp['A^']['W^E'][c]) for c in gp['C'])

    if printing:
        print('')
        print('Obtaining reward for constraint S...')
    model.objective = Objective(rule=objective_function, sense=maximize)
    rewards[num_constraints - 1] = solve_pyomo_model(instance, model, "GP")
    if printing:
        print('Reward:', rewards[num_constraints - 1])

    return rewards, penalties


# Optimization Model Helper Functions
def add_objective_measure_constraint(m, j, k, measure, numerator, p, vp):
    """
    Add an objective measure constraint to the model.

    This function takes the model (m), AFSC index (j), objective, objective measure, numerator of the function,
    problem parameters (p), and value parameters (vp) as inputs. It adds a constraint to the constraint list of the model
    based on the given objective measure.

    For objectives related to the number of cadets (such as Combined Quota, USAFA Quota, ROTC Quota), the minimum and
    maximum values of the measure are directly enforced.

    For objectives with constrained approximate measures, the function checks whether the constrained minimum number
    of cadets is lower than the Program Guidance Letter (PGL). If the constrained minimum is below the PGL, the
    objective constraint is based on that minimum value, otherwise, it is based on the PGL target.

    For objectives with constrained exact measures, the constraint is directly based on the minimum and maximum values
    multiplied by the count of cadets for the AFSC. (Numerator / Count) -> Objective Measure

    Args:
        m (ConcreteModel): The Pyomo model to which the constraint will be added.
        j (int): The index of the AFSC.
        k (int): The index of the objective.
        measure (Expression): The objective measure.
        numerator (Expression): The numerator of the objective measure function.
        p (dict): The problem parameters.
        vp (dict): The value parameters.

    Returns:
        ConcreteModel: The updated Pyomo model with the objective measure constraint added.
    """

    # Get count variables for this AFSC
    count = np.sum(m.x[i, j] for i in p['I^E'][j])

    # "Number of Cadets" objectives handled separately
    if vp['objectives'][k] in ['Combined Quota', 'USAFA Quota', 'ROTC Quota']:
        m.measure_constraints.add(expr=measure >= vp["objective_min"][j, k])
        m.measure_constraints.add(expr=measure <= vp["objective_max"][j, k])
    else:
        # Constrained Approximate Measure
        if vp['constraint_type'][j, k] == 1:

            # Take the smallest value between the PGL and constrained minimum number for this constraint
            m.measure_constraints.add(
                expr=numerator - vp["objective_min"][j, k] * min(p["pgl"][j], p['quota_min'][j]) >= 0)
            m.measure_constraints.add(
                expr=numerator - vp["objective_max"][j, k] * min(p["pgl"][j], p['quota_min'][j]) <= 0)

        # Constrained Exact Measure
        elif vp['constraint_type'][j, k] == 2:
            m.measure_constraints.add(expr=numerator - vp["objective_min"][j, k] * count >= 0)
            m.measure_constraints.add(expr=numerator - vp["objective_max"][j, k] * count <= 0)

    # Return the model
    return m


def common_optimization_handling(m, p, vp, mdl_p):
    """
    This function takes in the optimization model, parameters, and value parameters
    and then adds components that are common to all of my *main* optimization models
    (VFT and the 2 generalized assignment problem models).
    """

    # Define the x-variable
    m.x = Var(((i, j) for i in p['I'] for j in p['J^E'][i]), within=Binary)

    # Fixing variables if necessary
    for i in p['J^Fixed']:

        # Fix the variable
        m.x[i, p['J^Fixed'][i]].fix(1)

    # Cadets receive one and only one AFSC (Ineligibility constraint is always met as a result of the indexed sets)
    m.one_afsc_constraints = ConstraintList()
    for i in p['I']:
        m.one_afsc_constraints.add(expr=np.sum(m.x[i, j] for j in p['J^E'][i]) == 1)

    # Cadets with reserved AFSC slots get constrained so that the "worst" choice they can get is their reserved AFSC
    m.reserved_afsc_constraints = ConstraintList()
    for i in p['J^Reserved']:
        m.reserved_afsc_constraints.add(expr=np.sum(m.x[i, j] for j in p['J^Reserved'][i]) == 1)

    # "ALTERNATE LIST" SPECIAL TREATMENT SITUATION
    if 'J^Special' in p:

        # [1, 2, 3, 4, ..., num_reserved] for each special AFSC
        a_counts = {j: np.arange(p['num_reserved'][j]).astype(int) + 1 for j in p['J^Special']}

        # Dictionary with (key, value) pair as ("count" (defined above), cadet index)
        a_cadets = {}
        for j in p['J^Special']:
            a_cadets[j] = {}
            for c in a_counts[j]:
                a_cadets[j][c] = p['I^Alternate'][j][c - 1]

        # [1, 3, 6, 10, 15, ...] for each special AFSC (cumulative sums)
        a_cum_sums = {}
        for j in p['J^Special']:
            sums = []
            total = 0
            for c in a_counts[j]:
                total += c
                sums.append(total)
            a_cum_sums[j] = np.array(sums)  # Convert to numpy array

        # "Alternate Count" variable: 1 if we need "c" # of alternates for AFSC j
        m.alt = Var(((j, c) for j in p['J^Special'] for c in a_counts[j]), within=Binary)

        # List of constraints we need to enforce this concept
        m.alternate_list_constraints = ConstraintList()
        for j in p['J^Special']:  # Loop through each special AFSC to add the constraints we need

            # Number of alternates we need to pull (# Reserved slots - # reserved slots "filled")
            num_alternates = p['num_reserved'][j] - np.sum(m.x[i, j] for i in p['I^Reserved'][j])
            m.alternate_list_constraints.add(expr=np.sum(m.x[i, j] for i in p['I^Alternate'][j]) == num_alternates)

            # At most one "Alternate Count" variable can be activated (# of alternates matched)
            m.alternate_list_constraints.add(expr=np.sum(m.alt[j, c] for c in a_counts[j]) == 1)

            # Defining the "Alternate Count" variable in relation to the number of alternates
            m.alternate_list_constraints.add(  # Constraint determines which "alt" variable is activated
                expr=np.sum(a_counts[j][c - 1] * m.alt[j, c] for c in a_counts[j]) == num_alternates)

            # This constraint ensures that the cadets picked from the alternate list are at the top of the list
            m.alternate_list_constraints.add(expr=np.sum(
                a_counts[j][c - 1] * m.x[a_cadets[j][c], j] for c in a_counts[j]) == np.sum(
                a_cum_sums[j][c - 1] * m.alt[j, c] for c in a_counts[j]))

    # Cadets may sometimes be constrained to be part of one "Accessions Group" (probably just USSF)
    m.acc_grp_constraints = ConstraintList()
    if 'acc_grp_constraint' in p:
        for acc_grp in p['afscs_acc_grp']:
            for i in p['I^' + acc_grp]:
                m.acc_grp_constraints.add(expr=np.sum(m.x[i, j] for j in p['J^' + acc_grp] if j in p['J^E'][i]) == 1)

    # 5% cap on total percentage of USAFA cadets allowed into certain AFSCs
    if vp["J^USAFA"] is not None:
        cap = 0.05 * instance.mdl_p["real_usafa_n"]  # Total number of graduating USAFA cadets

        # USAFA 5% Cap Constraint
        def usafa_afscs_rule(m):
            """
            This is the 5% USAFA graduating class cap constraint for certain AFSCs (support AFSCs). I will note that
            as of Mar '23 this constraint is effectively null and void! Still here for documentation however and for any
            potential future experiment
            """
            return np.sum(np.sum(m.x[i, j] for i in p['usafa_cadets']) for j in vp["J^USAFA"]) <= cap

        m.usafa_afscs_constraint = Constraint(rule=usafa_afscs_rule)

    # Space Force PGL Constraint (Honor USSF SOC split)
    if mdl_p['ussf_soc_pgl_constraint']:

        # Necessary variables to calculate
        ussf_usafa_sum = np.sum(np.sum(m.x[i, j] for i in p['usafa_cadets'] if i in p['I^E'][j]) for j in p['J^USSF'])
        ussf_rotc_sum = np.sum(np.sum(m.x[i, j] for i in p['rotc_cadets'] if i in p['I^E'][j]) for j in p['J^USSF'])

        # SOC/USSF PGL Constraints
        m.soc_ussf_pgl_constraints = ConstraintList()
        m.soc_ussf_pgl_constraints.add(expr=ussf_usafa_sum >= p['ussf_usafa_pgl'] -
                                            mdl_p['ussf_soc_pgl_constraint_bound'] * p['ussf_usafa_pgl'])
        m.soc_ussf_pgl_constraints.add(expr=ussf_usafa_sum <= p['ussf_usafa_pgl'] +
                                            mdl_p['ussf_soc_pgl_constraint_bound'] * p['ussf_usafa_pgl'])
        m.soc_ussf_pgl_constraints.add(expr=ussf_rotc_sum >= p['ussf_rotc_pgl'] -
                                            mdl_p['ussf_soc_pgl_constraint_bound'] * p['ussf_rotc_pgl'])
        m.soc_ussf_pgl_constraints.add(expr=ussf_rotc_sum <= p['ussf_rotc_pgl'] +
                                            mdl_p['ussf_soc_pgl_constraint_bound'] * p['ussf_rotc_pgl'])

    # Space Force OM Constraint
    if vp['USSF OM']:

        # Necessary variables to calculate
        ussf_merit_sum = np.sum(np.sum(p['merit'][i] * m.x[i, j] for i in p['I^E'][j]) for j in p['J^USSF'])
        ussf_sum = np.sum(np.sum(m.x[i, j] for i in p['I^E'][j]) for j in p['J^USSF'])

        # Define constraint functions
        def ussf_om_upper_rule(m):
            """
            This is the 50% OM split constraint between the USAF and USSF (upper bound)
            """

            return ussf_merit_sum <= ussf_sum * (0.5 + mdl_p['ussf_merit_bound'])
        def ussf_om_lower_rule(m):
            """
            This is the 50% OM split constraint between the USAF and USSF (lower bound)
            """

            return ussf_merit_sum >= ussf_sum * (0.5 - mdl_p['ussf_merit_bound'])

        # Apply constraints
        m.ussf_om_constraint_upper = Constraint(rule=ussf_om_upper_rule)
        m.ussf_om_constraint_lower = Constraint(rule=ussf_om_lower_rule)

    # Cadet value constraint (Could work on any optimization model)
    m.min_cadet_value_constraints = ConstraintList()
    for i in p['I']:
        if vp['cadet_value_min'][i] != 0:
            m.min_cadet_value_constraints.add(expr=np.sum(
                p['cadet_utility'][i, j] * m.x[i, j] for j in p['J^E'][i]) >= vp['cadet_value_min'][i])

    return m  # Return the model


# Cadet Board Animation
def cadet_board_preprocess_model(b):
    """
    Builds a Pyomo optimization model to determine the x and y coordinates of AFSC squares on a chart.

    Parameters:
    -----------
    b : dict
        A dictionary containing configuration parameters for the model.
        The dictionary should include the following key-value pairs:
        - 'n^sorted' : numpy array
            Sorted values of the AFSC sizes (cadet box sizes).
        - 'M' : int
            The number of AFSCs (cadet boxes).
        - 'add_legend' : bool
            Whether to include a legend box in the chart.
        - 'simplified_model' : bool
            Whether to use a simplified model without positional constraints.
        - 'row_constraint' : bool
            Whether to incorporate a row constraint for AFSCs.
        - Additional bounds and constants used in the model.

    Returns:
    --------
    m : ConcreteModel
        The constructed Pyomo ConcreteModel instance representing the optimization model.

    Notes:
    ------
    This function creates an optimization model to determine the x and y coordinates of AFSC squares (cadet boxes)
    on a chart. The objective is to maximize the size of the cadet boxes, which are represented by the variable 'm.s'.
    The model seeks to find an optimal placement of the AFSC squares while satisfying various constraints.
    The specific constraints and objective function formulation depend on the configuration parameters provided in the 'b' dictionary.

    The function defines decision variables for the AFSC sizes ('m.s') and the x and y coordinates of each AFSC square ('m.x' and 'm.y').
    Depending on the configuration parameters, additional variables for the legend box and positional relationships between AFSCs may be included.

    Constraints are added to ensure that the AFSC squares stay within the chart borders, avoid overlapping with the legend box (if present),
    and meet any specified row constraints. The constraints vary based on whether the simplified model or the full model with positional
    relationships between AFSCs is used.

    The objective function aims to maximize the size of the cadet boxes ('m.s'), representing the objective of maximizing the visual prominence
    of each AFSC on the chart.
    """

    # Build Model
    m = ConcreteModel()

    # Use the "Sorted" values for J and n
    n = b['n^sorted']
    J = np.arange(b['M'])
    M = len(J)

    # Get desired tuples of AFSCs
    tuples = []
    for i in J:
        for j in J:
            if i != j and (j, i) not in tuples:
                tuples.append((i, j))

    # ______________________________VARIABLE DEFINITIONS______________________________
    m.s = Var(within=NonNegativeReals)  # Size of the cadet boxes (AFSC objective value)

    # Coordinates of bottom left corner of AFSC j box
    m.x = Var((j for j in J), within=NonNegativeReals)
    m.y = Var((j for j in J), within=NonNegativeReals)

    # ______________________________DUMMY VARIABLE DEFINITIONS______________________________
    pass  # Here so pycharm doesn't yell at me for the constraint line above

    if b['add_legend']:

        # 1 if AFSC j is to the right of the left edge of the legend box, 0 otherwise
        m.lga_r = Var((j for j in J), within=Binary)

        # 1 if AFSC j is above the bottom edge of the legend box, 0 otherwise
        m.lga_u = Var((j for j in J), within=Binary)

    if b['simplified_model']:

        # 1 if AFSC j is below AFSC j - 1, 0 otherwise
        m.q = Var((j for j in np.arange(1, M)), within=Binary)

    else:

        # 1 if AFSC i is to the left of AFSC j
        m.a_l = Var(((i, j) for i, j in tuples), within=Binary)

        # 1 if AFSC i is to the right of AFSC j
        m.a_r = Var(((i, j) for i, j in tuples), within=Binary)

        # 1 if AFSC i is above AFSC j
        m.a_u = Var(((i, j) for i, j in tuples), within=Binary)

        # 1 if AFSC i is below AFSC j
        m.a_d = Var(((i, j) for i, j in tuples), within=Binary)

        # Toggle for if we want to incorporate the "row constraint"
        if b['row_constraint']:

            # 1 if AFSC j is on row k, 0 otherwise
            m.lam = Var(((j, k) for j in J for k in range(b['n^rows'])), within=Binary)
            m.y_row = Var((k for k in range(b['n^rows'])), within=NonNegativeReals)

    # ______________________________OBJECTIVE FUNCTION______________________________
    def objective_function(m):
        return m.s

    m.objective = Objective(rule=objective_function, sense=maximize)

    # ____________________________________CONSTRAINTS_____________________________________
    pass  # Here so pycharm doesn't yell at me for the constraint line above

    # List of constraints that enforce AFSCs to stay within the borders
    m.border_constraints = ConstraintList()

    # List of constraints that enforce AFSCs to stay outside the legend box
    m.legend_constraints = ConstraintList()

    # More constraints
    if b['simplified_model']:

        # List of constraints that line up AFSCs in a nice grid
        m.grid_constraints = ConstraintList()
    else:

        # List of constraints that keep AFSCs from overlapping
        m.afsc_constraints = ConstraintList()

        if b['row_constraint']:

            # List of constraints that enforce the y row constraints
            m.y_row_constraints = ConstraintList()
            m.lam_constraints = ConstraintList()

    # Loop through each AFSC
    for j in J:

        # Border
        m.border_constraints.add(expr=m.x[j] >= b['bw^l'])  # Left Border
        m.border_constraints.add(expr=m.x[j] + m.s * n[j] <= b['fw'] - b['bw^r'])  # Right Border
        m.border_constraints.add(expr=m.y[j] >= b['bw^b'])  # Bottom Border
        m.border_constraints.add(expr=m.y[j] + m.s * n[j] <= b['fh'] - b['bw^t'])  # Top Border

        # Legend Dummy Definitions
        if b['add_legend']:

            m.legend_constraints.add(expr=m.x[j] + m.s * n[j] >= (b['fw'] - b['bw^r'] - b['lw']) * m.lga_r[j])
            m.legend_constraints.add(expr=m.x[j] + m.s * n[j] <= (b['fw'] - b['bw^r'] - b['lw']) * (1 - m.lga_r[j]))
            m.legend_constraints.add(expr=m.y[j] + m.s * n[j] >= (b['fh'] - b['bw^t'] - b['lh']) * m.lga_u[j])
            m.legend_constraints.add(expr=m.y[j] + m.s * n[j] <= (b['fw'] - b['bw^t'] - b['lh']) * (1 - m.lga_u[j]))

            # Enforce Legend Constraint
            m.legend_constraints.add(expr=m.y[j] + m.s * n[j] <= b['fh'] - b['bw^t'] - b['lh'] * m.lga_r[j])
            m.legend_constraints.add(expr=m.x[j] + m.s * n[j] <= b['fw'] - b['bw^r'] - b['lw'] * m.lga_u[j])

        # Toggle for if we want to incorporate the "row constraint"
        if b['row_constraint'] and not b['simplified_model']:

            # y row constraints
            m.y_row_constraints.add(
                expr=m.y[j] == np.sum(m.lam[j, k] * (m.y_row[k] - n[j] * m.s) for k in range(b['n^rows'])))
            m.lam_constraints.add(expr=np.sum(m.lam[j, k] for k in range(b['n^rows'])) == 1)

    if b['simplified_model']:

        # Pin the first AFSC to the left
        m.grid_constraints.add(expr=m.x[0] <= b['bw^l'])

        # Loop through each AFSC (after the first one)
        for j in np.arange(1, M):

            # Add the constraints to enforce the grid
            m.grid_constraints.add(expr=m.y[j] <= m.y[j - 1] - (m.s * n[j] + b['abw^ud']) * m.q[j])
            m.grid_constraints.add(expr=m.y[j] >= m.y[j - 1] * (1 - m.q[j]))
            m.grid_constraints.add(expr=m.x[j] >= (m.x[j - 1] + m.s * n[j - 1] + b['abw^lr']) * (1 - m.q[j]))
            m.grid_constraints.add(expr=m.x[j] <= b['bw^l'] * m.q[j] +
                                        (m.x[j - 1] + m.s * n[j - 1] + b['abw^lr']) * (1 - m.q[j]))

    else:

        # Loop through all AFSC "tuples"
        for i, j in tuples:

            # AFSC i is to the left of AFSC j (1) or not (0)
            m.afsc_constraints.add(expr=m.x[j] >= (m.x[i] + m.s * n[i] + b['abw^lr']) * m.a_l[i, j])

            # AFSC i is to the right of AFSC j (1) or not (0)
            m.afsc_constraints.add(expr=m.x[i] >= (m.x[j] + m.s * n[j] + b['abw^lr']) * m.a_r[i, j])

            # AFSC i is above AFSC j (1) or not (0)
            m.afsc_constraints.add(expr=m.y[i] >= (m.y[j] + m.s * n[j] + b['abw^ud']) * m.a_u[i, j])

            # AFSC i is below AFSC j (1) or not (0)
            m.afsc_constraints.add(expr=m.y[j] >= (m.y[i] + m.s * n[i] + b['abw^ud']) * m.a_d[i, j])

            # The positional relationship between AFSC i and AFSC j has to meet one of the above conditions
            m.afsc_constraints.add(expr=m.a_l[i, j] + m.a_r[i, j] + m.a_u[i, j] + m.a_d[i, j] >= 1)

    return m


def solve_cadet_board_model_direct_from_board_parameters(instance, filepath):
    """
    Solve the cadet board animation model using the provided instance parameters and save the results to a CSV file.

    Parameters:
    -----------
    instance : object
        An instance of the cadet board animation model.
        The instance should contain the necessary model parameters in the 'mdl_p' attribute.
        These parameters include board configuration details, such as size ratios, border widths, legend dimensions, etc.

    filepath : str
        The file path where the results will be saved as a CSV file.
        The file should have write permissions.

    Returns:
    --------
    None

    Notes:
    ------
    This function solves the cadet board animation model using the provided instance parameters.
    The instance parameters should include the necessary configuration details for the model, such as board dimensions, size ratios, solver information, etc.

    The function first initializes the board parameters ('b') by extracting them from the instance object.
    It calculates additional parameters, such as the figure height, border widths, AFSC border/buffer widths, and legend dimensions.

    The function then loads the AFSC data from a CSV file specified by the 'filepath' parameter.
    It assumes that the data is already sorted by the 'n' column.

    Next, the function creates the Pyomo optimization model using the 'cadet_board_preprocess_model' function,
    passing the board parameters ('b') as arguments.

    The solver executable path is set based on the 'b_solver_name' parameter from the board parameters ('b').

    The model is solved using the specified solver, and the solution time is printed.

    Finally, the x, y, and s (size) values from the solved model are extracted and stored in the AFSC dataframe.
    The updated dataframe is then saved to the specified CSV file.
    """

    # Initialize b
    b = instance.mdl_p

    # Figure Height
    b['fh'] = b['fw'] * b['fh_ratio']

    # Border Widths
    for i in ['t', 'l', 'r', 'b', 'u']:
        b['bw^' + i] = b['fw'] * b['bw^' + i + '_ratio']

    # AFSC border/buffer widths
    b['abw^lr'] = b['fw'] * b['abw^lr_ratio']
    b['abw^ud'] = b['fw'] * b['abw^ud_ratio']

    # Legend width/height
    if b['add_legend']:
        b['lw'] = b['fw'] * b['lw_ratio']
        b['lh'] = b['fw'] * b['lh_ratio']
    else:
        b['lw'], b['lh'] = 0, 0

    # Load in "b_df"
    b_df = afccp.core.globals.import_csv_data(filepath)

    # We assume this dataframe is already sorted by 'n'
    b['n^sorted'] = np.array(b_df['n'])
    b['M'] = len(b['n^sorted'])

    # Create model
    model = cadet_board_preprocess_model(b)

    # Get executable
    b["executable"] = afccp.core.globals.paths['solvers'] + b["b_solver_name"]

    # Solve Model
    print('Solving CadetBoard Model instance with solver ' + b["b_solver_name"] + '...')
    start_time = time.perf_counter()
    solver = SolverFactory(b["b_solver_name"], executable=b["executable"])
    solver.solve(model)
    print("Model solved in", round(time.perf_counter() - start_time, 2), "seconds.")

    # Get the values from the model and return them
    x, y, s = [], [], model.s.value
    for j in range(b['M']):
        x.append(model.x[j].value)
        y.append(model.y[j].value)

    # Load values back into dataframe
    b_df['x'], b_df['y'], b_df['s'] = x, y, s

    # Export to csv
    b_df.to_csv(filepath, index=False)




