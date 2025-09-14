"""
This module contains all **Pyomo-based optimization model logic** used to solve the
Air Force Cadet Career Problem (AFCCP), including the GUO (Global Utility Optimization),
VFT (Value Focused Thinking), Goal Programming, and CASTLE hybrid models. It also includes
support for infeasibility diagnostics, constraint logging, and solver configuration.

Contents
--------
- **Model Construction**
    - `assignment_model_build`: Builds the GUO or original AFPC assignment model.
    - `vft_model_build`: Builds the VFT model using piecewise-linear value functions.
    - `gp_model_build`: Constructs the Goal Programming (GP) model from Rebecca’s thesis.
- **Shared Components**
    - `common_optimization_handling`: Defines x-variable and shared constraints across models.
    - `base_training_model_handling`: Adds base/training (IST) variables and constraints.
    - `initialize_castle_value_curve_function_variables`: Adds CASTLE value function vars.
    - `initialize_value_function_constraint_lists`: Initializes constraint lists for VFT/CASTLE.
    - `add_objective_measure_constraint`: Adds measure constraints for constrained objectives.
    - `add_objective_value_function_constraints`: Adds CASTLE/VFT value function constraints.
    - `add_space_force_constraints`, `add_rated_alternates_constraints`: USSF- and rated-specific logic.
- **Solving & Diagnostics**
    - `solve_pyomo_model`: Unified wrapper to build, solve, and extract solutions from models.
    - `build_solver`, `execute_solver`: Configures and runs the selected Pyomo solver.
    - `handle_infeasible_model`, `log_constraint_violations_to_file`: Infeasibility diagnostics with LP parsing.
    - `parse_lp_file`, `parse_lp_file_debug`: Extract constraint structures and detect modeling issues.
- **Goal Programming Tools**
    - `calculate_rewards_penalties`: Calculates normalized GP weights via pre-optimization.
    - Handles weighted penalties/rewards (lambda/mu), satisfaction utilities, constraint-specific terms.
- **Bubble Chart Placement**
    - `bubble_chart_configuration_model`: Determines optimal (x, y) coordinates for AFSC bubble visuals.

Workflow
--------
1. **Model Building**
    - Determine objective type (`GUO`, `VFT`, `GP`, or `CASTLE` hybrid).
    - Construct Pyomo model (`ConcreteModel`) with cadet-AFSC variables and constraints.
    - Add base/training or CASTLE extensions depending on `mdl_p` settings.
1. **Solving**
    - Configure solver (Gurobi, CBC, IPOPT, etc.) and solve the model.
    - Display solver progress bar (optional), enforce timeouts, and handle warm starts (VFT only).
    - Extract `j_array`, `x`, and optional `b_array`, `c_array`, utility vectors, and warm start components.
1. **Infeasibility Handling**
    - On failure, write `.lp` file to `infeasibility_logs/`, generate constraint violation log.
    - Parse LP file for known issues (free/unbounded vars, hard equalities, infeasible bounds).
1. **CASTLE Integration**
    - Apply piecewise value curves to constrained CASTLE AFSCs (`J^CASTLE`).
    - Weight value curve and utility objective using `w^G` blending parameter.
1. **Goal Programming**
    - Penalize over-/under-achievement of constraints (e.g., ROTC quotas, merit, gender).
    - Maximize satisfaction utilities (`S`) for cadets via lexicographic weighted objectives.
    - Run multiple submodels to calibrate reward/penalty scales.

See Also
--------
- [`solutions.handling`](../../../reference/solutions/handling/#solutions.handling):
  Used for objective computation, constraint measures, and value evaluations.
- [`solutions.algorithms`](../../../reference/solutions/algorithms/#solutions.algorithms):
  Higher-level metaheuristics and solution workflows that invoke these optimization models.
- [`data.preferences`](../../../reference/data/preferences/#data.preferences):
  Provides eligible AFSC sets, alternate lists, and preference data structures used in modeling.
"""
import time
import numpy as np
import logging
import warnings
from pyomo.environ import *
from pyomo.core.expr.current import identify_variables
import os
import datetime
import sys
import re
import threading

# afccp modules
import afccp.globals
import afccp.solutions.handling

# Ignore warnings
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


# __________________________________ASSIGNMENT MODEL (GUO) MAIN COMPONENTS______________________________________________
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

    # *New* Utility/"Cost" Matrix based on CFM preferences and cadet preferences (GUO model)
    if mdl_p['assignment_model_obj'] == 'Global Utility':
        c = vp['global_utility']

        if printing:
            print("Building assignment problem (GUO) model...")

    # Original Model Utility/"Cost" Matrix  (Original model)
    else:  # This is the "legacy" AFPC model!
        c = build_old_afpc_model_utility_matrix(p)

        if printing:
            print("Building original assignment problem model...")

    # Build Model
    m = ConcreteModel()

    # ___________________________________VARIABLE DEFINITION_________________________________
    m = common_optimization_handling(m, p, vp, mdl_p)  # Define x along with additional functional constraints

    # Base/Training model extra components
    if mdl_p['solve_extra_components']:
        m = base_training_model_handling(m, p, mdl_p)

    # Initialize CASTLE value curve variables
    if mdl_p['solve_castle_guo']:
        m = initialize_castle_value_curve_function_variables(m, p, q=p['castle_q'])

    # ___________________________________OBJECTIVE FUNCTION__________________________________
    m.objective = assignment_model_objective_function_definition(m=m, p=p, vp=vp, mdl_p=mdl_p, c=c)

    # ________________________________________CONSTRAINTS_____________________________________
    m.measure_constraints = ConstraintList()  # AFSC Objective Measure Constraints (Optional decision-maker constraints)

    # Incorporate CASTLE value curve functional constraints
    if mdl_p['solve_castle_guo']:
        m = initialize_value_function_constraint_lists(m)

        # Loop through each CASTLE AFSC to add the constraints
        for castle_afsc, j_indices in p['J^CASTLE'].items():

            # Get the number of people assigned to each AFSC under this "CASTLE" AFSC umbrella (+ ADD OTS!!)
            if 'ots' in p['SOCs']:  # If OTS is in the model, we don't have to account for them
                measure = np.sum(np.sum(m.x[i, j] for i in p['I^E'][j]) for j in j_indices)
            else:  # If we are not matching OTS, we have to account for them
                measure = np.sum(np.sum(m.x[i, j] for i in p['I^E'][j]) for j in j_indices) + p['ots_counts'][
                    castle_afsc]

            # Add the value curve constraints for this "CASTLE" AFSC
            m = add_castle_value_curve_function_constraints(m, measure, afsc=castle_afsc, q=p['castle_q'])

    # Loop through all AFSCs to add AFSC objective measure constraints
    for j in p['J']:

        # Loop through all constrained AFSC objectives
        for k in vp['K^C'][j]:
            # Calculate AFSC objective measure components
            measure, numerator = afccp.solutions.handling.calculate_objective_measure_matrix(
                m.x, j, vp['objectives'][k], p, vp, approximate=True)

            # Add AFSC objective measure constraint
            m = add_objective_measure_constraint(m, j, k, measure, numerator, p, vp)

    if printing:
        print("Done. Solving model...")

    if mdl_p['usafa_soc_pilot_cross_in']:
        j = np.where(p['afscs'] == '11XX_U')[0][0]
        k = np.where(vp['objectives'] == 'Combined Quota')[0][0]
        if (vp["objective_min"][j, k] == vp["objective_max"][j, k]) and vp['constraint_type'][j, k] != 0:
            print("*****************WARNING**********************",
                  "\nModel designated to allow USAFA Pilot cross-flows but constraints are still activated that"
                  " would prevent this.\nEnsure you turn off the USAFA/ROTC pilot quota",
                  f"constraints in '{instance.data_name} VP.csv'")

    return m  # Return model


def build_old_afpc_model_utility_matrix(p):

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
    return c


# _______________________________COMMON OPTIMIZATION COMPONENTS (Across Models)_________________________________________
def assignment_model_objective_function_definition(m, p, vp, mdl_p, c):
    """
    Define the objective function for the assignment model.

    This function constructs the objective function for an assignment optimization model.
    The objective function varies depending on whether the model includes additional components
    (`solve_extra_components`) or only considers AFSC-based assignments.

    If `solve_extra_components` is enabled, the objective function consists of two weighted
    components:
    - A cadet-specific value function weighted by `cadets_overall_weight` and individual
      cadet weights.
    - An AFSC-specific utility function weighted by `afscs_overall_weight` and normalized by `1/N`.

    If `solve_extra_components` is disabled, the function simplifies to maximizing
    the AFSC-only utility values.

    Args:
        m (ConcreteModel): The Pyomo model to which the objective function will be added.
        p (dict): A dictionary containing problem parameters, including:
            - 'I': Set of cadets.
            - 'J^E': Set of available AFSCs for each cadet.
            - 'afsc_utility': Utility values for cadet-to-AFSC assignments.
            - 'N': Normalization factor for AFSC weight.
        vp (dict): A dictionary containing value parameters, including:
            - 'cadets_overall_weight': Overall weight for cadet value.
            - 'cadet_weight': Individual weights for cadets.
            - 'afscs_overall_weight': Overall weight for AFSC utility.
        mdl_p (dict): Model parameters containing:
            - 'solve_extra_components' (bool): Determines whether additional components
              (cadet-based value) are included in the objective.
        c (dict): A dictionary of AFSC-only utility values used when `solve_extra_components`
            is disabled.

    Returns:
        Objective: The Pyomo Objective function, maximizing either the full weighted sum
        of cadet and AFSC utility (if `solve_extra_components` is True) or just AFSC utility
        (if False).
    """

    # Do we incorporate base/training decision components to the model?
    if mdl_p['solve_extra_components']:

        # Base/Training model objective function
        def objective_function(m):
            return vp['cadets_overall_weight'] * np.sum(vp['cadet_weight'][i] * m.cadet_value[i] for i in p['I']) + \
                   1 / p['N'] * vp['afscs_overall_weight'] * np.sum(
                np.sum(p['afsc_utility'][i, j] * m.x[i, j] for j in p['J^E'][i]) for i in p['I'])

    else:  # If not, we solve the "AFSC-only" assignment problem model

        # AFSC-only objective function (GUO) i.e. (not base/training component considerations)
        z_guo = np.sum(np.sum(c[i, j] * m.x[i, j] for j in p["J^E"][i]) for i in p["I"]) / p['N^Match']

        def objective_function(m):  # Standard "GUO" function value "z^GUO"
            return z_guo

        # Determine whether we want to add "CASTLE" modeling components or not
        if mdl_p['solve_castle_guo']:
            if 'castle_q' not in p:  # If we don't have castle parameters, we can't solve the model
                print("CASTLE Parameters not found. We cannot solve model w/CASTLE modifications.")
                return Objective(rule=objective_function, sense=maximize)  # Return normal GUO function
            afscs = [afsc for afsc, _ in p['castle_afscs'].items()]

            # Define the Castle-informed objective function!
            z_castle_values = np.sum(m.f_value[afsc] for afsc in afscs)
            if mdl_p['w^G'] == 1:  # Weight on Castle solution is zero so we don't need to include it!
                def objective_function(m):  # Standard "GUO" function value "z^GUO"
                    return z_guo
            elif mdl_p['w^G'] == 0:  # Weight on GUO solution is zero so just use Castle information
                def objective_function(m):  # Castle-values portion of objective function
                    return z_castle_values
            else:
                def objective_function(m):  # Full weighted function of GUO and Castle
                    return mdl_p['w^G'] * z_guo + (1 - mdl_p['w^G']) * z_castle_values

    # Objective function has been defined
    return Objective(rule=objective_function, sense=maximize)


def common_optimization_handling(m, p, vp, mdl_p):
    """
    Adds optimization model components common to *main* optimization models like VFT and the generalized assignment
    problem models.

    Parameters:
    m (ConcreteModel): The Pyomo ConcreteModel instance to which the optimization model components will be added.
    p (dict): A dictionary containing problem-specific data, including cadet, AFSC, base, course, utility, and weight
             information.
    vp (dict): A dictionary containing value-specific parameters and information.
    mdl_p (dict): A dictionary containing model-specific parameters and configurations.

    Returns:
    ConcreteModel: The modified Pyomo ConcreteModel instance with added optimization model components.

    Notes:
    - This function extends the given Pyomo ConcreteModel (m) by adding optimization model components common to
      various main optimization models.
    - The parameters include:
        - m: The Pyomo ConcreteModel instance to be extended.
        - p: A dictionary containing various problem-specific data, such as cadet information, AFSCs, bases, courses,
             utility values, and weights.
        - vp: A dictionary containing value-specific parameters and information.
        - mdl_p: A dictionary containing model-specific parameters and configurations.
    - The added optimization model components include binary variables (x), constraints for cadet AFSC assignment,
      cadet value constraints, constraints for fixed variables, reserved AFSC constraints, and constraints for
      AFSC cadet percentages, among others.
    - Additional constraints handle special cases like alternate list rated addition, 5% cap on total percentage of
      USAFA cadets allowed in certain AFSCs, USSF SOC PGL constraint, and USSF OM constraint.
    - The given ConcreteModel (m) is modified in-place and returned for further use.
    """

    # Define the x-variable
    m.x = Var(((i, j) for i in p['I'] for j in p['J^E'][i]), within=Binary)

    # Cadets receive one and only one AFSC (Ineligibility constraint is always met as a result of the indexed sets)
    m.one_afsc_constraints = ConstraintList()  # ...except for OTS candidates- there is a lot of nuance there!
    m = add_x_variable_one_afsc_constraints(m, p, mdl_p)  # Modify the "one_afsc_constraints" list for each cadet

    # Cadets may sometimes be constrained to be part of one "Accessions Group" (probably just USSF)
    m.acc_grp_constraints = ConstraintList()
    if 'acc_grp_constraint' in p:
        for acc_grp in p['afscs_acc_grp']:
            for i in p['I^' + acc_grp]:
                m.acc_grp_constraints.add(
                    expr=np.sum(m.x[i, j] for j in p['J^' + acc_grp] if j in p['J^E'][i]) == 1)

    # Cadet value constraint (Could work on any optimization model)
    m.min_cadet_value_constraints = ConstraintList()
    for i in vp['I^C']:  # "J^Top_Choice is set of AFSCs at or above designated utility value (typically top 3)
        m.min_cadet_value_constraints.add(expr=np.sum(m.x[i, j] for j in vp['J^Top_Choice'][i]) == 1)

    # "AlTERNATE LIST" Rated Addition
    if mdl_p['rated_alternates'] and 'J^Preferred [usafa]' in p:  # If [usafa] version is here, [rotc] will be too
        m = add_rated_alternates_constraints(m, p, mdl_p)

    # Fixing variables if necessary
    for i in p['J^Fixed']:
        m.x[i, p['J^Fixed'][i]].fix(1)

    # Cadets with reserved AFSC slots get constrained so that the "worst" choice they can get is their reserved AFSC
    m.reserved_afsc_constraints = ConstraintList()
    for i in p['J^Reserved']:
        m.reserved_afsc_constraints.add(expr=np.sum(m.x[i, j] for j in p['J^Reserved'][i]) == 1)

    # 5% cap on total percentage of USAFA cadets allowed into certain AFSCs
    if mdl_p["USAFA-Constrained AFSCs"] is not None:
        cap = 0.05 * mdl_p["real_usafa_n"]  # Total number of graduating USAFA cadets

        # Convert list of AFSC names to list of AFSC indices
        constrained_afscs = [np.where(p['afscs'] == afsc)[0][0] for afsc in mdl_p["USAFA-Constrained AFSCs"]]

        # USAFA 5% Cap Constraint
        def usafa_afscs_rule(m):
            """
            This is the 5% USAFA graduating class cap constraint for certain AFSCs (support AFSCs). I will note that
            as of Mar '23 this constraint is effectively null and void! Still here for documentation however and for any
            potential future experiment
            """
            return np.sum(np.sum(m.x[i, j] for i in p['usafa_cadets']) for j in constrained_afscs) <= cap

        m.usafa_afscs_constraint = Constraint(rule=usafa_afscs_rule)

    # Space Force Special Constraints!
    m = add_space_force_constraints(m, p, mdl_p)

    # OTS accessions cap
    if 'ots' in p['SOCs']:
        # Determine the cap on OTS accessions
        print(f"OTS accessions capacity constraint: {p['ots_accessions']}")

        # Define and apply the constraint
        def ots_accessions_constraint_rule(m):
            """
            Constraint to ensure we stay under OTS accessions cap
            """
            return np.sum(np.sum(m.x[i, j] for j in p['J^E'][i]) for i in p['I^OTS']) <= p['ots_accessions']

        m.ots_accessions_constraint = Constraint(rule=ots_accessions_constraint_rule)

    # Return updated model
    return m


def add_x_variable_one_afsc_constraints(m, p, mdl_p):

    # Loop through every cadet to add the functional "x" constraints!
    for i in p['I']:

        # OTS candidates can go unmatched
        if 'I^OTS' in p:
            if i in p['I^OTS']:
                if mdl_p['ots_selected_preferences_only']:

                    # If this candidate doesn't have any eligible choices (and OTS select only is on), they go unmatched
                    if len(p['J^Selected-E'][i]) == 0:

                        # If they aren't eligible for any AFSC they selected, but they do still have eligible AFSCs...
                        if len(p['J^E'][i]) > 0:  # ...we must force them to go unmatched
                            m.one_afsc_constraints.add(expr=np.sum(m.x[i, j] for j in p['J^E'][i]) == 0)

                        # If they aren't eligible for any AFSCs, then there's no need to assign a variable constraint...
                        continue  # ... there is no "x_ij" variable defined for this candidate.

                    # If the OTS candidate is matched, they must be matched to an AFSC they selected as a preference
                    else:
                        m.one_afsc_constraints.add(expr=np.sum(m.x[i, j] for j in p['J^Selected-E'][i]) == \
                                                        np.sum(m.x[i, j] for j in p['J^E'][i]))

                # If we have to constrain the "must match" OTS candidates, they MUST be MATCHED!!
                if mdl_p['ots_constrain_must_match']:
                    if i in p['I^Must_Match']:  # This candidate must receive one and only one AFSC
                        m.one_afsc_constraints.add(expr=np.sum(m.x[i, j] for j in p['J^E'][i]) == 1)
                        continue

                # The candidate DOES have preferences, but they still COULD go unmatched
                m.one_afsc_constraints.add(expr=np.sum(m.x[i, j] for j in p['J^E'][i]) <= 1)
                continue  # Next!

        # This cadet must receive one and only one AFSC
        m.one_afsc_constraints.add(expr=np.sum(m.x[i, j] for j in p['J^E'][i]) == 1)

    return m


def add_space_force_constraints(m, p, mdl_p):

    # Space Force PGL Constraint (Honor USSF SOC split)
    if mdl_p['ussf_soc_pgl_constraint'] and "USSF" in p['afscs_acc_grp']:
        # Necessary variables to calculate
        ussf_usafa_sum = np.sum(np.sum(m.x[i, j] for i in p['usafa_cadets'] if i in p['I^E'][j]) for j in p['J^USSF'])
        ussf_rotc_sum = np.sum(np.sum(m.x[i, j] for i in p['rotc_cadets'] if i in p['I^E'][j]) for j in p['J^USSF'])

        # SOC/USSF PGL Constraints
        m.soc_ussf_pgl_constraints = ConstraintList()
        m.soc_ussf_pgl_constraints.add(
            expr=ussf_usafa_sum >= p['ussf_usafa_pgl'] - mdl_p['ussf_soc_pgl_constraint_bound'] * p['ussf_usafa_pgl'])
        m.soc_ussf_pgl_constraints.add(
            expr=ussf_usafa_sum <= p['ussf_usafa_pgl'] + mdl_p['ussf_soc_pgl_constraint_bound'] * p['ussf_usafa_pgl'])
        m.soc_ussf_pgl_constraints.add(
            expr=ussf_rotc_sum >= p['ussf_rotc_pgl'] - mdl_p['ussf_soc_pgl_constraint_bound'] * p['ussf_rotc_pgl'])
        m.soc_ussf_pgl_constraints.add(
            expr=ussf_rotc_sum <= p['ussf_rotc_pgl'] + mdl_p['ussf_soc_pgl_constraint_bound'] * p['ussf_rotc_pgl'])

    # Space Force OM Constraint
    if mdl_p['USSF OM'] and "USSF" in p['afscs_acc_grp']:
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

    return m


def add_rated_alternates_constraints(m, p, mdl_p):

    # Initialize list of blocking pairs constraints for alternate lists
    m.blocking_pairs_alternates = ConstraintList()

    # Subset of Rated AFSCs that have alternate constraints
    rated_afscs_with_constraints = []
    if mdl_p['rated_alternate_afscs'] is None:
        rated_afscs_with_constraints = p['J^Rated']
    else:
        for afsc in mdl_p['rated_alternate_afscs']:
            if afsc not in p['afscs']:
                raise ValueError("AFSC '" + afsc + "' not valid.")

            # Add the index of the AFSC
            rated_afscs_with_constraints.append(np.where(p['afscs'] == afsc)[0][0])

    # Do we want to allow ROTC to fill USAFA pilot slots? This is the VERY rare situation like in FY26 where
    if mdl_p['usafa_soc_pilot_cross_in']:  # USAFA targets were high but people didn't want to fill em

        # Remove pilot AFSC from standard alternate list blocking pair constraints (handle them in special case)
        j_pilot_u = np.where(p['afscs'] == '11XX_U')[0][0]
        j_pilot_r = np.where(p['afscs'] == '11XX_R')[0][0]
        rated_afscs_with_constraints.remove(j_pilot_u)
        rated_afscs_with_constraints.remove(j_pilot_r)

        # For USAFA, loop through every pilot qualified USAFA cadet and make sure their right to a...
        for i in p['I^E'][j_pilot_u]:  # ...pilot slot is protected from ROTC (they get something better)

            # Where did this USAFA cadet rank pilot?
            choice = np.where(p['cadet_preferences'][i] == j_pilot_u)[0][0]

            # Have we already reserved an AFSC for them?
            if i not in p['J^Reserved']:

                # If they're already fixed to something, skip them
                if i in p['J^Fixed']:
                    continue
                else:  # Reserve a pilot slot for them!
                    p['J^Reserved'][i] = p['cadet_preferences'][i][:choice + 1]

            else:  # If they're already reserved for something, honor their best interest scenario
                num_reserved_already = len(p['J^Reserved'][i])
                num_to_reserve = min(choice + 1, num_reserved_already)
                p['J^Reserved'][i] = p['cadet_preferences'][i][:num_to_reserve]

        # For ROTC, we introduce blocking pair constraints but include the deficit between USAFA pilot target and..
        # for i in p['I^Alternate [rotc]'][j_pilot_r]:  # ...USAFA pilot assigned
        for i in p['I^E'][j_pilot_r]:  # TODO: Figure out why alternate list isn't working

            # Are they already fixed to something?
            if i in p['J^Fixed']:

                # If we've already matched them to pilot, skip them
                if p['J^Fixed'][i] == j_pilot_r:
                    continue

                # If we fixed them to something undesirable, don't do that  TODO: Change this in the algorithm
                if p['c_pref_matrix'][i, j_pilot_r] < p['c_pref_matrix'][i, p['J^Fixed'][i]]:
                    p['J^Fixed'].pop(i)

            # Calculate number of USAFA pilot slots up for grabs
            num_extra = p['usafa_quota'][j_pilot_u] - np.sum(m.x[i, j_pilot_u] for i in p['I^E'][j_pilot_u])

            # Add the blocking pair constraint for the rated AFSC/cadet pair
            m.blocking_pairs_alternates.add(  # "j_p"/"i_p" indicate j/i "prime" or (')
                expr=p['rotc_quota'][j_pilot_r] *
                     (1 - np.sum(m.x[i, j_p] for j_p in p['J^Preferred [rotc]'][j_pilot_r][i])) <=
                     np.sum(m.x[i_p, j_pilot_r] for i_p in p['I^Preferred [rotc]'][j_pilot_r][i]) - num_extra)

        # USAFA-ROTC pilot totals constraint
        def usafa_rotc_pilot_totals(m):
            """
            Make sure we meet the collective pilot quota
            """
            usafa_pilot_totals = np.sum(m.x[i, j_pilot_u] for i in p['I^E'][j_pilot_u])
            rotc_pilot_totals = np.sum(m.x[i, j_pilot_r] for i in p['I^E'][j_pilot_r])
            return usafa_pilot_totals + rotc_pilot_totals == p['rotc_quota'][j_pilot_r] + \
                   p['usafa_quota'][j_pilot_u]

        m.usafa_rotc_pilot_totals_constraint = Constraint(rule=usafa_rotc_pilot_totals)

    # Loop through each SOC and rated AFSC
    for soc in p['SOCs']:
        for j in rated_afscs_with_constraints:

            # Loop through each cadet on this rated AFSC's alternate list for this SOC
            for i in p['I^Alternate [' + soc + ']'][j]:  # "j_p"/"i_p" indicate j/i "prime" or (')

                # Did this cadet (i) receive a more preferred AFSC than this one (j)?
                matched_better = np.sum(m.x[i, j_p] for j_p in p['J^Preferred [' + soc + ']'][j][i])

                # How many cadets matched to this AFSC (j) that are more preferred to the AFSC than this cadet (i)?
                better_cadet_count = np.sum(m.x[i_p, j] for i_p in p['I^Preferred [' + soc + ']'][j][i])

                # Add the blocking pair constraint for the rated AFSC/cadet pair
                m.blocking_pairs_alternates.add(expr=p[soc + '_quota'][j] * (1 - matched_better) <= better_cadet_count)

    return m


# ______________________________________________CONSTRAINT LOGGING______________________________________________________
def log_constraint_violations_to_file(model, filepath, tolerance=1e-4, max_violations=100):
    """
    Logs violated constraints to a text file, suppressing uninitialized variable warnings.

    Args:
        model (ConcreteModel): The Pyomo model to inspect.
        filepath (str): Path to the log file.
        tolerance (float): Numerical tolerance for feasibility.
        max_violations (int): Max number of violations to report.
    """
    def all_vars_initialized(expr):
        """Helper to check if all variables in an expression are initialized."""
        try:
            for var in identify_variables(expr, include_fixed=False):
                if var.value is None:
                    return False
            return True
        except:
            return False

    with open(filepath, 'w') as f:
        f.write("Constraint Violations Log\n\n")
        count = 0
        for constr in model.component_objects(Constraint, active=True):
            cdata = getattr(model, constr.name)
            for index in cdata:
                con = cdata[index]
                try:
                    if not all_vars_initialized(con.body):
                        continue  # Skip uninitialized expressions

                    lhs = value(con.body)
                    lb = value(con.lower) if con.lower is not None else -np.inf
                    ub = value(con.upper) if con.upper is not None else np.inf

                    if not (lb - tolerance <= lhs <= ub + tolerance):
                        f.write(f"❌ {constr.name}[{index}]\n")
                        f.write(f"    LHS = {lhs:.6f}\n")
                        f.write(f"    Bounds = [{lb:.6f}, {ub:.6f}]\n\n")
                        count += 1
                        if count >= max_violations:
                            f.write("⚠️ Max violations reached.\n")
                            return
                except Exception as e:
                    f.write(f"⚠️ Could not evaluate {constr.name}[{index}]: {e}\n\n")
        if count == 0:
            f.write("✅ No constraint violations detected.\n")


def handle_infeasible_model(model, results, output_dir="infeasibility_logs"):
    """
    Handles an infeasible model: prints message, logs diagnostics, and raises error.

    Args:
        model (ConcreteModel): The Pyomo model.
        results (SolverResults): The results from solver.solve().
        output_dir (str): Directory to write the LP file and violation log.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    infeasible_lp_path = os.path.join(output_dir, "infeasible_model.lp")
    violations_path = os.path.join(output_dir, "constraint_violations.txt")

    print("\n🚫 Model is infeasible. Writing diagnostics...")

    # Write model to LP file for further inspection
    model.write(infeasible_lp_path, io_options={"symbolic_solver_labels": True})
    print(f"📄 Model written to: {infeasible_lp_path}")

    # Log constraint violations
    log_constraint_violations_to_file(model, violations_path)
    print(f"📄 Constraint violations written to: {violations_path}")

    # Log LP analysis
    lp_analysis_path = os.path.join(output_dir, "lp_analysis.txt")
    parse_lp_file(infeasible_lp_path, output_log=lp_analysis_path)
    lp_analysis_debug_path = os.path.join(output_dir, "lp_analysis_debug.txt")
    parse_lp_file_debug(infeasible_lp_path, output_log=lp_analysis_debug_path)

    # Raise the error
    raise ValueError("Model is infeasible. Diagnostics have been saved.")


def parse_lp_file_debug(lp_path, output_log="lp_analysis_debug.txt"):
    """
    Parses and analyzes a .lp file to flag common infeasibility patterns.

    Args:
        lp_path (str): Path to the .lp file.
        output_log (str): Output file for diagnostic results.
    """
    with open(lp_path, 'r') as f:
        lines = f.readlines()

    in_constraints = False
    in_bounds = False
    in_general = False

    constraints = []
    bounds = []
    binary_vars = set()
    constraint_vars = set()
    variable_bounds = {}

    for line in lines:
        line = line.strip()

        # Section headers
        if line.startswith("Subject To"):
            in_constraints = True
            continue
        if line.startswith("Bounds"):
            in_constraints = False
            in_bounds = True
            continue
        if line.startswith("General") or line.startswith("Binary"):
            in_bounds = False
            in_general = True
            continue
        if line.startswith("End"):
            in_general = False

        # Constraints
        if in_constraints:
            constraints.append(line)
            constraint_vars.update(re.findall(r'\b\w+\b', line))

        # Bounds
        elif in_bounds:
            bounds.append(line)
            # Examples:
            # 0 <= x1 <= 1
            # x3 free
            parts = re.findall(r'[\w\.-]+', line)
            if "free" in line:
                variable_bounds[parts[0]] = ("-inf", "inf", "free")
            elif len(parts) == 5:
                variable_bounds[parts[2]] = (parts[0], parts[4], "bounded")

        # Binary/General
        elif in_general:
            binary_vars.update(re.findall(r'\b\w+\b', line))

    flagged_issues = []

    # Analyze bounds
    for var, (lb, ub, status) in variable_bounds.items():
        if status == "free":
            flagged_issues.append(f"⚠️ Variable {var} is free — may lead to unboundedness.")
        else:
            try:
                if float(lb) > float(ub):
                    flagged_issues.append(f"❌ Variable {var} has inconsistent bounds: {lb} > {ub}")
            except ValueError:
                flagged_issues.append(f"⚠️ Variable {var} has non-numeric bounds: {lb}, {ub}")

    # Analyze constraints
    eq_constraints = [c for c in constraints if re.search(r"= [\d\.-]+", c)]
    for c in eq_constraints:
        flagged_issues.append(f"⚠️ Hard equality constraint: {c.strip()}")

    # Check for undeclared variables
    for var in constraint_vars:
        if var not in variable_bounds and var not in binary_vars and not var.startswith("c"):
            flagged_issues.append(f"⚠️ Variable {var} used in constraints but not declared.")

    # Check for large coefficients
    for c in constraints:
        coeffs = re.findall(r'[\+\-]?\s*\d+\.\d+', c)
        for coeff in coeffs:
            val = float(coeff.replace(" ", ""))
            if abs(val) > 1e6 or abs(val) < 1e-6:
                flagged_issues.append(f"⚠️ Extreme coefficient ({val}) in: {c.strip()}")

    # Write log
    with open(output_log, 'w') as out:
        out.write("=== LP Debug Analysis ===\n\n")
        out.write("✅ Basic stats:\n")
        out.write(f"  - Constraints: {len(constraints)}\n")
        out.write(f"  - Bounds: {len(bounds)}\n")
        out.write(f"  - Binary/General vars: {len(binary_vars)}\n\n")

        if flagged_issues:
            out.write("❗ Potential Issues Found:\n")
            for issue in flagged_issues:
                out.write("  " + issue + "\n")
        else:
            out.write("✅ No obvious issues found.\n")

        out.write("\n=== Sample Constraints ===\n")
        for c in constraints[:10]:
            out.write("  " + c + "\n")

        out.write("\n=== Sample Bounds ===\n")
        for b in bounds[:10]:
            out.write("  " + b + "\n")

    print(f"📄 LP analysis written to: {output_log}")


def parse_lp_file(lp_path, output_log="lp_analysis.txt"):
    """
    Parses a .lp file and logs constraints and bounds for debugging.

    Args:
        lp_path (str): Path to the .lp file.
        output_log (str): File to write analysis log to.
    """
    with open(lp_path, 'r') as f:
        lines = f.readlines()

    in_constraints = False
    in_bounds = False
    in_general = False
    constraints = []
    bounds = []
    binary_vars = []

    for line in lines:
        line = line.strip()

        if line.startswith("Subject To"):
            in_constraints = True
            continue
        if line.startswith("Bounds"):
            in_constraints = False
            in_bounds = True
            continue
        if line.startswith("General") or line.startswith("Binary"):
            in_bounds = False
            in_general = True
            continue
        if line.startswith("End"):
            in_general = False

        if in_constraints:
            constraints.append(line)
        elif in_bounds:
            bounds.append(line)
        elif in_general:
            binary_vars.append(line)

    with open(output_log, 'w') as out:
        out.write("=== Constraint Section ===\n")
        for c in constraints[:50]:  # show only first 50 for brevity
            out.write(c + '\n')

        out.write("\n=== Bounds Section ===\n")
        for b in bounds[:50]:
            out.write(b + '\n')

        out.write("\n=== Binary/General Variables ===\n")
        for v in binary_vars:
            out.write(v + '\n')

        out.write("\n✅ Finished parsing LP file.\n")

    print(f"✔️ LP file parsed and analyzed. Results written to: {output_log}")


# _________________________________________SOLVE THE MODEL WITH A SOLVER________________________________________________
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
    if model_name == 'Bubbles':
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
    solver = build_solver(model_name, mdl_p, printing)

    # Solve Model
    start_time = time.perf_counter()
    model = execute_solver(model, solver, mdl_p)
    solve_time = round(time.perf_counter() - start_time, 2)

    # # Log infeasibility
    # if (results.solver.status != SolverStatus.ok) or (
    #         results.solver.termination_condition == TerminationCondition.infeasible):
    #     timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #     folder_path = f"{instance.export_paths['Analysis & Results']}infeasibility_logs {timestamp}"
    #     handle_infeasible_model(model, results, output_dir=folder_path)

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

    # "Bubble Chart" optimization model
    elif model_name == 'Bubbles':

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
            solution = {"method": model_name, "j_array": np.zeros(p['N']).astype(int), "x": np.zeros((p['N'], p['M'])),
                        'solve_time': solve_time, 'x_integer': True}

            # Loop through each cadet to determine what AFSC they're assigned
            for i in p['I']:
                found = False
                for j in p['J^E'][i]:
                    solution['x'][i, j] = model.x[i, j].value
                    try:
                        if round(solution['x'][i, j]):
                            solution['j_array'][i] = int(j)
                            found = True

                        if 0.01 < solution['x'][i, j] < 0.99:
                            solution['x_integer'] = False
                    except:
                        raise ValueError("Solution didn't come out right, likely model is infeasible.")

                # For some reason we may not have assigned a cadet to an AFSC in which case we just give them to one
                # they're eligible for and want (happens usually to only 1-3 people through VFT model)
                if not found:

                    # OTS candidates may go unmatched
                    if 'I^OTS' in p:
                        if i in p['I^OTS']:
                            solution['j_array'][i] = p['M']  # Cadet assigned to "unmatched" AFSC!
                            continue

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

            # Get objective value
            solution['pyomo_obj_value'] = round(model.objective(), 4)
            return solution

        # Obtain base/training solution components from the model
        def obtain_extra_solution_components(solution):
            """
            This nested function obtains the base/training variable components from the pyomo model
            """

            solution['b_array'] = np.zeros(p['N']).astype(int)
            solution['c_array'] = np.array([(0, 0) for _ in p['I']])
            solution['base_array'] = np.array([" " * 100 for _ in p['I']])
            solution['course_array'] = np.array([" " * 100 for _ in p['I']])
            solution['v'] = np.zeros((p['N'], p['S'])).astype(int)
            solution['q'] = np.zeros((p['N'], p['M'], max(p['T']))).astype(int)
            solution['cadet_value (Pyomo)'] = np.zeros(p['N'])
            solution['v_integer'], solution['q_integer'] = True, True

            # Loop through each cadet to determine what base they're assigned to
            for i in p['I']:
                found = False
                for b in p['B^E'][i]:
                    solution['v'][i, b] = model.v[i, b].value
                    try:
                        if round(solution['v'][i, b]):
                            solution['b_array'][i] = int(b)
                            found = True

                        if 0.01 < solution['v'][i, b] < 0.99:
                            warm_start['v_integer'] = False
                    except:
                        raise ValueError("Solution didn't come out right, likely model is infeasible.")

                if found:
                    solution['base_array'][i] = p['bases'][solution['b_array'][i]]
                else:  # Not matched to a base
                    solution['base_array'][i] = ""
                    solution['b_array'][i] = p['S']

            # Loop through each cadet to determine what course they're assigned to
            for i in p['I']:
                found = False
                for j in p['J^E'][i]:
                    for c in p['C^E'][i][j]:
                        solution['q'][i, j, c] = model.q[i, j, c].value
                        try:
                            if round(solution['q'][i, j, c]):
                                solution['c_array'][i] = (j, c)
                                found = True

                            if 0.01 < solution['q'][i, j, c] < 0.99:
                                warm_start['q_integer'] = False
                        except:
                            raise ValueError("Solution didn't come out right, likely model is infeasible.")

                if found:
                    solution['course_array'][i] = p['courses'][solution['c_array'][i][0]][solution['c_array'][i][1]]
                else:  # Not matched to a course
                    print('Cadet', i, 'not matched to a course for some reason. Something went wrong.')

            # Loop through each cadet to get their value from pyomo
            for i in p['I']:
                solution['cadet_value (Pyomo)'][i] = model.cadet_value[i].value

            return solution

        solution = obtain_solution()

        # Base/Training Model components
        if mdl_p['solve_extra_components']:
            solution = obtain_extra_solution_components(solution)

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
            warm_start = {'f(measure)': np.zeros([p['M'], vp['O']]), 'r^max': max_r + 1,
                          'lambda': np.zeros([p['M'], vp['O'], max_r + 1]),
                          'y': np.zeros([p['M'], vp['O'], max_r + 1]).astype(int), 'obj': model.objective(),
                          'y_original': np.zeros([p['M'], vp['O'], max_r + 1]), 'y_integer': True}

            # Load warm start variables
            for j in p['J']:
                for k in vp['K^A'][j]:
                    warm_start['f(measure)'][j, k] = model.f_value[j, k].value
                    for l in range(q['r'][j, k]):
                        warm_start['lambda'][j, k, l] = model.lam[j, k, l].value
                        if l < q['r'][j, k] - 1:
                            warm_start['y'][j, k, l] = round(model.y[j, k, l].value)
                            warm_start['y_original'][j, k, l] = model.y[j, k, l].value
                            if 0.01 < warm_start['y_original'][j, k, l] < 0.99:
                                warm_start['y_integer'] = False

            # Return the "warm start" dictionary
            return warm_start

        # Add additional components to solution dictionary
        if mdl_p["obtain_warm_start_variables"] and 'VFT' in model_name:
            warm_start = obtain_warm_start_variables()
            for key in warm_start:
                solution[key] = warm_start[key]

        if printing:
            timestamp = datetime.datetime.now().strftime('%B %d %Y %r')
            print(f"Model solved in {solve_time} seconds at {timestamp}. Pyomo reported objective value:",
                  solution['pyomo_obj_value'])

        # Return solution dictionary
        return solution


def build_solver(model_name, mdl_p, printing):

    # Determine how the solver is called here
    if mdl_p["executable"] is None:
        if mdl_p["provide_executable"]:
            if mdl_p["exe_extension"]:
                mdl_p["executable"] = afccp.globals.paths['solvers'] + mdl_p["solver_name"] + '.exe'
            else:
                mdl_p["executable"] = afccp.globals.paths['solvers'] + mdl_p["solver_name"]
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
                specific_model_name = "Approximate VFT Model"
            else:
                specific_model_name = "Exact VFT Model"
        else:
            specific_model_name = model_name + " Model"

        if mdl_p['solve_extra_components']:
            specific_model_name += " (w/base & training components)"
        if mdl_p['solve_castle_guo']:
            specific_model_name += " (w/CASTLE value curve modifications)"

        timestamp = datetime.datetime.now().strftime('%B %d %Y %r')
        print(f'Solving {specific_model_name} instance with solver {mdl_p["solver_name"]}...'
              f'\nStart Time: {timestamp}.')
    return solver


def execute_solver(model, solver, mdl_p):
    """
    Solves the Pyomo model with a progress bar if a time limit is specified.

    Args:
        model (ConcreteModel): The Pyomo model to solve.
        solver (SolverFactory): The solver instance.
        mdl_p (dict): Dictionary of model parameters including solver config.

    Returns:
        ConcreteModel: The solved model.
    """
    def print_progress_bar(duration, stop_event):
        start_time = time.time()
        bar_length = 40
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            percent = min(elapsed / duration, 0.99)
            filled_len = int(bar_length * percent)
            bar = '█' * filled_len + '-' * (bar_length - filled_len)
            eta = int(duration - elapsed) if elapsed < duration else 1
            sys.stdout.write(f'\r⏳ Solver running: |{bar}| {percent*100:5.1f}% ETA: ~{eta}s ')
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write(f'\r✅ Solver complete: |{"█" * bar_length}| 100.0%              \n')
        sys.stdout.flush()

    # If time limit is set, start the timer bar
    if mdl_p.get("pyomo_max_time"):
        stop_event = threading.Event()
        target_time = int(min(mdl_p["pyomo_max_time"] * 1.05, mdl_p["pyomo_max_time"] + 40))
        thread = threading.Thread(target=print_progress_bar, args=(target_time, stop_event))
        thread.start()

    # Solve Model
    try:
        if mdl_p["pyomo_max_time"] is not None:
            if mdl_p["solver_name"] == 'mindtpy':
                solver.solve(model, time_limit=mdl_p["pyomo_max_time"], tee=mdl_p['pyomo_tee'])
                # mip_solver='cplex_persistent', nlp_solver='ipopt')
            elif mdl_p["solver_name"] == 'gurobi':
                solver.solve(model, options={'TimeLimit': mdl_p["pyomo_max_time"],
                                             'IntFeasTol': 0.05}, tee=mdl_p['pyomo_tee'])
            elif mdl_p["solver_name"] == 'ipopt':
                solver.options['max_cpu_time'] = mdl_p["pyomo_max_time"]
                solver.solve(model, tee=mdl_p['pyomo_tee'])
            elif mdl_p["solver_name"] == 'cbc':
                solver.options['seconds'] = mdl_p["pyomo_max_time"]
                solver.solve(model, tee=mdl_p['pyomo_tee'])
            elif mdl_p["solver_name"] == 'baron':
                solver.solve(model, options={'MaxTime': mdl_p["pyomo_max_time"]}, tee=mdl_p['pyomo_tee'])
            else:
                solver.solve(model, tee=mdl_p['pyomo_tee'])
        else:
            if mdl_p["solver_name"] == 'mindtpy':
                model.pprint()
                solver.solve(model, mip_solver='cplex_persistent', nlp_solver='ipopt', tee=mdl_p['pyomo_tee'])
            else:
                solver.solve(model, tee=mdl_p['pyomo_tee'])

    finally:
        if mdl_p.get("pyomo_max_time"):
            stop_event.set()
            thread.join()

    return model


# _________________________________________CASTLE MARKET MODEL COMPONENTS_______________________________________________
def initialize_castle_value_curve_function_variables(m, p, q):
    # Castle AFSCs
    afscs = [afsc for afsc, _ in p['castle_afscs'].items()]
    m.f_value = Var((afsc for afsc in afscs), within=NonNegativeReals)  # AFSC objective value
    m.lam = Var(((afsc, l) for afsc in afscs for l in q['L'][afsc]),
                within=NonNegativeReals, bounds=(0, 1))  # Lambda and y variables for value functions
    m.y = Var(((afsc, l) for afsc in afscs for l in range(q['r'][afsc] - 1)), within=Binary)

    # Return updated model
    return m


def add_castle_value_curve_function_constraints(m, measure, afsc, q):
    # Add Linear Value Function Constraints
    m.measure_vf_constraints.add(expr=measure == np.sum(  # Measure Constraint for Value Function (20a)
        q['a'][afsc][l] * m.lam[afsc, l] for l in q['L'][afsc]))
    m.value_vf_constraints.add(expr=m.f_value[afsc] == np.sum(  # Value Constraint for Value Function (20b)
        q['f^hat'][afsc][l] * m.lam[afsc, l] for l in q['L'][afsc]))

    # Lambda .. y constraints (20c, 20d, 20e)
    m.lambda_y_constraint1.add(expr=m.lam[afsc, 0] <= m.y[afsc, 0])  # (20c)
    if q['r'][afsc] > 2:
        for l in range(1, q['r'][afsc] - 1):
            m.lambda_y_constraint2.add(expr=m.lam[afsc, l] <= m.y[afsc, l - 1] + m.y[afsc, l])  # (20d)
    m.lambda_y_constraint3.add(expr=m.lam[afsc, q['r'][afsc] - 1] <= m.y[afsc, q['r'][afsc] - 2])  # (20e)

    # Y sum to 1 constraint (20f)
    m.y_sum_constraint.add(expr=np.sum(m.y[afsc, l] for l in range(0, q['r'][afsc] - 1)) == 1)

    # Lambda sum to 1 constraint (20g)
    m.lambda_sum_constraint.add(expr=np.sum(m.lam[afsc, l] for l in q['L'][afsc]) == 1)

    # Lambda .. value positive constraint (20h) although the "f_value" constraint is implied in the thesis
    for l in q['L'][afsc]:
        m.lambda_positive.add(expr=m.lam[afsc, l] >= 0)
    m.f_value_positive.add(expr=m.f_value[afsc] >= 0)

    # Return updated model
    return m


# _____________________________________VALUE FOCUSED THINKING COMPONENTS________________________________________________
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
             "f^hat": [[[vp['f^hat'][j][k][l] for l in vp['L'][j][k]] for k in vp['K']] for j in
                       p['J']]}  # Values of bps

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
    if mdl_p['solve_extra_components']:

        # Base/Training model extra components
        m = base_training_model_handling(m, p, mdl_p)

        def objective_function(m):  # Z^VFT (w/base/training revision on cadet_value)
            """
            The objective function is to maximize "Z", the overall weighted sum of all VFT objectives
            """
            return vp['afscs_overall_weight'] * np.sum(vp['afsc_weight'][j] * np.sum(
                vp['objective_weight'][j, k] * m.f_value[j, k] for k in vp['K^A'][j]) for j in p['J']) + \
                   vp['cadets_overall_weight'] * np.sum(vp['cadet_weight'][i] * m.cadet_value[i] for i in p['I'])

    else:

        # AFSC-only objective function
        def objective_function(m):  # Z^VFT (Definition of variable in written formulation)
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

    # Value Function Constraints: Linking main methodology with value function methodology...
    m = initialize_value_function_constraint_lists(m)  # ...and then enforcing that methodology

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
                measure, numerator = afccp.solutions.handling.calculate_objective_measure_matrix(
                    m.x, j, objective, p, vp, approximate=instance.mdl_p['approximate'])

                # Add Value Function constraints (for functionality)
                m = add_objective_value_function_constraints(m, j, k, measure, q=q)

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


def initialize_value_function_constraint_lists(m):
    """
    Initialize constraint lists for the value function methodology.

    This function sets up empty constraint lists in the Pyomo model to enforce the relationships
    between the primary methodology and the value function methodology. These constraints
    are used later when defining the value function constraints.

    The constraint lists initialized correspond to the following formulations:

    - `measure_vf_constraints` (20a): Ensures the measure is computed using a weighted sum.
    - `value_vf_constraints` (20b): Computes the value function as a weighted sum.
    - `lambda_y_constraint1` (20c): Ensures the first lambda variable is bounded by y.
    - `lambda_y_constraint2` (20d): Ensures intermediate lambda variables are bounded by y variables.
    - `lambda_y_constraint3` (20e): Ensures the last lambda variable is bounded by y.
    - `y_sum_constraint` (20f): Ensures the y variables sum to 1.
    - `lambda_sum_constraint` (20g): Ensures the lambda variables sum to 1.
    - `lambda_positive` (20h): Enforces non-negativity on lambda variables.
    - `f_value_positive`: Enforces non-negativity on the AFSC objective value.

    Args:
        m (ConcreteModel): The Pyomo model to which the constraint lists will be added.

    Returns:
        ConcreteModel: The updated Pyomo model with initialized constraint lists.
    """

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

    # Return updated model
    return m


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

    try:
        # "Number of Cadets" objectives handled separately
        if vp['objectives'][k] in ['Combined Quota', 'USAFA Quota', 'ROTC Quota', 'OTS Quota']:
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

    except Exception as error:

        print("AFSC '" + p['afscs'][j] + "' Objective '" + vp['objectives'][k] + " constraint failed to add.")
        print("Exception:", error)

    # Return updated model
    return m


def add_objective_value_function_constraints(m, j, k, measure, q):
    """
    Add linear value function constraints to the Pyomo model.

    This function incorporates constraints related to the value function into the Pyomo optimization model.
    These constraints ensure that the measure and value function constraints are properly enforced, the lambda
    variables are bounded by y variables, and that summation and positivity constraints hold.

    The constraints implemented correspond to the following formulations:

    - Measure Constraint (20a): Ensures the measure is computed as a weighted sum of coefficients.
    - Value Function Constraint (20b): Computes the value function as a weighted sum of given parameters.
    - Lambda-Y Constraints (20c, 20d, 20e): Enforce relationships between lambda and y variables.
    - Y Summation Constraint (20f): Ensures the sum of y values equals 1.
    - Lambda Summation Constraint (20g): Ensures the sum of lambda values equals 1.
    - Lambda and Value Function Positivity Constraints (20h): Enforces non-negativity of lambda and the value function.

    Args:
        m (ConcreteModel): The Pyomo model to which the constraints will be added.
        j (int): The index representing the AFSC.
        k (int): The index representing the objective.
        measure (Expression): The measure variable in the value function.
        q (dict): A dictionary containing problem parameters, including:
            - 'a': Coefficients for the measure function.
            - 'f^hat': Coefficients for the value function.
            - 'L': Set of lambda indices.
            - 'r': The range parameter defining the number of lambda variables.

    Returns:
        ConcreteModel: The updated Pyomo model with the value function constraints added.
    """

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

    # Return updated model
    return m


# __________________________________________BASE & IST COMPONENTS_______________________________________________________
def base_training_model_handling(m, p, mdl_p):
    """
    Adds optimization model components to handle base and training (IST) assignments.

    Parameters:
    m (ConcreteModel): The Pyomo ConcreteModel instance to which the optimization model components will be added.
    p (dict): A dictionary containing problem data, including cadet, base, course, utility, and weight information.
    mdl_p (dict): A dictionary containing model-specific parameters and configurations.

    Returns:
    ConcreteModel: The modified Pyomo ConcreteModel instance with added optimization model components.

    Notes:
    - This function extends the given Pyomo ConcreteModel (m) by adding optimization model components to handle base and
      training (IST) assignments for cadets.
    - The parameters are as follows:
        - m: The Pyomo ConcreteModel instance to be extended.
        - p: A dictionary containing various problem data, such as cadet information, base information, course
          information, utility values, and weight information.
        - mdl_p: A dictionary containing model-specific parameters and configurations.
    - The added optimization model components include variables and constraints to handle cadet assignments to bases,
      courses, and values based on utility outcomes and constraints to ensure that assigned bases and courses do not
      exceed their capacities.
    - Cadet assignments are modeled using binary variables (v and q), which represent assignments to bases and courses,
      respectively.
    - Constraints are formulated to ensure cadet value calculations based on cadet states, base assignments, course
      assignments, and utility values.
    - Additionally, constraints ensure that cadet assignments to bases and courses do not exceed base and course
      capacities.

    Note: The given ConcreteModel (m) is modified in-place and returned for further use.
    """

    # Define the v and q-variables
    m.v = Var(((i, b) for i in p['I'] for b in p['B^E'][i]), within=Binary)
    m.q = Var(((i, j, c) for i in p['I'] for j in p['J^E'][i] for c in p['C^E'][i][j]), within=Binary)

    # Define the new cadet value variable
    m.cadet_value = Var((i for i in p['I']), within=NonNegativeReals, bounds=(0, 1))

    # Cadet Value Constraints. Define what the "cadet_value" variable should be based on which "state" the cadet is in.
    m.bc_cadet_value_constraints = ConstraintList()
    for i in p['I']:
        for d in p['D'][i]:

            # Calculate auxiliary variables for AFSC, base, course utility outcomes
            u = {
                'A': (1 / p['u^S'][i][d]) * np.sum(p['cadet_utility'][i, j] * m.x[i, j] for j in p['J^State'][i][d]),
                'C': np.sum(np.sum(
                    p['course_utility'][i][j][c] * m.q[i, j, c] for c in p['C^E'][i][j]) for j in p['J^State'][i][d])
            }

            # Weighted sum of cadet utilities in each area depends on if bases are involved
            if len(p['B^State'][i][d]) > 0:
                u['B'] = np.sum(p['base_utility'][i, b] * m.v[i, b] for b in p['B^State'][i][d])
                weighted_sum = p['w^A'][i][d] * u['A'] + p['w^B'][i][d] * u['B'] + p['w^C'][i][d] * u['C']
            else:
                weighted_sum = p['w^A'][i][d] * u['A'] + p['w^C'][i][d] * u['C']

            # This is the base/course state cadet value constraint. It ensures cadet_value will be the right value
            m.bc_cadet_value_constraints.add(expr=m.cadet_value[i] <= p['u^S'][i][d] * weighted_sum +
                                                  mdl_p['BIG M'] * (1 - np.sum(m.x[i, j] for j in p['J^State'][i][d])))

    # Cadet Base Constraints. If a cadet is assigned to a base, it has to be one that the AFSC is located at
    m.bc_cadet_base_constraints = ConstraintList()
    for i in p['I']:
        m.bc_cadet_base_constraints.add(expr=np.sum(m.v[i, b] for b in p['B^E'][i]) ==
                                             np.sum(m.x[i, j] for j in np.intersect1d(p['J^E'][i], p['J^B'])))
        for j in np.intersect1d(p['J^B'], p['J^E'][i]):
            m.bc_cadet_base_constraints.add(expr=m.x[i, j] <= np.sum(m.v[i, b] for b in p['B^A'][j]))

    # Cadet Course Constraints. Cadets have to be assigned to a course for their designated AFSC
    m.bc_cadet_course_constraints = ConstraintList()
    for i in p['I']:
        for j in p['J^E'][i]:
            m.bc_cadet_course_constraints.add(expr=m.x[i, j] == np.sum(m.q[i, j, c] for c in p['C^E'][i][j]))

    # Base Capacity Constraints. A base/AFSC pair cannot exceed its capacity
    m.bc_base_capacity_constraints = ConstraintList()
    for j in p['J^B']:
        for b in p['B^A'][j]:
            m.bc_base_capacity_constraints.add(expr=np.sum(m.v[i, b] for i in p['I^E'][j]) <= p['hi^B'][j][b])
            m.bc_base_capacity_constraints.add(expr=np.sum(m.v[i, b] for i in p['I^E'][j]) >= p['lo^B'][j][b])

    # Course Capacity Constraints. A course/AFSC pair cannot exceed its capacity
    m.bc_course_capacity_constraints = ConstraintList()
    for j in p['J']:
        for c in p['C'][j]:
            m.bc_course_capacity_constraints.add(expr=np.sum(m.q[i, j, c] for i in p['I^A'][j][c]) <= p['hi^C'][j][c])
            m.bc_course_capacity_constraints.add(expr=np.sum(m.q[i, j, c] for i in p['I^A'][j][c]) >= p['lo^C'][j][c])

    return m


# _____________________________________________BUBBLE CHART MODEL!______________________________________________________
def bubble_chart_configuration_model(b):
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

    # Initialize Model
    m = ConcreteModel()  # Concrete model allows native numpy/python objects
    n = b['n^sorted']  # Number of squares on edge of AFSC box (use sorted AFSCs)
    J = np.arange(b['M'])  # Use sorted AFSCs!
    M = len(J)

    # ______________________________VARIABLE DEFINITIONS________________________________________________________
    m.s = Var(within=NonNegativeReals)  # Size of the cadet boxes (AFSC objective value)
    m.x = Var((j for j in J), within=NonNegativeReals)  # X coordinate of bottom left corner of AFSC j box
    m.y = Var((j for j in J), within=NonNegativeReals)  # Y coordinate of bottom left corner of AFSC j box
    m.q = Var((j for j in np.arange(1, M)), within=Binary)  # 1 if AFSC j is below AFSC j - 1, 0 otherwise

    # ______________________________OBJECTIVE FUNCTION__________________________________________________________
    def objective_function(m):
        return m.s  # Objective is to maximize the size of the cadet boxes!

    m.objective = Objective(rule=objective_function, sense=maximize)

    # ____________________________________CONSTRAINTS___________________________________________________________
    # Loop through each AFSC to add the border constraints
    m.border_constraints = ConstraintList()  # List of constraints that enforce AFSCs to stay within the borders
    m.border_constraints.add(expr=m.x[0] <= b['bw^l'])  # Pin the first AFSC to the left
    for j in J:
        m.border_constraints.add(expr=m.x[j] >= b['bw^l'])  # Left Border
        m.border_constraints.add(expr=m.x[j] + m.s * n[j] <= b['fw'] - b['bw^r'])  # Right Border
        m.border_constraints.add(expr=m.y[j] >= b['bw^b'])  # Bottom Border
        m.border_constraints.add(expr=m.y[j] + m.s * n[j] <= b['fh'] - b['bw^t'])  # Top Border

    # Loop through each AFSC (after the first one) to add the grid constraints
    m.grid_constraints = ConstraintList()  # List of constraints that line up AFSCs in a nice grid
    for j in np.arange(1, M):
        m.grid_constraints.add(expr=m.y[j] <= m.y[j - 1] - (m.s * n[j] + b['abw^ud']) * m.q[j])
        m.grid_constraints.add(expr=m.y[j] >= m.y[j - 1] * (1 - m.q[j]))
        m.grid_constraints.add(expr=m.x[j] >= (m.x[j - 1] + m.s * n[j - 1] + b['abw^lr']) * (1 - m.q[j]))
        m.grid_constraints.add(expr=m.x[j] <= b['bw^l'] * m.q[j] +
                                    (m.x[j - 1] + m.s * n[j - 1] + b['abw^lr']) * (1 - m.q[j]))
    return m


# ______________________________GOAL PROGRAMMING (REBECCA THESIS) COMPONENTS____________________________________________
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