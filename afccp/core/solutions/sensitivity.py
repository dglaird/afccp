import time
import copy
import numpy as np
import logging
import warnings
import pandas as pd

# afccp modules
import afccp.core.globals
import afccp.core.solutions.handling
import afccp.core.solutions.optimization

# VFT Constraint Placing Algorithm
def determine_model_constraints(instance):
    """
    Iteratively evaluate the VFT (Value Focussed Thinking) model by adding constraints until a feasible solution is obtained,
    in order of importance.

    This function takes a problem instance containing parameters and value parameters as input. It starts with no constraints
    activated and gradually adds constraints in order of their importance. The function builds and solves the VFT model at each
    constraint iteration, evaluating the feasibility and objective value of the solution. The process continues until all
    constraints have been considered.

    Args:
        instance (ProblemInstance): An instance of the problem containing the problem parameters and value parameters.

    Returns:
        A tuple containing:
            - constraint_type (ndarray): The adjusted constraint type matrix, indicating the active constraints.
            - solutions_df (DataFrame): A DataFrame with the solutions for different constraint iterations.
            - report_df (DataFrame): A DataFrame containing the report of each constraint iteration, including information
                                     about the solution, the new constraint applied, the objective value, and if the solution
                                     failed or not.
    """

    # Shorthand
    p, vp, ip = instance.parameters, copy.deepcopy(instance.value_parameters), instance.mdl_p

    # Create a copy of the problem instance
    adj_instance = copy.deepcopy(instance)  # "Adjusted Instance"
    real_constraint_type = copy.deepcopy(vp["constraint_type"])  # All constraints (Not 0s)

    # Initially, we'll start with no constraints turned on
    vp["constraint_type"] = np.zeros([p["M"], vp["O"]])
    vp['K^C'] = {j: np.array([]) for j in p['J']}
    adj_instance.value_parameters = vp  # Set to instance

    # Initialize Report
    report_columns = ["Solution", "New Constraint", "Objective Value", "Failed"]
    report = {col: [] for col in report_columns}

    # Determine which model we're going to solve (either VFT or Assignment model)
    if ip['constraint_model_to_use'] == 'VFT':
        print("Initializing VFT Model Constraint Algorithm...")

        # Build the model
        model, q = afccp.core.solutions.optimization.vft_model_build(adj_instance)
        model_name, obj_metric = 'VFT', 'z'

    else:  # Assignment model
        print("Initializing Assignment Model Constraint Algorithm...")

        # Build the model
        adj_instance.mdl_p['assignment_model_obj'] = 'Global Utility'  # Force the correct objective function
        model, q = afccp.core.solutions.optimization.assignment_model_build(adj_instance), None
        model_name, obj_metric = 'Assignment', 'z^gu'
    print("Done. Solving model with no constraints active...")

    # Dictionary of solutions with different constraints!
    current_solution = afccp.core.solutions.optimization.solve_pyomo_model(
        adj_instance, model, model_name, q=q, printing=False)
    solutions = {0: current_solution}
    current_solution = afccp.core.solutions.handling.evaluate_solution(current_solution, p, vp)
    afsc_solutions = {0: current_solution['afsc_array']}

    # Add first solution to the report
    report["Solution"].append(0)
    report["Objective Value"].append(round(current_solution[obj_metric], 4))
    report["New Constraint"].append("None")
    report["Failed"].append(0)
    print("Done. New solution objective value:", str(report["Objective Value"][0]))

    # Get importance "list" based on multiplied weight
    afsc_weight = np.atleast_2d(vp["afsc_weight"]).T  # Turns 1d array into 2d column
    scaled_weights = afsc_weight * vp["objective_weight"]
    flat = np.ndarray.flatten(scaled_weights)  # flatten them
    tuples = [(j, k) for j in range(p['M']) for k in range(vp['O'])]  # get a list of tuples (0, 0), (0, 1) etc.
    tuples = np.array(tuples)
    sort_flat = np.argsort(flat)[::-1]
    importance_list = [(j, k) for (j, k) in tuples[sort_flat] if real_constraint_type[j, k] != 0]
    num_constraints = len(importance_list)

    # Begin the algorithm!
    cons = 0
    print("Running through " + str(num_constraints) + " total constraint iterations...")
    for (j, k) in importance_list:
        afsc = p["afscs"][j]
        objective = vp["objectives"][k]

        # Make a copy of the model (In case we have to remove a constraint)
        new_model = copy.deepcopy(model)

        # Calculate AFSC objective measure components
        measure, numerator = afccp.core.solutions.handling.calculate_objective_measure_matrix(
            new_model.x, j, objective, p, vp, approximate=True)

        # Add AFSC objective measure constraint
        vp['constraint_type'][j, k] = copy.deepcopy(real_constraint_type[j, k])
        new_model = afccp.core.solutions.optimization.add_objective_measure_constraint(
            new_model, j, k, measure, numerator, p, vp)
        num_measure_constraints = len(new_model.measure_constraints)

        # Update constraint type within the problem instance
        adj_instance.value_parameters['constraint_type'][j, k] = copy.deepcopy(real_constraint_type[j, k])

        # Print message
        cons += 1
        print_str = "\n------[" + str(cons) + "] AFSC " + afsc + " Objective " + objective
        print_str += "-" * (55 - len(print_str))
        print(print_str)

        # Loop through each of the constraints to validate how many are on
        num_activated = 0
        for i in list(new_model.measure_constraints):
            if new_model.measure_constraints[i].active:
                num_activated += 1
        print("Constraint", cons, "Active Constraints:", int(num_measure_constraints / 2),
                               "Validated:", int(num_activated / 2))

        # Variable to determine the outcome of this iteration
        feasible = True  # Assume model is feasible until proven otherwise

        # We can skip the quota constraint (leave it on without solving)
        if objective == "Combined Quota" and ip["skip_quota_constraint"]:
            print("Result: SKIPPED [Combined Quota]")
            solutions[cons] = copy.deepcopy(current_solution)
            skipped_obj = True

        # If our most current solution is already meeting this constraint, then we can skip this constraint
        elif vp['objective_min'][j, k] <= current_solution["objective_measure"][j, k] <= vp['objective_max'][j, k]:
            print("Result: SKIPPED [Measure:", str(round(current_solution["objective_measure"][j, k], 2)) + "], ",
                  "Range: (" + str(vp['objective_min'][j, k]) +",", str(vp['objective_max'][j, k]) + ")")
            solutions[cons] = copy.deepcopy(current_solution)
            skipped_obj = True

        # We can't skip the constraint, so we solve it
        else:
            skipped_obj = False

            # Solution Solved!
            try:
                solutions[cons] = copy.deepcopy(afccp.core.solutions.optimization.solve_pyomo_model(
                    adj_instance, new_model, model_name, q=q, printing=False))

            # Solution Failed :(
            except:
                solutions[cons] = copy.deepcopy(current_solution)
                feasible = False

        # Get solution information
        current_solution = copy.deepcopy(solutions[cons])
        current_solution = copy.deepcopy(afccp.core.solutions.handling.evaluate_solution(current_solution, p, vp))
        afsc_solutions[cons] = copy.deepcopy(current_solution['afsc_array'])

        # Add this solution to report
        report["Solution"].append(cons)
        report["New Constraint"].append(afsc + " " + objective)

        if feasible:
            report["Objective Value"].append(round(current_solution[obj_metric], 4))
            if not skipped_obj:
                print("Result: SOLVED [Z = " + str(report["Objective Value"][cons]) + "]")
            report["Failed"].append(0)
            model = copy.deepcopy(new_model)  # Save this model!

        else:
            print("Result: INFEASIBLE. Proceeding with next constraint.")
            report["Objective Value"].append(0)
            report["Failed"].append(1)

            # Update constraint type within the problem instance (Remove this constraint)
            vp["constraint_type"][j, k] = 0

        # Measure it again
        current_solution = copy.deepcopy(afccp.core.solutions.handling.evaluate_solution(current_solution, p, vp))

        # Validate solution meets the constraints:
        num_constraint_check = np.sum(vp["constraint_type"] != 0)
        print("Active Objective Measure Constraints:", num_constraint_check)
        print("Total Failed Constraints:", int(current_solution["total_failed_constraints"]))
        print("Current Objective Measure:", round(current_solution["objective_measure"][j, k], 2), "Range:",
              vp["objective_value_min"][j, k])

        for con_fail_str in current_solution["failed_constraints"]:
            print("Failed:", con_fail_str)

        # Check all other AFSC objectives to see if we're suddenly failing them now for some reason
        c = 0
        measure_fails = 0
        while c < cons:
            j_1, k_1 = importance_list[c]
            afsc_1, objective_1 = p["afscs"][j_1], vp["objectives"][k_1]
            if current_solution["objective_measure"][j_1, k_1] > (vp['objective_max'][j_1, k_1] * 1.05) or \
                    current_solution["objective_measure"][j_1, k_1] < (vp['objective_min'][j_1, k_1] * 0.95):
                print("Measure Fail:", afsc_1, objective_1, "Measure:",
                      round(current_solution["objective_measure"][j_1, k_1], 2), "Range:",
                      vp["objective_value_min"][j_1, k_1])
                measure_fails += 1
            c += 1
        print_str = "-" * 10 + " Objective Measure Fails:" + str(measure_fails)
        print(print_str + "-" * (55 - len(print_str)))

    # Build Report
    solutions_df = pd.DataFrame(afsc_solutions)
    report_df = pd.DataFrame(report)
    return vp["constraint_type"], solutions_df, report_df


# GA Population Initialization
def populate_initial_ga_solutions_from_vft_model(instance, printing=True):
    """
    This function takes a problem instance and creates several initial solutions for the genetic algorithm to evolve
    from
    :param instance: problem instance
    :param printing: whether to print something or not
    :return: initial population
    """

    if printing:
        print("Generating initial population of solutions for the genetic algorithm from the approximate VFT model...")

    # Load parameters/variables
    p, vp = instance.parameters, copy.deepcopy(instance.value_parameters)
    previous_estimate = p["quota_e"]
    initial_solutions = []

    # We get our first round of solutions by iterating on the estimated number of cadets
    if instance.mdl_p["iterate_from_quota"]:

        # Initialize variables
        deviations = np.ones(p["M"])
        quota_k = np.where(vp["objectives"] == 'Combined Quota')[0][0]
        i = 1
        while sum(deviations) > 0:

            if printing:
                print("\nSolving VFT model... (" + str(i) + ")")

            # Set the current estimated number of cadets
            current_estimate = p["quota_e"]

            try:

                # Build & solve the VFT model
                model, q = vft_model_build(instance)
                solution = solve_pyomo_model(instance, model, "VFT", q=q, printing=False)
                solution = afccp.core.solutions.handling.evaluate_solution(solution, p, vp)
                initial_solutions.append(solution['j_array'])

                # Save this estimate for quota
                previous_estimate = current_estimate

                # Update new quota information (based on the number of cadets assigned from this solution)
                instance.parameters["quota_e"] = solution["objective_measure"][:, quota_k].astype(int)

                # Validate the estimated number is within the appropriate range
                for j in p["J"]:

                    # Reset the parameter if necessary
                    if instance.parameters["quota_e"][j] < p["quota_min"][j]:
                        instance.parameters["quota_e"][j] = p["quota_min"][j]
                    elif instance.parameters["quota_e"][j] > p["quota_max"][j]:
                        instance.parameters["quota_e"][j] = p["quota_max"][j]

                # Calculate deviations and proceed with next iteration
                p = instance.parameters
                deviations = [abs(p["quota_e"][j] - current_estimate[j]) for j in p["J"]]
                i += 1

                if printing:
                    print("Current Number of Quota Differences:", sum(deviations), "with objective value of",
                          round(solution["z"], 4))

                # Don't solve this thing too many times (other stopping conditions)
                if i > instance.mdl_p["max_quota_iterations"]:
                    break

            except:

                if printing:
                    print("Something went wrong with this iteration, proceeding with overall weight technique...")

                # Revert to the previous quota estimate
                instance.parameters["quota_e"] = previous_estimate
                break

    # Solve for different overall weights on cadets/AFSCs
    weights = np.arange(instance.mdl_p["population_additions"])
    weights = weights / np.max(weights)
    for w in weights:

        if printing:
            print("\nSolving VFT model with 'w' of ", str(round(w, 2)) + "...")

        # Update overall weights
        instance.value_parameters["cadets_overall_weight"] = w
        instance.value_parameters["afscs_overall_weight"] = 1 - w

        # Solve model
        try:

            # Build & solve the model
            model, q = vft_model_build(instance)
            solution = solve_pyomo_model(instance, model, "VFT", q=q, printing=False)
            solution = afccp.core.solutions.handling.evaluate_solution(solution, p, vp)
            initial_solutions.append(solution['j_array'])

            if printing:
                print("Objective value of", round(solution["z"], 4), "obtained")
        except:

            if printing:
                print("Failed to solve. Going to next iteration...")

    instance.value_parameters = vp
    return np.array(initial_solutions)


def populate_initial_ga_solutions_from_assignment_model(instance, printing=True):
    """
    This function generates several initial solutions for the genetic algorithm to evolve from using the
    *new and improved* Assignment Problem Model as a heuristic
    """

    if printing:
        print("Generating initial population of solutions for the genetic algorithm from the approximate VFT model...")

    # Force the correct objective function
    instance.mdl_p['assignment_model_obj'] = 'Global Utility'

    # Load parameters/variables
    p, vp = instance.parameters, copy.deepcopy(instance.value_parameters)
    initial_solutions = []

    # Solve using different "global utility" matrices calculated from different overall weights on cadets/AFSCs
    weights = np.arange(instance.mdl_p["population_additions"])
    weights = weights / np.max(weights)
    for w in weights:

        if printing:
            print("\nSolving assignment model with 'w' of ", str(round(w, 2)) + "...")

        # Update global utility matrix
        instance.value_parameters['global_utility'] = np.zeros([p['N'], p['M'] + 1])
        for j in p['J']:
            instance.value_parameters['global_utility'][:, j] = w * p['cadet_utility'][:, j] + \
                                                                (1 - w) * p['afsc_utility'][:, j]

        # Solve model
        try:

            # Build & solve the model
            model = assignment_model_build(instance)
            solution = solve_pyomo_model(instance, model, "Assignment", printing=False)
            solution = afccp.core.solutions.handling.evaluate_solution(solution, p, vp)
            initial_solutions.append(solution['j_array'])

            if printing:
                print("Objective value of", round(solution["z"], 4), "obtained")
        except:

            if printing:
                print("Failed to solve. Going to next iteration...")

    instance.value_parameters = vp
    return np.array(initial_solutions)


# Optimization model "What If" Analysis
def optimization_what_if_analysis(instance, printing=True):
    """
    This function takes in an AFSC/cadet problem instance and performs some "What If" analysis based on the items listed
    in "What If List.csv". We manipulate the "value parameters" to meet these pre-defined conditions and then evaluate
    the model with the new constraints. We can then create a pareto frontier by modifying the weights on cadets/AFSCs.
    These results are all exported to a sub-folder called "What If" in the Analysis & Results folder.
    """
    pass