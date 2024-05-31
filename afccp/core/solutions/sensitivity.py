import time
import copy
import numpy as np
import logging
import warnings
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import io

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

    # Shorthand
    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p

    # Import dataframe
    df = afccp.core.globals.import_csv_data(instance.export_paths['Analysis & Results'] + "What If List.csv")

    # Make sure the first name is in the solutions dictionary
    if df.loc[0, 'Name'] not in instance.solutions:
        raise ValueError("Error. Constraint name '" + df.loc[0, 'Name'] + "' is the baseline solution and is currently"
                                                                          " not in the solutions dictionary.")

    # Get baseline solution and evaluate it
    baseline = instance.solutions[df.loc[0, 'Name']]
    baseline = afccp.core.solutions.handling.evaluate_solution(baseline, p, vp)

    # Dictionary of metric column names and their associated variable name in the solution dictionary
    metrics_dictionary = {'Effect on Global Utility': 'z^gu', 'Effect on Cadet Utility': 'cadet_utility_overall',
                          'Effect on AFSC Utility': 'afsc_utility_overall',
                          'Effect on USAFA Cadet Utility': 'usafa_cadet_utility',
                          'Effect on ROTC Cadet Utility': 'rotc_cadet_utility',
                          'Effect on USSF Cadet Utility': 'ussf_cadet_utility',
                          'Effect on USAF Cadet Utility': 'usaf_cadet_utility',
                          'Effect on USSF AFSC Utility': 'ussf_afsc_utility',
                          'Effect on USAF AFSC Utility': 'usaf_afsc_utility',
                          'Effect on USSF AFSC Norm Score': 'weighted_average_ussf_afsc_score',
                          'Effect on USAF AFSC Norm Score': 'weighted_average_usaf_afsc_score'}

    # Dictionary of AFSC solutions
    afsc_solutions = {}

    # Loop through each constraint type
    names, con = np.array(df['Name']), 0
    for name in names[1:]:  # Skip the baseline
        con += 1  # Next constraint iteration (skips baseline too)
        c_vp = copy.deepcopy(vp)  # set of value parameters for this constraint iteration

        # Print statement
        if printing:
            print('Iteration', con, name)

        # If we don't want to re-calculate something we can skip it
        if not df.loc[con, 'Calculate']:
            print("Skipped")
            continue

        # Set the appropriate value parameters for this iteration
        if name == 'Unconstrained':  # Turn off all the AFSC objective constraints
            c_vp['constraint_type'] = np.zeros([p['M'], vp['O']])

            # Turn off cadet constraints
            c_vp['cadet_value_min'] = np.zeros(p['N'])

        elif name == 'PGL Only':  # Turn of all the AFSC objective constraints except PGL
            c_vp['constraint_type'] = np.zeros([p['M'], vp['O']])

            # Turn on PGL constraints
            k = np.where(vp['objectives'] == 'Combined Quota')[0][0]
            c_vp['constraint_type'][:, k] = np.ones(p['M']) * 2

            # Turn off cadet constraints
            c_vp['cadet_value_min'] = np.zeros(p['N'])

        elif 'Top 10 First Choice' in name:  # Cadets from the top 10% of the class need to get first choice

            if "Replace" in name:  # Do we add these constraints on top of current cadet value constraints?

                # Turn off current cadet constraints
                c_vp['cadet_value_min'] = np.zeros(p['N'])

            # Constrained slots
            constrained_slots = np.zeros(p['M'])

            # Need an algorithm to see which constraints are possible based on GOM
            sorted_cadets = np.argsort(p['merit'])[::-1]
            for i in sorted_cadets:

                # Once we're passed the top 10% we're done
                if p['merit'][i] < 0.9:
                    break

                # Loop through choices until we find one under capacity
                for choice in [0, 1, 2, 3]:
                    j = p['cadet_preferences'][i][choice]

                    # Only constrain this preference if we're under constrained capacity
                    if constrained_slots[j] < p['pgl'][j]:

                        # Constrain the utility of this preference
                        utility = p['cadet_utility'][i, j]
                        c_vp['cadet_value_min'][i] = utility

                        # Increment constrained slots by one
                        constrained_slots[j] += 1

                        # Break out of this choice
                        break

        elif name == "USSF OM":  # Turn on USSF OM constraint
            c_vp["USSF OM"] = True

        elif name == "Strict AFOCD M Tier":  # Turn on mandatory AFOCD constraints

            # Loop through each degree tier to find AFSCs that have mandatory degree tier requirements
            for t in [0, 1, 2, 3]:
                j_indices = np.where(p["t_mandatory"][:, t])[0]
                vp['constraint_type'][j_indices] = 1  # Turn on these constraints

        else:  # Skip the rest of this loop
            print("Skipped")
            df.loc[con, "Result"] = "Skipped"
            continue

        # Set of constrained objectives for each AFSC
        c_vp['K^C'] = {}  # constrained objectives
        for j in p["J"]:
            c_vp['K^C'][j] = np.where(c_vp['constraint_type'][j, :] > 0)[0].astype(int)

        # Create a duplicate instance and set its value parameters
        c_instance = copy.deepcopy(instance)
        c_instance.value_parameters = copy.deepcopy(c_vp)

        # Solve the model
        model = afccp.core.solutions.optimization.assignment_model_build(c_instance)

        # Check for feasibility
        try:

            # Solve model and get solution
            solution = afccp.core.solutions.optimization.solve_pyomo_model(c_instance, model, 'Assignment',
                                                                           printing=printing)
            solution = afccp.core.solutions.handling.evaluate_solution(solution, p, c_vp)
            afsc_solutions[name] = copy.deepcopy(solution['afsc_array'])

            # Calculate metrics
            for key, value in metrics_dictionary.items():
                df.loc[con, key] = solution[value] - baseline[value]

            # It was feasible!
            df.loc[con, "Result"] = "Feasible"

            if printing:
                print("Feasible :)")

        except:

            # Empty solution
            afsc_solutions[name] = np.array(["*" for _ in p['I']])

            # It was infeasible!
            df.loc[con, "Result"] = "Infeasible"

            if printing:
                print("Infeasible :(")

    # Export main dataframe
    df.to_csv(instance.export_paths['Analysis & Results'] + "What If List.csv", index=False)

    # Create and export solutions dataframe
    solution_df = pd.DataFrame(afsc_solutions)
    solution_df.to_csv(instance.export_paths['Analysis & Results'] + "What If Solutions.csv", index=False)


def solve_pgl_capacity_sensitivity(instance, p_dict={}, printing=True):
    """
    Doc string here
    """

    def alter_quota_max(done_iterating):
        """

        :return:
        """

        # Determine which AFSCs were most over quota
        count = instance.solution['count']
        surplus = count - p['pgl']
        percentage = count / p['pgl']

        # Determine which AFSC to alter
        sorted_afscs = np.argsort(percentage)[::-1]
        for j in sorted_afscs:

            # If we're already at our "true max", pick the next AFSC in the list
            if count[j] <= true_max[j]:
                continue

            # Calculate what the new max should be
            new_max_val = np.floor(p['pgl'][j] + surplus[j] / 2)

            # If the difference is small enough, skip straight to true max
            difference = count[j] - new_max_val
            if difference <= 5:
                new_max_val = true_max[j]

            # If this would put the maximum under the "true maximum", force it to be the true maximum
            if new_max_val < true_max[j]:
                new_max_val = true_max[j]

            # Set the new maximum value for this AFSC
            if p['quota_max'][j] == new_max_val:
                print("Iterations complete.")
                done_iterating = True
            else:
                p['quota_max'][j] = new_max_val

            # Break out of the loop
            break

        return done_iterating

    # Shorthand
    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p

    # Make the main directory if needed
    folder_path = instance.export_paths['Analysis & Results']
    if "PGL Sensitivity Analysis" not in os.listdir(folder_path):
        os.mkdir(folder_path + '/' + 'PGL Sensitivity Analysis')
    folder_path += '/' + 'PGL Sensitivity Analysis/'

    # Adjust the chart settings
    p_dict["objective"] = "Combined Quota"
    p_dict["version"] = "quantity_bar"
    p_dict['macro_chart_kind'] = 'AFSC Chart'
    p_dict['save'] = False

    # Get contents of folder to determine sub-folder this analysis will be in
    folder = os.listdir(folder_path)

    # Settings for max quotas
    true_max = copy.deepcopy(p['quota_max'])

    # If we are starting where we left off and have provided a valid Analysis folder, we import there
    if mdl_p['import_pgl_analysis_folder'] in folder:
        sub_folder_name = mdl_p['import_pgl_analysis_folder']
        folder_path += sub_folder_name + "/"

        # Print statement
        if printing:
            print("Conducting PGL sensitivity analysis on this problem instance "
                  "from imported '" + sub_folder_name + "'...")

        # Import dataframes
        capacities_df = pd.read_csv(folder_path + "Capacities.csv")
        solutions_df = pd.read_csv(folder_path + "Solutions.csv")

        # Load dictionaries
        capacities_dict = {int(col): np.array(capacities_df[col]) for col in capacities_df}
        solutions_dict = {int(col): np.array(solutions_df[col]) for col in solutions_df}

        # Determine what our last iteration was
        iteration = len(capacities_df.columns) - 1

        # Set initial quota_max
        p['quota_max'] = capacities_dict[iteration]

        # Add the current solution to the instance
        instance.add_solution(solutions_dict[iteration])

        # Process that solution to get new quota max
        alter_quota_max(done_iterating=False)

        # Set the iterations
        iteration += 1
        iterations = np.arange(iteration, iteration + mdl_p['num_pgl_analysis_iterations'])

    # If we are starting from scratch, create the new analysis folder and start that way
    else:

        # Crate the new folder
        name_determined, i = False, 1
        while not name_determined:
            sub_folder_name = "Analysis " + str(i)
            if sub_folder_name not in folder:
                folder_path += sub_folder_name + "/"
                os.mkdir(folder_path)  # Make the folder
                name_determined = True
            else:
                i += 1

        # Print statement
        if printing:
            print("Conducting PGL sensitivity analysis on this problem instance "
                  "using new '" + sub_folder_name + "'...")

        # Set essentially no maximum value for each AFSC at the beginning
        p['quota_max'] = np.array([1000 if j in p['J^NRL'] else p['quota_max'][j] for j in p['J']])

        # Create dictionaries of solution/capacity arrays
        solutions_dict, capacities_dict = {},{}

        # Set the iterations
        iterations = np.arange(mdl_p['num_pgl_analysis_iterations'])

    # Loop through each iteration
    done_iterating = False
    for iteration in iterations:

        # Update the value parameters with the new quota max
        instance.update_value_parameters()

        # Run the model
        if printing:
            print("\n\nSolving iteration", iteration, "with capacities", p['quota_max'])
        instance.solve_guo_pyomo_model(p_dict, printing=True)

        # Save solution and capacities information
        capacities_dict[iteration] = copy.deepcopy(p['quota_max'])
        solutions_dict[iteration] = instance.solution['afsc_array']

        # Process the solution
        done_iterating = alter_quota_max(done_iterating)

        # If we're done iterating, stop. Otherwise, build the chart
        if done_iterating:
            break
        else:

            # Save dataframes at each step
            print("Saving dataframes..")
            capacities_df = pd.DataFrame(capacities_dict)
            capacities_df.to_csv(folder_path + "Capacities.csv", index=False)
            solutions_df = pd.DataFrame(solutions_dict)
            solutions_df.to_csv(folder_path + "Solutions.csv", index=False)
            print("Done.")


def build_pgl_sensitivity_chart(instance, folder_path, iteration, c_dict, s_dict):

    # Adjust instance plot parameters
    instance.mdl_p = afccp.core.data.support.determine_afsc_plot_details(instance, results_chart=True)

    # Create basic AFSC Chart
    chart = afccp.core.visualizations.charts.AFSCsChart(instance)
    chart.build(chart_type="Solution", printing=False)

    # Modify chart title
    chart.fig.suptitle("", fontsize=chart.ip['title_size'])

    # Import images
    gavel_hit = mpimg.imread(afccp.core.globals.paths['files'] + 'gavel_hit.png')
    gavel_swing = mpimg.imread(afccp.core.globals.paths['files'] + 'gavel_swing.png')
    # mole = mpimg.imread(afccp.core.globals.paths['files'] + 'mole.png')

    # Num iterations
    num_iterations = len(c_dict.keys())

    # AFSCs that we're swinging or hitting
    j_swing, j_hit = None, None

    # We're not on the last iteration
    if iteration < num_iterations - 1:

        # Determine which AFSC we hit next
        indices = np.where(c_dict[iteration] - c_dict[iteration + 1] != 0)[0]
        print('afscs swing', instance.parameters['afscs'][indices])
        if len(indices) == 1:
            j_swing = indices[0]

        else:
            print("SWING: No change in capacities between iterations", iteration, "and",
                  iteration + 1)

    # We're not on the first iteration
    if iteration != 0:

        # Determine which AFSC we hit this time
        indices = np.where(c_dict[iteration - 1] - c_dict[iteration] != 0)[0]
        print('afscs hit', instance.parameters['afscs'][indices])
        if len(indices) == 1:
            j_hit = indices[0]

        else:
            print("HIT: No change in capacities between iterations", iteration, "and",
                  iteration + 1)

    # If this is the first iteration, we have to create the initial image without wack a moles
    else:

        # Save chart
        filepath = folder_path + "Iteration Start.png"
        chart.fig.savefig(filepath)
        print("Saved figure to", filepath)

    # Put on the gavel hit image
    if j_hit is not None:
        y = instance.solution['count'][j_hit]
        loc = np.where(chart.c["afscs"] == instance.parameters['afscs'][j_hit])[0][0]

        # Add gavel
        imagebox = OffsetImage(gavel_hit, zoom=0.2)  # Adjust zoom as needed
        ab = AnnotationBbox(imagebox, (loc - 0.2, y - 1), xycoords='data', boxcoords="offset points", pad=0.5,
                            frameon=False)
        chart.ax.add_artist(ab)

    # Put on the gavel swing image
    if j_swing is not None and j_hit != j_swing:
        y = instance.solution['count'][j_swing]
        loc = np.where(chart.c["afscs"] == instance.parameters['afscs'][j_swing])[0][0]

        # Add gavel
        imagebox = OffsetImage(gavel_swing, zoom=0.2)  # Adjust zoom as needed
        ab = AnnotationBbox(imagebox, (loc - 0.2, y + 10), xycoords='data', boxcoords="offset points", pad=0.5,
                            frameon=False)
        chart.ax.add_artist(ab)

    # Save chart
    filepath = folder_path + "Iteration " + str(iteration) + ".png"
    chart.fig.savefig(filepath)
    print("Saved figure to", filepath)


def generate_pgl_capacity_charts(instance, p_dict={}, printing=True):

    # Shorthand
    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p

    # Make the main directory if needed
    folder_path = instance.export_paths['Analysis & Results']
    if "PGL Sensitivity Analysis" not in os.listdir(folder_path):
        os.mkdir(folder_path + '/' + 'PGL Sensitivity Analysis')
    folder_path += 'PGL Sensitivity Analysis/'

    # Get folder information if we have it, otherwise raise error
    folder = os.listdir(folder_path)
    if mdl_p['import_pgl_analysis_folder'] in folder:
        sub_folder_name = mdl_p['import_pgl_analysis_folder']
        folder_path += sub_folder_name + "/"

        # We want another sub-folder to contain all the images from the wack-a-mole stuff
        if "Snapshots" not in os.listdir(folder_path):
            os.mkdir(folder_path + '/' + 'Snapshots')
    else:
        raise ValueError("Error. Analysis sub-folder '",
                         mdl_p['import_pgl_analysis_folder'], 'not in ' + folder_path +
                         '. Please specify valid Analysis sub-folder (e.g. "Analysis 1") through model parameter '
                         '(mdl_p) "import_pgl_analysis_folder".')

    # Adjust the chart settings
    mdl_p["objective"] = "Combined Quota"
    mdl_p["version"] = "quantity_bar"
    mdl_p['macro_chart_kind'] = 'AFSC Chart'
    mdl_p['save'] = False

    # Print statement
    if printing:
        print("Conducting PGL sensitivity analysis on this problem instance "
              "from imported '" + sub_folder_name + "'...")

    # Import dataframes
    capacities_df = pd.read_csv(folder_path + "Capacities.csv")
    solutions_df = pd.read_csv(folder_path + "Solutions.csv")

    # Get into snapshots folder
    folder_path += 'Snapshots/'

    # Load dictionaries
    capacities_dict = {int(col): np.array(capacities_df[col]) for col in capacities_df}
    solutions_dict = {int(col): np.array(solutions_df[col]) for col in solutions_df}

    # Create snapshots
    for iteration in capacities_dict:

        # Add the current solution to the instance
        instance.add_solution(solutions_dict[iteration])

        # Build the chart and save it
        build_pgl_sensitivity_chart(instance, folder_path, iteration, capacities_dict, solutions_dict)









