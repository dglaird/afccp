# Import libraries
import numpy as np
import pandas as pd
import copy
import afccp.core.globals
import afccp.core.handling.preprocessing


# Fixed Parameter Procedures
def get_utility_preferences(parameters):
    """
    Converts utility matrix into two arrays of preferences and utilities (NxP for each)
    :param parameters: fixed cadet/afsc data
    :return: preference matrix and utilities matrix
    """
    preferences = np.array([[" " * 10 for _ in range(parameters['P'])] for _ in range(parameters['N'])])
    utilities_array = np.zeros([parameters['N'], parameters['P']])
    for i in range(parameters['N']):

        # Sort indices of nonzero utilities
        indices = parameters['utility'][i, :].nonzero()[0]
        sorted_init = np.argsort(parameters['utility'][i, :][indices])[::-1]
        sorted_indices = indices[sorted_init][:6]  # Just take the top 6 preferences for now

        # Put the utilities and preferences in the correct spots
        np.put(utilities_array[i, :], np.arange(len(sorted_indices)), parameters['utility'][i, :][sorted_indices])
        np.put(preferences[i, :], np.arange(len(sorted_indices)), parameters['afscs'][sorted_indices])

    return preferences, utilities_array


def get_utility_preferences_from_preference_array(parameters):
    """
    Takes the cadet preference matrix (NxM) of cadet "ranks" and converts it to preference columns (NxP) of AFSC names.
    Uses this alongside utility dataframe (NxP) to get the utility columns (NxP) as well.
    """

    # Shorthand
    p = parameters

    # Initialize data
    preference_matrix = copy.deepcopy(p["c_pref_matrix"])
    preferences = np.array([[" " * 10 for _ in range(p['P'])] for _ in range(p['N'])])
    utilities_array = np.zeros([p['N'], p['P']])
    for i in range(p['N']):

        # Eliminate AFSCs that weren't in the cadet's preference list (Change the choice to a large #)
        zero_indices = np.where(preference_matrix[i, :] == 0)[0]
        preference_matrix[i, zero_indices] = 100

        # Get the ordered list of AFSCs
        indices = np.argsort(preference_matrix[i, :])  # [::-1]  #.nonzero()[0]
        ordered_afscs = p["afscs"][indices][:p["M"] - len(zero_indices)][:p["P"]]
        ordered_utilities = p["utility"][i, indices][:p["M"] - len(zero_indices)][:p["num_util"]]

        # Put the utilities and preferences in the correct spots
        np.put(utilities_array[i, :], np.arange(len(ordered_utilities)), ordered_utilities)
        np.put(preferences[i, :], np.arange(len(ordered_afscs)), ordered_afscs)

    return preferences, utilities_array


def convert_utility_matrices_preferences(parameters):
    """
    This function converts the cadet and AFSC utility matrices into the preference dataframes
    """
    p = parameters

    # Loop through each cadet to get their preferences
    p["c_pref_matrix"] = np.zeros([p["N"], p["M"]]).astype(int)
    for i in p["I"]:

        # Sort the utilities to get the preference list
        utilities = p["utility"][i, :p["M"]]
        sorted_indices = np.argsort(utilities)[::-1]
        preferences = np.argsort(sorted_indices) + 1  # Add 1 to change from python index (at 0) to rank (start at 1)
        p["c_pref_matrix"][i, :] = preferences

    # Loop through each AFSC to get their preferences
    p["a_pref_matrix"] = np.zeros([p["N"], p["M"]]).astype(int)
    for j in p["J"]:

        # Sort the utilities to get the preference list
        utilities = p["afsc_utility"][:, j]
        sorted_indices = np.argsort(utilities)[::-1]
        preferences = np.argsort(sorted_indices)
        p["a_pref_matrix"][:, j] = preferences
    return p


def generate_fake_afsc_preferences(parameters, value_parameters=None):
    """
    This function generates fake AFSC utilities/preferences using AFOCD, merit, cadet preferences etc.
    :param value_parameters: set of cadet/AFSC weight and value parameters
    :param parameters: cadet/AFSC fixed data
    :return: parameters
    """
    p, vp = parameters, value_parameters
    N, M = p["N"], p["M"]

    # Create AFSC Utility Matrix
    p["afsc_utility"] = np.zeros([N, M])
    if vp is None:

        # If we don't have a set of value_parameters, we just make some assumptions
        weights = {"Merit": 80, "Mandatory": 100, "Desired": 50, "Permitted": 30, "Utility": 60}
        for objective in ['Merit', 'Mandatory', 'Desired', 'Permitted', 'Utility']:
            if objective.lower() in p:

                if objective == "Merit":
                    merit = np.tile(p['merit'], [M, 1]).T
                    p["afsc_utility"] += merit * weights[objective]
                elif objective == "Mandatory":
                    p["afsc_utility"] += p[objective.lower()][:, :M] * weights[objective]
    else:

        # If we do have a set of value_parameters, we incorporate them
        for objective in ['Merit', 'Mandatory', 'Desired', 'Permitted', 'Utility']:
            if objective in vp['objectives']:

                k = np.where(vp['objectives'] == objective)[0][0]
                if objective == "Merit":
                    merit = np.tile(p['merit'], [M, 1]).T
                    p["afsc_utility"] += merit * vp['objective_weight'][:, k].T
                else:
                    p["afsc_utility"] += p[objective.lower()][:, :M] * vp['objective_weight'][:, k].T

    p["afsc_utility"] *= p["eligible"]  # They have to be eligible!

    # Create AFSC Preference Matrix
    p["a_pref_matrix"] = np.zeros([p["N"], p["M"]]).astype(int)
    for j in p["J"]:

        # Sort the utilities to get the preference list
        utilities = p["afsc_utility"][:, j]
        sorted_indices = np.argsort(utilities)[::-1]
        preferences = np.argsort(sorted_indices)
        p["a_pref_matrix"][:, j] = preferences

    return p


def convert_afsc_preferences_to_percentiles(parameters):
    """
    This method takes the AFSC preference lists and turns them into normalized percentiles for each cadet for each
    AFSC.
    :param parameters: cadet/AFSC fixed data
    :return: parameters
    """

    # Shorthand
    p = parameters

    # Get normalized percentiles (Average of 0.5)
    p["afsc_utility"] = (p["N"] - p["a_pref_matrix"]) / p["N"]  # Too simple!

    # First weed out all those who are ineligible for each AFSC
    # p["afsc_utility"] = np.ones((p["N"], p["M"]))
    #
    # print(p["afsc_utility"])
    # for j in p["J"]:
    #
    #     p["afsc_utility"][:, j] = (p["num_eligible"][j] - p["a_pref_matrix"][:, j]) / p["num_eligible"][j]
    #
    #
    # All ineligible cadets are given percentiles of 0
    p["afsc_utility"] *= p["eligible"]
    #
    # print(p["afsc_utility"])

    return p


# Solution Handling Procedures
def swap_solution_shape(solution, M):
    """
    Changes the solution from a vector of length N containing AFSC indices to an NxM binary matrix, or vice versa
    depending on the input.
    :param M: number of AFSCs
    :param solution: either vector or matrix
    :return: either matrix or vector
    """
    N = len(solution)

    # if we're dealing with a matrix, convert to vector
    if len(solution.shape) == 2:
        changed_solution = np.where(solution == 1)[1]
        return changed_solution

    # if we're dealing with a vector, convert to matrix
    elif len(solution.shape) == 1:
        changed_solution = np.zeros([N, M])
        for i in range(N):
            changed_solution[i, int(solution[i])] = 1
        return changed_solution


def compare_solutions(baseline, compared, printing=False):
    """
    Takes two solutions (in vector form) to the same problem (must be same set of cadets/AFSCs) and returns how similar
    the compared solution is to the baseline in terms of what AFSCs cadets were assigned to
    :param printing: whether or not the procedure should print something
    :param baseline: solution 1
    :param compared: solution 2, compared against the baseline
    :return: the percentage of solution 2 that is the same as solution 1
    """
    percent_similar = (sum(baseline == compared * 1) / len(baseline))
    if printing:
        print("The two solutions are " + str(percent_similar) + "% the same.")
    return percent_similar


def value_function(a, f_a, r, x):
    """
    This is the AFSC objective value function
    :param r: number of breakpoints
    :param a: measure at each breakpoint
    :param f_a: value at each breakpoint
    :param x: actual AFSC objective measure
    :return: AFSC objective value
    """
    # Find which breakpoint is immediately before this measure
    indices = np.array([a[l] <= x <= a[l + 1] for l in range(r - 1)]) * 1
    l = np.where(indices)[0]

    # Obtain value
    if len(l) == 0:
        l = r - 1
        val = f_a[l]
    else:
        l = l[0]
        val = f_a[l + 1] - ((f_a[l + 1] - f_a[l]) / (a[l + 1] - a[l])) * (a[l + 1] - x)

    # Return value
    return val


def measure_solution_quality(solution, parameters, value_parameters, printing=False, approximate=False):
    """
    This procedure takes in a solution (either vector or matrix), as well as the fixed
    cadet/AFSC model parameters and the weight/value parameters, and then evaluates the solution. The solution metrics
    are returned. 
    :param approximate: whether we measure the approximate value or not (using target quota instead of count)
    :param printing: Whether or not the procedure should print the matrix
    :param solution: either vector or matrix of matched cadets to AFSCs
    :param parameters: fixed model cadet/AFSC input parameters
    :param value_parameters: weight/value parameters
    :return: solution metrics
    """

    # Shorthand notation
    p = parameters
    vp = value_parameters

    # Construct X matrix from vector
    if len(np.shape(solution)) == 1:
        x_matrix = False
        x = np.array([[1 if solution[i] == j else 0 for j in p['J']] for i in p['I']])
    else:
        x_matrix = True
        x = solution

    # Create metrics dictionary
    metrics = {'objective_measure': np.zeros([p['M'], vp['O']]), 'objective_value': np.ones([p['M'], vp['O']]),
               'afsc_value': np.zeros(p['M']), 'cadet_value': np.zeros(p['N']),
               'cadet_constraint_fail': np.zeros(p['N']), 'afsc_constraint_fail': np.zeros(p['M']),
               'objective_score': np.zeros(vp['O']), 'total_failed_constraints': 0, 'x': x, "failed_constraints": [],
               'objective_constraint_fail': np.array([[" " * 30 for _ in range(vp['O'])] for _ in range(p['M'])])}

    # Get certain objective indices
    obj_indices = {}
    p_lookup_dict = {"Norm Score": "a_pref_matrix", "Merit": "merit", "Male": "male", "Minority": "minority",
                     "Mandatory": "mandatory", "Desired": "desired", "Permitted": "permitted",
                     "USAFA Proportion": "usafa", "Utility": "utility"}
    for t in ['1', '2', '3', '4']:
        p_lookup_dict["Tier " + t] = "tier " + t

    # Loop through each potential objective
    for objective in p_lookup_dict:
        if p_lookup_dict[objective] in p and objective in vp["objectives"]:
            obj_indices[objective] = np.where(vp['objectives'] == objective)[0][0]

    # Loop through all AFSCs to assign their individual values
    for j in p['J']:

        # AFSC name
        afsc = p["afscs"][j]

        # Number of assigned cadets
        count = np.sum(x[i, j] for i in p['I^E'][j])

        # Only calculate value if we assigned at least one cadet (otherwise AFSC has a value of 0)
        if count != 0:

            # Get more variables for this AFSC
            if 'usafa' in p:
                usafa_count = np.sum(x[i, j] for i in p['I^D']['USAFA Proportion'][j])

            # Are we using approximate measures or not
            if approximate:
                num_cadets = p['quota_e'][j]
            else:
                num_cadets = count

            # Loop through all objectives that this AFSC cares about
            for k in vp['K^A'][j]:

                # Get the correct measure for this objective
                objective = vp['objectives'][k]
                if objective == 'Merit':
                    numerator = np.sum(p['merit'][i] * x[i, j] for i in p['I^E'][j])
                    metrics['objective_measure'][j, k] = numerator / num_cadets
                elif objective == 'Utility':
                    numerator = np.sum(p['utility'][i, j] * x[i, j] for i in p['I^E'][j])
                    metrics['objective_measure'][j, k] = numerator / num_cadets
                elif objective == 'Combined Quota':
                    metrics['objective_measure'][j, k] = count
                elif objective == 'USAFA Quota':
                    metrics['objective_measure'][j, k] = usafa_count
                elif objective == 'ROTC Quota':
                    metrics['objective_measure'][j, k] = count - usafa_count
                elif objective in vp['K^D']:
                    numerator = np.sum(x[i, j] for i in p['I^D'][objective][j])
                    metrics['objective_measure'][j, k] = numerator / num_cadets
                elif objective == "Norm Score":
                    best_range = range(count)
                    best_sum = np.sum(c for c in best_range)
                    worst_range = range(p["num_eligible"][j] - count, p["num_eligible"][j])
                    worst_sum = np.sum(c for c in worst_range)
                    achieved_sum = np.sum(p["a_pref_matrix"][i, j] * x[i, j] for i in p["I^E"][j])
                    metrics['objective_measure'][j, k] = 1 - (achieved_sum - best_sum) / (worst_sum - best_sum)

                # Get the correct value for this objective
                metrics['objective_value'][j, k] = value_function(a=vp['a'][j][k],
                                                                  f_a=vp['f^hat'][j][k],
                                                                  r=vp['r'][j][k],
                                                                  x=metrics['objective_measure'][j, k])

                # AFSC Objective Constraints
                if k in vp['K^C'][j]:

                    # Constrained Value (this isn't used though)
                    if vp['constraint_type'][j, k] == 1 or vp['constraint_type'][j, k] == 2:

                        # If the objective value is less than the lower bound on value, constraint is failed
                        if metrics['objective_value'][j, k] < float(vp['objective_value_min'][j, k]):
                            metrics['objective_constraint_fail'][j, k] = str(
                                round(metrics['objective_value'][j, k], 3)) + ' < ' + str(
                                float(vp['objective_value_min'][j, k]))
                            metrics['total_failed_constraints'] += 1
                            metrics["failed_constraints"].append(afsc + " " + objective)

                    # Constrained Approximate Measure
                    elif vp['constraint_type'][j, k] == 3:

                        # PGL should be more "forgiving" as a constraint
                        quota_j = p["pgl"][j]
                        value_list = vp['objective_value_min'][j, k].split(",")
                        min_measure = float(value_list[0].strip())
                        max_measure = float(value_list[1].strip())

                        # Get correct measure constraint
                        if objective not in ['Combined Quota', 'USAFA Quota', 'ROTC Quota']:
                            if numerator / quota_j < min_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    round(numerator / quota_j, 3)) + ' < ' + str(
                                    min_measure) + '. ' + str(round(100 * (numerator / quota_j) /
                                                                    min_measure, 2)) + '% Met.'
                                metrics['total_failed_constraints'] += 1
                                metrics["failed_constraints"].append(afsc + " " + objective)
                            elif numerator / quota_j > max_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    round(numerator / quota_j, 3)) + ' > ' + str(
                                    max_measure) + '. ' + str(round(100 * max_measure /
                                                                    (numerator / quota_j), 2)) + '% Met.'
                                metrics['total_failed_constraints'] += 1
                                metrics["failed_constraints"].append(afsc + " " + objective)
                        else:
                            if metrics['objective_measure'][j, k] < min_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    metrics['objective_measure'][j, k]) + ' < ' + str(
                                    min_measure) + '. ' + str(round(100 * (metrics['objective_measure'][j, k]) /
                                                                    min_measure, 2)) + '% Met.'
                                metrics['total_failed_constraints'] += 1
                                metrics["failed_constraints"].append(afsc + " " + objective)
                            elif metrics['objective_measure'][j, k] > max_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    metrics['objective_measure'][j, k]) + ' > ' + str(
                                    min_measure) + '. ' + str(round(100 * max_measure /
                                                                    metrics['objective_measure'][j, k], 2)) + '% Met.'
                                metrics['total_failed_constraints'] += 1
                                metrics["failed_constraints"].append(afsc + " " + objective)

                    # Constrained Exact Measure
                    elif vp['constraint_type'][j, k] == 4:

                        value_list = vp['objective_value_min'][j, k].split(",")
                        min_measure = float(value_list[0].strip())
                        max_measure = float(value_list[1].strip())

                        # Get correct measure constraint
                        if objective not in ['Combined Quota', 'USAFA Quota', 'ROTC Quota']:
                            if numerator / count < min_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    round(numerator / count, 3)) + ' < ' + str(
                                    min_measure) + '. ' + str(round(100 * (numerator / count) /
                                                                    min_measure, 2)) + '% Met.'
                                metrics['total_failed_constraints'] += 1
                                metrics["failed_constraints"].append(afsc + " " + objective)
                            elif numerator / count > max_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    round(numerator / count, 3)) + ' > ' + str(
                                    max_measure) + '. ' + str(round(100 * max_measure /
                                                                    (numerator / count), 2)) + '% Met.'
                                metrics['total_failed_constraints'] += 1
                                metrics["failed_constraints"].append(afsc + " " + objective)
                        else:
                            if metrics['objective_measure'][j, k] < min_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    metrics['objective_measure'][j, k]) + ' < ' + str(
                                    min_measure) + '. ' + str(round(100 * (metrics['objective_measure'][j, k]) /
                                                                    min_measure, 2)) + '% Met.'
                                metrics['total_failed_constraints'] += 1
                                metrics["failed_constraints"].append(afsc + " " + objective)
                            elif metrics['objective_measure'][j, k] > max_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    metrics['objective_measure'][j, k]) + ' > ' + str(
                                    max_measure) + '. ' + str(round(100 * max_measure /
                                                                    metrics['objective_measure'][j, k], 2)) + '% Met.'
                                metrics['total_failed_constraints'] += 1
                                metrics["failed_constraints"].append(afsc + " " + objective)

            # AFSC individual value
            metrics['afsc_value'][j] = np.dot(vp['objective_weight'][j, :], metrics['objective_value'][j, :])
            if metrics['afsc_value'][j] < vp['afsc_value_min'][j]:
                metrics['afsc_constraint_fail'][j] = 1
                metrics['total_failed_constraints'] += 1
                metrics["failed_constraints"].append(afsc + " Value")

            # Loop through each objective that we want to calculate objective measures for
            for objective in p_lookup_dict:
                if p_lookup_dict[objective] in p and objective in vp["objectives"]:
                    if obj_indices[objective] not in vp['K^A'][j]:

                        # Merit
                        if objective == "Merit":
                            numerator = np.sum(p['merit'][i] * x[i, j] for i in p['I^E'][j])
                            metrics['objective_measure'][j, obj_indices[objective]] = numerator / num_cadets

                        # Demographics
                        elif objective in ['USAFA Proportion', 'Male', 'Minority']:
                            numerator = np.sum(x[i, j] for i in p['I^D'][objective][j])
                            metrics['objective_measure'][j, obj_indices[objective]] = numerator / num_cadets

                        # Check to see if we determined a value function for these objectives
                        elif objective in ["Mandatory", "Desired", "Permitted", "Tier 1", "Tier 2", "Tier 3", "Tier 4"]:
                            if len(vp["f^hat"][j][obj_indices[objective]]) != 0:
                                numerator = np.sum(x[i, j] for i in p['I^D'][objective][j])
                                metrics['objective_measure'][j, obj_indices[objective]] = numerator / num_cadets

                        # Utility
                        elif objective == "Utility":
                            numerator = np.sum(p['utility'][i, j] * x[i, j] for i in p['I^E'][j])
                            metrics['objective_measure'][j, obj_indices[objective]] = numerator / num_cadets

                        # Norm Score
                        else:
                            best_range = range(count)
                            best_sum = np.sum(c for c in best_range)
                            worst_range = range(p["num_eligible"][j] - count, p["num_eligible"][j])
                            worst_sum = np.sum(c for c in worst_range)
                            achieved_sum = np.sum(p["a_pref_matrix"][i, j] * x[i, j] for i in p["I^E"][j])
                            metrics['objective_measure'][j, obj_indices[objective]] = 1 - (achieved_sum - best_sum) / \
                                                                                      (worst_sum - best_sum)

    # Loop through all cadets to assign their values
    for i in p['I']:
        metrics['cadet_value'][i] = np.sum(x[i, j] * p['utility'][i, j] for j in p['J^E'][i])
        if metrics['cadet_value'][i] < vp['cadet_value_min'][i]:
            metrics['cadet_constraint_fail'][i] = 1
            metrics['total_failed_constraints'] += 1
            metrics["failed_constraints"].append("Cadet " + str(p['cadets'][i]) + " Value")

    # Generate objective scores for each objective
    for k in vp['K']:
        new_weights = vp['afsc_weight'] * vp['objective_weight'][:, k]
        new_weights = new_weights / sum(new_weights)
        metrics['objective_score'][k] = np.dot(new_weights, metrics['objective_value'][:, k])

    # Solution Matched Vector
    metrics['afsc_solution'] = np.array([" " * 10 for _ in p['I']])

    # Get the AFSC solution (Modified to support "unmatched" cadets)
    metrics["num_unmatched"] = 0
    for i in p['I']:
        index = np.where(x[i, :])[0]
        if len(index) != 0:
            afsc_index = int(index[0])
        else:
            metrics["num_unmatched"] += 1
            afsc_index = 32
        metrics['afsc_solution'][i] = p['afscs'][afsc_index]

    # Initialize dictionaries for cadet choice based on demographics
    dd = {"usafa": ["USAFA", "ROTC"], "male": ["Male", "Female"]}
    demographic_dict = {cat: [dd[cat][0], dd[cat][1]] for cat in dd if cat in p}
    metrics["choice_counts"] = {"TOTAL": {}}
    for cat in demographic_dict:
        for dem in demographic_dict[cat]:
            metrics["choice_counts"][dem] = {}

    # Initialize arrays within the choice dictionaries for the AFSCs
    choice_categories = ["Top 3", "Next 3", "Non-Volunteers", "Total"]
    for dem in metrics["choice_counts"]:
        for c_cat in choice_categories:
            metrics["choice_counts"][dem][c_cat] = np.zeros(p["M"]).astype(int)
        for afsc in p["afscs"]:
            metrics["choice_counts"][dem][afsc] = np.zeros(p["P"]).astype(int)

    # Loop through each AFSC
    for j, afsc in enumerate(p["afscs"]):

        # Skip unmatched AFSC
        if afsc != "*":

            # The cadets that were assigned to this AFSC
            dem_cadets = {"TOTAL": np.where(metrics["afsc_solution"] == afsc)[0]}

            # The cadets with the demographic that were assigned to this AFSC
            for cat in demographic_dict:
                dem_1, dem_2 = demographic_dict[cat][0], demographic_dict[cat][1]
                dem_cadets[dem_1] = np.intersect1d(np.where(p[cat] == 1)[0], dem_cadets["TOTAL"])
                dem_cadets[dem_2] = np.intersect1d(np.where(p[cat] == 0)[0], dem_cadets["TOTAL"])

            # Loop through each choice and calculate the metric
            for choice in range(p["P"]):

                # The cadets that were assigned to this AFSC that placed it in their Pth choice
                choice_cadets = np.where(p["c_preferences"][:, choice] == afsc)[0]
                assigned_choice_cadets = np.intersect1d(choice_cadets, dem_cadets["TOTAL"])

                # The cadets that were assigned to this AFSC, placed it in their Pth choice, and had the specific demographic
                for dem in metrics["choice_counts"]:
                    metrics["choice_counts"][dem][afsc][choice] = len(
                        np.intersect1d(assigned_choice_cadets, dem_cadets[dem]))

            # Loop through each demographic
            for dem in metrics["choice_counts"]:
                metrics["choice_counts"][dem]["Total"][j] = int(len(dem_cadets[dem]))
                metrics["choice_counts"][dem]["Top 3"][j] = int(np.sum(metrics["choice_counts"][dem][afsc][:3]))
                metrics["choice_counts"][dem]["Next 3"][j] = int(np.sum(metrics["choice_counts"][dem][afsc][3:]))
                metrics["choice_counts"][dem]["Non-Volunteers"][j] = int(
                    len(dem_cadets[dem]) - np.sum(metrics["choice_counts"][dem][afsc]))


    # Define overall metrics
    metrics['cadets_overall_value'] = np.dot(vp['cadet_weight'], metrics['cadet_value'])
    metrics['afscs_overall_value'] = np.dot(vp['afsc_weight'], metrics['afsc_value'])
    metrics['z'] = vp['cadets_overall_weight'] * metrics['cadets_overall_value'] + \
                   vp['afscs_overall_weight'] * metrics['afscs_overall_value']
    metrics['num_ineligible'] = np.sum(x[i, j] * p['ineligible'][i, j] for j in
                                    p['J'] for i in p['I'])
    if printing:

        if approximate:
            model_type = 'approximate'
        else:
            model_type = 'exact'
        if x_matrix:
            solution_type = 'matrix'
        else:
            solution_type = 'vector'

        print("Measured " + model_type + " solution " + solution_type +
              " objective value: " + str(round(metrics['z'], 4)) +
              ". Unmatched cadets: " + str(metrics["num_unmatched"]))

    return metrics


def ga_fitness_function(chromosome, parameters, value_parameters, constraints='Fail', penalty_scale=1.3,
                        con_fail_dict=None, printing=False, first=True):
    """
    This function takes in a chromosome (solution vector) and evaluates it. This function is only here because I wanted
    to test the function to make sure it was returning the same objective value as the "measure_solution_quality"
    function since the two calculate the objective value a little differently. This one uses a vector of AFSCs and
    can therefore use sets more efficiently while the other one uses a binary solution matrix which is more like
    how the model actually calculates the objective value. Assuming integer solutions (and no constraints are violated),
    the two functions will return the same value.
    :param first: if this is a solution in the initial population
    :param con_fail_dict: dictionary used for constraints
    :param penalty_scale: how much to penalize failed constraints
    :param constraints: how we handle failed constraints
    :param printing: whether the procedure should print something
    :param value_parameters: weight and value parameters
    :param parameters: cadet/AFSC parameters
    :param chromosome: solution vector
    :return: fitness score
    """

    # Shorthand
    p = parameters
    vp = value_parameters
    objective_min = np.zeros((p['M'], vp['O']))
    objective_max = np.zeros((p['M'], vp['O']))

    # Initialize metrics
    metrics = {'objective_measure': np.zeros([p['M'], vp['O']]), 'objective_value': np.zeros([p['M'], vp['O']]),
               'afsc_value': np.zeros(p['M']), 'cadet_value': np.zeros(p['N']),
               'afsc_constraint_fail': np.zeros(p['M']), 'objective_constraint_fail': np.zeros([p['M'], vp['O']]),
               'total_failed_constraints': 0, 'cadet_constraint_fail': np.zeros(p['N'])}

    # Determine if we should calculate usafa count
    soc_counts = np.zeros(p['M'])
    for j in p['J']:

        # If we care about USAFA Count and ROTC Count
        if 'USAFA Count' in vp['objectives'][vp['K^A'][j]]:
            soc_counts[j] = 1

        # Get minimum/maximum values for constraints
        for k in vp['K^C'][j]:

            # Approximate or Exact Value Constraint
            if vp['objective_constraint_type'][j, k] == 1 or vp['objective_constraint_type'][j, k] == 2:
                objective_min[j, k] = float(vp['objective_value_min'][j, k])

            # Approximate or Exact Measure Constraint
            elif vp['objective_constraint_type'][j, k] == 3 or vp['objective_constraint_type'][j, k] == 4:
                value_list = vp['objective_value_min'][j, k].split(",")
                objective_min[j, k] = float(value_list[0].strip())
                objective_max[j, k] = float(value_list[1].strip())

    # If USAFA counts are used
    if sum(soc_counts) == 0:
        ignore_uc = True
    else:
        ignore_uc = False

    # Constraint penalty variables
    failed = False
    penalty = 0

    # Loop through all AFSCs to assign their values
    for j in p['J']:

        # list of indices of assigned cadets
        cadets = np.where(chromosome == j)[0]

        # Only calculate measures for AFSCs with at least one cadet
        count = len(cadets)
        usafa_count = count

        # If there's at least one cadet assigned to the AFSC, we can calculate a non-zero value
        if count > 0:
            if not ignore_uc:
                usafa_cadets = np.intersect1d(p['I^D']['USAFA Proportion'][j], cadets)
                usafa_count = len(usafa_cadets)

            # Loop through all AFSC objectives that apply to this AFSC
            for k in vp['K^A'][j]:
                objective = vp['objectives'][k]
                if objective == 'Merit':
                    metrics['objective_measure'][j, k] = np.mean(p['merit'][cadets])
                elif objective == 'Utility':
                    metrics['objective_measure'][j, k] = np.mean(p['utility'][cadets, j])
                elif objective == 'Combined Quota':
                    metrics['objective_measure'][j, k] = count
                elif objective == 'USAFA Quota':
                    metrics['objective_measure'][j, k] = usafa_count
                elif objective == 'ROTC Quota':
                    metrics['objective_measure'][j, k] = count - usafa_count
                elif objective in vp['K^D']:
                    metrics['objective_measure'][j, k] = len(np.intersect1d(p['I^D'][objective][j], cadets)) / count

                # Calculate AFSC objective value
                metrics['objective_value'][j, k] = value_function(vp['a'][j][k], vp['f^hat'][j][k], vp['r'][j][k],
                                                                  metrics['objective_measure'][j, k])

                # AFSC Objective Constraints
                if k in vp['K^C'][j]:

                    # We're really only ever going to constrain the approximate measure for Mandatory Tier
                    if objective == 'Mandatory':
                        constrained_measure = (metrics['objective_measure'][j, k] * count) / p['pgl'][j]
                    else:
                        constrained_measure = metrics['objective_measure'][j, k]

                    # Use the real constraint (potentially different as a result of approximate model)
                    constrained_min, constrained_max = objective_min[j, k], objective_max[j, k]

                    # This constraint fail dictionary keeps track of the constraints that are failed by the
                    # optimization model by maybe 1 constraint
                    if con_fail_dict is not None:
                        if (j, k) in con_fail_dict:
                            split_list = con_fail_dict[(j, k)].split(' ')
                            if split_list[0] == '>':
                                constrained_min = float(split_list[1])
                                constrained_max = objective_max[j, k]
                            else:
                                constrained_min = objective_min[j, k]
                                constrained_max = float(split_list[1])

                    if (constrained_measure < constrained_min or constrained_measure > constrained_max) \
                            and not first:

                        if con_fail_dict is not None and constraints == 'Fail':
                            failed = True
                            break
                        else:
                            if constrained_measure < constrained_min:
                                p_con_met = constrained_measure / constrained_min
                            else:
                                p_con_met = constrained_max / constrained_measure

                            if objective == 'USAFA Proportion':
                                adj_con_tolerance = 0.9
                            elif objective == 'Combined Quota':
                                adj_con_tolerance = min((constrained_min - 1) / constrained_min,
                                                        constrained_max / (constrained_max + 1))
                            elif objective == 'Mandatory':
                                adj_con_tolerance = min((constrained_min - (1 / p['quota'][j])) / constrained_min,
                                                        constrained_max / (constrained_max + (1 / p['pgl'][j])))
                            else:
                                adj_con_tolerance = 0.95

                            # Either we reduce z by some penalty or z is set to 0
                            if constraints == 'Penalty' and p_con_met < adj_con_tolerance:
                                penalty += vp['afscs_overall_weight'] * vp['afsc_weight'][j]
                            elif constraints == 'Fail' and p_con_met < adj_con_tolerance:
                                failed = True
                                break
            if failed:
                break
            else:

                # Calculate AFSC value
                metrics['afsc_value'][j] = np.dot(vp['objective_weight'][j, :], metrics['objective_value'][j, :])
                if j in vp['J^C']:
                    if metrics['afsc_value'][j] < vp['afsc_value_min'][j]:

                        # Either we reduce z by some penalty or z is set to 0
                        metrics['afsc_constraint_fail'][j] = 1
                        metrics['total_failed_constraints'] += 1
                        if constraints == 'Penalty':
                            penalty += vp['afscs_overall_weight'] * vp['afsc_weight'][j]
                        elif constraints == 'Fail':
                            failed = True
                            break

        else:

            # Either we reduce z by some penalty or z is set to 0
            metrics['afsc_constraint_fail'][j] = 1
            metrics['total_failed_constraints'] += 1
            if constraints == 'Penalty':
                penalty += vp['afscs_overall_weight'] * vp['afsc_weight'][j]
            elif constraints == 'Fail':
                failed = True
                break

    if not failed:
        metrics['cadet_value'] = np.array([p['utility'][i, int(chromosome[i])] for i in p['I']])
        for i in vp['I^C']:
            if metrics['cadet_value'][i] < vp['cadet_value_min'][i]:

                # Either we reduce z by some penalty or z is set to 0
                metrics['cadet_constraint_fail'][i] = 1
                metrics['total_failed_constraints'] += 1
                if constraints == 'Penalty':
                    penalty += vp['cadets_overall_weight'] * vp['cadet_weight'][i]

    # Solution Matched Vector
    metrics['afsc_solution'] = np.array([p['afscs'][int(chromosome[i])] for i in p['I']])

    # Calculate Overall Values
    metrics['cadets_overall_value'] = np.dot(vp['cadet_weight'], metrics['cadet_value'])
    metrics['afscs_overall_value'] = np.dot(vp['afsc_weight'], metrics['afsc_value'])
    metrics['z'] = vp['cadets_overall_weight'] * metrics['cadets_overall_value'] + \
                   vp['afscs_overall_weight'] * metrics['afscs_overall_value']
    metrics['num_ineligible'] = 0

    if failed:
        metrics['z'] = 0

    if constraints == 'Penalty':
        penalized_z = metrics['z'] - penalty ** (1 / penalty_scale)
        if printing:
            print("Measured solution vector fitness: " + str(round(metrics['z'], 4)))
            print("Measured solution vector penalized fitness: " + str(round(penalized_z, 4)))
        return metrics, penalized_z
    else:
        if printing:
            print("Measured solution vector fitness: " + str(round(metrics['z'], 4)))
        return metrics


def find_solution_ineligibility(parameters, solution=None, printing=True):
    """
    This procedure takes a solution, presumably an original AFPC solution, and finds the cadets that were matched
    to an ineligible AFSC
    :param solution: afsc solution vector
    :param parameters: cadet afsc parameters
    :param printing: whether the procedure should print something
    """

    if printing:
        print('Finding ineligibility of matched cadets...')

    for i in parameters['I']:
        afsc_index = int(solution[i])
        if afsc_index not in parameters['J^E'][i]:
            if printing:
                print('cadet', i, '->', parameters['afscs'][afsc_index])


def solution_similarity_coordinates(similarity_matrix):
    """
    This procedure takes in a similarity matrix then performs MDS and returns the coordinates
    to plot the solutions in terms of how similar they are to each other
    :param similarity_matrix: similarity matrix
    :return: coordinates
    """
    # Change similarity matrix into distance matrix
    distances = 1 - similarity_matrix

    # Get coordinates
    if use_manifold:
        mds = manifold.MDS(n_components=2, dissimilarity='precomputed', random_state=10)
        results = mds.fit(distances)
        coords = results.embedding_
    else:
        coords = np.zeros([len(distances), 2])
        print('Sklearn manifold not available')

    return coords


def parameter_sanity_check(instance):
    """
    This function runs through all the different parameters and sanity checks them to make sure that they make
    sense and don't break the model
    """

    # Shorthand
    p, vp = instance.parameters, instance.value_parameters

    if vp is None:
        raise ValueError("Cannot sanity check parameters without specifying which value_parameters to use.")

    # Initialization
    print("Sanity checking the instance parameters...")
    issue = 0

    # Loop through each AFSC to check various elements
    for j, afsc in enumerate(p["afscs"][:p["M"]]):

        if p["num_eligible"][j] < p["quota_min"][j]:
            issue += 1
            print(issue, "ISSUE: AFSC '" + afsc + "' quota constraint invalid. " + str(p["quota_min"][j]) +
                  " (min) > " + str(p["num_eligible"][j]) + " (number of eligible cadets).")
        elif p["num_eligible"][j] == p["quota_min"][j]:
            issue += 1
            print(issue, "WARNING: AFSC '" + afsc +
                  "' has a lower quota that is the same as its number of eligible cadets (" +
                  str(p["quota_min"][j]) + "). All eligible cadets for this AFSC will be assigned to it.")

        if p["quota_min"][j] > p["quota_max"][j]:
            issue += 1
            print(issue, "ISSUE: AFSC '" + afsc + "' quota constraint invalid. " + str(p["quota_min"][j]) +
                  " (min) > " + str(p["quota_max"][j]) + " (max).")

        quota_k = np.where(vp["objectives"] == "Combined Quota")[0][0]
        if p["quota_d"][j] != vp["objective_target"][j, quota_k]:
            issue += 1
            print(issue, "ISSUE: AFSC '" + afsc + "' quota desired target of " + str(p["quota_d"][j]) +
                  " from AFSCs Fixed does not match its objective target (" + str(vp["objective_target"][j, quota_k]) +
                  ") in the value parameters.")

        if p["quota_d"][j] < p["quota_min"][j] or p["quota_d"][j] > p["quota_max"][j]:
            issue += 1
            print(issue, "ISSUE: AFSC '" + afsc + "' quota desired target of " + str(p["quota_d"][j]) +
                  " is outside the specified range on the number of cadets (" + str(p["quota_min"][j]) + ", " +
                  str(p["quota_max"][j]) + ").")

        # Validate AFOCD tier objectives
        for objective in ["Mandatory", "Desired", "Permitted", "Tier 1", "Tier 2", "Tier 3", "Tier 4"]:

            # Make sure this is a valid objective for this problem instance
            if objective not in vp["objectives"]:
                continue  # goes to the next objective

            # Get index
            k = np.where(vp["objectives"] == objective)[0][0]

            # Check if the AFSC is constraining this objective
            if k not in vp["K^C"][j]:
                continue

            # Make sure there are cadets that have this degree tier
            if len(p["I^D"][objective][j]) == 0:
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective +
                      "'-Tier does not exist. No cadets have degrees that fit in this tier.")

            # Make sure objective has valid target
            if vp["objective_target"][j, k] == 0:
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective +
                      "'-Tier target cannot be 0 when it has a nonzero weight.")

        # Make sure all constrained objectives have appropriate constraints
        for k in vp["K^C"][j]:
            objective = vp["objectives"][k]

            try:
                lb = float(vp["objective_value_min"][j, k].split(",")[0])
                ub = float(vp["objective_value_min"][j, k].split(",")[1])
                assert lb <= ub
            except:
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective +
                      "' constraint range (objective_value_min) '" + vp["objective_value_min"][j, k] +
                      "' is invalid. This constraint is currently activated.")

        # Make sure value functions are valid
        for k in vp["K^A"][j]:
            objective = vp["objectives"][k]
            vf_string_start = vp["value_functions"][j, k].split("|")[0]

            # VF String validation
            if vf_string_start not in ["Min Increasing", "Min Decreasing", "Balance", "Quota_Direct",
                                       "Quota_Normal"]:
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective + "' value function string '" +
                      vp["value_functions"][j, k] + "' is invalid.")

            # Validate number of breakpoints
            if vp["r"][j, k] == 0:
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective +
                      "' does not have any value function breakpoints. 'a':", vp["a"][j][k])
                continue

            # Value function should have same number of x and y coordinates
            if len(vp["a"][j][k]) != len(vp["f^hat"][j][k]):
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective +
                      "' value function breakpoint coordinates do not align. 'a' has length of " + len(vp["a"][j][k]) +
                      " while 'f^hat' has length of " + len(vp["f^hat"][j][k]) + ".")
                continue

            # Ensure that the breakpoint "x" coordinates are always getting bigger
            current_x = -1
            valid_x_bps = True
            for l in vp["L"][j][k]:
                if vp["a"][j][k][l] < current_x:
                    valid_x_bps = False
                    break
                else:
                    current_x = vp["a"][j][k][l]

            if not valid_x_bps:
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective +
                      "' value function x coordinates do not continuously increase along x-axis. 'a':", vp["a"][j][k],
                      "'vf_string':", vp["value_functions"][j, k])

    # Loop through each objective to see if there are any null values in the objective target array
    for k, objective in enumerate(vp["objectives"]):
        num_null = pd.isnull(vp["objective_target"][:, k]).sum()
        if num_null > 0:
            issue += 1
            print(issue, "ISSUE: Objective '" + objective + "' contains " +
                  str(num_null) + " null target values ('objective_target').")

    print('Done,', issue, "issues found.")


# Export solution metrics
def pyomo_measures_to_excel(x, measures, values, parameters, value_parameters, filepath=None, printing=False):
    """
    Exports x matrix to excel along with objective values and measures
    :param values: objective values
    :param measures: objective measures
    :param parameters: cadet parameters
    :param value_parameters: value parameters
    :param X: X matrix
    :param filepath: filepath
    :param printing: if we print something out
    :return: None
    """
    if printing:
        print('Exporting Pyomo measures to excel...')

    x_df = pd.DataFrame({'Encrypt_PII': parameters['SS_encrypt']})
    for j, afsc in enumerate(parameters['afscs']):
        x_df[afsc] = x[:, j]

    measures_df = pd.DataFrame({'AFSC': parameters['afscs']})
    values_df = pd.DataFrame({'AFSC': parameters['afscs']})
    for k, objective in enumerate(value_parameters['objectives']):
        measures_df[objective] = measures[:, k]
        values_df[objective] = values[:, k]

    if filepath is None:
        filepath = paths['results'] + 'X_Matrix.xlsx'

    with pd.ExcelWriter(filepath) as writer:  # Export to excel
        x_df.to_excel(writer, sheet_name="X", index=False)
        measures_df.to_excel(writer, sheet_name="Measures", index=False)
        values_df.to_excel(writer, sheet_name="Values", index=False)




