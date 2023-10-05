import numpy as np
import pandas as pd
import copy
import time

# afccp modules
import afccp.core.globals

# Import sklearn.manifold if it is installed
if afccp.core.globals.use_manifold:
    from sklearn import manifold

# Primary Solution Evaluation Functions
def evaluate_solution(solution, parameters, value_parameters, approximate=False, re_calculate_x=True, printing=False):
    """
    Evaluate a solution (either a vector or a matrix) by calculating various metrics.

    Parameters:
        solution (numpy.ndarray): The solution to evaluate, represented as a vector or a matrix.
        parameters (dict): The fixed cadet/AFSC model parameters.
        value_parameters (dict): The weight/value parameters.
        approximate (bool, optional): Whether the solution is approximate or exact. Defaults to False.
        re_calculate_x (bool, optional): If we want to force re-calculation of x as integer matrix. Defaults to True.
        printing (bool, optional): Whether to print the evaluated metrics. Defaults to False.

    Returns:
        solution (dict): A dictionary containing the solution core elements and evaluated metrics.

    Note:
        This function evaluates a solution by calculating various metrics, including objective measures, objective values,
        AFSC values, cadet values, constraint failures, overall values, and additional useful metrics.
    """

    # Shorthand
    p, vp = parameters, value_parameters

    # Get X matrix
    if 'x' not in solution or re_calculate_x:
        solution['x'] = np.array([[1 if solution['j_array'][i] == j else 0 for j in p['J']] for i in p['I']])
    x = solution['x']

    # Initialize solution metrics to be added to solution dictionary
    metrics = {'objective_measure': np.zeros([p['M'], vp['O']]),  # AFSC objective "raw" measure
               'objective_value': np.ones([p['M'], vp['O']]),  # AFSC objective value determined through value function
               'afsc_value': np.zeros(p['M']), 'cadet_value': np.zeros(p['N']),  # AFSC/Cadet Individual values
               'cadet_constraint_fail': np.zeros(p['N']),  # 1-N binary array indicating cadet constraint failures
               'afsc_constraint_fail': np.zeros(p['M']),  # 1-M binary array indicating AFSC constraint failures
               'objective_score': np.zeros(vp['O']),  # "Flipped" score for the AFSC objective

               # Constraint data metrics
               'total_failed_constraints': 0, "failed_constraints": [],
               'objective_constraint_fail': np.array([[" " * 30 for _ in range(vp['O'])] for _ in range(p['M'])]),
               'con_fail_dict': {}  # Dictionary containing the new minimum/maximum value we need to adhere to
               }
    for key in metrics:
        solution[key] = metrics[key]

    # Loop through all AFSCs to assign their "individual" values
    for j in p['J']:
        afsc = p["afscs"][j]

        # Loop through all AFSC objectives
        for k, objective in enumerate(vp["objectives"]):

            # Calculate AFSC objective measure
            solution['objective_measure'][j, k], _ = calculate_objective_measure_matrix(
                solution['x'], j, objective, p, vp, approximate=approximate)

            # Calculate AFSC objective value
            if k in vp["K^A"][j]:
                solution['objective_value'][j, k] = value_function(
                    vp['a'][j][k], vp['f^hat'][j][k], vp['r'][j][k], solution['objective_measure'][j, k])

            # Update metrics dictionary with failed AFSC objective constraint information
            if k in vp['K^C'][j]:
                solution = calculate_failed_constraint_metrics(j, k, solution, p, vp)

        # AFSC individual value
        solution['afsc_value'][j] = np.dot(vp['objective_weight'][j, :], solution['objective_value'][j, :])
        if solution['afsc_value'][j] < vp['afsc_value_min'][j]:
            solution['afsc_constraint_fail'][j] = 1
            solution['total_failed_constraints'] += 1
            solution["failed_constraints"].append(afsc + " Value")

    # Loop through all cadets to assign their values
    for i in p['I']:
        solution['cadet_value'][i] = np.sum(x[i, j] * p['cadet_utility'][i, j] for j in p['J^E'][i])
        if solution['cadet_value'][i] < vp['cadet_value_min'][i]:
            solution['cadet_constraint_fail'][i] = 1
            solution['total_failed_constraints'] += 1
            solution["failed_constraints"].append("Cadet " + str(p['cadets'][i]) + " Value")

    # Variables used to help verify that cadets are receiving the AFSCs they need to if specified
    num_fixed_correctly = 0
    num_reserved_correctly = 0

    # Get the AFSC solution (Modified to support "unmatched" cadets)
    solution["num_unmatched"] = 0
    solution['afsc_array'] = np.array([" " * 10 for _ in p['I']])
    for i in p['I']:
        j = np.where(x[i, :])[0]
        if len(j) != 0:
            j = int(j[0])
        else:
            solution["num_unmatched"] += 1
            j = p['M']  # Last index (*)
        solution['afsc_array'][i] = p['afscs'][j]

        # Check if this AFSC was "fixed" for this cadet
        if i in p['J^Fixed']:
            if j == p['J^Fixed'][i]:
                num_fixed_correctly += 1

        # Check if this AFSC was reserved for this cadet
        if i in p['J^Reserved']:
            if j in p['J^Reserved'][i]:
                num_reserved_correctly += 1

    # Alternate list situation
    if 'J^Special' in p:
        solution['num_alternates_allowed'] = {}  # Number of cadets with reserved slots that got something else
        solution[
            'num_correct_alternates'] = {}  # Number of cadets in the alternate list that correctly received AFSC
        for j in p['J^Special']:
            solution['num_alternates_allowed'][j] = len(
                [i for i in p['I^Reserved'][j] if solution['j_array'][i] != j])
            solution['num_correct_alternates'][j] = len(
                [i for i in p['I^Alternate'][j][:solution['num_alternates_allowed'][j]] if
                 solution['j_array'][i] == j])

        total_allowed = int(np.sum(solution['num_alternates_allowed'][j] for j in p['J^Special']))
        total_correct = int(np.sum(solution['num_correct_alternates'][j] for j in p['J^Special']))
        solution['alternate_list_metric'] = str(total_correct) + " / " + str(total_allowed)
    else:
        solution['alternate_list_metric'] = "0 / 0"  # Not applicable here

    # Verification that AFSCs are being assigned properly to work with J^Fixed and J^Reserved
    num_fixed_needed = len(p['J^Fixed'].keys())
    num_reserved_needed = len(p['J^Reserved'].keys())
    solution['cadets_fixed_correctly'] = str(num_fixed_needed) + ' / ' + str(num_fixed_needed)
    solution['cadets_reserved_correctly'] = str(num_reserved_correctly) + ' / ' + str(num_reserved_needed)

    # Define overall metrics
    solution['cadets_overall_value'] = np.dot(vp['cadet_weight'], solution['cadet_value'])
    solution['afscs_overall_value'] = np.dot(vp['afsc_weight'], solution['afsc_value'])
    solution['z'] = vp['cadets_overall_weight'] * solution['cadets_overall_value'] + \
                   vp['afscs_overall_weight'] * solution['afscs_overall_value']
    solution['num_ineligible'] = np.sum(x[i, j] * p['ineligible'][i, j] for j in p['J'] for i in p['I'])

    # Add additional metrics components (Non-VFT stuff)
    solution = calculate_additional_useful_metrics(solution, p, vp)

    # Calculate blocking pairs
    if 'a_pref_matrix' in p:
        solution['blocking_pairs'] = calculate_blocking_pairs(p, solution)
        solution['num_blocking_pairs'] = len(solution['blocking_pairs'])

    # Print statement
    if printing:
        if approximate:
            model_type = 'approximate'
        else:
            model_type = 'exact'
        if 'name' in solution:
            print_str = "Solution Evaluated: " + solution['name'] + "."
        else:
            print_str = "New Solution Evaluated."
        print_str += "\nMeasured " + model_type + " VFT objective value: " + str(round(solution['z'], 4))
        if 'z^gu' in solution:
            print_str += ".\nGlobal Utility Score: " + str(round(solution['z^gu'], 4))
        print_str += ". " + solution['cadets_fixed_correctly'] + ' AFSCs fixed. ' + \
                     solution['cadets_reserved_correctly'] + ' AFSCs reserved'
        print_str += ". " + solution['alternate_list_metric'] + ' cadets correctly pulled from alternate list'
        if 'num_blocking_pairs' in solution:
            print_str += ".\nBlocking pairs: " + str(solution['num_blocking_pairs'])
        print_str += ". Unmatched cadets: " + str(solution["num_unmatched"])
        print_str += ". Ineligible cadets: " + str(solution['num_ineligible']) + "."
        print(print_str)

    # Return the solution/metrics
    return solution

def fitness_function(chromosome, p, vp, mp, con_fail_dict=None):
    """
    Evaluates a chromosome (solution vector) and returns its fitness score.

    Parameters:
        chromosome (array-like): The chromosome representing the solution vector.
        p (dict): Parameters used in the calculations.
        vp (dict): Value parameters used in the calculations.
        mp (dict): Model parameters.
        con_fail_dict (dict, optional): Dictionary to store failed constraints for efficient evaluation.
                                        Defaults to None.

    Returns:
        fitness_score (float): The fitness score of the chromosome.

    Note:
        This function is relatively time-consuming and should be as efficient as possible.
        The fitness score is calculated based on the provided chromosome and parameters.
    """

    # 5% cap on total percentage of USAFA cadets allowed into certain AFSCs
    if vp["J^USAFA"] is not None:

        # This is a pretty arbitrary constraint and will only be used for real class years (but has since been removed)
        cap = 0.05 * mp["real_usafa_n"]
        u_count = 0
        for j in vp["J^USAFA"]:
            cadets = np.where(chromosome == j)[0]
            usafa_cadets = np.intersect1d(p['I^D']['USAFA Proportion'][j], cadets)
            u_count += len(usafa_cadets)

        # If we fail this constraint, we return an objective value of 0
        if u_count > int(cap + 1):
            return 0

    # Calculate AFSC individual values
    afsc_value = np.zeros(p['M'])
    for j in p['J']:

        # Initialize objective measures and values
        measure = np.zeros(vp['O'])
        value = np.zeros(vp['O'])

        # Indices of cadets assigned to this AFSC
        cadets = np.where(chromosome == j)[0]

        # Only calculate measures for AFSCs with at least one cadet
        count = len(cadets)
        if count > 0:

            # Loop through all AFSC objectives
            for k in vp["K"]:

                # If this AFSC is constraining this objective or only has it in the objective function
                if k in vp["K^A"][j]:

                    # Calculate AFSC objective measure
                    measure[k] = calculate_objective_measure_chromosome(cadets, j, vp['objectives'][k], p, vp, count)

                    # Assign AFSC objective value
                    value[k] = value_function(vp['a'][j][k], vp['f^hat'][j][k], vp['r'][j][k], measure[k])

                    # Check failed AFSC objective
                    if k in vp['K^C'][j]:
                        if check_failed_constraint_chromosome(j, k, measure[k], count, p, vp, con_fail_dict):
                            return 0

            # Calculate AFSC value
            afsc_value[j] = np.dot(vp['objective_weight'][j, :], value)
            if j in vp['J^C']:

                # If we fail this constraint, we return an objective value of 0
                if afsc_value[j] < vp['afsc_value_min'][j]:
                    return 0

        # No cadets assigned to the AFSC means it failed
        else:
            return 0

    # Calculate Cadet Value
    cadet_value = np.array([p['cadet_utility'][i, int(chromosome[i])] for i in p['I']])
    for i in vp['I^C']:

        # If we fail this constraint, we return an objective value of 0
        if cadet_value[i] < vp['cadet_value_min'][i]:
            return 0

    # Return fitness value
    return vp['cadets_overall_weight'] * np.dot(vp['cadet_weight'], cadet_value) + \
           vp['afscs_overall_weight'] * np.dot(vp['afsc_weight'], afsc_value)

def calculate_blocking_pairs(parameters, solution, only_return_count=False):
    """
    Calculate blocking pairs
    """

    # Shorthand
    p = parameters

    # Dictionary of cadets matched to each AFSC in this solution
    cadets_matched = {j: np.where(solution['j_array'] == j)[0] for j in p['J']}

    # Loop through all cadets and their assigned AFSCs
    blocking_pairs = []
    blocking_pair_count = 0

    # Loop through each cadet, AFSC pair
    for i, j in enumerate(solution['j_array']):

        # Unmatched cadets are blocking pairs by definition
        if j == p['M']:
            if only_return_count:
                blocking_pair_count += 1
            else:
                blocking_pairs.append((i, j))
                blocking_pair_count += 1

        # Matched cadets need to be calculated
        else:
            cadet_choice = np.where(p['cadet_preferences'][i] == j)[0][0]

            # Loop through more desirable AFSCs than current matched
            for j_compare in p['cadet_preferences'][i][:cadet_choice]:

                # Where is this cadet ranked in the AFSC list?
                afsc_choice_of_this_cadet = p['a_pref_matrix'][i, j_compare]
                matched_cadet_ranks = p['a_pref_matrix'][cadets_matched[j_compare], j_compare]

                # No one has been assigned to this more desirable AFSC (another blocking pair situation)
                if len(matched_cadet_ranks) == 0:
                    if only_return_count:
                        blocking_pair_count += 1
                    else:
                        blocking_pairs.append((i, j))
                        blocking_pair_count += 1
                    break

                # The lowest rank of the assigned cadet
                afsc_choice_of_worst_cadet = np.max(matched_cadet_ranks)

                # Check for blocking pairs
                if afsc_choice_of_this_cadet < afsc_choice_of_worst_cadet:
                    if only_return_count:
                        blocking_pair_count += 1
                    else:
                        blocking_pairs.append((i, j))
                        blocking_pair_count += 1
                    break

    if only_return_count:
        return blocking_pair_count
    else:
        return blocking_pairs

# Secondary Solution Evaluation Functions
def value_function(a, f_a, r, x):
    """
    Calculates the AFSC objective value based on the provided parameters.

    Parameters:
        a (array-like): Measure at each breakpoint.
        f_a (array-like): Value at each breakpoint.
        r (int): Number of breakpoints.
        x (float): Actual AFSC objective measure.

    Returns:
        value (float): AFSC objective value.

    Note:
        This function finds the appropriate breakpoint based on the measure and calculates the objective value
        using linear interpolation.
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

def value_function_points(a, fhat):
    """
    Takes the linear function parameters and returns the approximately non-linear coordinates
    :param a: function breakpoints
    :param fhat: function breakpoint values
    :return: x, y
    """
    x = (np.arange(1001) / 1000) * a[len(a) - 1]
    y = np.array([value_function(a, fhat, len(a), i) for i in x])
    return x, y

def calculate_afsc_norm_score(cadets, j, p, count=None):
    """
    This function simply calculates the AFSC "Norm Score" value and returns it
    """

    # Re-calculate count if necessary
    if count is None:
        count = len(cadets)

    # Best score sum we could achieve
    best_sum = np.sum(c for c in range(count))

    # Worst score sum we could receive
    worst_range = range(p["num_eligible"][j] - count, p["num_eligible"][j])
    worst_sum = np.sum(c for c in worst_range)

    # Score sum we did receive
    achieved_sum = np.sum(p["a_pref_matrix"][cadets, j])

    # Normalize this score and return it
    return 1 - (achieved_sum - best_sum) / (worst_sum - best_sum)

def calculate_afsc_norm_score_general(ranks, achieved_ranks):
    """
    This function simply calculates the AFSC "Norm Score" value and returns it for
    any given inputs of ranks
    """
    # Number of cadets assigned here
    count = len(achieved_ranks)

    # Only consider eligible cadets
    eligible_indices = np.where(ranks != 0)[0]
    eligible_ranks = ranks[eligible_indices]

    # Determine the best and worst set of rankings in this list
    worst_indices = np.argsort(eligible_ranks)[-count:][::-1]
    worst_ranks = eligible_ranks[worst_indices]
    best_indices = np.argsort(eligible_ranks)[:count]
    best_ranks = eligible_ranks[best_indices]

    # Calculate the sums
    best_sum, achieved_sum, worst_sum = np.sum(best_ranks), np.sum(achieved_ranks), np.sum(worst_ranks)

    # Normalize this score and return it
    return 1 - (achieved_sum - best_sum) / (worst_sum - best_sum)

def calculate_additional_useful_metrics(solution, p, vp):
    """
    Add additional components to the "metrics" dictionary based on the parameters and value parameters.

    Parameters:
        solution (dict): The dictionary containing the existing metrics.
        p (dict): The parameters dictionary.
        vp (dict): The value parameters dictionary.

    Returns:
        solution (dict): The updated metrics dictionary.

    Note:
        This function adds additional components to the "solution" dictionary based on the provided parameters
        and value parameters. The purpose is to enhance the information and analysis of the solution/metrics.
    """

    # Only calculate these metrics if we have the right parameters
    if 'c_pref_matrix' in p and 'a_pref_matrix' in p:

        # Calculate various metrics achieved
        solution['cadet_choice'] = np.zeros(p["N"]).astype(int)
        solution['afsc_choice'] = np.zeros(p['N']).astype(int)
        solution['cadet_utility_achieved'] = np.zeros(p['N'])
        solution['afsc_utility_achieved'] = np.zeros(p['N'])
        solution['global_utility_achieved'] = np.zeros(p['N'])
        for i, j in enumerate(solution['j_array']):
            if j in p['J']:
                solution['cadet_choice'][i] = p['c_pref_matrix'][i, j]  # Assigned cadet choice

                # Cadet is not in the AFSC's preferences
                if i not in p['afsc_preferences'][j]:
                    print('Cadet', i, 'not in ' + p['afscs'][j] + "'s preferences. This will cause an error.")
                    continue
                solution['afsc_choice'][i] = np.where(p['afsc_preferences'][j] == i)[0][0] + 1  # Where is the cadet ranked
                solution['cadet_utility_achieved'][i] = p['cadet_utility'][i, j]
                solution['afsc_utility_achieved'][i] = p['afsc_utility'][i, j]
                solution['global_utility_achieved'][i] = vp['global_utility'][i, j]
            else:
                solution['cadet_choice'][i] = np.max(p['c_pref_matrix'][i, :]) + 1  # Unassigned cadet choice
        solution['average_cadet_choice'] = round(np.mean(solution['cadet_choice']), 2)

        # Calculate average cadet choice for each AFSC individually
        solution['afsc_average_cadet_choice'] = np.zeros(p['M'])
        for j in p['J']:
            cadets = np.where(solution['j_array'] == j)[0]
            solution['afsc_average_cadet_choice'][j] = np.mean(p['c_pref_matrix'][cadets, j])

        # Calculate overall utility scores
        solution['z^gu'] = round(np.mean(solution['global_utility_achieved']), 4)
        solution['cadet_utility_overall'] = round(np.mean(solution['cadet_utility_achieved']), 4)
        solution['afsc_utility_overall'] = round(np.mean(solution['afsc_utility_achieved']), 4)

        # Calculate cadet utility based on SOC
        solution['usafa_cadet_utility'] = round(np.mean(solution['cadet_utility_achieved'][p['usafa_cadets']]), 4)
        solution['rotc_cadet_utility'] = round(np.mean(solution['cadet_utility_achieved'][p['rotc_cadets']]), 4)

    # Cadet Choice Counts (For exporting solution file to excel)
    solution['cadet_choice_counts'] = {}
    for choice in np.arange(1, 11):  # Just looking at top 10
        solution['cadet_choice_counts'][choice] = len(np.where(solution['cadet_choice'] == choice)[0])
    solution['cadet_choice_counts']['All Others'] = int(p['N'] - sum(
        [solution['cadet_choice_counts'][choice] for choice in np.arange(1, 11)]))

    # Save the counts for each AFSC separately from the objective_measure matrix
    quota_k = np.where(vp['objectives'] == 'Combined Quota')[0][0]
    solution['count'] = solution['objective_measure'][:, quota_k]

    # Assigned cadets
    solution['cadets_assigned'] = {j: np.where(solution['j_array'] == j)[0] for j in p['J']}

    # Cadets assigned to each accession group
    con_acc_grp_numerator = 0
    con_acc_grp_denominator = 0
    for acc_grp in p['afscs_acc_grp']:
        solution['I^' + acc_grp] = np.array([i for i in p['I'] if solution['j_array'][i] in p['J^' + acc_grp]])

        # Determine if we constrained Accessions groups properly
        if "I^" + acc_grp in p:
            acc_grp_constrained = len(p['I^' + acc_grp])

            # Calculate metrics if we actually constrained some people from this group
            if acc_grp_constrained > 0:
                acc_grp_correct = len(np.intersect1d(p['I^' + acc_grp], solution['I^' + acc_grp]))
                con_acc_grp_numerator += acc_grp_correct
                con_acc_grp_denominator += acc_grp_constrained
    solution['constrained_acc_grp_target'] = str(con_acc_grp_numerator) + " / " + str(con_acc_grp_denominator)

    # Air Force Cadets
    if 'USSF' in p['afscs_acc_grp']:
        solution['I^USAF'] = np.array([i for i in p['I'] if i not in solution['I^USSF']])

    # Calculate USSF Merit Distribution
    solution['ussf_om'] = 0  # Just to have something to show
    if 'USSF' in p['afscs_acc_grp']:

        # Necessary variables to calculate
        ussf_merit_sum = np.sum(np.sum(p['merit'][i] * solution['x'][i, j] for i in p['I^E'][j]) for j in p['J^USSF'])
        ussf_sum = np.sum(np.sum(solution['x'][i, j] for i in p['I^E'][j]) for j in p['J^USSF'])

        # Calculate metric
        solution['ussf_om'] = round(ussf_merit_sum / ussf_sum, 3)

        # USSF/USAF cadet distinctions
        solution['ussf_cadets'] = np.array([i for i in p['I'] if solution['j_array'][i] in p['J^USSF']])
        solution['usaf_cadets'] = np.array([i for i in p['I'] if solution['j_array'][i] not in p['J^USSF']])

        # Calculate cadet/AFSC utility relative to USSF/USAF cadets
        for service in ['ussf', 'usaf']:
            for entity in ['cadet', 'afsc']:

                if len(solution[service + '_cadets']) > 0:
                    solution[service + '_' + entity + '_utility'] = round(
                        np.mean(solution[entity + '_utility_achieved'][solution[service + '_cadets']]), 4)
                else:
                    solution[service + '_' + entity + '_utility'] = 0

        # USSF SOC Breakout
        solution['ussf_usafa_cadets'] = np.intersect1d(solution['ussf_cadets'], p['usafa_cadets'])
        solution['ussf_rotc_cadets'] = np.intersect1d(solution['ussf_cadets'], p['rotc_cadets'])
        solution['ussf_usafa_cadets_count'] = len(solution['ussf_usafa_cadets'])
        solution['ussf_rotc_cadets_count'] = len(solution['ussf_rotc_cadets'])

        # Metrics that will be printed to excel
        solution['ussf_usafa_pgl_target'] = str(solution['ussf_usafa_cadets_count']) + " / " + str(p['ussf_usafa_pgl'])
        solution['ussf_rotc_pgl_target'] = str(solution['ussf_rotc_cadets_count']) + " / " + str(p['ussf_rotc_pgl'])

    # Calculate weighted average AFSC choice (based on Norm Score)
    if 'Norm Score' in vp['objectives']:
        k = np.where(vp['objectives'] == 'Norm Score')[0][0]

        # Individual norm scores for each AFSC
        solution['afsc_norm_score'] = solution['objective_measure'][:, k]

        # Weighted average AFSC choice
        weights = solution['count'] / np.sum(solution['count'])
        solution['weighted_average_afsc_score'] = np.dot(weights, solution['afsc_norm_score'])

        # Space Force and Air Force differences
        if 'USSF' in p['afscs_acc_grp']:

            # Weighted average AFSC choice for USSF AFSCs (SFSCs)
            weights = solution['count'][p['J^USSF']] / np.sum(solution['count'][p['J^USSF']])
            solution['weighted_average_ussf_afsc_score'] = np.dot(weights, solution['afsc_norm_score'][p['J^USSF']])

            # Weighted average AFSC choice for USAF AFSCs (AFSCs)
            weights = solution['count'][p['J^USAF']] / np.sum(solution['count'][p['J^USAF']])
            solution['weighted_average_usaf_afsc_score'] = np.dot(weights, solution['afsc_norm_score'][p['J^USAF']])

    # Generate objective scores for each objective
    for k in vp['K']:
        new_weights = vp['afsc_weight'] * vp['objective_weight'][:, k]
        new_weights = new_weights / sum(new_weights)
        solution['objective_score'][k] = np.dot(new_weights, solution['objective_value'][:, k])

    # SOC/Gender proportions across AFSCs
    solution['usafa_proportion_afscs'] = np.array(
        [round(np.mean(p['usafa'][solution['cadets_assigned'][j]]), 2) for j in p['J']])
    if 'male' in p:
        solution['male_proportion_afscs'] = np.array(
            [round(np.mean(p['male'][solution['cadets_assigned'][j]]), 2) for j in p['J']])

    # SOC/Gender proportions across each Accession group
    for acc_grp in p['afscs_acc_grp']:
        if len(solution['I^' + acc_grp]) > 0:
            solution['usafa_proportion_' + acc_grp] = np.around(np.mean(p['usafa'][solution['I^' + acc_grp]]), 2)
            if 'male' in p:
                solution['male_proportion_' + acc_grp] = np.around(np.mean(p['male'][solution['I^' + acc_grp]]), 2)

    # Simpson index
    if 'race' in p:
        races = p['race_categories']  # Shorthand (easier to type)

        # Calculate Simpson diversity index for each AFSC
        solution['simpson_index'] = np.zeros(p['M'])  # Initialize index array for all the AFSCs
        for j in p['J']:
            n = solution['count'][j]  # Just grabbing "n" as the number of cadets assigned to this AFSC

            # "AFSC Cadets Race" dictionary of the number of cadets that were assigned to this AFSC from each race
            acr = {race: len(np.intersect1d(p['I^' + race], solution['cadets_assigned'][j])) for race in races}

            # Calculate simpson diversity index for this AFSC
            solution['simpson_index'][j] = round(1 - np.sum([(acr[r] * (acr[r] - 1)) / (n * (n - 1)) for r in races]), 2)

        # Calculate Simpson diversity index for each Accessions Group
        for acc_grp in p['afscs_acc_grp']:
            n = len(solution['I^' + acc_grp])  # Just grabbing "n" as the number of cadets assigned to this acc group

            # "Accessions Cadets Race" dictionary of the number of cadets that were assigned to this grp from each race
            acr = {race: len(np.intersect1d(p['I^' + race], solution['I^' + acc_grp])) for race in races}

            # Calculate simpson diversity index for this accessions group
            try:
                solution['simpson_index_' + acc_grp] = round(1 - np.sum(
                    [(acr[r] * (acr[r] - 1)) / (n * (n - 1)) for r in races]), 2)
            except:
                solution['simpson_index_' + acc_grp] = 0

    # Simpson index (Ethnicity)
    if 'ethnicity' in p:
        eths = p['ethnicity_categories']  # Shorthand (easier to type)

        # Calculate Simpson diversity index for each AFSC
        solution['simpson_index_eth'] = np.zeros(p['M'])  # Initialize index array for all the AFSCs
        for j in p['J']:
            n = solution['count'][j]  # Just grabbing "n" as the number of cadets assigned to this AFSC

            # "AFSC Cadets Ethnicity" dictionary of the number of cadets that were assigned to this AFSC from each eth
            ace = {eth: len(np.intersect1d(p['I^' + eth], solution['cadets_assigned'][j])) for eth in eths}

            # Calculate simpson diversity index for this AFSC
            solution['simpson_index_eth'][j] = round(
                1 - np.sum([(ace[eth] * (ace[eth] - 1)) / (n * (n - 1)) for eth in eths]), 2)

        # Calculate Simpson diversity index for each Accessions Group
        for acc_grp in p['afscs_acc_grp']:
            n = len(
                solution['I^' + acc_grp])  # Just grabbing "n" as the number of cadets assigned to this acc group

            # "Accessions Cadets Ethnicity" dictionary of the number of cadets that were assigned to this grp/from eth
            ace = {eth: len(np.intersect1d(p['I^' + eth], solution['I^' + acc_grp])) for eth in eths}

            # Calculate simpson diversity index for this accessions group
            try:
                solution['simpson_index_eth_' + acc_grp] = round(1 - np.sum(
                    [(ace[eth] * (ace[eth] - 1)) / (n * (n - 1)) for eth in eths]), 2)
            except:
                solution['simpson_index_eth_' + acc_grp] = 0

    # Calculate STEM proportions in each AFSC
    if 'stem' in p:
        pass

    # Initialize dictionaries for cadet choice based on demographics
    dd = {"usafa": ["USAFA", "ROTC"], "male": ["Male", "Female"]}  # Demographic Dictionary
    demographic_dict = {cat: [dd[cat][0], dd[cat][1]] for cat in dd if cat in p}  # Demographic Dictionary (For this instance)
    solution["choice_counts"] = {"TOTAL": {}}  # Everyone
    for cat in demographic_dict:
        for dem in demographic_dict[cat]:
            solution["choice_counts"][dem] = {}

    # Top 3 Choices from USSF and USAF (and ROTC/USAFA)
    if 'USSF' in p['afscs_acc_grp']:
        for cat in ['USSF', 'USAF']:

            # Might not have anyone assigned from this group
            if len(solution['I^' + cat]) == 0:
                solution['top_3_' + cat.lower() + '_count'] = 0
                continue

            # Calculate actual top 3 count
            arr = np.array([i for i in solution['I^' + cat] if solution['j_array'][i] in p['cadet_preferences'][i][:3]])
            solution['top_3_' + cat.lower() + '_count'] = round(len(arr) / len(solution['I^' + cat]), 4)
    for cat in ['USAFA', 'ROTC']:
        arr = np.array([i for i in p['I^' + cat] if solution['j_array'][i] in p['cadet_preferences'][i][:3]])
        solution['top_3_' + cat.lower() + '_count'] = round(len(arr) / len(p['I^' + cat]), 4)

    # Initialize arrays within the choice dictionaries for the AFSCs
    choice_categories = ["Top 3", "Next 3", "All Others", "Total"]
    for dem in solution["choice_counts"]:
        for c_cat in choice_categories:
            solution["choice_counts"][dem][c_cat] = np.zeros(p["M"]).astype(int)
        for afsc in p["afscs"]:
            solution["choice_counts"][dem][afsc] = np.zeros(p["P"]).astype(int)

    # Loop through each AFSC
    for j, afsc in enumerate(p["afscs"][:p['M']]):  # Skip unmatched AFSC

        # The cadets that were assigned to this AFSC
        dem_cadets = {"TOTAL": np.where(solution["afsc_array"] == afsc)[0]}

        # The cadets with the demographic that were assigned to this AFSC
        for cat in demographic_dict:
            dem_1, dem_2 = demographic_dict[cat][0], demographic_dict[cat][1]
            dem_cadets[dem_1] = np.intersect1d(np.where(p[cat] == 1)[0], dem_cadets["TOTAL"])
            dem_cadets[dem_2] = np.intersect1d(np.where(p[cat] == 0)[0], dem_cadets["TOTAL"])

        # Loop through each choice and calculate the metric
        for choice in range(p["P"]):

            # The cadets that were assigned to this AFSC and placed it in their Pth choice
            assigned_choice_cadets = np.intersect1d(p["I^Choice"][choice][j], dem_cadets["TOTAL"])

            # The cadets that were assigned to this AFSC, placed it in their Pth choice, and had the specific demographic
            for dem in solution["choice_counts"]:
                solution["choice_counts"][dem][afsc][choice] = len(
                    np.intersect1d(assigned_choice_cadets, dem_cadets[dem]))

        # Loop through each demographic
        for dem in solution["choice_counts"]:
            solution["choice_counts"][dem]["Total"][j] = int(len(dem_cadets[dem]))
            solution["choice_counts"][dem]["Top 3"][j] = int(np.sum(solution["choice_counts"][dem][afsc][:3]))
            solution["choice_counts"][dem]["Next 3"][j] = int(np.sum(solution["choice_counts"][dem][afsc][3:6]))
            solution["choice_counts"][dem]["All Others"][j] = int(len(
                dem_cadets[dem]) - solution["choice_counts"][dem]["Top 3"][j] - solution["choice_counts"][dem]["Next 3"][j])

    return solution

# AFSC Objective Measure Calculation Functions
def calculate_objective_measure_chromosome(cadets, j, objective, p, vp, count):
    """
    Calculates the AFSC objective measure based on the provided parameters.

    Parameters:
        cadets (list): List of cadets.
        j (int): AFSC index.
        objective (str): Objective for which to calculate the measure.
        p (dict): Parameters used in the calculations.
        vp (dict): Value parameters used in the calculations.
        count (int): Number of cadets.

    Returns:
        measure (float): The calculated AFSC objective measure.

    Note:
        The function assumes an "exact" model since it's used in the fitness function.
        The measure is calculated based on the objective and the provided inputs.

    """

    # Objective to balance some demographic of the cadets (binary indicator)
    if objective in vp['K^D']:
        return len(np.intersect1d(p['I^D'][objective][j], cadets)) / count

    # Balancing Merit
    elif objective == 'Merit':
        return np.mean(p['merit'][cadets])

    # "Number of Cadets" Objectives
    elif objective == 'Combined Quota':
        return count
    elif objective == 'USAFA Quota':
        return len(np.intersect1d(p['I^D']['USAFA Proportion'][j], cadets))
    elif objective == 'ROTC Quota':
        return count - len(np.intersect1d(p['I^D']['USAFA Proportion'][j], cadets))

    # Maximize cadet utility
    elif objective == 'Utility':
        return np.mean(p['cadet_utility'][cadets, j])

    # New objective to evaluate CFM preference lists
    elif objective == "Norm Score":
        return calculate_afsc_norm_score(cadets, j, p, count=count)

def calculate_objective_measure_matrix(x, j, objective, p, vp, approximate=True):
    """
    Calculates the AFSC objective measure based on the provided parameters.

    Parameters:
        x (ndarray): Matrix representing the assignment of cadets to AFSCs.
        j (int): AFSC index.
        objective (str): Objective for which to calculate the measure.
        p (dict): Parameters used in the calculations.
        vp (dict): Value parameters used in the calculations.
        approximate (bool, optional): Flag indicating whether to use an approximate measure (divide by estimated number
            of cadets, not the REAL number af cadets assigned to the AFSC. Defaults to True.

    Returns:
        measure (float): The calculated AFSC objective measure.
        numerator (float or None): The numerator used in the calculation of the measure.
            It is None for certain objectives.

    Raises:
        ValueError: If the provided objective does not have a means of calculation in the VFT model.

    Note:
        The measure and numerator are calculated based on the objective and the provided inputs.
        The numerator is the value used in the calculation of the measure (sum of cadets with some feature over the
        "num_cadets" variable which is either the actual number of cadets assigned (count) or estimated (quota_e).

    """

    # Get count variables for this AFSC
    count = np.sum(x[i, j] for i in p['I^E'][j])
    if approximate:
        num_cadets = int(p['quota_e'][j])  # estimated number of cadets
    else:
        num_cadets = count  # actual number of cadets

    # Objective to balance some demographic of the cadets (binary indicator)
    if objective in vp['K^D']:
        numerator = np.sum(x[i, j] for i in p['I^D'][objective][j])
        return numerator / num_cadets, numerator # Measure, Numerator

    # Balancing Merit
    elif objective == "Merit":
        numerator = np.sum(p['merit'][i] * x[i, j] for i in p['I^E'][j])
        return numerator / num_cadets, numerator # Measure, Numerator

    # "Number of Cadets" Objectives
    elif objective == "Combined Quota":
        return count, None # Measure, Numerator
    elif objective == "USAFA Quota":
        return np.sum(x[i, j] for i in p['I^D']['USAFA Proportion'][j]), None # Measure, Numerator
    elif objective == "ROTC Quota":
        return count - np.sum(x[i, j] for i in p['I^D']['USAFA Proportion'][j]), None # Measure, Numerator

    # Maximize cadet utility
    elif objective == "Utility":
        numerator = np.sum(p['cadet_utility'][i, j] * x[i, j] for i in p['I^E'][j])
        return numerator / num_cadets, numerator  # Measure, Numerator

    # New objective to evaluate CFM preference lists
    elif objective == "Norm Score":

        # Proxy for constraint purposes
        numerator = np.sum(p['afsc_utility'][i, j] * x[i, j] for i in p['I^E'][j])

        # Temporary placeholder- need better methodology for calculating Norm Score in "Exact Model"!
        if type(num_cadets) not in [int, np.int64]:
            num_cadets = int(p['quota_e'][j])

        # Actual objective measure
        best_range = range(num_cadets)
        best_sum = np.sum(c for c in best_range)
        worst_range = range(p["num_eligible"][j] - num_cadets, p["num_eligible"][j])
        worst_sum = np.sum(c for c in worst_range)
        achieved_sum = np.sum(p["a_pref_matrix"][i, j] * x[i, j] for i in p["I^E"][j])
        return 1 - (achieved_sum - best_sum) / (worst_sum - best_sum), numerator  # Measure, Numerator

    # Unrecognized objective
    else:
        raise ValueError("Error. Objective '" + objective + "' does not have a means of calculation in the"
                                                            " VFT model. Please adjust.")

# AFSC Objective Measure Constraint Functions
def calculate_failed_constraint_metrics(j, k, solution, p, vp):
    """
    Calculate failed constraint metrics for an AFSC objective and return the updated metrics dictionary.

    Parameters:
        j (int): Index of the AFSC objective.
        k (int): Index of the objective measure.
        solution (dict): The solution/metrics dictionary.
        p (dict): The fixed cadet/AFSC model parameters.
        vp (dict): The weight/value parameters.

    Returns:
        solution (dict): The updated solution/metrics dictionary.

    Note:
        This function calculates the failed constraint metrics for an AFSC objective and updates the metrics dictionary
        with the newly calculated values.
    """

    # Constrained Approximate Measure (Only meant for degree tier constraints)
    if vp["constraint_type"][j, k] == 1:  # Should be an "at least constraint"

        # Get count variable
        count = np.sum(solution['x'][i, j] for i in p['I^E'][j])
        constrained_measure = (solution['objective_measure'][j, k] * count) / min(p['quota_min'][j], p['pgl'][j])

    # Constrained Exact Measure
    elif vp["constraint_type"][j, k] == 2:  # Should be either an "at most constraint" or simple valid range (0.2, 0.4)
        constrained_measure = solution['objective_measure'][j, k]

    else:
        pass

    # Measure is below the range
    if constrained_measure < vp['objective_min'][j, k]:
        solution['objective_constraint_fail'][j, k] = \
            str(round(constrained_measure, 2)) + ' < ' + str(vp['objective_min'][j, k]) + '. ' + \
            str(round(100 * constrained_measure / vp['objective_min'][j, k], 2)) + '% Met.'
        solution['total_failed_constraints'] += 1
        solution["failed_constraints"].append(p['afscs'][j] + " " + vp['objectives'][k])
        solution["con_fail_dict"][(j, k)] = '> ' + str(round(constrained_measure, 4))

    # Measure is above the range
    elif constrained_measure > vp['objective_max'][j, k]:
        solution['objective_constraint_fail'][j, k] = \
            str(round(constrained_measure, 2)) + ' > ' + str(vp['objective_max'][j, k]) + '. ' + \
            str(round(100 * vp['objective_max'][j, k] / constrained_measure, 2)) + '% Met.'
        solution['total_failed_constraints'] += 1
        solution["failed_constraints"].append(p['afscs'][j] + " " + vp['objectives'][k])
        solution["con_fail_dict"][(j, k)] = '< ' + str(round(constrained_measure, 4))

    return solution  # Return *updated* solution/metrics

def check_failed_constraint_chromosome(j, k, measure, count, p, vp, con_fail_dict):
    """
    This function takes in the AFSC index, objective index, AFSC objective measure, number of cadets assigned (count),
    parameters, value parameters, and the constraint fail dictionary and determines if we've failed the constraint or not.

    :param j: Index of the AFSC (Air Force Specialty Code).
    :param k: Index of the objective.
    :param measure: Measure of the AFSC objective.
    :param count: Number of cadets assigned.
    :param p: Dictionary of parameters.
        - 'quota_min': Array of minimum quotas for each AFSC.
        - 'pgl': Array of Program Guidance Letter (PGL) targets for each AFSC.
    :param vp: Dictionary of value parameters.
        - 'constraint_type': Array representing the constraint type for each AFSC and objective.
            - 1 represents Constrained Approximate Measure.
            - 2 represents Constrained Exact Measure.
        - 'objective_min': Array representing the minimum objective value for each AFSC and objective.
        - 'objective_max': Array representing the maximum objective value for each AFSC and objective.
    :param con_fail_dict: Dictionary containing information about failed constraints (optional).
        - Keys are tuples (j, k) representing AFSC and objective indices.
        - Values are strings representing adjusted min/max values for the failed constraint.
            - If the string starts with '>', it means the minimum value needed should be lowered.
            - Otherwise, it means the maximum value allowed should be raised.

    :return: A boolean indicating whether the constraint is failed or not.
        - True if the measure is outside the constrained range (constraint failed).
        - False if the measure is within the constrained range (constraint passed).
    """

    # Constrained Approximate Measure (Only meant for degree tier constraints)
    if vp["constraint_type"][j, k] == 1:  # Should be an "at least constraint"
        constrained_measure = (measure * count) / min(p['quota_min'][j], p['pgl'][j])

    # Constrained Exact Measure
    elif vp["constraint_type"][j, k] == 2:  # Should be either an "at most constraint" or simple valid range (0.2, 0.4)
        constrained_measure = measure

    # The constrained min and max values as specified by the value parameters
    constrained_min, constrained_max = vp["objective_min"][j, k], vp["objective_max"][j, k]

    # We adjust the constrained min and max based on the pyomo solution since it could be a little off due to rounding
    if con_fail_dict is not None:
        if (j, k) in con_fail_dict:

            # Split up the value in the dictionary to get the new min and max
            split_list = con_fail_dict[(j, k)].split(' ')
            if split_list[0] == '>':  # We lower the minimum value needed
                constrained_min = float(split_list[1])
            else:  # We raise the maximum value allowed
                constrained_max = float(split_list[1])

    # Round everything to stay consistent
    constrained_min, constrained_measure, constrained_max = round(constrained_min, 4), round(constrained_measure, 4), \
                                                            round(constrained_max, 4)

    # Check if we failed the constraint, and return a boolean
    if constrained_min <= constrained_measure <= constrained_max:
        return False  # Measure is in the range, we DID NOT fail the constraint (failed = False)
    else:
        return True  # Measure is outside the range, we DID fail the constraint (failed = True)

# Solution Comparison Functions
def compare_solutions(baseline, compared, printing=False):
    """
    Compare two solutions (in vector form) to the same problem and determine the similarity between them based on the
    AFSCs assigned to cadets.

    Parameters:
        baseline (numpy.ndarray): The first solution (baseline) to compare.
        compared (numpy.ndarray): The second solution to compare against the baseline.
        printing (bool, optional): Whether to print the similarity percentage. Defaults to False.

    Returns:
        percent_similar (float): The percentage of the compared solution that is the same as the baseline solution.

    Note:
        This function compares two solutions represented as vectors and calculates the percentage of similarity
        between them in terms of the AFSCs assigned to cadets. The solutions must be for the same set of cadets
        and AFSCs.

    Example:
        baseline = np.array([0, 1, 2, 1, 0])
        compared = np.array([1, 0, 2, 1, 0])
        similarity = compare_solutions(baseline, compared, printing=True)
        # Output: The two solutions are 60.0% the same (3/5).
    """

    percent_similar = (sum(baseline == compared * 1) / len(baseline))
    if printing:
        print("The two solutions are " + str(percent_similar) + "% the same.")
    return percent_similar

def similarity_coordinates(similarity_matrix):
    """
    This procedure takes in a similarity matrix then performs MDS and returns the coordinates
    to plot the solutions in terms of how similar they are to each other
    :param similarity_matrix: similarity matrix
    :return: coordinates
    """
    # Change similarity matrix into distance matrix
    distances = 1 - similarity_matrix

    # Get coordinates
    if afccp.core.globals.use_manifold:
        mds = manifold.MDS(n_components=2, dissimilarity='precomputed', random_state=10)
        results = mds.fit(distances)
        coordinates = results.embedding_
    else:
        coordinates = np.zeros([len(distances), 2])
        print('Sklearn manifold not available')

    return coordinates

# Solution Preparation Functions
def incorporate_rated_results_in_parameters(instance, printing=True):
    """
    This function extracts the results from the two Rated solutions (for both USAFA & ROTC)
    and fixes those cadets that were "matched" from the algorithm and constrains the individuals
    that had "reserved" slots from the algorithm
    """

    if printing:
        print("Incorporating rated algorithm results...")

    # Shorthand
    p, vp, solutions = instance.parameters, instance.value_parameters, instance.solutions
    mdl_p = instance.mdl_p

    # Make sure we have the solutions from both SOCs with matches and reserves
    for soc in ['USAFA', 'ROTC']:
        for kind in ['Reserves', 'Matches']:
            solution_name = "Rated " + soc.upper() + " HR (" + kind + ")"
            if solution_name not in solutions:
                return p  # We don't have the required solutions!

    # Matched cadets get fixed in the solution!
    counts = {'Reserved': {"USAFA": 0, "ROTC": 0}, 'Matched': {"USAFA": 0, "ROTC": 0}}
    for soc in ['USAFA', 'ROTC']:
        solution = solutions["Rated " + soc.upper() + " HR (Matches)"]
        matched_cadets = np.where(solution['j_array'] != p['M'])[0]
        for i in matched_cadets:
            p['J^Fixed'][i] = solution['j_array'][i]
            counts["Matched"][soc] += 1

    # Reserved cadets AFSC selection is constrained to be AT LEAST their reserved Rated slot
    p['J^Reserved'] = {}
    for soc in ['USAFA', 'ROTC']:
        solution = solutions["Rated " + soc.upper() + " HR (Reserves)"]
        reserved_cadets = np.where(solution['j_array'] != p['M'])[0]
        for i in reserved_cadets:
            j = solution['j_array'][i]
            choice = np.where(p['cadet_preferences'][i] == j)[0][0]
            p['J^Reserved'][i] = p['cadet_preferences'][i][:choice + 1]
            counts["Reserved"][soc] += 1

    # "ALTERNATE LIST" SPECIAL TREATMENT SITUATION
    if mdl_p['special_afscs_with_alternates'] is not None:
        p['J^Special'] = np.array(
            [np.where(p['afscs'] == afsc)[0][0] for afsc in mdl_p['special_afscs_with_alternates']])

        # *Mostly* already defined components of the reserved/matched algorithm idea
        p['I^Reserved'] = {}  # Indexed set of cadets with reserved slots for each AFSC j
        p['I^Matched'] = {}  # Indexed set of cadets that were definitively matched to AFSC j
        p['num_reserved'] = {}  # Number of cadets with reserved slots for each AFSC j

        # Indexed set of cadets (ordered by the AFSC's ranking) on the "alternate" list for each AFSC j
        p['I^Alternate'] = {}

        # Loop through each "Special" AFSC and develop components for enforcing the "alternate list" concept
        for j in p['J^Special']:

            # Defined components from above
            p['I^Reserved'][j] = np.array(
                [i for i in p['J^Reserved'] if p['cadet_preferences'][i][len(p['J^Reserved'][i]) - 1] == j])
            p['I^Matched'][j] = np.array([i for i in p['J^Fixed'] if j == p['J^Fixed'][i]])
            p['num_reserved'][j] = len(p['I^Reserved'][j])

            # Need to get ordered alternate list
            alternate_count, extra_reserved_count = 0, 0
            alternates = []
            for i in p['afsc_preferences'][j]:  # Loop through each cadet in order of the AFSC's preference

                # If this cadet is "next in line" on the AFSC's preference list after the matched/reserved people
                if i not in p['I^Reserved'][j] and i not in p['I^Matched'][j]:

                    # If this AFSC was the cadet's first choice, we add them to the alternate list
                    if p['cadet_preferences'][i][0] == j:
                        alternates.append(i)
                        alternate_count += 1

                    # This AFSC was not the cadet's first choice, so we make sure they get one of their other choices
                    elif i not in p['J^Fixed'] and i not in p['J^Reserved']:  # Make sure this is legitimate
                        extra_reserved_count += 1
                        choice = np.where(p['cadet_preferences'][i] == j)[0][0]  # Where the cadet ranked AFSC j
                        p['J^Reserved'][i] = p['cadet_preferences'][i][:choice]  # They have to get a better preference

                # If we've exhausted our alternate list, we're done here!
                if alternate_count >= p['num_reserved'][j]:
                    break
            p['I^Alternate'][j] = np.array(alternates)  # Convert to numpy array

            if printing:
                print("AFSC '" + p['afscs'][j] + "' alternate list:", p['I^Alternate'][j])
                print(alternate_count, "alternates;", extra_reserved_count, "extra reserved.")

    else:

        # Calculate additional rated algorithm result information for both SOCs
        for soc in ['rotc', 'usafa']:
            p = augment_rated_algorithm_results(p, soc=soc, printing=printing)

    # Print statement
    if printing:
        print_str = "Rated SOC Algorithm Results ["
        for soc in ['USAFA', 'ROTC']:
            for kind in ["Matched", "Reserved"]:
                print_str += soc + ' ' + kind + ' Cadets: ' + str(int(counts[kind][soc])) + '. '
        print(print_str[:-1] + "]")

    return p  # Return the parameters!

def augment_rated_algorithm_results(p, soc='rotc', printing=False):
    """
    Analyzes the rated algorithm results and gathers more information to include alternates
    as well as definitively matching additional people
    """

    # Start with a full list of cadets eligible for this AFSC
    possible_cadets = {j: list(p['I^E'][j]) for j in p['J^Rated']}

    # Main algorithm
    iteration = 0
    while iteration < 8:

        # Set of cadets reserved or matched to each AFSC
        p['I^Reserved'] = {j: np.array([i for i in p['J^Reserved'] if p['cadet_preferences'][i][
            len(p['J^Reserved'][i]) - 1] == j and p[soc][i]]) for j in p['J^Rated']}
        p['I^Matched'] = {j: np.array([
            i for i in p['J^Fixed'] if j == p['J^Fixed'][i] and p[soc][i]]) for j in p['J^Rated']}

        # Print Statement
        if printing:
            print("Iteration", iteration)
            print("Possible", {p['afscs'][j]: len(possible_cadets[j]) for j in p['J^Rated']})
            print("Reserved", {p['afscs'][j]: len(p['I^Reserved'][j]) for j in p['J^Rated']})
            print("Matched", {p['afscs'][j]: len(p['I^Matched'][j]) for j in p['J^Rated']})

        # Number of alternates (number of reserved slots)
        num_reserved = {j: len(p['I^Reserved'][j]) for j in p['J^Rated']}

        # Need to determine who falls into each category of alternates
        hard_alternates = {j: [] for j in p['J^Rated']}
        soft_r_alternates = {j: [] for j in p['J^Rated']}
        soft_n_alternates = {j: [] for j in p['J^Rated']}
        alternates = {j: [] for j in p['J^Rated']}  # all the cadets ordered here

        # Loop through each rated AFSC to determine alternates
        for j in p['J^Rated']:

            # Loop through each cadet in order of the AFSC's preference from this SOC
            for i in p['afsc_preferences'][j]:
                if not p[soc][i]:
                    continue

                # Assume this cadet is "next in line" until proven otherwise
                next_in_line = True

                # Is the cadet already fixed to something else?
                if i in p['J^Fixed']:
                    next_in_line = False

                # Is this cadet reserved for something?
                if i in p['J^Reserved']:

                    # If they're already reserved for this AFSC or something better, they're not considered
                    if len(p['J^Reserved'][i]) <= p['c_pref_matrix'][i, j]:  # <= temporary
                        next_in_line = False

                # If this cadet is next in line
                if next_in_line:
                    alternates[j].append(i)

                    # Loop through the cadet's preferences:
                    for j_c in p['cadet_preferences'][i]:

                        if j_c == j:
                            hard_alternates[j].append(i)
                            break
                        elif j_c in p['J^Rated']:
                            if i in possible_cadets[j_c]:
                                soft_r_alternates[j].append(i)
                                break
                            else:
                                continue
                        else:
                            soft_n_alternates[j].append(i)
                            break

                # We're done with this AFSC after we've exhausted our "hard alternates"
                if len(hard_alternates[j]) >= num_reserved[j]:
                    break

        # Loop through each rated AFSC to potentially turn "reserved" slots into "matched" slots
        for j in p['J^Rated']:

            # Loop through each cadet in order of the AFSC's preference from this SOC
            for i in p['afsc_preferences'][j]:
                if not p[soc][i]:
                    continue

                # Does this cadet have a reserved slot for something?
                if i in p['J^Reserved']:

                    # Is this cadet reserved for this AFSC?
                    if len(p['J^Reserved'][i]) == p['c_pref_matrix'][i, j]:

                        # Determine if there's any possible way this cadet might not be matched to this AFSC
                        inevitable_match = True
                        for j_c in p['J^Reserved'][i][:-1]:
                            if j_c not in p['J^Rated']:
                                inevitable_match = False
                            else:  # Rated
                                if i in alternates[j_c]:
                                    inevitable_match = False
                                else:
                                    if i in possible_cadets[j_c]:
                                        possible_cadets[j_c].remove(i)  # Remove this cadet as a possibility!

                        # If still inevitable, change from reserved to fixed
                        if inevitable_match:
                            p['J^Fixed'][i] = j
                            p['J^Reserved'].pop(i)

        iteration += 1

    # Incorporate alternate lists
    if 'J^Alternates (Hard)' not in p:
        p['J^Alternates (Hard)'] = {}
    if 'J^Alternates (Soft)' not in p:
        p['J^Alternates (Soft)'] = {}
    for i in p['I']:
        if p[soc][i]:
            for j in p['J^Rated']:
                if i in hard_alternates[j]:
                    p['J^Alternates (Hard)'][i] = j
                elif i in soft_r_alternates[j] or i in soft_n_alternates[j]:
                    p['J^Alternates (Soft)'][i] = j

    # Return updated parameters (and alternate lists)
    return p




