# Import Libraries
import numpy as np
import pandas as pd
import copy
import time
import afccp.core.globals

# Solution Evaluation Functions
def evaluate_solution(solution, parameters, value_parameters, approximate=False, printing=False):
    """
    Evaluate a solution (either a vector or a matrix) by calculating various metrics.

    Parameters:
        solution (numpy.ndarray): The solution to evaluate, represented as a vector or a matrix.
        parameters (dict): The fixed cadet/AFSC model parameters.
        value_parameters (dict): The weight/value parameters.
        approximate (bool, optional): Whether the solution is approximate or exact. Defaults to False.
        printing (bool, optional): Whether to print the evaluated metrics. Defaults to False.

    Returns:
        metrics (dict): A dictionary containing the evaluated metrics of the solution.

    Note:
        This function evaluates a solution by calculating various metrics, including objective measures, objective values,
        AFSC values, cadet values, constraint failures, overall values, and additional useful metrics.
    """


    # Shorthand
    p, vp = parameters, value_parameters

    # Convert to X matrix
    x = swap_solution_shape(solution, p['M'], to_matrix=True)

    # Initialize metrics dictionary
    metrics = {'objective_measure': np.zeros([p['M'], vp['O']]),  # AFSC objective "raw" measure
               'objective_value': np.ones([p['M'], vp['O']]),  # AFSC objective value determined through value function
               'afsc_value': np.zeros(p['M']), 'cadet_value': np.zeros(p['N']),
               'cadet_constraint_fail': np.zeros(p['N']),  # 1-N binary array indicating cadet constraint failures
               'afsc_constraint_fail': np.zeros(p['M']),  # 1-M binary array indicating AFSC constraint failures
               'objective_score': np.zeros(vp['O']),  # "Flipped" score for the AFSC objective

               # Constraint data metrics
               'total_failed_constraints': 0, 'x': x, "failed_constraints": [],
               'objective_constraint_fail': np.array([[" " * 30 for _ in range(vp['O'])] for _ in range(p['M'])]),
               'con_fail_dict': {}  # Dictionary containing the new minimum/maximum value we need to adhere to
               }

    # Loop through all AFSCs to assign their "individual" values
    for j in p['J']:
        afsc = p["afscs"][j]

        # Loop through all AFSC objectives
        for k, objective in enumerate(vp["objectives"]):

            # Calculate AFSC objective measure
            metrics['objective_measure'][j, k], _ = calculate_objective_measure_matrix(
                x, j, objective, p, vp, approximate=approximate)

            # Calculate AFSC objective value
            if k in vp["K^A"][j]:
                metrics['objective_value'][j, k] = value_function(
                    vp['a'][j][k], vp['f^hat'][j][k], vp['r'][j][k], metrics['objective_measure'][j, k])

            # Update metrics dictionary with failed AFSC objective constraint information
            if k in vp['K^C'][j]:
                metrics = calculate_failed_constraint_metrics(j, k, x, metrics, p, vp)

        # AFSC individual value
        metrics['afsc_value'][j] = np.dot(vp['objective_weight'][j, :], metrics['objective_value'][j, :])
        if metrics['afsc_value'][j] < vp['afsc_value_min'][j]:
            metrics['afsc_constraint_fail'][j] = 1
            metrics['total_failed_constraints'] += 1
            metrics["failed_constraints"].append(afsc + " Value")

    # Loop through all cadets to assign their values
    for i in p['I']:
        metrics['cadet_value'][i] = np.sum(x[i, j] * p['utility'][i, j] for j in p['J^E'][i])
        if metrics['cadet_value'][i] < vp['cadet_value_min'][i]:
            metrics['cadet_constraint_fail'][i] = 1
            metrics['total_failed_constraints'] += 1
            metrics["failed_constraints"].append("Cadet " + str(p['cadets'][i]) + " Value")

    # Get the AFSC solution (Modified to support "unmatched" cadets)
    metrics["num_unmatched"] = 0
    metrics['afsc_solution'] = np.array([" " * 10 for _ in p['I']])
    for i in p['I']:
        j = np.where(x[i, :])[0]
        if len(j) != 0:
            j = int(j[0])
        else:
            metrics["num_unmatched"] += 1
            j = 32
        metrics['afsc_solution'][i] = p['afscs'][j]

    # Define overall metrics
    metrics['cadets_overall_value'] = np.dot(vp['cadet_weight'], metrics['cadet_value'])
    metrics['afscs_overall_value'] = np.dot(vp['afsc_weight'], metrics['afsc_value'])
    metrics['z'] = vp['cadets_overall_weight'] * metrics['cadets_overall_value'] + \
                   vp['afscs_overall_weight'] * metrics['afscs_overall_value']
    metrics['num_ineligible'] = np.sum(x[i, j] * p['ineligible'][i, j] for j in p['J'] for i in p['I'])
    if printing:
        if approximate:
            model_type = 'approximate'
        else:
            model_type = 'exact'
        print("Measured " + model_type + " solution objective value: " + str(round(metrics['z'], 4)) +
              ". Unmatched cadets: " + str(metrics["num_unmatched"]) +
              ". Ineligible cadets: " + str(metrics['num_ineligible']) + ".")

    # Add additional metrics components
    metrics = calculate_additional_useful_metrics(metrics, p, vp)

    # Return the metrics
    return metrics

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
    cadet_value = np.array([p['utility'][i, int(chromosome[i])] for i in p['I']])
    for i in vp['I^C']:

        # If we fail this constraint, we return an objective value of 0
        if cadet_value[i] < vp['cadet_value_min'][i]:
            return 0

    # Return fitness value
    return vp['cadets_overall_weight'] * np.dot(vp['cadet_weight'], cadet_value) + \
           vp['afscs_overall_weight'] * np.dot(vp['afsc_weight'], afsc_value)

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
        return np.mean(p['utility'][cadets, j])

    # New objective to evaluate CFM preference lists
    elif objective == "Norm Score":
        best_sum = np.sum(c for c in range(count))
        worst_range = range(p["num_eligible"][j] - count, p["num_eligible"][j])
        worst_sum = np.sum(c for c in worst_range)
        achieved_sum = np.sum(p["a_pref_matrix"][cadets, j])
        return 1 - (achieved_sum - best_sum) / (worst_sum - best_sum)

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
        num_cadets = p['quota_e'][j]  # estimated number of cadets
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
        numerator = np.sum(p['utility'][i, j] * x[i, j] for i in p['I^E'][j])
        return numerator / num_cadets, numerator  # Measure, Numerator

    # New objective to evaluate CFM preference lists
    elif objective == "Norm Score":
        best_range = range(num_cadets)
        best_sum = np.sum(c for c in best_range)
        worst_range = range(p["num_eligible"][j] - num_cadets, p["num_eligible"][j])
        worst_sum = np.sum(c for c in worst_range)
        achieved_sum = np.sum(p["a_pref_matrix"][i, j] * x[i, j] for i in p["I^E"][j])
        return 1 - (achieved_sum - best_sum) / (worst_sum - best_sum), None  # Measure, Numerator

    # Unrecognized objective
    else:
        raise ValueError("Error. Objective '" + objective + "' does not have a means of calculation in the"
                                                            " VFT model. Please adjust.")

# AFSC Objective Measure Constraint Functions
def calculate_failed_constraint_metrics(j, k, x, metrics, p, vp):
    """
    Calculate failed constraint metrics for an AFSC objective and return the updated metrics dictionary.

    Parameters:
        j (int): Index of the AFSC objective.
        k (int): Index of the objective measure.
        x (numpy.ndarray): The solution matrix.
        metrics (dict): The metrics dictionary.
        p (dict): The fixed cadet/AFSC model parameters.
        vp (dict): The weight/value parameters.

    Returns:
        metrics (dict): The updated metrics dictionary.

    Note:
        This function calculates the failed constraint metrics for an AFSC objective and updates the metrics dictionary
        with the newly calculated values.
    """


    # Shorthand
    m = metrics

    # Constrained Approximate Measure (Only meant for degree tier constraints)
    if vp["constraint_type"][j, k] == 1:  # Should be an "at least constraint"

        # Get count variable
        count = np.sum(x[i, j] for i in p['I^E'][j])
        constrained_measure = (m['objective_measure'][j, k] * count) / min(p['quota_min'][j], p['pgl'][j])

    # Constrained Exact Measure
    elif vp["constraint_type"][j, k] == 2:  # Should be either an "at most constraint" or simple valid range (0.2, 0.4)
        constrained_measure = m['objective_measure'][j, k]

    else:
        pass

    # Measure is below the range
    if constrained_measure < vp['objective_min'][j, k]:
        m['objective_constraint_fail'][j, k] = \
            str(round(constrained_measure, 2)) + ' < ' + str(vp['objective_min'][j, k]) + '. ' + \
            str(round(100 * constrained_measure / vp['objective_min'][j, k], 2)) + '% Met.'
        m['total_failed_constraints'] += 1
        m["failed_constraints"].append(p['afscs'][j] + " " + vp['objectives'][k])
        m["con_fail_dict"][(j, k)] = '> ' + str(round(constrained_measure, 4))

    # Measure is above the range
    elif constrained_measure > vp['objective_max'][j, k]:
        m['objective_constraint_fail'][j, k] = \
            str(round(constrained_measure, 2)) + ' > ' + str(vp['objective_max'][j, k]) + '. ' + \
            str(round(100 * vp['objective_max'][j, k] / constrained_measure, 2)) + '% Met.'
        m['total_failed_constraints'] += 1
        m["failed_constraints"].append(p['afscs'][j] + " " + vp['objectives'][k])
        m["con_fail_dict"][(j, k)] = '< ' + str(round(constrained_measure, 4))

    return m  # Return *updated* metrics

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

# Additional Solution Helper Functions
def swap_solution_shape(solution, M, to_matrix=False):
    """
    Changes the solution from a vector of length N containing AFSC indices to an NxM binary matrix, or vice versa
    depending on the input. If "to_matrix" is specified, then we ONLY swap if it's to change a vector into a matrix.
    Otherwise, we return the matrix
    """
    N = len(solution)

    # If we have a vector of length N, turn it into a binary x matrix of shape (N, M)
    if np.shape(solution) == (N, ):
        return np.array([[1 if solution[i] == j else 0 for j in range(M)] for i in range(N)])

    # If we have a matrix of shape (N, M), turn it into a vector of length N (IF we don't "force" a returned matrix)
    elif np.shape(solution) == (N, M):
        if to_matrix:
            return solution  # Binary matrix of shape (N, M)
        else:
            return np.where(solution == 1)[1]  # Vector of length N

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
        # Output: The two solutions are 60.0% the same.
    """

    percent_similar = (sum(baseline == compared * 1) / len(baseline))
    if printing:
        print("The two solutions are " + str(percent_similar) + "% the same.")
    return percent_similar

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

def calculate_additional_useful_metrics(metrics, p, vp):
    """
    Add additional components to the "metrics" dictionary based on the parameters and value parameters.

    Parameters:
        metrics (dict): The dictionary containing the existing metrics.
        p (dict): The parameters dictionary.
        vp (dict): The value parameters dictionary.

    Returns:
        metrics (dict): The updated metrics dictionary.

    Note:
        This function adds additional components to the "metrics" dictionary based on the provided parameters
        and value parameters. The purpose is to enhance the information and analysis of the metrics.
    """


    # Generate objective scores for each objective
    for k in vp['K']:
        new_weights = vp['afsc_weight'] * vp['objective_weight'][:, k]
        new_weights = new_weights / sum(new_weights)
        metrics['objective_score'][k] = np.dot(new_weights, metrics['objective_value'][:, k])

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
                metrics["choice_counts"][dem]["Next 3"][j] = int(np.sum(metrics["choice_counts"][dem][afsc][3:6]))
                metrics["choice_counts"][dem]["Non-Volunteers"][j] = int(
                    len(dem_cadets[dem]) - np.sum(metrics["choice_counts"][dem][afsc]))

    return metrics

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

