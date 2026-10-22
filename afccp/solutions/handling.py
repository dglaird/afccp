"""
This module provides the core logic for **evaluating, validating, and comparing solutions**
within the `afccp` framework. It includes comprehensive tools for computing cadet and AFSC value
scores, applying VFT constraints, and generating solution diagnostics for use in optimization,
genetic algorithms, and post-hoc analysis.

Contents
--------
- **Solution Evaluation**
    - `evaluate_solution`: Evaluate a proposed solution (matrix/vector) against AFCCP objectives.
    - `fitness_function`: Calculate the scalar fitness score for use in meta-heuristics.
- **Metrics and Constraint Evaluation**
    - `calculate_objective_measure_matrix`: Compute objective measures across cadets/AFSCs.
    - `calculate_failed_constraint_metrics`: Annotate constraint violations for debugging.
    - `value_function`, `value_function_points`: Calculate piecewise linear VFT scores.
- **Additional Metrics**
    - `calculate_additional_useful_metrics`: Compute fairness, choice rank, gender/race equity, utility scores.
    - `calculate_base_training_metrics`: Integrate base/training preference satisfaction.
    - `calculate_castle_solution_metrics`: Evaluate CASTLE-specific value curves and GUO tradeoffs.
    - `calculate_ots_specific_metrics`: Add OTS-specific merit and preference analytics.
- **Stability and Comparison**
    - `calculate_blocking_pairs`: Identify cadet–AFSC blocking pairs based on stable matching criteria.
    - `compare_solutions`: Measure assignment similarity between two `j_array` vectors.
    - `similarity_coordinates`: Use multidimensional scaling (MDS) to visualize solution clusters.
- **Rated SOC Integration**
    - `incorporate_rated_results_in_parameters`: Lock matched/reserved cadets from SOC rated algorithms.
    - `augment_rated_algorithm_results`: Expand alternate lists and enforce USAFA/ROTC rated constraints.

Workflow
--------
1. **Evaluation of a Solution**
    - Construct `x`, `j_array`, and `afsc_array`.
    - Calculate AFSC/cadet utility values and objective scores.
    - Apply value functions and check for failed constraints.
    - Compute global metrics: `z`, `z^GUO`, blocking pairs, Simpson diversity, and match quality.
1. **Constraint Handling**
    - Differentiate between hard and soft constraints (AFSC value floors, cadet preference thresholds).
    - Track failed constraints using `objective_constraint_fail` and `con_fail_dict`.
1. **Post-Evaluation Analysis**
    - Calculate Simpson indices (race/ethnicity), choice distributions, and demographic breakdowns.
    - Decompose metrics by SOC, gender, service branch (USSF/USAF), and AFSC.
1. **Solution Comparison**
    - Visualize solution similarity with MDS or compute `compare_solutions` overlap %.
    - Count and list blocking pairs to assess matching stability.
1. **Integration with SOC-Rated Workflows**
    - Lock rated matches/reserves and update alternate lists.
    - Define alternate eligibility (hard/soft), preference sets, and blocking pair-safe configurations.

See Also
--------
- [`afccp.solutions.algorithms`](../../../reference/solutions/algorithms/#solutions.algorithms):
  Contains the matching and meta-heuristic algorithms whose outputs are evaluated here.
- [`afccp.data.preferences`](../../../reference/data/preferences/#data.preferences):
  Helper functions to compute cadet and AFSC preferences, including rated eligibility sets.
"""
import numpy as np
import copy
import pandas as pd

# afccp modules
import afccp.globals

# Import sklearn.manifold if it is installed
if afccp.globals.use_manifold:
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

    # Set of cadets that have been matched to AFSCs
    solution['I^Match'] = np.where(solution['j_array'] != p['M'])[0]
    solution['Num Matched'] = len(solution['I^Match'])

    # Loop through all cadets to assign their values
    for i in p['I']:
        solution['cadet_value'][i] = np.sum(x[i, j] * p['cadet_utility'][i, j] for j in p['J^E'][i])
        if solution['cadet_value'][i] < vp['cadet_value_min'][i]:
            solution['cadet_constraint_fail'][i] = 1
            solution['total_failed_constraints'] += 1
            solution["failed_constraints"].append("Cadet " + str(p['cadets'][i]) + " Value")

    # AFSC array!
    solution['afsc_array'] = np.array([p['afscs'][j] for j in solution['j_array']])

    # Define overall metrics
    solution['cadets_overall_value'] = np.dot(vp['cadet_weight'][solution['I^Match']],
                                              solution['cadet_value'][solution['I^Match']])
    solution['afscs_overall_value'] = np.dot(vp['afsc_weight'], solution['afsc_value'])
    solution['z^VFT'] = vp['cadets_overall_weight'] * solution['cadets_overall_value'] + \
                        vp['afscs_overall_weight'] * solution['afscs_overall_value']
    solution['num_ineligible'] = np.sum(x[i, j] * p['ineligible'][i, j] for j in p['J'] for i in solution['I^Match'])

    # Add additional metrics components (Non-VFT stuff)
    solution = calculate_additional_useful_metrics(solution, p, vp)

    # Calculate the SOC algorithm reserve/matches metrics!
    if 'cadets_solved_for' not in solution:  # Don't run this for a "partial" solution!
        solution = calculate_soc_algorithm_metrics(solution, p)

    # Add base/training components if applicable
    if 'base_array' in solution:
        solution = calculate_base_training_metrics(solution, p, vp)

    # Add Castle solution metrics
    if 'castle_q' in p:
        solution = calculate_castle_solution_metrics(solution, p)

    # Calculate FINAL "Z" Value!
    if 'z^CASTLE' in solution:  # CASTLE IS included, so weigh GUO and CASTLE scores for Z
        solution['z'] = round(p['w^G'] * solution['z^GUO'] + (1 - p['w^G']) * solution['z^CASTLE'], 4)
    else:  # CASTLE IS NOT included, so Z is just the GUO portion
        solution['z'] = solution['z^GUO']

    if 'ots' in p['SOCs']:
        solution = calculate_ots_specific_metrics(solution, p)

    # Calculate blocking pairs
    if 'a_pref_matrix' in p:
        try:
            solution['blocking_pairs'] = calculate_blocking_pairs(p, solution)
            solution['num_blocking_pairs'] = len(solution['blocking_pairs'])
        except:
            solution['blocking_pairs'] = []
            solution['num_blocking_pairs'] = np.nan
            print("WARNING. Blocking pair calculation received an error (likely preference list eligibility issues)")

    # Print solution evaluation summary
    if printing:
        # ANSI colors
        GREEN, YELLOW, RED, CYAN, BOLD, RESET = (
            '\033[92m', '\033[93m', '\033[91m',
            '\033[96m', '\033[1m', '\033[0m'
        )

        name = solution.get('name', 'New Solution')
        if 'VFT' in name:
            name += ' (Approximate)' if approximate else ' (Exact)'

        print(f"\n{BOLD}{CYAN}{'=' * 65}{RESET}")
        print(f"{BOLD}  SOLUTION EVALUATED: {name}{RESET}")
        print(f"{CYAN}{'=' * 65}{RESET}")

        # Objective function metrics
        print(f"\n{BOLD}OBJECTIVE FUNCTION{RESET}")
        for label, key in [('Overall Objective (z)', 'z'), ('Global Utility (z^GUO)', 'z^GUO'),
                           ('CASTLE Value (z^CASTLE)', 'z^CASTLE')]:
            if key in solution:
                print(f"  {label:<24} {solution[key]:.4f}")

        # Assignment metrics
        print(f"\n{BOLD}ASSIGNMENT METRICS{RESET}")
        matched, total = solution['Num Matched'], p['N']
        match_color = GREEN if matched == total else YELLOW

        print(f"  Matched Cadets           {match_color}{matched:,} / {total:,}{RESET}")
        print(f"  Unmatched Cadets         {solution['num_unmatched']:,}")
        print(f"  Ineligible Cadets        {solution['num_ineligible']:,}")

        if 'num_blocking_pairs' in solution:
            count = solution['num_blocking_pairs']
            print(f"  Blocking Pairs           {count:,}")

        # Rated algorithm metrics by SOC
        if 'num_fixed_constraints_met' in solution and 'num_fixed_constraints' in p:
            print(f"\n{BOLD}RATED ALGORITHM CONSTRAINTS{RESET}")
            print(f"  {'Constraint':<18} {'Overall':<15}" +
                  "".join(f"{soc.upper():<15}" for soc in p['SOCs']))
            print(f"  {'-' * (33 + 15 * len(p['SOCs']))}")

            for metric, label in [('fixed', 'Fixed Matches'), ('reserved', 'Reserved'),
                                  ('alternate', 'Alternates')]:
                met = solution[f'num_{metric}_constraints_met']
                targets = p[f'num_{metric}_constraints']

                total_met = sum(met[soc] for soc in p['SOCs'])
                total_target = sum(targets[soc] for soc in p['SOCs'])

                def colored_ratio(n, d):
                    color = GREEN if n == d else RED if n == 0 and d > 0 else YELLOW
                    return f"{color}{n}/{d}{RESET}"

                print(f"  {label:<18} {colored_ratio(total_met, total_target):<24}" +
                      "".join(f"{colored_ratio(met[soc], targets[soc]):<24}" for soc in p['SOCs']))

            # Show constraint violations separately, grouped by SOC.
            violations = [
                ('Fixed Matches', 'pairs_breaking_fixed_constraints'),
                ('Reserved', 'pairs_breaking_reserved_constraints'),
                ('Alternates', 'pairs_breaking_alternate_constraints')
            ]

            for label, key in violations:
                if key not in solution:
                    continue

                for soc, pairs in solution[key].items():
                    if pairs:
                        print(f"\n  {RED}{soc.upper()} {label} Violations ({len(pairs)}):{RESET}")
                        print(f"    {pairs}")

        # OTS-specific metrics
        if 'ots' in p['SOCs']:
            print(f"\n{BOLD}OTS METRICS{RESET}")

            if 'matched_out_of_must_match' in solution:
                print(f"  Must-Match Candidates    {solution['matched_out_of_must_match']}")

            if 'OTS Average Merit' in solution:
                print(f"  Average Merit            {solution['OTS Average Merit']:.4f}")

            if 'OTS Selected Pref Count' in solution and 'I^Match-OTS' in solution:
                selected = solution['OTS Selected Pref Count']
                total = len(solution['I^Match-OTS'])
                print(f"  Received Selected AFSC   {selected:,} / {total:,}")

        print(f"\n{CYAN}{'=' * 65}{RESET}\n")

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
    Calculate blocking pairs in a given solution.

    Parameters:
    - parameters (dict): The parameters of the matching problem.
    - solution (dict): The current matching solution.
    - only_return_count (bool): If True, return the count of blocking pairs; if False,
      return the list of blocking pairs.

    Returns:
    - blocking_pairs (list or int): A list of blocking pairs (or count of blocking pairs).

    Description:
    This function calculates the blocking pairs in a given matching solution based on
    the stable matching community's definition. A blocking pair consists of an unmatched
    cadet and a more preferred AFSC that is also unmatched or assigned to a cadet with
    lower preference.

    Parameters Dictionary Structure:
    - 'cadet_preferences': An array representing cadet preferences.
    - 'a_pref_matrix': A matrix of AFSC preferences.
    - 'J': The set of all AFSCs.
    - 'M': A special symbol representing an unmatched cadet.

    Solution Dictionary Structure:
    - 'j_array': An array representing the assignment of AFSCs to cadets.

    Dependencies:
    - NumPy

    Reference:
    - Gale, D., & Shapley, L. S. (1962). College Admissions and the Stability of
      Marriage. American Mathematical Monthly, 69(1), 9-15.
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
                        blocking_pairs.append((i, j_compare))
                        blocking_pair_count += 1
                    break

                # The lowest rank of the assigned cadet
                afsc_choice_of_worst_cadet = np.max(matched_cadet_ranks)

                # Check for blocking pairs
                if afsc_choice_of_this_cadet < afsc_choice_of_worst_cadet:
                    if only_return_count:
                        blocking_pair_count += 1
                    else:
                        blocking_pairs.append((i, j_compare))
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
    Calculate the Normalized Score for an AFSC assignment.

    Parameters:
    - cadets (list or numpy.ndarray): A list of cadets assigned to the AFSC.
    - j (int): The index of the AFSC for which the score is calculated.
    - p (dict): The problem parameters including preferences and AFSC data.
    - count (int, optional): The number of cadets assigned to the AFSC. If not provided,
      it is calculated from the length of the 'cadets' list.

    Returns:
    - norm_score (float): The normalized score for the AFSC assignment, ranging from 0 to 1.

    Description:
    This function calculates the normalized score for an assignment of cadets to an AFSC.
    The score reflects how well the cadets are matched to their preferences for the given AFSC.
    A higher score indicates a better match, while a lower score suggests a less favorable assignment.

    The calculation involves comparing the achieved score (sum of cadet preferences) to the
    best and worst possible scores for the AFSC assignment. The result is then normalized to
    a range between 0 and 1, with 1 being the best possible score and 0 being the worst.

    Parameters Dictionary Structure:
    - 'a_pref_matrix': A matrix of AFSC preferences.
    - 'num_eligible': A dictionary with the number of eligible cadets for each AFSC.

    Dependencies:
    - NumPy

    Returns:
    - norm_score (float): The normalized score for the AFSC assignment, ranging from 0 to 1.
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
    Calculate the Normalized Score for an AFSC assignment using custom ranks.

    Parameters:
    - ranks (numpy.ndarray): An array containing the preference ranks for eligible cadets
      for the specific AFSC.
    - achieved_ranks (numpy.ndarray): An array of achieved ranks, indicating the ranks
      at which cadets were assigned to the AFSC.

    Returns:
    - norm_score (float): The normalized score for the AFSC assignment, ranging from 0 to 1.

    Description:
    This function calculates the normalized score for an assignment of cadets to an AFSC.
    The score reflects how well the cadets are matched to their preferences for the given AFSC.
    A higher score indicates a better match, while a lower score suggests a less favorable assignment.

    The calculation involves comparing the achieved ranks of cadets to the best and worst possible ranks
    for the AFSC assignment. The result is then normalized to a range between 0 and 1, with 1 being the
    best possible score and 0 being the worst.

    Dependencies:
    - NumPy

    Returns:
    - norm_score (float): The normalized score for the AFSC assignment, ranging from 0 to 1.
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
        solution['average_cadet_choice'] = round(np.mean(solution['cadet_choice'][solution['I^Match']]), 2)

        # Calculate average cadet choice for each AFSC individually
        solution['afsc_average_cadet_choice'] = np.zeros(p['M'])
        for j in p['J']:
            cadets = np.where(solution['j_array'] == j)[0]
            solution['afsc_average_cadet_choice'][j] = np.mean(p['c_pref_matrix'][cadets, j])

        # Calculate overall utility scores (z^GUO) from matched cadets
        solution['z^GUO'] = round(np.mean(solution['global_utility_achieved'][solution['I^Match']]), 4)
        solution['cadet_utility_overall'] = round(np.mean(solution['cadet_utility_achieved'][solution['I^Match']]), 4)
        solution['afsc_utility_overall'] = round(np.mean(solution['afsc_utility_achieved'][solution['I^Match']]), 4)

        # Calculate cadet utility based on SOC
        for soc in p['SOCs']:
            cadets = np.intersect1d(p[f'{soc}_cadets'], solution['I^Match'])
            solution[f'{soc}_cadet_utility'] = (
                round(float(np.mean(solution['cadet_utility_achieved'][cadets])), 4)
                if len(cadets) else None
            )

    # Cadet Choice Counts (For exporting solution file to excel)
    solution['cadet_choice_counts'] = {}
    for choice in np.arange(1, 11):  # Just looking at top 10
        choice_cadets = np.where(solution['cadet_choice'] == choice)[0]
        solution['cadet_choice_counts'][choice] = len(np.intersect1d(choice_cadets, solution['I^Match']))
    solution['cadet_choice_counts']['All Others'] = int(solution['Num Matched'] - sum(
        [solution['cadet_choice_counts'][choice] for choice in np.arange(1, 11)]))

    # Save the counts for each AFSC separately from the objective_measure matrix
    quota_k = np.where(vp['objectives'] == 'Combined Quota')[0][0]
    solution['count'] = solution['objective_measure'][:, quota_k]

    # Assigned cadets
    solution['cadets_assigned'] = {j: np.where(solution['j_array'] == j)[0] for j in p['J']}
    solution['num_unmatched'] = p['N'] - len(solution['I^Match'])

    # Cadets assigned to each accession group
    for acc_grp in p['afscs_acc_grp']:
        solution['I^' + acc_grp] = np.array(
            [i for i in p['I'] if solution['j_array'][i] in p['J^' + acc_grp]], dtype=int)

    # Air Force & USSF Cadets
    if 'USSF' in p['afscs_acc_grp']:
        solution['I^USAF'] = np.array([i for i in solution['I^Match'] if i not in solution['I^USSF']], dtype=int)
    else:
        solution['I^USAF'] = solution['I^Match']  # Everyone matched is USAF if USSF is not in this problem instance

    # Calculate USSF/USAF Metrics
    services = ['USAF', 'USSF'] if 'USSF' in p['afscs_acc_grp'] else ['USAF']
    for service in services:
        service_cadets = solution[f'I^{service}']
        solution[f'{service.lower()}_om'] = (
            round(np.mean(p['merit'][service_cadets]), 3) if len(service_cadets) > 0 else np.nan)

        # Calculate Service-SOC Specific Metrics
        for soc in p['SOCs']:
            cadets = np.intersect1d(service_cadets, p[f'{soc}_cadets'])
            solution[f'{service.lower()}_{soc}_count'] = len(cadets)
            solution[f'{service.lower()}_{soc}_pgl_target'] = \
                str(len(cadets)) + ' / ' + str(p[f'{service.lower()}_{soc}_pgl'])

        # Calculate cadet/AFSC metrics
        for entity in ['cadet', 'afsc']:
            if len(service_cadets) > 0:
                solution[f'{service}_{entity}_utility'] = \
                    round(np.mean(solution[f'{entity}_utility_achieved'][service_cadets]), 4)
            else:
                solution[f'{service}_{entity}_utility'] = 0

    # Calculate weighted average AFSC choice (based on Norm Score)
    if 'Norm Score' in vp['objectives']:
        k = np.where(vp['objectives'] == 'Norm Score')[0][0]

        # Individual AFSC normalized scores; unassigned AFSCs receive zero.
        scores = np.asarray(solution['objective_measure'][:, k], dtype=float).copy()
        counts = np.asarray(solution['count'], dtype=float)
        scores[counts == 0] = 0
        solution['afsc_norm_score'] = scores

        # Weighted average across all AFSCs.
        total = counts.sum()
        solution['weighted_average_afsc_score'] = (
            np.dot(counts, scores) / total if total > 0 else 0.0
        )

        # Weighted average for each accession group.
        for acc_grp in p['afscs_acc_grp']:
            indices = p[f'J^{acc_grp}']
            group_counts = counts[indices]
            group_total = group_counts.sum()
            solution[f'weighted_average_{acc_grp.lower()}_afsc_score'] = (
                np.dot(group_counts, scores[indices]) / group_total
                if group_total > 0 else 0.0
            )

        # Space Force and Air Force scores.
        if 'USSF' in p['afscs_acc_grp']:
            for service in ['USSF', 'USAF']:
                indices = p[f'J^{service}']
                service_counts = counts[indices]
                service_total = service_counts.sum()
                solution[f'weighted_average_{service.lower()}_afsc_score'] = (
                    np.dot(service_counts, scores[indices]) / service_total
                    if service_total > 0 else 0.0
                )

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
    dd = {"usafa": [soc.upper() for soc in p['SOCs']], "male": ["Male", "Female"]}  # Demographic Dictionary
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
    for cat in dd['usafa']:
        arr = np.array([i for i in p['I^' + cat] if solution['j_array'][i] in p['cadet_preferences'][i][:3]])
        solution['top_3_' + cat.lower() + '_count'] = round(len(arr) / len(p['I^' + cat]), 4)

    # Initialize arrays within the choice dictionaries for the AFSCs
    choice_categories = ["Top 3", "Next 3", "All Others", "Total"]
    for dem in solution["choice_counts"]:
        for c_cat in choice_categories:
            solution["choice_counts"][dem][c_cat] = np.zeros(p["M"]).astype(int)
        for afsc in p["afscs"]:
            solution["choice_counts"][dem][afsc] = np.zeros(p["P"]).astype(int)

    # AFSC choice
    solution['afsc_choice_counts'] = {}

    cat_bound_dict = {'90-100%': 0.9, '80-89%': 0.8, '70-79%': 0.7, '60-69%': 0.6, '50-59%': 0.5,
                      '40-49%': 0.4, '30-39%': 0.3, '20-29%': 0.2, '10-19%': 0.1, '0-10%': 0}

    # Loop through each AFSC
    for j, afsc in enumerate(p["afscs"][:p['M']]):  # Skip unmatched AFSC

        # The cadets that were assigned to this AFSC
        dem_cadets = {"TOTAL": np.where(solution["afsc_array"] == afsc)[0]}

        # The cadets with the demographic that were assigned to this AFSC
        for cat in demographic_dict:
            dem_1, dem_2 = demographic_dict[cat][0], demographic_dict[cat][1]
            dem_cadets[dem_1] = np.intersect1d(np.where(p[cat] == 1)[0], dem_cadets["TOTAL"])
            dem_cadets[dem_2] = np.intersect1d(np.where(p[cat] == 0)[0], dem_cadets["TOTAL"])

        afsc_utilities = p['afsc_utility'][dem_cadets['TOTAL'], j]
        solution['afsc_choice_counts'][afsc] = {}
        for cat, bound in cat_bound_dict.items():

            if cat == '90-100%':
                condition = (afsc_utilities >= bound) & (afsc_utilities <= bound + 0.1)
            else:
                condition = (afsc_utilities >= bound) & (afsc_utilities < bound + 0.1)
            cadets = np.where(condition)[0]
            solution['afsc_choice_counts'][afsc][cat] = len(cadets)

        # Loop through each choice and calculate the metric
        for choice in range(p["P"]):

            # The cadets that were assigned to this AFSC and placed it in their Pth choice
            assigned_choice_cadets = np.intersect1d(p["I^Choice"][choice][j], dem_cadets["TOTAL"])

            # The cadets that were assigned to this AFSC, placed it in their Pth choice, and had the demographic
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

    # Top 3 Choice Percentages
    matched = solution['I^Match']
    top3 = sum(1 <= solution['cadet_choice'][i] <= 3 for i in matched)
    solution['top_3_choice_percent'] = (round(top3 / len(matched), 3) if len(matched) else None)

    # Top 3 Choices by SOC
    for soc in p['SOCs']:
        cadets = np.intersect1d(p[f'{soc}_cadets'], matched)
        top3 = sum(1 <= solution['cadet_choice'][i] <= 3 for i in cadets)
        solution[f'{soc}_top_3_choice_percent'] = (round(top3 / len(cadets), 3) if len(cadets) else None)

    # Top 3 Choices by Service
    for service in ['USSF', 'USAF']:
        key = f'I^{service}'
        if key in solution:
            cadets = solution[key]
            top3 = sum(1 <= solution['cadet_choice'][i] <= 3 for i in cadets)
            solution[f'top_3_{service.lower()}_choice_percent'] = (round(top3 / len(cadets), 3) if len(cadets) else None)

    return solution


def calculate_base_training_metrics(solution, p, vp):
    """
    Add additional base/training components to the "solution" dictionary based on the parameters and value parameters.
    """

    # Initialize arrays
    solution['base_choice'] = np.zeros(p['N']).astype(int)
    solution['base_utility_achieved'] = np.zeros(p['N'])
    solution['course_utility_achieved'] = np.zeros(p['N'])
    solution['cadet_state_achieved'] = np.zeros(p['N']).astype(int)
    solution['cadet_value_achieved'] = np.zeros(p['N'])

    # Weights Implemented
    solution['afsc_weight_used'] = np.zeros(p['N'])
    solution['base_weight_used'] = np.zeros(p['N'])
    solution['course_weight_used'] = np.zeros(p['N'])
    solution['state_utility_used'] = np.zeros(p['N'])

    # Loop through each cadet to load in their values to each of the above
    for i, j in enumerate(solution['j_array']):

        # If this candidate was not matched to an AFSC, don't calculate any metrics! They get nothing!
        if p['afscs'][j] == '*':
            continue
        b, c = solution['b_array'][i], int(solution['c_array'][i][1])
        t = p['tau'][j]

        # Determine what state this cadet achieved
        d = [d for d in p['D'][i] if j in p['J^State'][i][d]][0]
        solution['cadet_state_achieved'][i] = d

        # Extract base information if it's there
        solution['base_weight_used'][i] = p['w^B'][i][d]
        if b in p['B']:
            solution['base_choice'][i] = p['b_pref_matrix'][i, b]
            solution['base_utility_achieved'][i] = p['base_utility'][i, b]

        # Extract course information if it's there
        solution['course_weight_used'][i] = p['w^C'][i][d]
        if t in p['T']:
            if c in p['C'][t]:
                solution['course_utility_achieved'][i] = p['course_utility'][i][t][c]

        # AFSC/Overall information
        solution['afsc_weight_used'][i] = p['w^A'][i][d]
        solution['state_utility_used'][i] = p['u^S'][i][d]

        # Calculate Cadet Value
        solution['cadet_value_achieved'][i] = p['u^S'][i][d] * (
                p['w^A'][i][d] * (p['cadet_utility'][i, j] / p['u^S'][i][d]) +
                solution['base_weight_used'][i] * solution['base_utility_achieved'][i] +
                p['w^C'][i][d] * solution['course_utility_achieved'][i])

    # Calculate adjusted Z value (VFT) and associated metrics
    solution['cadet_value'] = solution['cadet_value_achieved']
    solution['cadets_overall_value'] = np.dot(vp['cadet_weight'], solution['cadet_value'])
    solution['z^VFT'] = vp['cadets_overall_weight'] * solution['cadets_overall_value'] + \
                    vp['afscs_overall_weight'] * solution['afsc_utility_overall']

    # Update cadet utility overall
    solution['cadet_utility_overall'] = round(np.mean(solution['cadet_value_achieved'][solution['I^Match']]), 4)

    # Calculate adjusted Z value (GUO)
    solution['z^GUO'] = vp['afscs_overall_weight'] * solution['afsc_utility_overall'] + \
                        vp['cadets_overall_weight'] * solution['cadet_utility_overall']

    return solution


def calculate_castle_solution_metrics(solution, p):
    """
    Add CASTLE-specific solution metrics
    """

    # Shorthand
    x, q = solution['x'], p['castle_q']

    # Loop through each CASTLE AFSC to calculate the numbers of people assigned
    solution['castle_counts'], solution['castle_v'] = {}, {}
    afscs = [afsc for afsc, _ in p['J^CASTLE'].items()]
    for afsc, j_indices in p['J^CASTLE'].items():

        # Get the number of people assigned to each AFSC under this "CASTLE" AFSC umbrella
        measure = np.sum(np.sum(x[i, j] for i in p['I^E'][j]) for j in j_indices)

        # (+ ADD OTS if they're not being matched!!)
        if 'ots' not in p['SOCs']:
            measure += p['ots_counts'][afsc]

        # Save the measure
        solution['castle_counts'][afsc] = measure

        # Get the value from this AFSC's curve
        v = value_function(q['a'][afsc], q['f^hat'][afsc], q['r'][afsc], measure)
        solution['castle_v'][afsc] = v

    # Calculate 'z^CASTLE'
    solution['z^CASTLE'] = round(np.sum(solution['castle_v'][afsc] for afsc in afscs), 4)
    return solution


def calculate_ots_specific_metrics(solution, p):

    # Must Match summary
    if 'I^Must_Match' in p:
        solution['matched_out_of_must_match'] = \
            f"{len(np.intersect1d(p['I^Must_Match'], solution['I^Match']))} / {len(p['I^Must_Match'])}"

    # OTS Quality
    solution['I^Match-OTS'] = np.intersect1d(solution['I^Match'], p['I^OTS'])
    solution['OTS Average Merit'] = np.around(np.mean(p['merit'][solution['I^Match-OTS']]), 3)

    # Number of OTS candidates that received an AFSC they selected as a preference
    num_desired = 0
    for i in solution['I^Match-OTS']:
        if solution['j_array'][i] in p['J^Selected'][i]:
            num_desired += 1
    solution['OTS Selected Pref Count'] = num_desired

    # Average cadet/AFSC utility of matched OTS candidates
    solution['OTS Average AFSC Utility'] = \
        np.around(np.mean(solution['afsc_utility_achieved'][solution['I^Match-OTS']]), 3)
    solution['OTS Average Cadet Utility'] = \
        np.around(np.mean(solution['cadet_utility_achieved'][solution['I^Match-OTS']]), 3)

    return solution


def calculate_soc_algorithm_metrics(solution, p):

    # Shorthand
    s = solution

    # Set up rated algorithm constraints met counter
    s['num_alternate_constraints_met'] = {soc: 0 for soc in p['SOCs']}
    s['num_reserved_constraints_met'] = {soc: 0 for soc in p['SOCs']}
    s['num_fixed_constraints_met'] = {soc: 0 for soc in p['SOCs']}

    # Setup rated algorithm break cadet-AFSC pair lists
    s['pairs_breaking_fixed_constraints'] = {soc: [] for soc in p['SOCs']}
    s['pairs_breaking_reserved_constraints'] = {soc: [] for soc in p['SOCs']}
    s['pairs_breaking_alternate_constraints'] = {soc: [] for soc in p['SOCs']}

    # Loop through each SOC
    for soc in p['SOCs']:
        for i in p[f'{soc}_cadets']:
            j = s['j_array'][i]

            # Was this cadet matched to a rated AFSC properly?
            if i in p['J^Fixed']:
                if j == p['J^Fixed'][i]:
                    s['num_fixed_constraints_met'][soc] += 1
                else:
                    j_fixed = p['J^Fixed'][i]
                    s['pairs_breaking_fixed_constraints'][soc].append((i, p['afscs'][j_fixed]))

            # Was this cadet reserved for a rated AFSC properly?
            if i in p['J^Reserved']:
                if j in p['J^Reserved'][i]:
                    s['num_reserved_constraints_met'][soc] += 1
                else:
                    num_reserved = len(p['J^Reserved'][i])
                    j_reserved = p['J^Reserved'][i][num_reserved - 1]
                    s['pairs_breaking_reserved_constraints'][soc].append((i, p['afscs'][j_reserved]))

            # Was this cadet on an alternate list with no blocking pairs?
            if 'I^Alternates' in p:
                if i in p['I^Alternates']:
                    j_alternate = p['I^Alternates'][i]

                    # If the cadet is matched to the AFSC they're an alternate for, the rated-cadet pair is cadet-stable
                    if j == j_alternate:
                        cadet_stable = True

                    # If the cadet is matched to an AFSC they prefer over this rated AFSC, it's stable
                    elif j in p['J^Preferred [' + soc + ']'][j_alternate][i]:
                        cadet_stable = True

                    # If the cadet is matched to an AFSC that is less than desirable than this rated AFSC, then
                    else:  # the rated AFSC better be filled with higher ranked cadets
                        cadet_stable = False

                    cadets_more_preferred = p[f'I^Preferred [{soc}]'][j_alternate][i]
                    soc_cadets_assigned = np.intersect1d(p[f'{soc}_cadets'], s['cadets_assigned'][j_alternate])
                    cadets_more_preferred_matched = np.intersect1d(soc_cadets_assigned, cadets_more_preferred)

                    # If AFSC does not have all PGL slots filled with cadets it prefers over this cadet, it's not stable
                    if len(cadets_more_preferred_matched) < p[soc + '_quota'][j_alternate]:
                        afsc_stable = False

                    else:  # This AFSC has at least as many more preferred cadets as the PGL quota
                        afsc_stable = True

                    # Is this a blocking pair?
                    if not cadet_stable and not afsc_stable:
                        s['pairs_breaking_alternate_constraints'][soc].append((i, p['afscs'][j_alternate]))
                    else:
                        s['num_alternate_constraints_met'][soc] += 1

    # Rated algorithm metrics
    for metric in ['fixed', 'reserved', 'alternate']:
        count_key = f'num_{metric}_constraints_met'
        parameter_key = f'num_{metric}_constraints'

        # Skip metrics when the rated stability feature has not been incorporated.
        if parameter_key not in p or count_key not in solution:
            continue
        counts = solution[count_key]
        quotas = p[parameter_key]
        if not isinstance(counts, dict) or not isinstance(quotas, dict):
            continue

        # Overall results
        numerator = sum(counts.get(soc, 0) for soc in p['SOCs'])
        denominator = sum(quotas.get(soc, 0) for soc in p['SOCs'])
        solution[f'total_{metric}_constraints_met'] = f'{numerator} / {denominator}'

        # SOC-specific results
        solution[f'soc_{metric}_constraints_met'] = {
            soc: f'{counts.get(soc, 0)} / {quotas.get(soc, 0)}'
            for soc in p['SOCs']
        }

    return s


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
        return len(np.intersect1d(p['usafa_cadets'], cadets))
    elif objective == 'ROTC Quota':
        return len(np.intersect1d(p['rotc_cadets'], cadets))
    elif objective == 'OTS Quota':
        return len(np.intersect1d(p['ots_cadets'], cadets))

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
        return np.sum(x[i, j] for i in np.intersect1d(p['usafa_cadets'], p['I^E'][j])), None # Measure, Numerator
    elif objective == "ROTC Quota":
        return np.sum(x[i, j] for i in np.intersect1d(p['rotc_cadets'], p['I^E'][j])), None  # Measure, Numerator
    elif objective == "OTS Quota":
        return np.sum(x[i, j] for i in np.intersect1d(p['ots_cadets'], p['I^E'][j])), None  # Measure, Numerator

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
    Perform Multidimensional Scaling (MDS) on a similarity matrix to obtain coordinates
    representing the solutions' similarity relationships.

    Parameters:
    - similarity_matrix (numpy.ndarray): A square similarity matrix where each element
      (i, j) measures the similarity between solutions i and j.

    Returns:
    - coordinates (numpy.ndarray): An array of 2D coordinates representing the solutions
      in a space where the distance between solutions reflects their similarity.

    Description:
    This function takes in a similarity matrix and performs Multidimensional Scaling (MDS)
    to obtain coordinates representing the solutions in a lower-dimensional space. The
    purpose of MDS is to transform similarity data into distances. In the resulting 2D
    space, solutions that are similar to each other will be closer together, while
    dissimilar solutions will be farther apart.

    MDS is particularly useful for visualizing the similarity relationships among solutions.
    These coordinates can be used for plotting or further analysis to gain insights into
    how solutions relate to each other based on their similarities.

    Note: Ensure that you have the required libraries, such as NumPy and Scikit-learn, installed.
    """

    # Change similarity matrix into distance matrix
    distances = 1 - similarity_matrix

    # Get coordinates
    if afccp.globals.use_manifold:
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
    and incorporates them into the problem's parameters. It fixes cadets who were "matched" by the algorithm
    to specific AFSCs and constrains individuals who had "reserved" slots.

    Parameters:
    - instance: An instance of the problem, containing parameters and algorithm results.
    - printing (bool, optional): A flag to control whether to print information during execution.
      Set to True to enable printing, and False to suppress it. Default is True.

    Returns:
    - parameters (dict): The updated parameters dictionary reflecting the rated algorithm results.

    Description:
    This function is used to integrate the outcomes of the Rated SOC (Source of Commissioning) algorithm into
    the problem's parameters. It processes the results for both USAFA (United States Air Force Academy)
    and ROTC (Reserve Officers' Training Corps) categories.

    - The "Matched" cadets are assigned to specific AFSCs based on the algorithm results. These assignments
      are recorded in the 'J^Fixed' array within the parameters.

    - The "Reserved" cadets have their AFSC selections constrained based on their reserved slots. The
      'J^Reserved' dictionary is updated to enforce these constraints.

    - Special treatment is provided for AFSCs with an "alternate list" concept. Cadets who did not receive
      one of their top preferences but are next in line for a particular AFSC are assigned to the "alternate list."
      Cadets on this list may be given preferences or reserved slots, depending on availability.

    This function aims to ensure that the problem's parameters align with the Rated SOC algorithm's results,
    facilitating further decision-making and analysis.

    Note: Detailed information on the Rated SOC algorithm results is assumed to be available within the 'instance.'
    """

    if printing:
        print("Incorporating rated algorithm results...")

    # Shorthand
    p, vp, solutions = instance.parameters, instance.value_parameters, instance.solutions
    mdl_p = instance.mdl_p
    if mdl_p['socs_to_use'] is None:
        socs_to_use = p['SOCs']
    else:
        socs_to_use = mdl_p['socs_to_use']
    upper_socs = [soc.upper() for soc in socs_to_use]

    # Make sure we have the solutions from both SOCs with matches and reserves
    for soc in upper_socs:
        for kind in ['Reserves', 'Matches']:
            solution_name = "Rated " + soc + " HR (" + kind + ")"
            if solution_name not in solutions:
                return p  # We don't have the required solutions!

    # Matched cadets get fixed in the solution!
    if 'J^Fixed' not in p:
        p['J^Fixed'] = {}
    for soc in upper_socs:
        solution = solutions["Rated " + soc + " HR (Matches)"]
        matched_cadets = np.where(solution['j_array'] != p['M'])[0]
        for i in matched_cadets:
            p['J^Fixed'][i] = solution['j_array'][i]

    # Reserved cadets AFSC selection is constrained to be AT LEAST their reserved Rated slot
    p['J^Reserved'] = {}
    for soc in upper_socs:
        solution = solutions["Rated " + soc + " HR (Reserves)"]
        reserved_cadets = np.where(solution['j_array'] != p['M'])[0]
        for i in reserved_cadets:
            j = solution['j_array'][i]
            choice = np.where(p['cadet_preferences'][i] == j)[0][0]
            p['J^Reserved'][i] = p['cadet_preferences'][i][:choice + 1]

    # Calculate additional rated algorithm result information for both SOCs (alternate lists)
    p['I^Alternates'] = {soc: {} for soc in p['SOCs']}
    for soc in socs_to_use:

        # Do we want to potentially allow ROTC to fill USAFA pilot slots?
        if mdl_p['usafa_soc_pilot_cross_in'] and soc == 'rotc':

            # If we want to allow cross-flow, we need to add USAFA reserved pilot slots to be potentially filled
            j_pilot_u = np.where(p['afscs'] == '11XX_U')[0][0]  # by ROTC (increase their alternate list)

            # First we calculate the number of reserved USAFA pilot slots there are
            u_reserved_pilot = \
                [i for i in p['J^Reserved'] if
                 p['cadet_preferences'][i][len(p['J^Reserved'][i]) - 1] == j_pilot_u and p['usafa'][i]]
            num_reserved_u_pilot = len(u_reserved_pilot)

            # We also need the number of hard alternates for USAFA pilot
            u_hard_alternates_pilot = [i for i, j in p['J^Alternates (Hard)'].items() if j == j_pilot_u]
            num_hard_alternates_u_pilot = len(u_hard_alternates_pilot)

            # The number of potential additions for ROTC pilot is going to be the total number of resered USAFA slots
            num_additions_rotc_pilot = num_reserved_u_pilot - num_hard_alternates_u_pilot  # minus hard alternates

        else:
            num_additions_rotc_pilot = 0
        p = augment_rated_algorithm_results(p, soc=soc, printing=instance.mdl_p['alternate_list_iterations_printing'],
                                            num_additions_rotc_pilot=num_additions_rotc_pilot)

    # Set up variables
    p['num_alternate_constraints'] = {
        soc: len([i for i in p['I^Alternates'] if i in p[f'{soc}_cadets']]) for soc in socs_to_use}
    p['num_reserved_constraints'] = {
        soc: len([i for i in p['J^Reserved'] if i in p[f'{soc}_cadets']])for soc in socs_to_use}
    p['num_fixed_constraints'] = {
        soc: len([i for i in p['J^Fixed'] if i in p[f'{soc}_cadets']])for soc in socs_to_use}

    # Print statement
    if printing:

        # Matched/Reserved Lists
        print_str = "Rated SOC Algorithm Results:\n"
        for soc in upper_socs:
            for kind in ["Fixed", "Reserved"]:
                count = str(len([i for i in p[soc.lower() + "_cadets"] if i in p['J^' + kind]]))
                print_str += soc + ' ' + kind + ' Cadets: ' + count + ', '
        print(print_str[:-2])

        # Alternate Lists
        count_u = str(int(sum([len(p['I^Alternate [usafa]'][j]) for j in p['J^Rated']])))
        count_r = str(int(sum([len(p['I^Alternate [rotc]'][j]) for j in p['J^Rated']])))
        print_str = "USAFA Rated Alternates: " + count_u + ", ROTC Rated Alternates: " + count_r
        if 'ots' in socs_to_use:
            count_o = str(int(sum([len(p['I^Alternate [ots]'][j]) for j in p['J^Rated']])))
            print_str += ", OTS Rated Alternates: " + count_o
        print(print_str)

    return p  # Return the parameters!


def augment_rated_algorithm_results(p, soc='rotc', printing=False, num_additions_rotc_pilot: int = 0):
    """
    Analyzes the results of the Rated SOC algorithm for a specific SOC (Source of Commissioning),
    such as ROTC or USAFA, and augments the system's parameters. This analysis includes identifying
    alternates and definitively matching additional individuals to AFSCs.

    Parameters:
    - p (dict): The problem's parameters containing relevant data.
    - soc (str, optional): The SOC to analyze and augment results for. Default is 'rotc'.
    - printing (bool, optional): A flag to control whether to print information during execution.
      Set to True to enable printing, and False to suppress it. Default is False.

    Returns:
    - parameters (dict): The updated parameters dictionary reflecting the rated algorithm results,
      including alternates and definitively matched individuals.

    Description:
    This function processes the results of the Rated SOC algorithm for a specific SOC category,
    identifying alternates and definitively matching additional individuals to AFSCs. The primary goal
    is to ensure the system's parameters accurately reflect the outcomes of the algorithm, which
    aids in further analysis and decision-making.

    - 'Reserved' cadets have their AFSC selections constrained to match their reserved slots.
    - 'Matched' cadets are assigned specific AFSCs based on the algorithm results.
    - 'Alternates' are cadets who did not receive one of their top AFSC preferences but are next in line
      for specific AFSCs based on the algorithm's execution. Alternates may be given preferences or
      reserved slots, depending on availability.

    Note: Detailed information on the Rated SOC algorithm results is assumed to be available within the 'parameters.'
    """

    # Start with a full list of cadets eligible for each AFSC from this SOC
    possible_cadets = {j: list(np.intersect1d(p['I^E'][j], p[soc + '_cadets'])) for j in p['J^Rated']}

    # Used for stopping conditions
    last_reserves, last_matches, last_alternates_h = np.array([1000 for _ in p['J^Rated']]), \
                                                     np.array([1000 for _ in p['J^Rated']]), \
                                                     np.array([1000 for _ in p['J^Rated']])

    if printing:
        print("\nSOC:", soc.upper())
        print()

    # Main algorithm
    iteration, iterating = 0, True
    while iterating:

        # Set of cadets reserved or matched to each AFSC
        p['I^Reserved'] = {j: np.array([i for i in p['J^Reserved'] if p['cadet_preferences'][i][
            len(p['J^Reserved'][i]) - 1] == j and p[soc][i]]) for j in p['J^Rated']}
        p['I^Matched'] = {j: np.array([
            i for i in p['J^Fixed'] if j == p['J^Fixed'][i] and p[soc][i]]) for j in p['J^Rated']}

        # Number of alternates (number of reserved slots)
        num_reserved = {j: len(p['I^Reserved'][j]) for j in p['J^Rated']}

        # Need to determine who falls into each category of alternates
        hard_alternates = {j: [] for j in p['J^Rated']}
        soft_r_alternates = {j: [] for j in p['J^Rated']}
        soft_n_alternates = {j: [] for j in p['J^Rated']}
        alternates = {j: [] for j in p['J^Rated']}  # all the cadets ordered here

        # Loop through each rated AFSC to determine alternates
        for j in p['J^Rated']:

            # We may want to increase the length of ROTC pilot alternate list to account for USAFA
            if p['afscs'][j] == '11XX_R':
                num_additions = copy.deepcopy(num_additions_rotc_pilot)
            else:
                num_additions = 0

            # Loop through each cadet in order of the AFSC's preference from this SOC
            for i in p['afsc_preferences'][j]:
                if not p[soc][i]:
                    continue

                # Assume this cadet is "next in line" until proven otherwise
                next_in_line = True

                # Is the cadet already fixed to something else?
                if i in p['J^Fixed']:
                    next_in_line = False
                    if i in possible_cadets[j]:
                        possible_cadets[j].remove(i)

                # Is this cadet reserved for something?
                if i in p['J^Reserved']:

                    # If they're already reserved for this AFSC or something better, they're not considered
                    if len(p['J^Reserved'][i]) <= p['c_pref_matrix'][i, j]:
                        next_in_line = False
                        if i in possible_cadets[j]:
                            possible_cadets[j].remove(i)

                # If this cadet is next in line (and we still have alternates to assign)
                if next_in_line and len(hard_alternates[j]) < (num_reserved[j] + num_additions):
                    alternates[j].append(i)

                    # Loop through the cadet's preferences:
                    for j_c in p['cadet_preferences'][i]:

                        # Determine what kind of alternate this cadet is
                        if j_c == j:  # Hard Rated Alternate
                            hard_alternates[j].append(i)
                            break
                        elif j_c in p['J^Rated']:
                            if i in possible_cadets[j_c]:  # Soft Rated Alternate
                                soft_r_alternates[j].append(i)
                                break
                            else:  # Can't be matched, go to next preference
                                continue
                        else:  # Soft Non-Rated Alternate
                            soft_n_alternates[j].append(i)
                            break

                # We've run out of hard alternates to assign (thus, we're done assigning alternates)
                elif len(hard_alternates[j]) >= (num_reserved[j] + num_additions):
                    if i in possible_cadets[j]:
                        possible_cadets[j].remove(i)

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

                # This cadet cannot receive this AFSC
                if i not in alternates[j] and i in possible_cadets[j]:
                    possible_cadets[j].remove(i)

        # Print Statement
        if printing:
            print("Iteration", iteration)
            print("Possible", {p['afscs'][j]: len(possible_cadets[j]) for j in p['J^Rated']})
            print("Matched", {p['afscs'][j]: len(p['I^Matched'][j]) for j in p['J^Rated']})
            print("Reserved", {p['afscs'][j]: len(p['I^Reserved'][j]) for j in p['J^Rated']})
            print("Alternates (Hard)", {p['afscs'][j]: len(hard_alternates[j]) for j in p['J^Rated']})
            print("Alternates (Soft)", {p['afscs'][j]: len(soft_n_alternates[j]) +
                                                       len(soft_r_alternates[j]) for j in p['J^Rated']})

        # Once we stop changing from the algorithm, we're done!
        current_matched = np.array([len(p['I^Matched'][j]) for j in p['J^Rated']])
        current_reserved = np.array([len(p['I^Reserved'][j]) for j in p['J^Rated']])
        current_alternates_h = np.array([len(hard_alternates[j]) for j in p['J^Rated']])
        if np.sum(current_matched - last_matches + current_reserved -
                  last_reserves + current_alternates_h - last_alternates_h) == 0:
            iterating = False
        else:
            last_matches, last_reserves, last_alternates_h = current_matched, current_reserved, current_alternates_h

        # Next iteration
        iteration += 1

    # Incorporate alternate lists (broken down by Hard/Soft)
    if 'J^Alternates (Hard)' not in p:
        p['J^Alternates (Hard)'] = {}
    if 'J^Alternates (Soft)' not in p:
        p['J^Alternates (Soft)'] = {}
    for i in p['Rated Cadets'][soc]:  # Loop through all rated cadets
        for j in p['Rated Choices'][soc][i]:  # Loop through rated preferences in order
            if i in hard_alternates[j]:
                p['J^Alternates (Hard)'][i] = j
                p['I^Alternates'][i] = j
            elif i in soft_r_alternates[j] or i in soft_n_alternates[j]:
                p['J^Alternates (Soft)'][i] = j
                p['I^Alternates'][i] = j
                break # Next cadet

    # Alternate List Optimization Formulation Sets Needed
    p['J^Preferred [' + soc + ']'], p['I^Preferred [' + soc + ']'], p['I^Alternate [' + soc + ']'] = {}, {}, {}
    for j in p['J^Rated']:

        # Empty sets for each AFSC
        p['I^Alternate [' + soc + ']'][j] = []
        p['I^Preferred [' + soc + ']'][j] = {}
        p['J^Preferred [' + soc + ']'][j] = {}

        # Loop through each cadet in order of the AFSC's preference
        for i in p['afsc_preferences'][j]:

            # Where this cadet ranked this AFSC
            cadet_rank_afsc = np.where(p['cadet_preferences'][i] == j)[0][0]

            # Set of more preferred AFSCs (including this AFSC too) for this cadet
            p['J^Preferred [' + soc + ']'][j][i] = p['cadet_preferences'][i][:cadet_rank_afsc + 1]

            # Where this AFSC ranked this cadet
            afsc_rank_cadet = np.where(p['afsc_preferences'][j] == i)[0][0]

            # Set of more preferred cadets (including this cadet too) for this AFSC
            p['I^Preferred [' + soc + ']'][j][i] = np.intersect1d(
                p['afsc_preferences'][j][:afsc_rank_cadet + 1], p[soc + '_cadets'])

            # Is this cadet an alternate from this SOC?
            if i in p[soc + '_cadets'] and (i in p['J^Alternates (Hard)'] or i in p['J^Alternates (Soft)']):

                # This cadet needs to be an alternate specifically for this AFSC
                alternate = False
                if i in p['J^Alternates (Hard)']:
                    if p['J^Alternates (Hard)'][i] == j:
                        alternate = True
                elif i in p['J^Alternates (Soft)']:
                    if p['J^Alternates (Soft)'][i] == j:
                        alternate = True
                if not alternate:
                    continue

                # Add the cadet to the alternate list for this AFSC
                p['I^Alternate [' + soc + ']'][j].append(i)

        # Convert to numpy array
        p['I^Alternate [' + soc + ']'][j] = np.array(p['I^Alternate [' + soc + ']'][j])

    # Return updated parameters (and alternate lists)
    return p


# Solution Metadata resolutions
def reconcile_solution_metadata(current_metadata, existing_metadata=None):
    """
    Reconcile metadata for a newly generated or previously existing solution.

    If the solution is new, initialize its run history. If it duplicates an
    existing solution, preserve the original metadata and update the latest
    run information.

    Parameters
    ----------
    current_metadata : dict
        Metadata generated from the current Pyomo optimization run.

    existing_metadata : dict, optional
        Metadata associated with an existing identical solution. If None,
        the current solution is treated as new.

    Returns
    -------
    dict
        Updated solution metadata containing:
        - new_solution
        - run_count
        - latest_run

    Notes
    -----
    Original optimization metadata is preserved when an identical solution
    is found. Only the run-tracking fields are updated.
    """

    import copy

    # ___________________________________________________INITIALIZATION_________________________________________________
    run_fields = ['start_time', 'finish_time', 'solve_time', 'pyomo_obj_value']
    latest_run = {key: current_metadata.get(key) for key in run_fields if key in current_metadata}

    # ___________________________________________________NEW SOLUTION_________________________________________________
    if existing_metadata is None:
        metadata = copy.deepcopy(current_metadata)
        metadata['new_solution'] = True
        metadata['run_count'] = 1
        metadata['latest_run'] = copy.deepcopy(latest_run)
        return metadata

    # ________________________________________________EXISTING SOLUTION_________________________________________________
    # Preserve the original metadata, including the original feature configuration
    metadata = copy.deepcopy(existing_metadata)

    # Increment the number of times this solution has been generated
    metadata['run_count'] = metadata.get('run_count', 1) + 1

    # Update the latest optimization run information
    metadata['latest_run'] = copy.deepcopy(latest_run)
    metadata['latest_run']['features'] = copy.deepcopy(current_metadata.get('features', {}))

    # Mark that the latest optimization reproduced an existing solution
    metadata['new_solution'] = False

    return metadata


# _____________________________________________MODEL FEATURE EVALUATION_________________________________________________
def _feature_evaluation_status(met, total):
    """
    Determine the outcome of a model feature evaluation.

    Parameters
    ----------
    met : int
        Number of individual requirements satisfied.
    total : int
        Total number of applicable requirements.

    Returns
    -------
    tuple
        (status, compliance), where compliance is a proportion from 0 to 1.

    Notes
    -----
    - Fully Met: All applicable requirements satisfied.
    - Partially Met: Some, but not all, requirements satisfied.
    - Not Met: No applicable requirements satisfied.
    - Not Applicable: No applicable requirements exist.

    Missing evaluation data is handled separately as 'Unable to Evaluate'.
    """
    if total == 0: return 'Not Applicable', np.nan
    if met == total: return 'Fully Met', 1.0
    if met == 0: return 'Not Met', 0.0
    return 'Partially Met', met / total


def _feature_metadata_configuration(solution, feature_id, current_feature):
    """
    Retrieve the feature configuration associated with a solution.

    The solution's saved feature configuration takes precedence over the current
    model feature registry. Any configuration fields absent from metadata fall
    back to their current registry values.

    Feature activation is handled separately because a missing activation record
    must be interpreted as inactive, not inherited from the current checklist.

    Parameters
    ----------
    solution : dict
        Solution containing optional metadata.
    feature_id : str
        Unique feature registry identifier.
    current_feature : dict
        Current feature definition from model_features.

    Returns
    -------
    tuple
        (configuration, solution_active)

        configuration : dict
            Effective feature configuration used for evaluation.
        solution_active : bool
            Whether the feature was active in the solution's latest GUO run.

    Notes
    -----
    Expected metadata structure:

        solution['metadata'] = {
            'method': 'GUO',
            'features': {
                'meet_afsc_pgl': {'active': True},
                'ussf_om_balance': {'active': True, 'threshold': 0.02}
            },
            'latest_run': {
                'features': {...}
            }
        }

    The latest_run feature record is preferred when present, including when
    the latest run reproduced an existing solution.

    Both 'active' and 'activated' are recognized as activation keys. Boolean
    feature records are also supported.

    A missing feature record is treated as inactive.
    """
    metadata = solution.get('metadata') or {}
    if not isinstance(metadata, dict): metadata = {}

    # Begin with the current registry, which provides default feature settings.
    configuration = copy.deepcopy(current_feature)

    # Original and latest-run feature configurations are applied in that order.
    original_features = metadata.get('features') or {}
    latest_run = metadata.get('latest_run') or {}
    latest_features = latest_run.get('features') or {}

    if not isinstance(original_features, dict): original_features = {}
    if not isinstance(latest_features, dict): latest_features = {}

    original = original_features.get(feature_id)
    latest = latest_features.get(feature_id)

    # Preserve historical configuration, then apply any newer saved settings.
    for record in (original, latest):
        if isinstance(record, dict):
            configuration.update({
                key: copy.deepcopy(value)
                for key, value in record.items()
                if key not in ('active', 'activated', 'enabled')
            })

    # Only GUO solutions can claim that model features were activated.
    if str(metadata.get('method', '')).upper() != 'GUO':
        return configuration, False

    # Latest-run activation takes precedence over the original run.
    record = latest if latest is not None else original

    if isinstance(record, dict):
        active = record.get('active', record.get('activated', record.get('enabled', False)))
    elif isinstance(record, (bool, np.bool_)):
        active = bool(record)
    else:
        active = False

    return configuration, bool(active)


def _feature_quota_evaluation(solution, p, j_indices, targets, label):
    """
    Evaluate AFSC production against the specified quota targets.

    An AFSC satisfies its target when its actual production is greater than
    or equal to the target. This evaluation deliberately ignores any relaxed
    quota_min or objective_min values used by the optimization model.

    Zero-target AFSCs are excluded because they do not represent positive
    production requirements.

    Returns
    -------
    tuple
        (met, total, details)
    """
    assignments = np.asarray(solution['j_array'], dtype=int)
    records, failures = [], []

    for j in j_indices:
        j = int(j)
        target = int(targets[j])
        if target <= 0: continue

        actual = int(np.count_nonzero(assignments == j))
        records.append((j, actual, target))

        if actual < target:
            failures.append(f'{p["afscs"][j]}: {actual}/{target} (short {target - actual})')

    total = len(records)
    met = total - len(failures)

    details = [
        f'{label} Requirements Met: {met}/{total}',
        f'Total Production: {sum(r[1] for r in records):,}/{sum(r[2] for r in records):,}',
        f'AFSCs Below Target: {len(failures)}'
    ]

    if failures:
        details.extend(['AFSC Shortfalls:', *(f'  - {item}' for item in failures)])

    return met, total, '\n'.join(details)


def _feature_rated_stability_evaluation(solution, p, solutions=None):
    """
    Evaluate rated stability requirements by commissioning source.

    Three components are evaluated independently for each SOC:

    1. Fixed Matches
       Cadets must receive their original rated match.

    2. Reserved Assignments
       Cadets must receive their reserved rated AFSC or a more-preferred AFSC.

    3. Alternate-List Requirements
       Rated alternate-list inequalities must hold for applicable cadets.

    Source priority
    ---------------
    Fixed/reserved obligations:
        1. Original 'Rated {SOC} HR (Matches/Reserves)' solutions, if available.
        2. Current J^Fixed/J^Reserved parameter dictionaries otherwise.

    Alternate-list obligations:
        Current J^Preferred, I^Preferred, and I^Alternate structures.

    Manual overrides
    ----------------
    An original rated obligation remains part of the historical stability
    evaluation even when a manual constraint overrides it.

    Violations involving manually overridden cadets are reported separately.
    They remain violations of original rated stability, but are identified as
    authorized overrides rather than unexplained assignment discrepancies.

    Returns
    -------
    tuple
        (met, total, details)

    Notes
    -----
    SOC-specific original rated solutions use global cadet indices and p['M']
    to represent cadets without preliminary matches or reservations.

    The reserved AFSC is the final element of the permitted preference array.

    The alternate-list inequality is evaluated against actual assignments
    rather than the x matrix, allowing independent post-hoc evaluation.
    """

    # _________________________________________________INITIALIZATION_________________________________________________

    assignments = np.asarray(solution['j_array'], dtype=int)
    solutions = solutions or {}

    fixed = p.get('J^Fixed', {})
    reserved = p.get('J^Reserved', {})

    # Identify manual overrides from the resolved constraint pipeline.
    manual_overrides = set(int(i) for i in p.get('I^Manual Override', []))

    # Also recognize raw constraint-method dictionaries when available.
    for i, methods in p.get('J^Constraint Methods', {}).items():
        if 'Manual' in methods and len(methods) > 1:
            manual_overrides.add(int(i))

    # Store SOC-specific results before assembling the final report.
    soc_records = []
    all_met, all_total = 0, 0

    def afsc_name(j):
        """Return an AFSC name, including the unmatched sentinel."""
        j = int(j)
        if j == p['M']: return 'Unmatched'
        return str(p['afscs'][j])

    def cadet_name(i):
        """Return the cadet identifier associated with a global cadet index."""
        return str(p['cadets'][int(i)])

    # _________________________________________________EVALUATE EACH SOC______________________________________________

    for soc in p['SOCs']:
        soc = str(soc).lower()
        soc_indices = set(int(i) for i in p[f'{soc}_cadets'])

        fixed_name = f'Rated {soc.upper()} HR (Matches)'
        reserved_name = f'Rated {soc.upper()} HR (Reserves)'

        # Original rated results are authoritative when both are available.
        original_available = fixed_name in solutions and reserved_name in solutions

        if original_available:
            matched_array = np.asarray(solutions[fixed_name]['j_array'], dtype=int)
            reserved_array = np.asarray(solutions[reserved_name]['j_array'], dtype=int)

            # Preserve original preliminary matches.
            fixed_requirements = {
                int(i): int(matched_array[i])
                for i in soc_indices
                if matched_array[i] != p['M']
            }

            # Reconstruct the permitted reservation preference set from the
            # original reserved AFSC and the cadet's ranked preferences.
            reserved_requirements = {}

            for i in soc_indices:
                j = int(reserved_array[i])
                if j == p['M']: continue

                preferences = np.asarray(p['cadet_preferences'][i], dtype=int)
                positions = np.where(preferences == j)[0]

                if len(positions) == 0:
                    raise ValueError(
                        f'{soc.upper()} cadet {cadet_name(i)}: '
                        f'reserved AFSC {afsc_name(j)} missing from preferences.'
                    )

                reserved_requirements[i] = preferences[:int(positions[0]) + 1]

            source = 'Original SOC Rated Matching Solutions'

        else:
            # Fall back to currently incorporated rated obligations.
            fixed_requirements = {
                int(i): int(j)
                for i, j in fixed.items()
                if int(i) in soc_indices
            }

            reserved_requirements = {
                int(i): np.asarray(allowed, dtype=int)
                for i, allowed in reserved.items()
                if int(i) in soc_indices
            }

            source = 'Current Rated Parameter Sets'

        # __________________________________________FIXED MATCHES____________________________________________________

        fixed_failures = []
        fixed_overrides = []

        for i, required_j in fixed_requirements.items():
            actual_j = int(assignments[i])

            if actual_j != required_j:
                description = (
                    f'{cadet_name(i)}: {afsc_name(actual_j)} '
                    f'(required {afsc_name(required_j)})'
                )

                fixed_failures.append(description)

                if i in manual_overrides:
                    fixed_overrides.append(description)

        fixed_total = len(fixed_requirements)
        fixed_met = fixed_total - len(fixed_failures)

        # __________________________________________RESERVATIONS_____________________________________________________

        reserved_failures = []
        reserved_overrides = []

        for i, allowed in reserved_requirements.items():
            actual_j = int(assignments[i])
            allowed = np.asarray(allowed, dtype=int)

            if actual_j not in allowed:
                description = (
                    f'{cadet_name(i)}: {afsc_name(actual_j)} '
                    f'(permitted: {", ".join(afsc_name(j) for j in allowed)})'
                )

                reserved_failures.append(description)

                if i in manual_overrides:
                    reserved_overrides.append(description)

        reserved_total = len(reserved_requirements)
        reserved_met = reserved_total - len(reserved_failures)

        # __________________________________________ALTERNATE LISTS__________________________________________________

        alternate_failures = []
        alternate_overrides = []
        alternate_total, alternate_met = 0, 0

        preferred_j = p.get(f'J^Preferred [{soc}]')
        preferred_i = p.get(f'I^Preferred [{soc}]')
        alternate_i = p.get(f'I^Alternate [{soc}]')

        alternate_available = all(
            isinstance(item, dict)
            for item in (preferred_j, preferred_i, alternate_i)
        )

        if alternate_available:

            for j in p['J^Rated']:
                j = int(j)

                for i in alternate_i.get(j, []):
                    i = int(i)
                    alternate_total += 1

                    # More-preferred AFSCs, including the rated AFSC itself.
                    preferred_afscs = np.asarray(preferred_j[j][i], dtype=int)

                    # Cadets ranked at least as highly by this rated AFSC.
                    preferred_cadets = np.asarray(preferred_i[j][i], dtype=int)

                    # Equivalent to the existing SOC alternate-list inequality.
                    cadet_satisfied = int(assignments[i] in preferred_afscs)
                    higher_ranked_assigned = int(
                        np.count_nonzero(assignments[preferred_cadets] == j)
                    )

                    required_slots = int(p[f'{soc}_quota'][j])

                    satisfied = (
                        required_slots * (1 - cadet_satisfied)
                        <= higher_ranked_assigned
                    )

                    if satisfied:
                        alternate_met += 1

                    else:
                        description = (
                            f'{cadet_name(i)} / {afsc_name(j)}: '
                            f'{higher_ranked_assigned}/{required_slots} '
                            f'higher-ranked cadets assigned'
                        )

                        alternate_failures.append(description)

                        if i in manual_overrides:
                            alternate_overrides.append(description)

        # __________________________________________SOC SUMMARY______________________________________________________

        soc_met = fixed_met + reserved_met + alternate_met
        soc_total = fixed_total + reserved_total + alternate_total

        all_met += soc_met
        all_total += soc_total

        soc_records.append({
            'soc': soc.upper(),
            'source': source,
            'fixed_met': fixed_met,
            'fixed_total': fixed_total,
            'reserved_met': reserved_met,
            'reserved_total': reserved_total,
            'alternate_met': alternate_met,
            'alternate_total': alternate_total,
            'alternate_available': alternate_available,
            'fixed_failures': fixed_failures,
            'reserved_failures': reserved_failures,
            'alternate_failures': alternate_failures,
            'fixed_overrides': fixed_overrides,
            'reserved_overrides': reserved_overrides,
            'alternate_overrides': alternate_overrides
        })

    # _________________________________________________CONSTRUCT REPORT________________________________________________

    details = [
        f'Overall Rated Stability: {all_met}/{all_total}',
        f'Manually Overridden Cadets: {len(manual_overrides)}'
    ]

    for record in soc_records:
        soc = record['soc']

        details.extend([
            '',
            f'{soc}:',
            f'  Source: {record["source"]}',
            f'  Fixed Matches: {record["fixed_met"]}/{record["fixed_total"]}',
            f'  Reservations: {record["reserved_met"]}/{record["reserved_total"]}',
            f'  Alternate Lists: {record["alternate_met"]}/{record["alternate_total"]}'
        ])

        if not record['alternate_available']:
            details.append('  Alternate List Data: Unavailable')

        # Show individual violations, grouped by requirement type.
        for label, key in [
            ('Fixed Match Violations', 'fixed_failures'),
            ('Reservation Violations', 'reserved_failures'),
            ('Alternate List Violations', 'alternate_failures')
        ]:
            if record[key]:
                details.append(f'  {label}:')
                details.extend(f'    - {item}' for item in record[key])

        # Identify which violations were associated with manual overrides.
        for label, key in [
            ('Manually Overridden Fixed Matches', 'fixed_overrides'),
            ('Manually Overridden Reservations', 'reserved_overrides'),
            ('Manually Overridden Alternates', 'alternate_overrides')
        ]:
            if record[key]:
                details.append(f'  {label}: {len(record[key])}')

    return all_met, all_total, '\n'.join(details)


def calculate_model_feature_metrics(solution, parameters, model_features,
                                    feature_checklist=None, solutions=None):
    """
    Evaluate a solution against every registered One Market model feature.

    This function evaluates feature outcomes independently of feature activation.
    Every feature is evaluated regardless of whether it was enabled in the
    current model or when the solution was generated.

    Results are stored in solution['feature_metrics'] as a pandas DataFrame.

    Parameters
    ----------
    solution : dict
        Evaluated solution containing j_array and optional metadata.

        The solution's algorithm is identified by:

            solution['metadata']['method']

        Only method='GUO' can report activated solution features.

    parameters : dict
        Problem parameters used to evaluate assignments.

    model_features : dict
        Current model feature registry, typically self.model_features.

        The registry supplies feature names and default configurations.

    feature_checklist : dict, optional
        Current model instance checklist, typically self.feature_checklist.

        This controls the 'Current Checklist' column only.

    solutions : dict, optional
        Dictionary of available solutions, typically self.solutions.

        Used to retrieve original SOC rated matching solutions when available.

    Returns
    -------
    dict
        Updated solution containing a DataFrame under 'feature_metrics'.

    DataFrame Columns
    -----------------
    Feature ID
        Unique feature registry identifier.

    Feature
        Human-readable feature name.

    Current Checklist
        Whether the feature is currently enabled in the model instance.

    Solution Checklist
        Whether the feature was activated in the solution's GUO metadata.

    Solution Status
        Fully Met, Partially Met, Not Met, Not Applicable, or Unable to Evaluate.

    Compliance
        Fraction of applicable requirements satisfied, between 0 and 1.

    Met
        Number of requirements satisfied.

    Total
        Number of applicable requirements.

    Details
        Multiline explanation of actual solution outcomes.

    Notes
    -----
    Feature configuration is selected using the following precedence:

        1. Solution metadata, latest_run.features
        2. Solution metadata, features
        3. Current model_features registry

    Missing configuration fields fall back to the current registry.

    Activation states are handled independently:
        - Current Checklist comes from feature_checklist.
        - Solution Checklist comes from GUO solution metadata.
        - Solution Status comes exclusively from actual assignments.

    No model parameters or feature activation states are modified.
    """

    # _________________________________________________INITIALIZATION_________________________________________________

    p = parameters
    assignments = np.asarray(solution['j_array'], dtype=int)
    checklist = feature_checklist or {}
    rows = []

    # Validate the fundamental assignment vector before evaluating features.
    if len(assignments) != p['N']:
        raise ValueError(
            f'Solution has {len(assignments)} assignments, '
            f'but the problem contains {p["N"]} cadets.'
        )

    if np.any((assignments < 0) | (assignments > p['M'])):
        raise ValueError('Solution contains invalid AFSC assignment indices.')

    def add_result(feature_id, name, current_active, solution_active,
                   met=None, total=None, details='', error=None):
        """
        Append a standardized feature evaluation record.

        Errors are represented explicitly rather than being interpreted as
        feature failures or successful compliance.
        """
        if error is not None:
            status, compliance = 'Unable to Evaluate', np.nan
            met, total = np.nan, np.nan
            details = f'Evaluation unavailable: {error}'
        else:
            status, compliance = _feature_evaluation_status(met, total)

        rows.append({
            'Feature ID': feature_id,
            'Feature': name,
            'Current Checklist': 'Active' if current_active else 'Inactive',
            'Solution Checklist': 'Active' if solution_active else 'Inactive',
            'Solution Status': status,
            'Compliance': compliance,
            'Met': met,
            'Total': total,
            'Details': details
        })

    # _________________________________________________EVALUATE FEATURES______________________________________________

    for feature_id, current_feature in model_features.items():

        # The current checklist and historical activation are independent.
        current_active = bool(checklist.get(feature_id, False))

        # Historical metadata supplies the feature's evaluation configuration
        # and whether it was activated in the latest GUO run.
        feature, solution_active = _feature_metadata_configuration(
            solution, feature_id, current_feature
        )

        name = str(feature.get('name', current_feature.get('name', feature_id)))

        try:

            # __________________________________________MEET AFSC PGL_________________________________________________

            if feature_id == 'meet_afsc_pgl':

                met, total, details = _feature_quota_evaluation(
                    solution, p, p['J'], p['pgl'], 'AFSC PGL'
                )

            # __________________________________________MEET RATED PGL________________________________________________

            elif feature_id == 'meet_rated_pgl':

                met, total, details = _feature_quota_evaluation(
                    solution, p, p['J^Rated'], p['pgl'], 'Rated PGL'
                )

            # __________________________________________MEET RATED SOC PGL____________________________________________

            elif feature_id == 'meet_rated_soc_pgl':

                records, failures = [], []

                for j in p['J^Rated']:
                    j = int(j)
                    afsc = str(p['afscs'][j])

                    # Every rated AFSC has a combined production requirement.
                    required = [('Combined', int(p['pgl'][j]))]

                    # Non-SOC-specific rated AFSCs also have SOC quota targets.
                    if not afsc.endswith(('_U', '_R', '_O')):
                        required.extend(
                            (soc.upper(), int(p[f'{soc}_quota'][j]))
                            for soc in p['SOCs']
                        )

                    for category, target in required:
                        if target <= 0: continue

                        if category == 'Combined':
                            actual = int(np.count_nonzero(assignments == j))

                        else:
                            soc_indices = np.asarray(
                                p[f'{category.lower()}_cadets'], dtype=int
                            )

                            actual = int(np.count_nonzero(
                                assignments[soc_indices] == j
                            ))

                        records.append((afsc, category, actual, target))

                        if actual < target:
                            failures.append(
                                f'{afsc} / {category}: {actual}/{target} '
                                f'(short {target - actual})'
                            )

                total = len(records)
                met = total - len(failures)

                details = [
                    f'Rated Quota Requirements Met: {met}/{total}',
                    f'Combined/SOC Quota Shortfalls: {len(failures)}'
                ]

                # Report successful and unsuccessful SOC-specific requirements.
                for soc in ['Combined'] + [s.upper() for s in p['SOCs']]:
                    subset = [r for r in records if r[1] == soc]

                    if subset:
                        subset_met = sum(r[2] >= r[3] for r in subset)
                        details.append(
                            f'{soc} Requirements: {subset_met}/{len(subset)}'
                        )

                if failures:
                    details.extend([
                        'Quota Shortfalls:',
                        *(f'  - {item}' for item in failures)
                    ])

                details = '\n'.join(details)

            # __________________________________________RATED STABILITY SOC____________________________________________

            elif feature_id == 'rated_stability_soc':

                met, total, details = _feature_rated_stability_evaluation(
                    solution, p, solutions=solutions
                )

            # __________________________________________USSF OM BALANCE________________________________________________

            elif feature_id == 'ussf_om_balance':

                if 'USSF' not in p['afscs_acc_grp']:
                    met, total = 0, 0
                    details = 'USSF is not included in this problem.'

                else:
                    threshold = float(feature['threshold'])
                    lower, upper = 0.5 - threshold, 0.5 + threshold

                    ussf_indices = np.asarray(p['J^USSF'], dtype=int)
                    ussf_cadets = np.where(
                        np.isin(assignments, ussf_indices)
                    )[0]

                    if len(ussf_cadets) == 0:
                        met, total = 0, 0
                        details = 'No cadets were assigned to USSF career fields.'

                    else:
                        achieved = float(np.mean(
                            np.asarray(p['merit'])[ussf_cadets]
                        ))

                        met = int(lower - 1e-9 <= achieved <= upper + 1e-9)
                        total = 1

                        details = '\n'.join([
                            f'USSF Cadets Assigned: {len(ussf_cadets):,}',
                            f'Average USSF Merit: {achieved:.4f}',
                            f'Target Merit: 0.5000',
                            f'Permitted Range: {lower:.4f}–{upper:.4f}',
                            f'Difference from Target: {achieved - 0.5:+.4f}'
                        ])

            # __________________________________________USAFA SUPPORT CAP______________________________________________

            elif feature_id == 'usafa_support_cap':

                constrained_afscs = list(feature['constrained_afscs'])
                percent_cap = float(feature['percent_cap'])

                missing = sorted(
                    set(constrained_afscs) - set(p['afscs'])
                )

                if missing:
                    raise ValueError(
                        f'Configured AFSCs not found: {", ".join(missing)}'
                    )

                constrained_indices = np.where(
                    np.isin(p['afscs'], constrained_afscs)
                )[0]

                usafa_cadets = np.asarray(
                    p['usafa_cadets'], dtype=int
                )

                num_usafa = len(usafa_cadets)

                if num_usafa == 0:
                    met, total = 0, 0
                    details = 'No USAFA cadets are included in this problem.'

                else:
                    actual = int(np.count_nonzero(
                        np.isin(
                            assignments[usafa_cadets],
                            constrained_indices
                        )
                    ))

                    maximum = int(np.ceil(num_usafa * percent_cap))  # Round up

                    met = int(actual <= maximum)
                    total = 1

                    details = '\n'.join([
                        f'Constrained AFSCs: {", ".join(constrained_afscs)}',
                        f'USAFA Cadets: {num_usafa:,}',
                        f'Assigned to Constrained AFSCs: {actual:,}',
                        f'Maximum Permitted: {maximum:,}',
                        f'Configured Cap: {percent_cap:.1%}',
                        f'Actual Percentage: {actual / num_usafa:.2%}',
                        f'Excess Assignments: {max(0, actual - maximum):,}'
                    ])

            # __________________________________________HIGH OM PREFERENCE_____________________________________________

            elif feature_id == 'high_om_receive_preference':

                percentile_cutoff = float(feature['percentile_cutoff'])
                preference_number = int(feature['preference_number'])

                # Use the actual merit threshold, not an empirical cohort
                # percentile or the feature's potentially inactive parameter sets.
                qualifying = np.where(
                    np.asarray(p['merit']) >= 1 - percentile_cutoff
                )[0]

                failures = []

                for i in qualifying:
                    preferred = np.asarray(
                        p['cadet_preferences'][int(i)][:preference_number],
                        dtype=int
                    )

                    if assignments[i] not in preferred:
                        assigned = (
                            'Unmatched' if assignments[i] == p['M']
                            else str(p['afscs'][assignments[i]])
                        )

                        failures.append(
                            f'{p["cadets"][i]}: {assigned}'
                        )

                total = len(qualifying)
                met = total - len(failures)

                details = [
                    f'Merit Threshold: {1 - percentile_cutoff:.1%}',
                    f'Guaranteed Preference Range: Top {preference_number}',
                    f'Qualifying Cadets: {total:,}',
                    f'Received Top {preference_number}: {met:,}/{total:,}',
                    f'Outside Top {preference_number}: {len(failures):,}'
                ]

                if failures:
                    details.extend([
                        'Cadets Outside Preference Range:',
                        *(f'  - {item}' for item in failures)
                    ])

                details = '\n'.join(details)

            # __________________________________________UNRECOGNIZED FEATURE___________________________________________

            else:
                add_result(
                    feature_id, name, current_active, solution_active,
                    error='No evaluation method is defined for this feature.'
                )
                continue

            # Store the completed feature evaluation.
            add_result(
                feature_id, name, current_active, solution_active,
                met=met, total=total, details=details
            )

        except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError) as exc:
            add_result(
                feature_id, name, current_active, solution_active,
                error=f'{type(exc).__name__}: {exc}'
            )

    # _________________________________________________CONSTRUCT DATAFRAME____________________________________________

    columns = [
        'Feature ID',
        'Feature',
        'Current Checklist',
        'Solution Checklist',
        'Solution Status',
        'Compliance',
        'Met',
        'Total',
        'Details'
    ]

    solution['feature_metrics'] = pd.DataFrame(rows, columns=columns)

    return solution



