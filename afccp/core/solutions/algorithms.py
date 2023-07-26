import time
import numpy as np
import random
import copy

# afccp modules
import afccp.core.globals
import afccp.core.solutions.handling

# Matching algorithms
def hand_jam_missiles_fix(instance):
    """
    This is the 13N bandaid fix for FY24 NOT RECOMMENDED :P
    """

    # Shorthand
    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p

    # Run this to get list of rated folks we can't take from
    print("Running classic HR purely for list of Rated cadets we can't take...")
    solution = classic_hr(instance, printing=False)

    # These cadets received a Rated AFSC (so we can't put them into 13N)
    rated_cadets = np.array([i for i in p['I'] if solution['j_array'][i] in p['J^Rated']])

    # Initialize solution dictionary
    solution = {'method': '13N_HR_LQ_Fix', 'j_array': np.array([p['M'] for _ in p['I']])}

    # Pre-process 13N
    print('Pre-processing 13N cadets...')
    j_13N = np.where(p['afscs'] == '13N')[0][0]
    def preprocess_missiles():
        """
        This algorithm purely preprocesses the missiles career field based on preferences for FY24. All 140
        slots get filled from 13N people who weren't eligible for anything else that we would take from.
        """

        # Get a list of potential cadets to pre-process into 13N
        potential_cadets = []
        for i in p['I']:

            # 13N has to be in top 8 choices and this cannot be a rated cadet
            if p['c_pref_matrix'][i, j_13N] <= 8 and i not in rated_cadets:

                # Get a list of AFSCs that "disqualify" the potential 13N cadet (62E afscs)
                j_62E = [j for j in p['J'] if '62E' in p['afscs'][j]]

                # Check to see if this cadet is eligible for any of the above defined AFSCs
                missiles_potential = True
                for j in j_62E:
                    if i in p['I^E'][j]:
                        missiles_potential = False
                        break

            # 13N wasn't in the top 6 choices
            else:
                missiles_potential = False

            # Cadet is considered for pre-processing
            if missiles_potential:
                potential_cadets.append(i)

        # 13N choices
        potential_cadets = np.array(potential_cadets)  # So we can select a subset of a numpy array using indices below
        missiles_cadet_pref = p['c_pref_matrix'][potential_cadets, j_13N]

        # Print statements!
        print(len(potential_cadets), 'potential 13N cadets.')
        for choice in np.arange(1, 9):
            count = len(np.where(p['c_pref_matrix'][potential_cadets, j_13N] == choice)[0])
            print(count, 'potential cadets had 13N as choice', choice)

        # Sort the list of cadets in order of their preference for 13N
        sorted_indices = np.argsort(missiles_cadet_pref)
        potential_cadets = potential_cadets[sorted_indices]  # Sorts cadets in order by their preference for 13N

        # Fill 13N cadets until we've met the PGL
        count = 0
        for i in potential_cadets:
            solution['j_array'][i] = j_13N
            count += 1

            # As soon as we hit the PGL, stop
            if count == p['pgl'][j_13N]:
                break

        # Print statements!
        cadets_13N = np.where(solution['j_array'] == j_13N)[0]
        print('Average choice of 13N preprocessed cadets:', np.mean(p['c_pref_matrix'][cadets_13N, j_13N]))
        for choice in np.arange(1, 9):
            count = len(np.where(p['c_pref_matrix'][cadets_13N, j_13N] == choice)[0])
            print(count, 'preprocessed cadets had 13N as choice', choice)
        print('Done.', len(cadets_13N), 'cadets preprocessed to 13N.', p['N'] - len(cadets_13N), 'cadets left.')

        # Return 13N pre-processed solution
        return solution, cadets_13N
    solution, cadets_13N = preprocess_missiles()

    # Algorithm initialization
    total_slots = p[mdl_p['capacity_parameter']]

    # Array to keep track of what AFSC choice in their list the cadets are proposing to (python index at 0)
    cadet_proposal_choice = np.zeros(p['N']).astype(int)  # Everyone proposes to their first choice initially

    # Begin the simple Hospital/Residents Algorithm (omitting 13N)
    total_rejections = np.zeros(p['M'])  # Number of rejections for each AFSC
    total_matched = np.zeros(p['M'])  # Number of accepted cadets for each AFSC
    total_matched[j_13N] = p['pgl'][j_13N]  # Fixing total matched for 13N
    exhausted_cadets = []  # Will contain the cadets that have exhausted (been rejected by) all of their preferences
    iteration = 0  # First iteration of the algorithm
    while np.sum(total_matched) + len(exhausted_cadets) < p['N'] and iteration < 42:  # Stopping conditions

        # Cadets propose to their top choice that hasn't been rejected
        exhausted_cadets = np.where(cadet_proposal_choice >= p['num_cadet_choices'])[0]
        proposals = np.array([p['cadet_preferences'][i][cadet_proposal_choice[i]] if i not in exhausted_cadets
                              else p['M'] for i in p['I']])
        for i in cadets_13N:
            proposals[i] = j_13N  # Fix the 13N people!

        # Print statement!
        print("\nIteration", iteration + 1)
        counts = {p['afscs'][j]: len(np.where(proposals == j)[0]) for j in p['J']}

        print('Total Matched', np.sum(total_matched), 'Total Exhausted', len(exhausted_cadets))
        print('Matched:', total_matched)

        # Initialize matches information for this iteration
        total_matched = np.zeros(p['M'])
        total_matched[j_13N] = p['pgl'][j_13N]  # Fixing total matched for 13N

        # AFSCs accept their best cadets and reject the others
        for j in p['J']:

            # Skip 13N
            if j == j_13N:
                for i in proposals:

                    # If the cadet is proposing to 13N, but wasn't preprocessed, we reject them
                    if proposals[i] == j and i not in cadets_13N:

                        # Essentially "delete" the preference from the cadet's list
                        cadet_proposal_choice[i] += 1
                        proposals[i] = p['M']  # index of the unmatched AFSC (*)

                        # Collect additional information
                        iteration_rejections += 1
                        total_rejections[j] += 1

            # Loop through their preferred cadets from top to bottom
            iteration_rejections = 0
            for i in p['afsc_preferences'][j]:

                # If this is a 13N cadet, ignore them (tough...better luck next time, cadet!)
                if i in cadets_13N:
                    continue  # This AFSC is forced to reject them

                # If the cadet is proposing to this AFSC, we have two options
                if proposals[i] == j:

                    # We haven't hit capacity, so we accept this cadet
                    if total_matched[j] < total_slots[j]:
                        total_matched[j] += 1

                    # We're at capacity, so we reject this cadet
                    else:

                        # Essentially "delete" the preference from the cadet's list
                        cadet_proposal_choice[i] += 1
                        proposals[i] = p['M']  # index of the unmatched AFSC (*)

                        # Collect additional information
                        iteration_rejections += 1
                        total_rejections[j] += 1

        # Print statement!
        print('Proposals:', counts)
        print('Matched', {p['afscs'][j]: int(total_matched[j]) for j in p['J']})
        print('Rejected', {p['afscs'][j]: int(total_rejections[j]) for j in p['J']})

        iteration += 1 # Next iteration!

    # Return solution
    solution['j_array'] = proposals
    solution['afsc_array'] = np.array([p['afscs'][j] for j in solution['j_array']])
    return solution


def classic_hr(instance, capacities=None, printing=True):
    """
    Matches cadets and AFSCs across all rated, space, and NRL positions using the Hospitals/Residents algorithm.

    Parameters:
        instance (CadetCareerProblem): The instance of the CadetCareerProblem class.
        capacities (numpy.ndarray or None): The capacities of AFSCs. If None, the capacities are taken from the
            instance parameters. Default is None.
        printing (bool): Whether to print status updates or not. Default is True.

    Returns:
        dict: The solution dictionary containing the assigned AFSCs for each cadet.

    This function implements the Hospitals/Residents algorithm to match cadets and AFSCs across all rated, space,
    and NRL positions. It takes an instance of the CadetCareerProblem class as input and an optional parameter
    `capacities` to specify the capacities of AFSCs. If `capacities` is None, the capacities are taken from the
    instance parameters. By default, the function prints status updates during the matching process.

    The algorithm initializes the necessary variables and dictionaries. It then proceeds with the Hospitals/Residents
    algorithm by having cadets propose to their top choices and AFSCs accept or reject cadets based on their preferences
    and capacities. The matching process continues until all cadets are matched or have exhausted their preferences.
    The function updates the matches and rejections for each AFSC and tracks the progress through iterations.

    The function returns a solution dictionary containing the assigned AFSCs for each cadet.

    Example usage:
        solution = classic_hr(instance, capacities=capacities, printing=True)
    """

    if printing:
        print("Modeling this as an H/R problem and solving with DAA...")

    # Shorthand
    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p

    # Algorithm initialization
    if capacities is None:
        total_slots = p[mdl_p['capacity_parameter']]
    else:  # In case this is used in a genetic algorithm
        total_slots = capacities

    # Array to keep track of what AFSC choice in their list the cadets are proposing to (python index at 0)
    cadet_proposal_choice = np.zeros(p['N']).astype(int)  # Everyone proposes to their first choice initially

    # Initialize solution dictionary
    solution = {'method': 'HR'}

    # Dictionary of parameters used for the "CadetBoardFigure" object (animation)
    if mdl_p['collect_solution_iterations']:
        solution['iterations'] = {'type': 'HR'}
        for key in ['proposals', 'matches', 'names']:
            solution['iterations'][key] = {}

    # Begin the simple Hospital/Residents Algorithm
    total_rejections = np.zeros(p['M'])  # Number of rejections for each AFSC
    total_matched = np.zeros(p['M'])  # Number of accepted cadets for each AFSC
    exhausted_cadets = []  # Will contain the cadets that have exhausted (been rejected by) all of their preferences
    iteration = 0  # First iteration of the algorithm
    while np.sum(total_matched) + len(exhausted_cadets) < p['N']:  # Stopping conditions

        # Cadets propose to their top choice that hasn't been rejected
        exhausted_cadets = np.where(cadet_proposal_choice >= p['num_cadet_choices'])[0]
        proposals = np.array([p['cadet_preferences'][i][cadet_proposal_choice[i]] if i not in exhausted_cadets
                              else p['M'] for i in p['I']])

        # Solution Iteration components (Proposals) and print statement
        if mdl_p['collect_solution_iterations']:
            solution['iterations']['proposals'][iteration] = copy.deepcopy(proposals)
        if mdl_p['ma_printing']:
            print("\nIteration", iteration + 1)
            counts = {p['afscs'][j]: len(np.where(proposals == j)[0]) for j in p['J']}

        # Initialize matches information for this iteration
        total_matched = np.zeros(p['M'])

        # AFSCs accept their best cadets and reject the others
        for j in p['J']:

            # Loop through their preferred cadets from top to bottom
            iteration_rejections = 0
            for i in p['afsc_preferences'][j]:

                # If the cadet is proposing to this AFSC, we have two options
                if proposals[i] == j:

                    # We haven't hit capacity, so we accept this cadet
                    if total_matched[j] < total_slots[j]:
                        total_matched[j] += 1

                    # We're at capacity, so we reject this cadet
                    else:

                        # Essentially "delete" the preference from the cadet's list
                        cadet_proposal_choice[i] += 1
                        proposals[i] = p['M']  # index of the unmatched AFSC (*)

                        # Collect additional information
                        if mdl_p['ma_printing']:
                            iteration_rejections += 1
                            total_rejections[j] += 1

        # Solution Iteration components
        if mdl_p['collect_solution_iterations']:
            solution['iterations']['matches'][iteration] = copy.deepcopy(proposals)
            solution['iterations']['names'][iteration] = 'Round ' + str(iteration + 1)

        # Specific matching algorithm print statement
        if mdl_p['ma_printing']:
            print('Proposals:', counts)
            print('Matched', {p['afscs'][j]: total_matched[j] for j in p['J']})
            print('Rejected', {p['afscs'][j]: total_rejections[j] for j in p['J']})

        iteration += 1 # Next iteration!

    # Last solution iteration
    solution['iterations']['last_s'] = iteration - 1

    # Return solution
    solution['j_array'] = proposals
    solution['afsc_array'] = np.array([p['afscs'][j] for j in solution['j_array']])
    return solution


def hr_lower_quota_fix(instance, solution, capacities=None, printing=True):
    """
    Insert Chatgpt here
    """

    impossible_quotas = []
    cadets_moved = []

    if printing:
        print("Fixing the H/R algorithm lower quota issue...")

    # Shorthand
    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p

    # Algorithm initialization
    if capacities is None:
        total_slots = p[mdl_p['capacity_parameter']]
    else:  # In case this is used in a genetic algorithm
        total_slots = capacities

    # Adjust solution method
    if '13N' not in solution['method']:
        solution['method'] = 'HR_LQ_Fix'

    # Let's match 62EXE manually
    matched_rated_cadets = []
    for i in p['cadet_preferences'].keys():
        if solution['afsc_array'][i] in [rated_afsc for rated_afsc in p['afscs_acc_grp']['Rated']]:
            matched_rated_cadets.append(i)

    cadets_desiring_62EXE = []
    for i in (p['cadet_preferences'].keys()):
        if i not in matched_rated_cadets and solution['afsc_array'][i] != '32EXE' and solution['afsc_array'][
            i] != '62E1E1S' and solution['afsc_array'][i] != '62EXE':
            any_62EXE_pref_cadet = p['cadet_preferences'][i]  # Extract all of the preferences
            if any(pref == 40 for pref in any_62EXE_pref_cadet):
                cadets_desiring_62EXE.append(i)

    counter_62EXE = 0 + len(np.where(solution['j_array'] == 40)[0])
    cadets_to_move = []  # Temporary list to store cadets to move
    for i in cadets_desiring_62EXE:
        if i not in cadets_moved:
            # Get dictionary of cadet preferences for the AFSC that cadet is matched to
            a = {i: p['c_pref_matrix'][i, solution['j_array'][i]] for i in cadets_desiring_62EXE}

            b = {i: p['c_pref_matrix'][i, 40] for i in cadets_desiring_62EXE}

            # Get dictionary of AFSC preferences on the cadets that were unassigned to this AFSC (but were eligible)
            c = {i: p['a_pref_matrix'][i, 40] for i in cadets_desiring_62EXE}

            d = {i: b[i] - a[i] for i in cadets_desiring_62EXE}
            vals = [d[i] for i in cadets_desiring_62EXE]

            minimum_cadets = sorted(cadets_desiring_62EXE, key=lambda x: d[x])
            for i in minimum_cadets:
                if counter_62EXE >= p['quota_min'][40]:
                    break  # Stop adding cadets once the quota_min value is reached
                cadets_to_move.append(i)
                counter_62EXE += 1
                print('counter_total', counter_62EXE)
                cadet = i
                current_afsc = p['afscs'][solution['j_array'][i]]
                solution['j_array'][i] = 40
                new_afsc = solution['afsc_array'][i] = '62EXE'
                print('Moved cadet', cadet, 'from', current_afsc, 'to AFSC', new_afsc)
    counts_62EXE = {40: len(np.where(solution['j_array'] == 40)[0])}

    print(counts_62EXE)
    # Add the moved cadets to cadets_moved list after the loop. This will also prevent movement of cadets below.
    cadets_moved.extend(cadets_to_move)

    print(cadets_moved)

    impossible_quotas = []

    # Need to keep track of unfilled AFSCs (from the start)
    counts = {j: len(np.where(solution['j_array'] == j)[0]) for j in p['J']}
    percent_of_lower_quota = np.array([counts[j] / total_slots[j] * 100 for j in p['J']])
    unfilled_j = np.where(percent_of_lower_quota < 98)[0]
    print(p['afscs'][unfilled_j])

    # Determine which AFSCs we can pull from that aren't rated and are the largest 12 AFSCs
    j_sorted = np.argsort(p['pgl'])[::-1]
    j_selects = np.array([j for j in j_sorted if p['afscs'][j] not in p['afscs_acc_grp'][
        'Rated'] and j != 20 and j != 21 and j != 22 and j != 23 and j != 24 and j != 25 and j != 33 and j != 39 and j != 40 and j != 41])

    # Iterate until all AFSCs are at or over quota
    iteration = 0
    cadets_moved = []

    while True:
        counts = {j: len(np.where(solution['j_array'] == j)[0]) for j in p['J']}
        percent_of_lower_quota = np.array([counts[j] / total_slots[j] * 100 for j in p['J']])
        unfilled_j = np.where((percent_of_lower_quota < 98.5) & (~np.isin(p['J'], np.array(impossible_quotas))))[0]
        unfilled_percent_of_lower_quota = {j: percent_of_lower_quota[j] for j in unfilled_j if
                                           j not in impossible_quotas}
        # Check if we have filled all AFSCs to their quotas
        if len(unfilled_j) == 0 and len(np.where(solution['j_array'] == p['M'])[0]) == 0:
            break
        elif len(unfilled_j) != 0:
            pass
            # Breaking up the code to show that we only execute this loop in order to force the rest of the unmatched cadets to match.
            # -------------------------------------------------------------------------------------------------------------#
        else:
            # We have met the 98 threshold. Move up to 99 and pull unmatched cadets.
            unfilled_j = np.where((percent_of_lower_quota < 100) & (~np.isin(p['J'], np.array(impossible_quotas))))[0]
            unfilled_percent_of_lower_quota = {j: percent_of_lower_quota[j] for j in unfilled_j if
                                               j not in impossible_quotas}
            j_u = min(unfilled_percent_of_lower_quota, key=unfilled_percent_of_lower_quota.get)
            assigned_cadets = np.where(solution['j_array'] == j_u)[0]
            unmatched_cadets = np.where(solution['j_array'] == p['M'])[0]
            filtered_cadets = filter(lambda i: (i in unmatched_cadets
                                                and i in p['I^E'][j_u]
                                                and i not in assigned_cadets
                                                and i not in cadets_moved), p['I^E'][j_u])

            # Convert the filter object to a list and store it in potential_cadets
            potential_cadets = list(filtered_cadets)

            if len(potential_cadets) == 0:
                impossible_quotas.append(j_u)
                j_u_index = np.where(unfilled_j == j_u)
                unfilled_j = np.delete(unfilled_j, j_u_index)
                if len(unfilled_j) > 0:
                    continue
                else:
                    if len(np.where(solution['j_array'] == p['M'])[0]) == 0:
                        break

            # Get dictionary of cadet preferences for the AFSC that cadet is matched to (for all unassigned eligible cadets)
            a = {}
            for i in potential_cadets:
                if solution['j_array'][i] in j_selects:
                    a[i] = p['c_pref_matrix'][i, solution['j_array'][i]]
                else:
                    a[i] = 100

            # Get dictionary of cadet preferences for this AFSC of the cadets that were unassigned to this AFSC (but eligible)
            b = {i: p['c_pref_matrix'][i, j_u] for i in potential_cadets}

            # Get dictionary of AFSC preferences on the cadets that were unassigned to this AFSC (but were eligible)
            c = {i: p['a_pref_matrix'][i, j_u] for i in potential_cadets}

            # Get cadet(s) that minimize drop in preference between matched and j_u
            d = {i: b[i] - a[i] for i in potential_cadets}
            vals = [d[i] for i in potential_cadets]
            min_val = min(vals)
            minimum_cadets = [i for i in d if d[i] == min_val]

            # Get dictionary of this AFSC's preference on the "minimum cadets"
            minimum_cadets_j_u_choice = {i: c[i] for i in minimum_cadets}
            cadet = min(minimum_cadets_j_u_choice, key=minimum_cadets_j_u_choice.get)

            # Adjust solution and keep track of cadet
            cadets_moved.append(cadet)
            current_afsc = p['afscs'][solution['j_array'][cadet]]
            solution['j_array'][cadet] = j_u
            solution['afsc_array'][cadet] = p['afscs'][j_u]

            if len(np.where(solution['j_array'] == p['M'])[0]) == 0:
                break

        # ----------------------------------------------------------------------------------------------------------------------#
        j_u = min(unfilled_percent_of_lower_quota, key=unfilled_percent_of_lower_quota.get)

        # Check if we have filled all AFSCs to their quotas
        if len(unfilled_j) == 0 and len(np.where(solution['j_array'] == p['M'])[0]) == 0:
            break

        # Lists of cadets that were assigned to "j_u"
        assigned_cadets = np.where(solution['j_array'] == j_u)[0]
        unmatched_cadets = np.where(solution['j_array'] == p['M'])[0]

        filtered_cadets = filter(lambda i: (i in p['I^E'][j_u]
                                            and i not in assigned_cadets
                                            and i not in cadets_moved
                                            and solution['j_array'][i] in j_selects)
                                           or (i in unmatched_cadets
                                               and i in p['I^E'][j_u]
                                               and i not in assigned_cadets
                                               and i not in cadets_moved), p['I^E'][j_u])

        # Convert the filter object to a list and store it in potential_cadets
        potential_cadets = list(filtered_cadets)

        # When we exhaust all possible cadets to move to unfilled AFSCs, remove this AFSC -- it's an impossible quota.
        if len(potential_cadets) == 0:
            impossible_quotas.append(j_u)
            j_u_index = np.where(unfilled_j == j_u)
            unfilled_j = np.delete(unfilled_j, j_u_index)
            if len(unfilled_j) > 0:
                continue
            else:
                if len(np.where(solution['j_array'] == p['M'])[0]) == 0:
                    break

        # Get dictionary of cadet preferences for the AFSC that cadet is matched to (for all unassigned eligible cadets)
        a = {}
        for i in potential_cadets:
            if solution['j_array'][i] in j_selects:
                a[i] = p['c_pref_matrix'][i, solution['j_array'][i]]
            else:
                a[i] = 100

        # Get dictionary of cadet preferences for this AFSC of the cadets that were unassigned to this AFSC (but eligible)
        b = {i: p['c_pref_matrix'][i, j_u] for i in potential_cadets}

        # Get dictionary of AFSC preferences on the cadets that were unassigned to this AFSC (but were eligible)
        c = {i: p['a_pref_matrix'][i, j_u] for i in potential_cadets}

        # Get cadet(s) that minimize drop in preference between matched and j_u
        d = {i: b[i] - a[i] for i in potential_cadets}
        vals = [d[i] for i in potential_cadets]
        min_val = min(vals)
        minimum_cadets = [i for i in d if d[i] == min_val]

        # Get dictionary of this AFSC's preference on the "minimum cadets"
        minimum_cadets_j_u_choice = {i: c[i] for i in minimum_cadets}
        cadet = min(minimum_cadets_j_u_choice, key=minimum_cadets_j_u_choice.get)

        # Adjust solution and keep track of cadet
        cadets_moved.append(cadet)
        current_afsc = p['afscs'][solution['j_array'][cadet]]
        solution['j_array'][cadet] = j_u
        solution['afsc_array'][cadet] = p['afscs'][j_u]

        # Iteration print statement
        if printing:
            print('Iteration', iteration, '\n', 'Cadet chosen:', cadet, 'AFSC moved to:', p['afscs'][j_u],
                  'AFSC moved from:', current_afsc)

        iteration += 1
        if iteration > 1000:
            break

    # Final print statement
    if printing:
        counts = {j: len(np.where(solution['j_array'] == j)[0]) for j in p['J']}
        percent_of_lower_quota = np.array([counts[j] / total_slots[j] * 100 for j in p['J']])
        print('Finished.', iteration, 'iterations processed. Final percent filled:', percent_of_lower_quota)
        for afsc, j in zip(p['afscs'], p['J']):
            print(f"AFSC: {afsc}, % Lower Quota: {percent_of_lower_quota[j]}")

    # Get the indices where afsc_solution is equal to '*'
    indices_with_48 = np.where(solution['j_array'] == 48)[0]
    indices_with_asterisk = np.where(solution['afsc_array'] == '*')[0]
    # Print the indices
    print(indices_with_48)
    print(indices_with_asterisk)

    return solution


# SOC specific algorithms
def rotc_rated_board_original(instance, printing=False):
    """
    Assigns Rated AFSCs to ROTC cadets based on their preferences and the existing quotas for each AFSC using the
    current rated board algorithm.

    Parameters:
        instance (CadetCareerProblem): The instance of the CadetCareerProblem class.
        printing (bool): Whether to print status updates or not. Default is False.

    Returns:
        dict: The solution dictionary containing the assigned AFSCs for each cadet.

    This function assigns Rated AFSCs to ROTC cadets based on their preferences and the existing quotas for each AFSC.
    It follows the current rated board algorithm. The function takes an instance of the CadetCareerProblem class as
    input and an optional parameter `printing` to specify whether to print status updates. By default, `printing` is
    set to False. The function initializes the necessary variables and dictionaries for the algorithm. It then goes
    through each phase of the rated board algorithm, considering cadets' order of merit and interest levels for each
    AFSC. Cadets are assigned AFSCs based on availability and eligibility. The function updates the assigned AFSCs
    for each cadet and tracks the number of matched cadets for each AFSC. Finally, it converts the assigned AFSCs into
    a solution dictionary and returns it.

    Example usage:
        solution = rotc_rated_board_original(instance, printing=True)
    """


    if printing:
        print("Running status quo ROTC rated algorithm...")

    # Shorthand
    p = instance.parameters

    # Cadets/AFSCs and their preferences
    cadet_indices = p['Rated Cadets']['rotc']  # indices of the cadets in the full set of cadets
    cadets, N = np.arange(len(cadet_indices)), len(cadet_indices)
    rated_J = np.array([j for j in p['J^Rated'] if '_U' not in p['afscs'][j]])
    afscs, M = p['afscs'][rated_J], len(rated_J)
    afsc_indices = np.array([np.where(p['afscs'] == afsc)[0][0] for afsc in afscs])
    afsc_om = {afscs[j]: p['rr_om_matrix'][:, j] for j in range(M)}  # list of OM percentiles for each AFSC
    afsc_interest = {afscs[j]: p['rr_interest_matrix'][:, j] for j in range(M)}  # list of Rated interest from cadets
    eligible = {afscs[j]: p['rr_om_matrix'][:, j] > 0 for j in range(M)}  # binary eligibility column for each AFSC

    # Dictionary of dictionaries of cadets within each order of merit "level" for each AFSC
    om_cl = {"High": (0.4, 1), "Med": (0.2, 0.4), "Low": (0.1, 0.2)}
    om = {afsc: {level: np.where((afsc_om[afsc] >= om_cl[level][0]) &
                                 (afsc_om[afsc] < om_cl[level][1]))[0] for level in om_cl} for afsc in afscs}

    # Dictionary of dictionaries of cadets with each interest "level" for each AFSC
    interest_levels = ["High", "Med", "Low", "None"]
    interest = {afsc: {level: np.where(afsc_interest[afsc] == level)[0] for level in interest_levels} for afsc in afscs}

    # Algorithm initialization
    total_slots = {afscs[idx]: p['rotc_quota'][j] for idx, j in enumerate(afsc_indices)}
    total_matched = {afsc: 0 for afsc in afscs}  # Number of matched cadets for each AFSC
    matching = {afsc: True for afsc in afscs}  # Used in stopping conditions
    assigned_afscs = {cadet: "" for cadet in cadets}

    # Initialize solution dictionary
    solution = {'cadets_solved_for': 'ROTC Rated', 'afscs_solved_for': 'Rated', 'method': 'ROTCRatedBoard'}

    # Dictionary of parameters used for the "CadetBoardFigure" object (animation)
    if mdl_p['collect_solution_iterations']:
        solution['iterations'] = {'type': 'ROTC Rated Board'}
        for key in ['matches', 'names']:
            solution['iterations'][key] = {}

    # Re-order AFSCs if necessary
    if instance.mdl_p['rotc_rated_board_afsc_order'] is not None:
        afscs = instance.mdl_p['rotc_rated_board_afsc_order']  # Need to be ordered list of ROTC Rated AFSCs

    # Phases of the Rated board where each tuple represents the level for (OM, Interest)
    phases = [("High", "High"), ("High", "Med"), ("Med", "High"), ("Med", "Med"), ("Low", "High"), ("High", "Low"),
              ("Low", "Med"), ("Med", "Low"), ("Low", "Low"), ("High", "None"), ("Med", "None"), ("Low", "None")]
    phase_num = 0
    s = 0  # for solution iterations
    while any(matching.values()) and phase_num < len(phases):
        phase = phases[phase_num]
        om_level, interest_level = phase[0], phase[1]
        phase_num += 1
        if printing:
            print("\nPhase", phase_num, om_level, "OM", "&", interest_level, "Interest")

        # Loop through each Rated AFSC
        for afsc in afscs:

            # Get the ordered list of cadets we're considering in this phase
            phase_cadets = np.intersect1d(om[afsc][om_level], interest[afsc][interest_level])
            om_phase_cadets = afsc_om[afsc][phase_cadets]
            indices = np.argsort(om_phase_cadets)[::-1]
            ordered_cadets = phase_cadets[indices]

            # Loop through each eligible cadet to assign them this AFSC if applicable
            eligible_cadets = np.where(eligible[afsc])[0]
            counter = 0
            for cadet in ordered_cadets:

                # The cadet has to be eligible for the AFSC to be considered
                if cadet not in eligible_cadets:
                    continue

                # If we didn't already assign them an AFSC, and we've still got open slots left, the cadet gets matched
                if assigned_afscs[cadet] == "" and total_matched[afsc] < total_slots[afsc]:
                    assigned_afscs[cadet] = afsc
                    total_matched[afsc] += 1
                    counter += 1

                # We've reached capacity for this AFSC
                if total_matched[afsc] == total_slots[afsc]:
                    matching[afsc] = False
                    break

            if counter != 0 and printing:
                print(afsc, "Phase Matched:", counter, "  --->   Total Matched:", total_matched[afsc], "/",
                      total_slots[afsc])

            # Solution Iteration components
            s += 1
            if mdl_p['collect_solution_iterations']:
                afsc_solution = np.array([" " * 10 for _ in p['I']])
                for cadet, i in enumerate(cadet_indices):
                    if assigned_afscs[cadet] in afscs:
                        afsc_solution[i] = assigned_afscs[cadet]
                indices = np.where(afsc_solution == " " * 10)[0]
                afsc_solution[indices] = "*"
                solution['iterations']['matches'][s] = \
                    np.array([np.where(p['afscs'] == afsc)[0][0] for afsc in afsc_solution])
                solution['iterations']['names'][s] = \
                    "Phase " + str(phase_num) + " (" + om_level + ", " + interest_level + ") [" + afsc + "]"

    # Solution Iteration components
    if mdl_p['collect_solution_iterations']:
        solution['iterations']['last_s'] = s - 1

    # Convert it back to a full solution with all cadets (anyone not matched to a Rated AFSC is unmatched)
    afsc_solution = np.array([" " * 10 for _ in p['I']])
    for cadet, i in enumerate(cadet_indices):
        if assigned_afscs[cadet] in afscs:
            afsc_solution[i] = assigned_afscs[cadet]
    indices = np.where(afsc_solution == " " * 10)[0]
    afsc_solution[indices] = "*"
    solution['j_array'] = np.array([np.where(p['afscs'] == afsc)[0][0] for afsc in afsc_solution])
    return solution


def soc_rated_matching_algorithm(instance, soc='usafa', printing=True):
    """
    Matches or reserves cadets to their Rated AFSCs based on the Source of Commissioning (SOC) using the Hospitals/Residents algorithm.

    Parameters:
        instance (CadetCareerProblem): The instance of the CadetCareerProblem class.
        soc (str): The SOC for which to perform the matching algorithm. Options are 'usafa' (United States Air Force Academy)
                   or 'rotc' (Reserve Officer Training Corps). Default is 'usafa'.
        printing (bool): Whether to print status updates or not. Default is True.

    Returns:
        tuple: A tuple containing three solution dictionaries: the overall solution, the reserves solution,
            and the matches solution.

    This function implements the Hospitals/Residents algorithm to match or reserve cadets to their Rated AFSCs
    based on the Source of Commissioning (SOC). It takes an instance of the CadetCareerProblem class as input and
    an optional parameter `soc` to specify the SOC for which the matching algorithm should be performed. The available
    options for `soc` are 'usafa' (United States Air Force Academy) and 'rotc' (Reserve Officer Training Corps). By
    default, the SOC is set to 'usafa'. The function also takes an optional parameter `printing` to control whether
    status updates are printed during the matching process.

    The algorithm initializes the necessary variables and dictionaries. It then proceeds with the Hospitals/Residents
    algorithm by having cadets propose to their top choices and AFSCs accept or reject cadets based on their preferences
    and capacities. The matching process continues until all cadets are matched or have exhausted their preferences.
    The function tracks the progress through iterations and collects information on both reserved and matched AFSCs.

    The function returns a tuple containing three solution dictionaries: the overall solution, the reserves solution,
    and the matches solution. Each solution dictionary contains the assigned AFSCs for each cadet. The reserves
    solution only includes cadets with reserved slots, the matches solution only includes cadets with matched slots,
    and the overall solution includes both cadets with reserved and matched slots.

    Example usage:
        solution, reserves, matches = soc_rated_matching_algorithm(instance, soc='usafa', printing=True)
    """

    if printing:
        print("Solving the rated matching algorithm for " + soc.upper() + " cadets...")

    # Shorthand
    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p

    # Slight change to Rated AFSCs (Remove SOC specific slots)
    if soc == 'usafa':
        rated_J = np.array([j for j in p['J^Rated'] if '_R' not in p['afscs'][j]])
    else:
        rated_J = np.array([j for j in p['J^Rated'] if '_U' not in p['afscs'][j]])

    # Algorithm initialization
    total_slots = {j: p[soc + "_quota"][j] for j in rated_J}
    cadets = p['Rated Cadets'][soc]
    N = len(cadets)

    # "REAL" first choice of the cadet
    first_choice = {i: p['cadet_preferences'][i][0] for i in cadets}

    # Dictionary to keep track of what AFSC choice in their list the cadets are proposing to
    cadet_proposal_choice = {i: 0 for i in cadets}  # Initially all propose to their top Rated preference!

    # Initialize solution dictionary for all 3 solutions (reserves, matches, combined)
    solution_reserves = {'cadets_solved_for': soc.upper() + ' Rated', 'afscs_solved_for': 'Rated',
                         'method': 'Rated ' + soc.upper() + ' HR (Reserves)'}
    solution_matches = {'cadets_solved_for': soc.upper() + ' Rated', 'afscs_solved_for': 'Rated',
                        'method': 'Rated ' + soc.upper() + ' HR (Matches)'}
    solution = {'cadets_solved_for': soc.upper() + ' Rated', 'afscs_solved_for': 'Rated',  # Combined Solution
                'method': 'Rated ' + soc.upper() + ' HR'}

    # Dictionary of parameters used for the "CadetBoardFigure" object (animation)
    if mdl_p['collect_solution_iterations']:
        solution['iterations'] = {'type': 'Rated SOC HR'}
        for key in ['proposals', 'matches', 'reserves', 'matched', 'names']:
            solution['iterations'][key] = {}

    # Begin the simple Hospital/Residents Algorithm
    total_rejections = {j: 0 for j in rated_J}  # Number of rejections for each AFSC
    total_matched = {j: 0 for j in rated_J}  # Number of accepted cadets for each AFSC
    exhausted_cadets = []  # Will contain the cadets that have exhausted (been rejected by) all of their preferences
    iteration = 0  # First iteration of the algorithm
    while sum([total_matched[j] for j in rated_J]) + len(exhausted_cadets) < N:  # Stopping conditions

        # Cadets propose to their top choice that hasn't been rejected
        proposals = {i: p['Rated Choices'][
            soc][i][cadet_proposal_choice[i]] if i not in exhausted_cadets else p['M'] for i in cadets}
        proposal_array = np.array([proposals[i] if i in cadets else p['M'] for i in p['I']])
        counts = {p['afscs'][j]: len(np.where(proposal_array == j)[0]) for j in rated_J}

        # Solution Iteration components (Proposals) and print statement
        if mdl_p['collect_solution_iterations']:
            solution['iterations']['proposals'][iteration] = proposal_array
        if mdl_p['ma_printing']:
            print("\nIteration", iteration + 1)

        # Initialize matches information for this iteration
        total_matched = {j: 0 for j in rated_J}

        # AFSCs accept their best cadets and reject the others
        for j in rated_J:

            # Loop through their preferred cadets from top to bottom
            iteration_rejections = 0
            for i in p['afsc_preferences'][j]:
                if i not in cadets:
                    continue  # Other SOC (we don't care about them right now)

                # If the cadet is proposing to this AFSC, we have two options
                if proposals[i] == j:

                    # We haven't hit capacity, so we accept this cadet
                    if total_matched[j] < total_slots[j]:
                        total_matched[j] += 1

                    # We're at capacity, so we reject this cadet
                    else:

                        # Essentially "delete" the preference from the cadet's list
                        cadet_proposal_choice[i] += 1
                        proposals[i] = p['M']  # index of the unmatched AFSC (*)

                        # Collect additional information
                        if mdl_p['ma_printing']:
                            iteration_rejections += 1
                            total_rejections[j] += 1

        # Solution Iteration components
        if mdl_p['collect_solution_iterations']:

            # Rated matches from this iteration
            solution['iterations']['matches'][iteration] = \
                np.array([proposals[i] if i in cadets else p['M'] for i in p['I']])
            solution['iterations']['names'][iteration] = 'Iteration ' + str(iteration + 1)

            # Collect information on this iteration's reserved slots and actual matched slots
            reserves = np.zeros(p['N']).astype(int)
            matches = np.zeros(p['N']).astype(int)
            for i in p['I']:

                # Default to unmatched
                reserves[i], matches[i] = p['M'], p['M']
                if i in cadets:
                    if proposals[i] in rated_J:
                        if first_choice[i] == proposals[i]:
                            matches[i] = proposals[i]
                        else:
                            reserves[i] = proposals[i]

            # Set of cadets with reserved or matched slots
            solution['iterations']['matched'][iteration] = np.where(matches != p['M'])[0]
            solution['iterations']['reserves'][iteration] = np.where(reserves != p['M'])[0]

        # Specific matching algorithm print statement
        if mdl_p['ma_printing']:
            print('Proposals:', counts)
            print('Matched', {p['afscs'][j]: total_matched[j] for j in p['J']})
            print('Rejected', {p['afscs'][j]: total_rejections[j] for j in p['J']})

        # Check exhausted cadets
        exhausted_cadets = []
        for i in cadets:
            if cadet_proposal_choice[i] >= p['Num Rated Choices'][soc][i]:
                exhausted_cadets.append(i)

        iteration += 1 # Next iteration!

    # Last solution iteration
    solution['iterations']['last_s'] = iteration - 1

    # Collect information on all 3 solutions: reserves, matches, and combined
    solution_reserves['j_array'] = np.zeros(p['N']).astype(int)
    solution_matches['j_array'] = np.zeros(p['N']).astype(int)
    solution['j_array'] = np.zeros(p['N']).astype(int)
    for i in p['I']:

        # Default to unmatched
        solution_matches['j_array'][i], solution_reserves['j_array'][i] = p['M'], p['M']
        solution['j_array'][i] = p['M']
        if i in cadets:
            if proposals[i] in rated_J:
                solution['j_array'][i] = proposals[i]
                if first_choice[i] == proposals[i]:
                    solution_matches['j_array'][i] = proposals[i]
                else:
                    solution_reserves['j_array'][i] = proposals[i]

    # Add information to the solution matches and reserves components
    solution['matches'] = np.where(solution_matches['j_array'] != p['M'])[0]
    solution['reserves'] = np.where(solution_reserves['j_array'] != p['M'])[0]

    # Return solution, reserved array, and solution iterations
    return solution, solution_reserves, solution_matches


# Meta-heuristics
def vft_genetic_algorithm(instance, initial_solutions=None, con_fail_dict=None, printing=False):
    """
    Solves the optimization problem using a genetic algorithm.

    Parameters:
        instance (CadetCareerProblem): An instance of the CadetCareerProblem class representing the optimization problem.
        initial_solutions (ndarray or None): An optional array of initial solutions in the population. If provided, it
            should be a numpy ndarray of shape (pop_size, N) where pop_size is the size of the population and N is the
            number of cadets. Default is None.
        con_fail_dict (dict or None): An optional dictionary containing information about constraints that failed for
            the initial solutions. It should be a dictionary where the keys are the indices of the initial solutions
            (0-based) and the values are lists of constraint indices that failed for that solution. Default is None.
        printing (bool): A flag indicating whether to print status updates during the genetic algorithm execution.
            Default is False.

    Returns:
        tuple: A tuple containing the best solution and the time evaluation dataframe (if time evaluation is enabled).

    This function implements a genetic algorithm to solve the optimization problem defined by the CadetCareerProblem
    instance. The genetic algorithm works by iteratively evolving a population of candidate solutions through selection,
    crossover, and mutation operations. The fitness of each solution is evaluated using the Value-Focused Thinking (VFT)
    objective function.

    The genetic algorithm operates as follows:
    1. Initialize the population: If initial_solutions are provided, they are used as the initial population. Otherwise,
       a random population is generated.
    2. Evaluate the fitness of each solution in the population using the VFT objective function.
    3. Sort the population based on the fitness scores in descending order.
    4. Create the next generation of solutions:
       - The top two solutions (best fitness) from the current population are automatically included in the next generation.
       - For the remaining solutions, select two parents based on their fitness scores using rank selection.
       - Apply multi-point crossover to generate two offspring solutions from the selected parents.
       - Perform mutation on the offspring solutions to introduce small random changes.
       - Add the offspring solutions to the next generation.
    5. Repeat steps 2-4 until the termination condition is met (e.g., maximum time limit).

    The best solution found during the genetic algorithm execution is returned as the output. If time evaluation is
    enabled, a time evaluation dataframe is also returned, containing the objective values at different time points
    during the algorithm execution.

    Example usage:
        solution, time_eval_df = vft_genetic_algorithm(instance, initial_solutions, con_fail_dict, printing=True)
    """


    def multi_point_crossover(genome1, genome2):
        """
        Take two parent genomes, crossover the genes at multiple points and return two offspring solutions
        :param genome1: first parent genome
        :param genome2: second parent genome
        :return: offspring
        """
        points = np.sort(np.random.choice(crossover_positions, size=mp["num_crossover_points"], replace=False))
        start_points = np.append(0, points)
        stop_points = np.append(points, p['N'] - 1)
        child1 = np.zeros(p['N']).astype(int)
        child2 = np.zeros(p['N']).astype(int)
        flip = 1
        for i in range(len(start_points)):
            if flip == 1:
                child1[start_points[i]:stop_points[i] + 1] = genome2[start_points[i]:stop_points[i] + 1]
                child2[start_points[i]:stop_points[i] + 1] = genome1[start_points[i]:stop_points[i] + 1]
            else:
                child1[start_points[i]:stop_points[i] + 1] = genome1[start_points[i]:stop_points[i] + 1]
                child2[start_points[i]:stop_points[i] + 1] = genome2[start_points[i]:stop_points[i] + 1]

            flip = flip * -1

        return child1, child2

    def mutation(genome):
        """
        Takes a genome, and picks a random cadet index to mutate with some probability.
        This means we can swap an AFSC for one cadet individually
        :param genome: solution vector
        :return: mutated genome
        """
        for _ in range(mp["num_mutations"]):
            i = np.random.choice(p['I^Variable'])  # Pick a random cadet that doesn't have a "fixed" AFSC

            if mp['mutation_function'] == 'cadet_choice':

                # Determine what set of AFSCs we can choose from (Coin flip on if we're going to select more preferred ones)
                if np.random.uniform() > mp['preference_mutation_rate']:

                    # Current preference that the cadet received
                    current_choice = p['c_pref_matrix'][i, genome[i]]

                    # All AFSCs that are at least as preferred as current assigned
                    possible_afscs = p['cadet_preferences'][i][:current_choice]

                else:

                    # All AFSCs that the cadet is eligible for
                    possible_afscs = p['J^E'][i]
            else:

                # All AFSCs that the cadet is eligible for
                possible_afscs = p['J^E'][i]

            # Fix the possible AFSCs to select from if this cadet has a reserved Rated slot
            if i in p['J^Reserved']:
                possible_afscs = p['J^Reserved'][i]

            # Pick a random AFSC
            j = np.random.choice(possible_afscs)

            # Mutate if applicable
            genome[i] = j if (np.random.uniform() < mp["mutation_rate"]) else genome[i]

        return genome

    # Shorthand
    p = instance.parameters
    vp = instance.value_parameters
    mp = instance.mdl_p

    # Cadets that aren't "fixed" in the solution
    p['I^Variable'] = np.array([i for i in p['I'] if i not in p['J^Fixed']])

    # Rank Selection Parameters
    rank_weights = (np.arange(1, mp["pop_size"] + 1)[::-1]) ** 1.2
    rank_weights = rank_weights / sum(rank_weights)
    rank_choices = np.arange(mp["pop_size"])

    # Multi-Point Crossover Parameters
    crossover_positions = np.arange(1, p['N'] - 1)

    # Initialize Population
    population = np.array([[np.random.choice(p['J^E'][i]) for i in p['I']] for _ in range(mp["pop_size"])]).astype(int)
    if initial_solutions is not None:

        # Get fitness of initial_solutions in case there are feasibility issues or there are too many initial solutions
        num_initial = len(initial_solutions)
        fitness = np.zeros(num_initial)
        for s, chromosome in enumerate(initial_solutions):
            fitness[s] = afccp.core.solutions.handling.fitness_function(chromosome, p, vp, mp, con_fail_dict)

        # Sort Initial solutions by Fitness
        sorted_indices = fitness.argsort()[::-1]
        initial_solutions = initial_solutions[sorted_indices]

        # Insert these solutions into the population
        for s, chromosome in enumerate(initial_solutions):

            # Make sure there aren't too many initial solutions
            if s < mp["pop_size"]:
                population[s] = chromosome

    # Initialize Fitness Scores
    fitness = np.zeros(mp["pop_size"])
    for s, chromosome in enumerate(population):
        fitness[s] = afccp.core.solutions.handling.fitness_function(chromosome, p, vp, mp, con_fail_dict)

    # Sort Population by Fitness
    sorted_indices = fitness.argsort()[::-1]
    fitness = fitness[sorted_indices]
    population = population[sorted_indices]

    # Print updates
    if mp["ga_printing"]:
        step = mp["ga_max_time"] / (100 / mp["percent_step"])
        checkpoint = step
        completed = 0
        print('Initial Fitness Scores', fitness)

    # Time Evaluation Initialization
    if mp["time_eval"]:
        time_step = mp["ga_max_time"] / mp["num_time_points"]
        step_num = 1
        times = [0]
        scores = [fitness[0]]

    # Main Loop
    start_time = time.perf_counter()
    eval_times = []
    gen_times = []
    generating = True
    generation = 1
    while generating:

        # Start time
        gen_start_time = time.perf_counter()

        # Evaluate Population
        for index in range(2, mp["pop_size"]):
            fitness[index] = afccp.core.solutions.handling.fitness_function(population[index], p, vp, mp, con_fail_dict)
        eval_times.append(time.perf_counter() - gen_start_time)

        # Sort Population by Fitness
        sorted_indices = fitness.argsort()[::-1]
        fitness = fitness[sorted_indices]
        population = population[sorted_indices]
        best_score = fitness[0]

        # Printing updates
        if mp["ga_printing"]:
            if (time.perf_counter() - start_time) > checkpoint:
                completed += mp["percent_step"]
                checkpoint += step
                print(str(completed) + "% complete. Best solution value: " + str(round(best_score, 4)))
                avg_eval_time = np.mean(np.array(eval_times[0:generation]))
                print('Average evaluation time for ' + str(mp["pop_size"]) + ' solutions: ' +
                      str(round(avg_eval_time, 4)) + ' seconds.')
                avg_gen_time = np.mean(np.array(gen_times[0:generation]))
                print('Average generation time: ' + str(round(avg_gen_time, 4)) + ' seconds.')

        # Create next generation
        next_generation = population[0:2]  # the best two solutions are kept for the next generation
        for twins in range(int((mp["pop_size"] / 2) - 1)):  # create the offspring

            # Select parents for mating
            index_1, index_2 = np.random.choice(rank_choices, size=2, replace=False, p=rank_weights)
            parent_1, parent_2 = population[index_1], population[index_2]

            # Apply crossover function
            offspring_1, offspring_2 = multi_point_crossover(parent_1, parent_2)

            # Mutate genomes of offspring
            offspring_1 = mutation(offspring_1)
            offspring_2 = mutation(offspring_2)

            # Add this pair to the next generation
            offsprings = np.vstack((offspring_1, offspring_2))
            next_generation = np.vstack((next_generation, offsprings))

        # Time Eval
        if mp["time_eval"]:
            if (time.perf_counter() - start_time) > (time_step * step_num):
                times.append(time.perf_counter() - start_time)
                scores.append(best_score)
                step_num += 1

        # Check stopping criteria
        if (time.perf_counter() - start_time) > mp["ga_max_time"]:
            if mp["ga_printing"]:
                end_time = round(time.perf_counter() - start_time, 2)
                print('End time reached in ' + str(end_time) + ' seconds.')
            generating = False

        # Next Generation
        population = next_generation
        gen_times.append(time.perf_counter() - gen_start_time)
        generation += 1

    # Acquire solution dictionary for the top chromosome in the population
    solution = {'method': 'VFT_Genetic', 'j_array': population[0]}

    # Time Eval
    if mp["time_eval"]:

        # Create time_eval_df
        time_eval_df = pd.DataFrame({'Time': times, 'Objective Value': scores})

        if printing:
            print(time_eval_df)
        return solution, time_eval_df

    else:

        # Return best solution
        return solution, None


def genetic_matching_algorithm(instance, printing=False):
    """
    Genetic algorithm that determines optimal capacities to the classic deferred acceptance algorithm to minimize
    blocking pairs

    Parameters:
        instance (CadetCareerProblem): An instance of the CadetCareerProblem class representing the optimization problem.
        printing (bool): A flag indicating whether to print additional information during the algorithm execution.
            Default is False.

    Returns:
        ndarray: An array representing the optimal capacities determined by the genetic algorithm.

    This function implements a genetic algorithm to determine the optimal capacities for the classic deferred acceptance
    algorithm. The goal is to minimize the number of blocking pairs in the matching process.

    The genetic algorithm works as follows:
    1. Initialize the population of capacities randomly. Each capacity is selected within the valid range for the
       corresponding AFSC.
    2. Evaluate the fitness of each capacity configuration using the classic deferred acceptance algorithm with the
       given capacities. The fitness is determined by the number of blocking pairs in the resulting matching.
    3. Sort the population based on fitness scores in descending order.
    4. Create the next generation of capacities:
       - The two best capacities (lowest fitness) from the current population are automatically included in the next
         generation.
       - For the remaining capacities, select two parents based on their fitness scores using rank selection.
       - Apply multi-point crossover to generate two offspring capacities from the selected parents.
       - Perform mutation on the offspring capacities to introduce small random changes.
       - Add the offspring capacities to the next generation.
    5. Repeat steps 2-4 until a termination condition is met (e.g., maximum time or number of generations).

    The best capacity configuration found during the genetic algorithm execution is returned as the output.

    Example usage:
        optimal_capacities = genetic_matching_algorithm(instance, printing=True)
    """


    # Shorthand
    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p

    # Define functions
    def initialize_population():
        """
        Function to initialize all the "chromosomes" for this GA
        """
        population = np.array([np.zeros(p['M']) for _ in range(mdl_p['gma_pop_size'])])
        for c in range(mdl_p['gma_pop_size']):
            for j in p['J']:
                capacity = int(random.choice(capacity_options[j]))
                population[c, j] = capacity

        return population

    def fitness_function(chromosome):
        """
        Evaluates the chromosome (capacities for HR)
        """

        # Run the algorithm using these capacities
        solution = classic_hr(instance, capacities=chromosome, printing=False)

        # Evaluate blocking pairs
        return afccp.core.solutions.handling.calculate_blocking_pairs(p, solution, only_return_count=True)

    def multi_point_crossover(genome1, genome2):
        """
        Take two parent genomes, crossover the genes at multiple points and return two offspring solutions
        :param genome1: first parent genome
        :param genome2: second parent genome
        :return: offspring
        """
        points = np.sort(np.random.choice(crossover_positions, size=mdl_p["gma_num_crossover_points"], replace=False))
        start_points = np.append(0, points)
        stop_points = np.append(points, p['M'] - 1)
        child1 = np.zeros(p['M']).astype(int)
        child2 = np.zeros(p['M']).astype(int)
        flip = 1
        for i in range(len(start_points)):
            if flip == 1:
                child1[start_points[i]:stop_points[i] + 1] = genome2[start_points[i]:stop_points[i] + 1]
                child2[start_points[i]:stop_points[i] + 1] = genome1[start_points[i]:stop_points[i] + 1]
            else:
                child1[start_points[i]:stop_points[i] + 1] = genome1[start_points[i]:stop_points[i] + 1]
                child2[start_points[i]:stop_points[i] + 1] = genome2[start_points[i]:stop_points[i] + 1]

            flip = flip * -1

        return child1, child2

    def mutation(genome):
        """
        Takes a genome, and picks a random cadet index to mutate with some probability.
        This means we can swap an AFSC for one cadet individually
        :param genome: solution vector
        :return: mutated genome
        """
        for _ in range(mdl_p["gma_mutations"]):
            j = np.random.choice(p['J'])  # Random AFSC

            # Pick random new capacity for AFSC j
            min, max = p['quota_min'][j], p['quota_max'][j]
            capacity_options = np.arange(min, max + 1).astype(int)
            capacity = int(random.choice(capacity_options))
            genome[j] = capacity if (np.random.uniform() < mdl_p["gma_mutation_rate"]) else genome[j]

        return genome

    # Determine range of capacities for all AFSCs
    capacity_options = {}
    for j in p['J']:
        min, max = p['quota_min'][j], p['quota_max'][j]
        capacity_options[j] = np.arange(min, max + 1).astype(int)

    # Rank Selection Parameters
    rank_weights = (np.arange(1, mdl_p["gma_pop_size"] + 1)[::-1]) ** 1.2
    rank_weights = rank_weights / sum(rank_weights)
    rank_choices = np.arange(mdl_p["gma_pop_size"])

    # Multi-Point Crossover Parameters
    crossover_positions = np.arange(1, p['M'] - 1)

    # Initialize population
    population = initialize_population()

    # Initialize fitness scores
    fitness = np.zeros(mdl_p['gma_pop_size'])
    for c in range(mdl_p['gma_pop_size']):
        fitness[c] = fitness_function(population[c])

    # Sort Population by Fitness
    sorted_indices = fitness.argsort()
    fitness = fitness[sorted_indices]
    population = population[sorted_indices]

    # Main Loop
    start_time = time.perf_counter()
    generation = 0
    generating = True
    while generating:

        # Evaluate population
        for c in range(2, mdl_p['gma_pop_size']):
            fitness[c] = fitness_function(population[c])

        # Sort Population by Fitness
        sorted_indices = fitness.argsort()
        fitness = fitness[sorted_indices]
        population = population[sorted_indices]

        # Print statements
        if mdl_p['gma_printing']:
            print('Generation', generation, 'Fitness', fitness)

        # Create next generation
        next_generation = population[:2]  # the best two solutions are kept for the next generation
        for twins in range(int((mdl_p["gma_pop_size"] / 2) - 1)):  # create the offspring

            # Select parents for mating
            c1, c2 = np.random.choice(rank_choices, size=2, replace=False, p=rank_weights)
            parent_1, parent_2 = population[c1], population[c2]

            # Apply crossover function
            offspring_1, offspring_2 = multi_point_crossover(parent_1, parent_2)

            # Mutate genomes of offspring
            offspring_1 = mutation(offspring_1)
            offspring_2 = mutation(offspring_2)

            # Add this pair to the next generation
            offsprings = np.vstack((offspring_1, offspring_2))
            next_generation = np.vstack((next_generation, offsprings))

        # Next Generation
        population = next_generation
        generation += 1

        # Stopping conditions
        if mdl_p['stopping_conditions'] == 'Time':
            if (time.perf_counter() - start_time) < mdl_p['gma_max_time']:
                generating = False
        elif mdl_p['stopping_conditions'] == 'Generations':
            if generation >= mdl_p['gma_num_generations']:
                generating = False

    if printing:
        print("Final capacities:", population[0])

    # Return the capacities
    return population[0]


