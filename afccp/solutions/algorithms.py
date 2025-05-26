import time
import numpy as np
import random
import copy
import pandas as pd

# afccp modules
import afccp.globals
import afccp.solutions.handling
from afccp.data.preferences import determine_soc_rated_afscs


# Matching algorithms
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

    # Dictionary of parameters used for the "BubbleChart" object (animation)
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
    if mdl_p['collect_solution_iterations']:
        solution['iterations']['last_s'] = iteration - 1

    # Return solution
    solution['j_array'] = proposals
    solution['afsc_array'] = np.array([p['afscs'][j] for j in solution['j_array']])
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
    p, mdl_p = instance.parameters, instance.mdl_p

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

    # Dictionary of parameters used for the "BubbleChart" object (animation)
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
    p, mdl_p = instance.parameters, instance.mdl_p

    # Slight change to Rated AFSCs (Remove SOC specific slots)
    rated_afscs = determine_soc_rated_afscs(soc, all_rated_afscs=p['afscs_acc_grp']["Rated"])
    rated_J = np.array([np.where(p['afscs'] == afsc)[0][0] for afsc in rated_afscs])

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

    # Dictionary of parameters used for the "BubbleChart" object (animation)
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
            print('Matched', {p['afscs'][j]: total_matched[j] for j in rated_J})
            print('Rejected', {p['afscs'][j]: total_rejections[j] for j in rated_J})

        # Check exhausted cadets
        exhausted_cadets = []
        for i in cadets:
            if cadet_proposal_choice[i] >= p['Num Rated Choices'][soc][i]:
                exhausted_cadets.append(i)

        iteration += 1 # Next iteration!

    # Last solution iteration
    if mdl_p['collect_solution_iterations']:
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


def allocate_ots_candidates_original_method(instance, printing=False):

    if printing:
        print("Running status quo OTS matching/selection algorithm...")

    # Shorthand
    p, mdl_p = instance.parameters, instance.mdl_p

    # Make sure we're dealing with the original ROTC/USAFA solution with unmatched OTS candidates!
    if 'One Market ROTC_USAFA' not in instance.solutions.keys():
        raise ValueError('Error. Solution "One Market ROTC_USAFA" not present in solutions dictionary.\n'
                         'We need this solution which contains matched ROTC/USAFA cadets and unmatched OTS candidates.')
    solution = instance.solutions['One Market ROTC_USAFA']

    # Initialize new solution to add OTS into the mix
    new_solution = copy.deepcopy(solution)
    new_solution['method'] = 'One Market OTS Addition-Status Quo'
    new_solution['cadets_solved_for'] = 'OTS Cadets'
    new_solution['afscs_solved_for'] = 'OTS-AFSCs'

    # Dictionary of parameters used for the "BubbleChart" object (animation)
    new_solution['iterations'] = {'type': 'OTS Status Quo Algorithm'}
    for key in ['rejections', 'matches', 'names', 'new_match', 'cadets_matched']:
        new_solution['iterations'][key] = {}

    # Sort OTS candidates in order of merit
    ordered_ots = np.argsort(p['merit'])[::-1]
    mask = np.isin(ordered_ots, p['I^OTS'])
    ordered_ots = ordered_ots[mask]

    # Loop through each OTS candidate (in order of merit) to assign an AFSC
    counts = copy.deepcopy(p['ots_quota'])
    iteration = 1
    keep = True
    stop_iteration = 14
    sequence = 'Initial'
    resume_iteration = 1000000
    resume_interval = 200
    for i in ordered_ots:

        # Loop through each AFSC in order of the candidate's preferences
        choice = 0
        for j in p['cadet_preferences'][i]:
            if j not in p['J^Selected'][i]:  # This AFSC has to be one that the cadet selected
                continue
            else:
                choice += 1

            if iteration >= resume_iteration:
                keep = True
                stop_iteration = iteration + 8
                resume_interval *= 2
                resume_iteration = iteration + resume_interval

            # If there are still slots to give out, give them one
            if counts[j] > 0:

                # Assign this AFSC to this cadet
                new_solution['j_array'][i] = j

                # Decrement the remaining slots by 1... on to the next candidate
                counts[j] -= 1

                # Solution iterations handling
                if iteration >= stop_iteration:
                    keep = False

                # Keep solution iterations
                if keep:
                    new_solution['iterations']['names'][iteration] = f'I: {iteration}. ' \
                                                                     f'Cadet "{i}" matched to {p["afscs"][j]}.'
                    new_solution['iterations']['matches'][iteration] = copy.deepcopy(new_solution['j_array'])
                    new_solution['iterations']['new_match'][iteration] = (i, j)
                iteration += 1
                break
            else:

                # This is the first reject!
                if sequence == 'Initial':
                    keep = True
                    sequence = 'Continue'
                    stop_iteration = iteration + 8
                    resume_iteration = iteration + resume_interval

                # Keep solution iterations
                if keep and iteration <= 1000:
                    new_solution['iterations']['names'][iteration] = f'I: {iteration}. ' \
                                                                     f'Cadet "{i}" rejected from {p["afscs"][j]}.'
                    new_solution['iterations']['matches'][iteration] = copy.deepcopy(new_solution['j_array'])
                    new_solution['iterations']['matches'][iteration][i] = j
                    new_solution['iterations']['rejections'][iteration] = (i, j)
                    new_solution['iterations']['new_match'][iteration] = (i, j)
                iteration += 1

    # Last iteration!
    new_solution['iterations']['names'][iteration] = f'I: {iteration}. Final Solution.'
    new_solution['iterations']['matches'][iteration] = copy.deepcopy(new_solution['j_array'])
    new_solution['iterations']['matches'][iteration][i] = j
    new_solution['iterations']['rejections'][iteration] = (i, j)
    new_solution['iterations']['new_match'][iteration] = (i, j)

    # Save last iteration info
    iterations = list(new_solution['iterations']['new_match'].keys())
    new_solution['iterations']['last_s'] = iterations[len(iterations) - 1]

    # Return the new solution with OTS included!
    return new_solution


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
            fitness[s] = afccp.solutions.handling.fitness_function(chromosome, p, vp, mp, con_fail_dict)

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
        fitness[s] = afccp.solutions.handling.fitness_function(chromosome, p, vp, mp, con_fail_dict)

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
            fitness[index] = afccp.solutions.handling.fitness_function(population[index], p, vp, mp, con_fail_dict)
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
        return afccp.solutions.handling.calculate_blocking_pairs(p, solution, only_return_count=True)

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
            if (time.perf_counter() - start_time) > mdl_p['gma_max_time']:
                generating = False
        elif mdl_p['stopping_conditions'] == 'Generations':
            if generation >= mdl_p['gma_num_generations']:
                generating = False

        # We have no blocking pairs!
        if fitness[0] == 0:
            generating = False

    if printing:
        print("Final capacities:", population[0])

    # Return the capacities
    return population[0]


