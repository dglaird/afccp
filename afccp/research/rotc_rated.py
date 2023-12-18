# Import Libraries
import afccp.core.globals
from typing import Any
import os
import pandas as pd
import numpy as np
import math
import datetime
import glob
import copy
import time
import warnings

warnings.filterwarnings('ignore')


# noinspection PyDictCreation
def process_data_in_parameters(df, parameters):
    """
    This function takes in a dataframe (must be of the "ROTC Rated Data" format), processes it,
    and loads it into the appropriate "parameters" data dictionary. The "parameters" dictionary is then returned
    :param df: dataframe in "ROTC Rated Data" format
    :param parameters: initialized parameters dictionary
    :return: parameters dictionary
    """

    # Shorthand
    p = parameters

    # Collect AFSC preferences
    p['rr_om_matrix'] = np.array(df.loc[:, p['rated_afscs'][0]: p['rated_afscs'][len(p['rated_afscs']) - 1]])
    p['afscs_preferences'] = {}
    for idx, j in enumerate(p['J^Rated']):

        sorted_cadets = np.argsort(p['rr_om_matrix'][:, idx])[::-1]
        zero_cadets = np.where(p['rr_om_matrix'][:, idx] == 0)[0]
        num_zero = len(zero_cadets)
        p['afscs_preferences'][j] = sorted_cadets[:p['N'] - num_zero]

    # Collect Cadet preferences
    p['c_preferences'] = np.array(df.loc[:, 'Pref_1':])
    p['cadets_preferences'] = {}
    p['rated_only_preferences'] = {}
    p['cadet_first_choice'] = {}
    p['num_eligible'] = {}
    for i in p['I']:

        # "Real" first choice AFSC
        p['cadet_first_choice'][i] = p['c_preferences'][i, 0]

        # Ordered list of AFSC indices
        p['rated_only_preferences'][i] = np.array(
            [np.where(p['afscs'] == afsc)[0][0] for afsc in p['c_preferences'][i] if afsc in p['rated_afscs']])
        p['cadets_preferences'][i] = np.array(
            [np.where(p['afscs'] == afsc)[0][0] for afsc in p['c_preferences'][i] if afsc in p['afscs']])

        # Make sure cadet is on each of their desired AFSCs' lists
        cadet_not_on_afsc_lists = []
        for j in p['rated_only_preferences'][i]:
            if i not in p['afscs_preferences'][j]:
                cadet_not_on_afsc_lists.append(p['afscs'][j])

                # We remove this AFSC from this cadet's preferences
                idx = np.where(p['rated_only_preferences'][i] == j)[0][0]
                p['rated_only_preferences'][i] = np.delete(p['rated_only_preferences'][i], idx)

                # We remove this AFSC from this cadet's preferences
                idx = np.where(p['cadets_preferences'][i] == j)[0][0]
                p['cadets_preferences'][i] = np.delete(p['cadets_preferences'][i], idx)

        # Print update
        if len(cadet_not_on_afsc_lists) > 0:
            print_afscs = ", ".join(cadet_not_on_afsc_lists)
            print("[" + str(i) + "] Cadet '" + str(p['cadets'][i]) + "' not eligible for " + print_afscs +
                  ". AFSC(s) have been removed from the cadet's preferences.")

        # See if this cadet is on the AFSC's list but the AFSC is not in their preferences
        afsc_not_on_cadet_list = []
        for j in [j for j in p['J^Rated'] if j not in p['rated_only_preferences'][i]]:
            if i in p['afscs_preferences'][j]:
                afsc_not_on_cadet_list.append(p['afscs'][j])

                # We remove this cadet from this AFSC's preferences
                idx = np.where(p['afscs_preferences'][j] == i)[0][0]
                p['afscs_preferences'][j] = np.delete(p['afscs_preferences'][j], idx)

        # Print update
        if len(afsc_not_on_cadet_list) > 0:
            print_afscs = ", ".join(afsc_not_on_cadet_list)
            print("[" + str(i) + "] Cadet '" + str(p['cadets'][i]) + "' on preferences lists for " + print_afscs +
                  ". AFSC(s) not in cadet's list. Cadet has been removed from the AFSCs' preferences.")

        # Number of eligible AFSCs for this cadet
        p['num_eligible'][i] = len(p['rated_only_preferences'][i])

        # Make sure this cadet is eligible for at least one rated AFSC
        if p['num_eligible'][i] == 0:
            raise ValueError("Error at index '" + str(i) + "'. Cadet '" + str(p['cadets'][i]) +
                             "' not eligible for any rated AFSCs but is in the dataset. Please adjust.")

    # Return parameters
    return p

def rotc_rated_hr_algorithm(parameters, printing=True):
    """
    This function performs the classic hospital/residents algorithm to match ROTC rated cadets. The ideas of
    reserved/fixed AFSCs is also imposed here.
    :param parameters: dictionary of model parameters pertaining to the ROTC rated matching process
    :param printing: whether to print status updates
    :return: solution augmented parameters
    """

    # Shorthand
    p = parameters

    # Dictionary to keep track of what AFSC choice in their list the cadets are proposing to
    cadet_proposal_choice = {i: 0 for i in p['I']} # Initially all propose to their top Rated preference!

    # Begin the simple Hospital/Residents Algorithm
    total_rejections = {j: 0 for j in p['J^Rated']}  # Number of rejections for each rated AFSC
    total_matched = {j: 0 for j in p['J^Rated']}  # Number of accepted cadets for each rated AFSC
    exhausted_cadets = []  # Will contain the cadets that have exhausted (been rejected by) all of their preferences
    iteration = 1 # First iteration of the algorithm
    while sum([total_matched[j] for j in p['J^Rated']]) + len(exhausted_cadets) < p['N']:  # Stopping conditions

        # Cadets propose to their top choice that hasn't been rejected
        proposals = {i: p['rated_only_preferences'][i][
            cadet_proposal_choice[i]] if i not in exhausted_cadets else p['M'] for i in p['I']}
        proposal_array = np.array([proposals[i] if i in p['I'] else p['M'] for i in p['I']])
        counts = {p['afscs'][j]: len(np.where(proposal_array == j)[0]) for j in p['J^Rated']}

        # Initialize matches information for this iteration
        total_matched = {j: 0 for j in p['J^Rated']}

        # AFSCs accept their best cadets and reject the others
        for j in p['J^Rated']:

            # Loop through their preferred cadets from top to bottom
            iteration_rejections = 0
            for i in p['afscs_preferences'][j]:

                # If the cadet is proposing to this AFSC, we have two options
                if proposals[i] == j:

                    # We haven't hit capacity, so we accept this cadet
                    if total_matched[j] < p['total_slots'][j]:
                        total_matched[j] += 1

                    # We're at capacity, so we reject this cadet
                    else:

                        # Essentially "delete" the preference from the cadet's list
                        cadet_proposal_choice[i] += 1
                        proposals[i] = p['M']  # index of the unmatched AFSC (*)

                        # Collect additional information
                        if printing:
                            iteration_rejections += 1
                            total_rejections[j] += 1

        # Check exhausted cadets
        exhausted_cadets = []
        for i in p['I']:
            if cadet_proposal_choice[i] >= p['num_eligible'][i]:
                exhausted_cadets.append(i)

        # Print statement
        if printing:
            print("\nIteration", iteration)
            print('Proposals:', counts)
            print('Matched', {p['afscs'][j]: total_matched[j] for j in p['J^Rated']})
            print('Rejected', {p['afscs'][j]: total_rejections[j] for j in p['J^Rated']})
            total_matched_sum = sum([total_matched[j] for j in p['J^Rated']])
            exhausted_sum = len(exhausted_cadets)
            print("Total Exhausted (" + str(exhausted_sum) + ") + Total Matched (" + str(total_matched_sum)
                  + ") = Total Accounted (" + str(exhausted_sum + total_matched_sum) + ") || N: " + str(p['N']) + ".")

        iteration += 1  # Next iteration!

    # Create "Matched" and "Reserved" solution arrays
    p['matches'], p['reserves'] = np.array([p['M'] for _ in p['I']]), np.array([p['M'] for _ in p['I']])
    p['J^Fixed'], p['J^Reserved'] = {}, {}
    for i in p['I']:
        j = proposal_array[i]
        if j in p['J^Rated']:
            if p['cadet_first_choice'][i] == p['afscs'][j]:
                p['matches'][i] = j
                p['J^Fixed'][i] = j
            else:
                p['reserves'][i] = j
                choice = np.where(p['cadets_preferences'][i] == j)[0][0]
                p['J^Reserved'][i] = p['cadets_preferences'][i][:choice + 1]




    return p

def rotc_rated_alternates_algorithm(parameters, printing=True):
    """
    This function takes in the results from the ROTC Rated HR algorithm and then augments it with the
    alternate list logic
    :param parameters: dictionary of model parameters pertaining to the ROTC rated matching process
    :param printing: whether to print status updates
    :return: augmented parameters containing the solution w/alternates
    """

    # Shorthand
    p = parameters

    # Start with a full list of cadets eligible for each AFSC
    possible_cadets = {j: list(p['afscs_preferences'][j]) for j in p['J^Rated']}

    # Used for stopping conditions
    last_reserves, last_matches, last_alternates_h = np.array([1000 for _ in p['J^Rated']]), \
                                                     np.array([1000 for _ in p['J^Rated']]), \
                                                     np.array([1000 for _ in p['J^Rated']])

    # Main algorithm
    iteration, iterating = 0, True
    while iterating:

        # Set of cadets reserved or matched to each rated AFSC
        p['I^Reserved'] = {j: np.array([i for i in p['J^Reserved'] if p['reserves'][i] == j]) for j in p['J^Rated']}
        p['I^Matched'] = {j: np.array([i for i in p['J^Fixed'] if p['matches'][i] == j]) for j in p['J^Rated']}

        # Number of alternates (number of reserved slots)
        num_reserved = {j: len(p['I^Reserved'][j]) for j in p['J^Rated']}

        # Need to determine who falls into each category of alternates
        hard_alternates = {j: [] for j in p['J^Rated']}
        soft_alternates = {j: [] for j in p['J^Rated']}
        alternates = {j: [] for j in p['J^Rated']}  # all the cadets ordered here

        # Loop through each rated AFSC to determine alternates
        for j in p['J^Rated']:

            # Loop through each cadet in order of the AFSC's preference
            for i in p['afscs_preferences'][j]:

                # Assume this cadet is "next in line" until proven otherwise
                next_in_line = True

                # Is the cadet already fixed to something else?
                if i in p['J^Fixed']:
                    next_in_line = False
                    if i in possible_cadets[j]:
                        possible_cadets[j].remove(i)

                # Is this cadet reserved for something?
                if i in p['J^Reserved']:

                    # Where did the cadet rank their reserved AFSC?
                    reserved_choice = np.where(p['rated_only_preferences'][i] == p['reserves'][i])[0][0]

                    # Where did the cadet rank this rated AFSC?
                    this_choice = np.where(p['rated_only_preferences'][i] == j)[0][0]

                    # If they're already reserved for this rated AFSC or something better, they're not considered
                    if reserved_choice <= this_choice:
                        next_in_line = False
                        if i in possible_cadets[j]:
                            possible_cadets[j].remove(i)

                # If this cadet is next in line (and we still have alternates to assign)
                if next_in_line and len(hard_alternates[j]) < num_reserved[j]:
                    alternates[j].append(i)

                    # Loop through the cadet's preferences:
                    for j_c in p['cadets_preferences'][i]:

                        # Determine what kind of alternate this cadet is
                        if j_c == j:  # Hard Rated Alternate
                            hard_alternates[j].append(i)
                            break
                        elif j_c in p['J^Rated']:
                            if i in possible_cadets[j_c]:  # Soft Rated Alternate
                                soft_alternates[j].append(i)
                                break
                            else:  # Can't be matched, go to next preference
                                continue
                        else:  # Soft Non-Rated Alternate
                            soft_alternates[j].append(i)
                            break

                # We've run out of hard alternates to assign (thus, we're done assigning alternates)
                elif len(hard_alternates[j]) >= num_reserved[j]:
                    if i in possible_cadets[j]:
                        possible_cadets[j].remove(i)

        # Loop through each AFSC to potentially turn "reserved" slots into "matched" slots
        for j in p['J^Rated']:

            # Loop through each cadet in order of the AFSC's preference
            for i in p['afscs_preferences'][j]:

                # Does this cadet have a reserved slot for something?
                if i in p['J^Reserved']:

                    # Is this cadet reserved for this AFSC?
                    if p['reserves'][i] == j:

                        # Determine if there's any possible way this cadet might not be matched to this AFSC
                        inevitable_match = True
                        for j_c in p['J^Reserved'][i][:-1]:  # Loop through all more preferred AFSCs than this one
                            if j_c not in p['J^Rated']:  # Non-Rated
                                inevitable_match = False
                            else:  # Rated
                                if i in alternates[j_c]:  # They're an alternate for something more preferred
                                    inevitable_match = False
                                else:  # They're not an alternate for that more preferred AFSC...
                                    if i in possible_cadets[j_c]:  # ...they cannot be matched to that AFSC
                                        possible_cadets[j_c].remove(i)  # Remove this cadet as a possibility!

                        # If still inevitable, change from reserved to fixed
                        if inevitable_match:
                            p['J^Fixed'][i], p['matches'][i], p['reserves'][i] = j, j, p['M']
                            p['J^Reserved'].pop(i)

                # This cadet cannot receive this AFSC
                if i not in alternates[j] and i in possible_cadets[j]:
                    possible_cadets[j].remove(i)

        # Print Statement
        if printing:
            print("\nIteration", iteration)
            print("Possible", {p['afscs'][j]: len(possible_cadets[j]) for j in p['J^Rated']})
            print("Matched", {p['afscs'][j]: len(p['I^Matched'][j]) for j in p['J^Rated']})
            print("Reserved", {p['afscs'][j]: len(p['I^Reserved'][j]) for j in p['J^Rated']})
            print("Alternates (Hard)", {p['afscs'][j]: len(hard_alternates[j]) for j in p['J^Rated']})
            print("Alternates (Soft)", {p['afscs'][j]: len(soft_alternates[j]) for j in p['J^Rated']})

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
    p['alternates_hard'] = np.array([p['M'] for _ in p['I']])
    p['alternates_soft'] = np.array([p['M'] for _ in p['I']])
    for i in p['I']:  # Loop through all rated cadets
        for j in p['rated_only_preferences'][i]:  # Loop through rated preferences in order
            if i in hard_alternates[j]:
                p['alternates_hard'][i] = j
            elif i in soft_alternates[j]:
                p['alternates_soft'][i] = j
                break  # Next cadet

    # Return updated parameters (and alternate lists)
    return p

def rotc_rated_match(filename, afsc_quotas, printing=True):
    """
    This is the main function to match ROTC rated cadets to their AFSCs prior to the main "One Market" model
    match. It calls all necessary functions to process the information and export it back to excel
    :param filename: name/path to file
    :param afsc_quotas: dictionary of ROTC capacities for the rated AFSCs
    :param printing: if we should print status updates
    :return: None. Export results to excel
    """

    # Load in data
    df = afccp.core.globals.import_csv_data(filename)

    # Initialize parameter dictionary
    parameters = {'rated_afscs': np.array([afsc for afsc in afsc_quotas]), 'cadets': np.array(df['Cadet']),
                  'afscs': np.array(['11U', '11XX_R', '11XX_U', '12XX', '13B', '13H', '13M', '13N',
                          '13S1S', '14F', '14N', '14N1S', '15A', '15W', '17S1S', '17X',
                          '21A', '21M', '21R', '31P', '32EXA', '32EXC', '32EXE', '32EXF',
                          '32EXG', '32EXJ', '35P', '38F', '61C', '61D', '62E1A1S', '62E1B1S',
                          '62E1C1S', '62E1E1S', '62E1G1S', '62E1H1S', '62E1I1S', '62EXA',
                          '62EXB', '62EXC', '62EXE', '62EXG', '62EXH', '62EXI', '63A', '63A1S',
                          '64P', '65F'])}

    # Additional information
    parameters['N'], parameters['M'] = len(parameters['cadets']), len(parameters['afscs'])
    parameters['I'], parameters['J'] = np.arange(parameters['N']), np.arange(parameters['M'])
    parameters['J^Rated'] = np.array([np.where(afsc == parameters['afscs'])[0][0] for afsc in parameters['rated_afscs']])
    parameters['total_slots'] = {j: afsc_quotas[parameters['afscs'][j]] for j in parameters['J^Rated']}

    if printing:
        print("\nProcessing Data...\n")

    # Process data
    parameters = process_data_in_parameters(df, parameters)

    if printing:
        print("\nROTC Rated Board Algorithm...")

    # Run the rated board algorithm
    parameters = rotc_rated_hr_algorithm(parameters, printing)

    if printing:
        print("\nROTC Rated Alternates Algorithm...\n")

    # Run the rated alternates algorithm
    parameters = rotc_rated_alternates_algorithm(parameters, printing)

    # Adjust solution arrays and add them into dataframe
    df['Matches'] = [parameters['afscs'][j] if j in parameters['J'] else '' for j in parameters['matches']]
    df['Reserves'] = [parameters['afscs'][j] if j in parameters['J'] else '' for j in parameters['reserves']]
    df['Alternates (H)'] = [parameters['afscs'][j] if j in parameters['J'] else '' for j in
                            parameters['alternates_hard']]
    df['Alternates (S)'] = [parameters['afscs'][j] if j in parameters['J'] else '' for j in
                            parameters['alternates_soft']]

    # Export back to csv
    df.to_csv(filename, index=False)





