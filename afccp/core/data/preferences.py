import numpy as np
import pandas as pd
import copy


# Preference functions
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
        sorted_indices = indices[sorted_init][:parameters['P']]  # Take the top "P" preferences (all preferences)

        # Put the utilities and preferences in the correct spots
        np.put(utilities_array[i, :], np.arange(len(sorted_indices)), parameters['utility'][i, :][sorted_indices])
        np.put(preferences[i, :], np.arange(len(sorted_indices)), parameters['afscs'][sorted_indices])

    return preferences, utilities_array


def get_utility_preferences_from_preference_array(parameters):
    """
    Convert cadet preference matrix to preference columns of AFSC names and calculate utility columns.

    This function takes the cadet preference matrix (NxM) where cadet "ranks" are specified and converts it to
    preference columns (NxP) of AFSC names, where P is the number of AFSC preferences for each cadet. It uses
    this preference information alongside the utility dataframe (NxP) to extract the utility columns (NxP) as well.

    Args:
        parameters (dict): A dictionary containing the following elements:
            - "c_pref_matrix" (numpy.ndarray): Cadet preference matrix (NxM) with cadet ranks.
            - "P" (int): Number of AFSC preferences for each cadet.
            - "N" (int): Total number of cadets.
            - "I" (list): List of cadet indices.
            - "M" (int): Total number of AFSCs.
            - "afscs" (numpy.ndarray): Array of AFSC names.
            - "num_util" (int): Number of utility values to extract.
            - "utility" (numpy.ndarray): Utility dataframe (NxM) containing utility values for cadets and AFSCs.

    Returns:
        tuple: A tuple containing two elements:
            - preferences (numpy.ndarray): Cadet preference columns (NxP) with AFSC names.
            - utilities_array (numpy.ndarray): Utility columns (NxP) for each cadet and AFSC preference.

    """

    # Shorthand
    p = parameters

    # Initialize data
    preference_matrix = copy.deepcopy(p["c_pref_matrix"])
    preferences = np.array([[" " * 10 for _ in range(p['P'])] for _ in range(p['N'])])
    utilities_array = np.zeros([p['N'], p['P']])
    for i in p['I']:

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


def update_first_choice_cadet_utility_to_one(parameters, printing=True):

    # Shorthand
    p = parameters

    # Loop through each cadet and make their first choice AFSC have a utility of 1
    fixed_cadets = []
    for i in p['I']:

        # If this cadet does not have any preferences, we skip them (must be an OTS candidate)
        if len(p['cadet_preferences'][i]) == 0:
            continue

        # Fix the first choice
        if p['utility'][i, p['cadet_preferences'][i][0]] != 1:
            p['utility'][i, p['cadet_preferences'][i][0]] = 1
            fixed_cadets.append(i)

    if printing:
        print_str = f'Fixed {len(fixed_cadets)} first choice cadet utility values to 100%.\nCadets: {fixed_cadets}'
        print(print_str)

    return p['utility']


def convert_utility_matrices_preferences(parameters, cadets_as_well=False):
    """
    This function converts the cadet and AFSC utility matrices into the preference dataframes
    """
    p = parameters

    # Loop through each AFSC to get their preferences
    if 'afsc_utility' in p:
        p["a_pref_matrix"] = np.zeros([p["N"], p["M"]]).astype(int)
        for j in p["J"]:

            # Sort the utilities to get the preference list
            utilities = p["afsc_utility"][:, j]
            sorted_indices = np.argsort(utilities)[::-1]
            preferences = np.argsort(sorted_indices)
            p["a_pref_matrix"][:, j] = preferences

    # Loop through each cadet to get their preferences
    if cadets_as_well:
        p["c_pref_matrix"] = np.zeros([p["N"], p["M"]]).astype(int)
        for i in p["I"]:

            # Sort the utilities to get the preference list
            utilities = p["cadet_utility"][i, :p["M"]]
            sorted_indices = np.argsort(utilities)[::-1]
            preferences = np.argsort(
                sorted_indices) + 1  # Add 1 to change from python index (at 0) to rank (start at 1)
            p["c_pref_matrix"][i, :] = preferences
    return p


def generate_fake_afsc_preferences(parameters, value_parameters=None, fix_cadet_eligibility=False):
    """
    This function generates fake AFSC utilities/preferences using AFOCD, merit, cadet preferences etc.
    :param fix_cadet_eligibility: if we want to fix the cadet preferences based on eligibility
    :param value_parameters: set of cadet/AFSC weight and value parameters
    :param parameters: cadet/AFSC fixed data
    :return: parameters
    """
    # Shorthand
    p, vp = parameters, value_parameters

    # Create AFSC Utility Matrix
    p["afsc_utility"] = np.zeros([p["N"], p["M"]])
    if vp is None:

        # If we don't have a set of value_parameters, we just make some assumptions
        weights = {"Merit": 80, "Tier 1": 100, "Tier 2": 50, "Tier 3": 30, "Tier 4": 0, "Utility": 60}
        for objective in weights:
            if objective.lower() in p:

                if objective == "Merit":
                    merit = np.tile(p['merit'], [p["M"], 1]).T
                    p["afsc_utility"] += merit * weights[objective]
                else:
                    p["afsc_utility"] += p[objective.lower()][:, :p["M"]] * weights[objective]
    else:

        # If we do have a set of value_parameters, we incorporate them
        for objective in ['Merit', 'Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Utility']:
            if objective in vp['objectives']:

                k = np.where(vp['objectives'] == objective)[0][0]
                if objective == "Merit":
                    merit = np.tile(p['merit'], [p["M"], 1]).T
                    p["afsc_utility"] += merit * vp['objective_weight'][:, k].T
                else:
                    p["afsc_utility"] += p[objective.lower()][:, :p["M"]] * vp['objective_weight'][:, k].T
    p["afsc_utility"] *= p["eligible"]  # They have to be eligible!

    if fix_cadet_eligibility:  # We just start over from scratch with cadet preferences
        p['c_pref_matrix'] = np.zeros([p["N"], p["M"]]).astype(int)
        p['cadet_preferences'] = {}

        # Add a column to the eligible matrix for the unmatched AFSC (just to get the below multiplication to work)
        eligible = copy.deepcopy(p['eligible'])
        eligible = np.hstack((eligible, np.array([[0] for _ in range(p["N"])])))
        p['cadet_utility'] *= eligible  # They have to be eligible!
        for i in p["I"]:

            # Sort the utilities to get the preference list
            utilities = p["cadet_utility"][i, :p["M"]]
            ineligible_indices = np.where(eligible[i, :p["M"]] == 0)[0]
            sorted_indices = np.argsort(utilities)[::-1][:p['M'] - len(ineligible_indices)]
            p['cadet_preferences'][i] = sorted_indices

            # Since 'cadet_preferences' is an array of AFSC indices, we can do this
            p['c_pref_matrix'][i, p['cadet_preferences'][i]] = np.arange(1, len(p['cadet_preferences'][i]) + 1)

    # Create AFSC Preferences
    p["a_pref_matrix"] = np.zeros([p["N"], p["M"]]).astype(int)
    p['afsc_preferences'] = {}
    for j in p["J"]:

        # Loop through each cadet one more time to fix them on the AFSC list
        for i in p['I']:
            if p['c_pref_matrix'][i, j] == 0:
                p['afsc_utility'][i, j] = 0

        # Sort the utilities to get the preference list
        utilities = p["afsc_utility"][:, j]
        ineligible_indices = np.where(utilities == 0)[0]
        sorted_indices = np.argsort(utilities)[::-1][:p['N'] - len(ineligible_indices)]
        p['afsc_preferences'][j] = sorted_indices

        # Since 'afsc_preferences' is an array of AFSC indices, we can do this
        p['a_pref_matrix'][p['afsc_preferences'][j], j] = np.arange(1, len(p['afsc_preferences'][j]) + 1)

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
    p["afsc_utility"] = np.zeros([p['N'], p['M']])
    for j in p['J']:
        p['afsc_utility'][p['afsc_preferences'][j], j] = \
            np.arange(1, len(p['afsc_preferences'][j]) + 1)[::-1] / len(p['afsc_preferences'][j])

    return p


def generate_rated_data(parameters):
    """
    This function generates a dataset of ROTC Rated "interest" and OM data and also USAFA Rated OM data
    """

    # Shorthand
    p = parameters
    if 'usafa' not in p['Rated Cadets']:
        return p  # No Rated AFSCs to add

    # ROTC Rated Interest Matrix (Only generate random data if we don't already have it)
    if 'rr_interest_matrix' not in p:
        p['rr_interest_matrix'] = np.array(
            [[np.random.choice(['High', 'Med', 'Low', 'None']) for _ in
              p['afscs_acc_grp']['Rated']] for _ in range(len(p['Rated Cadets']['rotc']))])

    # Loop through each SOC to generate OM data (based on AFSC preferences) if we don't already have it
    dataset_dict = {soc: f'{soc[0]}r_om_matrix' for soc in p['SOCs']}
    for soc in dataset_dict:
        dataset = dataset_dict[soc]  # SOC specific dataset name for Rated OM data

        # Collect useful information on the Rated cadets (for this SOC) and the Rated AFSCs
        rated_cadets, rated_afscs = p['Rated Cadets'][soc], p['afscs_acc_grp']['Rated']
        num_rated_cadets, num_rated_afscs = len(rated_cadets), len(rated_afscs)
        if dataset not in p:  # Only generate data if we don't already have it

            # Loop through each Rated AFSC to construct their OM data
            p[dataset] = np.zeros([num_rated_cadets, num_rated_afscs])
            for afsc_index, afsc in enumerate(p['afscs_acc_grp']['Rated']):
                j = np.where(p['afscs'] == afsc)[0][0]

                # Percentiles sorted from top (1) to bottom (0)
                rated_afsc_eligible_cadets = np.intersect1d(p['I^E'][j], rated_cadets)
                percentiles = \
                    ((np.arange(len(rated_afsc_eligible_cadets)) + 1) / (len(rated_afsc_eligible_cadets)))[::-1]

                # Loop through each cadet in order of preference and give them the highest percentile based on SOC
                count = 0
                for i in p['afsc_preferences'][j]:
                    if i in rated_afsc_eligible_cadets:
                        cadet_index = p['Rated Cadet Index Dict'][soc][i]
                        p[dataset][cadet_index, afsc_index] = percentiles[count]
                        count += 1
    return p


def determine_soc_rated_afscs(soc, all_rated_afscs):

    # Rated AFSCs for this SOC
    other_letters = [l for l in ['_U', '_R', '_O'] if l != f'_{soc[0].upper()}']
    rated_afscs = []
    for afsc in all_rated_afscs:
        include = True
        for l in other_letters:
            if l in afsc:
                include = False
                break
        if include:
            rated_afscs.append(afsc)

    return rated_afscs


def construct_rated_preferences_from_om_by_soc(parameters):
    """
    This method takes the two OM Rated matrices (from both SOCs) and then zippers them together to
    create a combined "1-N" list for the Rated AFSCs. The AFSC preference matrix is updated as well as the
    AFSC preference lists
    """

    # Shorthand
    p = parameters

    # Loop through all the parameters we need and make sure we have everything
    for key in ['rr_om_matrix', 'ur_om_matrix', 'afsc_preferences', 'a_pref_matrix']:
        if key not in p:
            raise ValueError("Error. Parameter '" + key + "' not in the parameter dictionary. Please make sure you have"
                                                          "AFSC preferences and Rated preferences from both sources "
                                                          "of commissioning.")

    # Need to construct a "combined Rated OM" matrix with ALL cadets which also contains 0 if the cadet is ineligible
    all_rated_afscs = p['afscs_acc_grp']["Rated"]
    dataset_dict = {soc: f'{soc[0]}r_om_matrix' for soc in p['SOCs']}
    cadets_dict = {soc: f'{soc[0]}r_om_cadets' for soc in p['SOCs']}
    combined_rated_om = {afsc: np.zeros(p['N']) for afsc in all_rated_afscs}

    # Loop through all sources of commissioning
    rated_cadets, rated_cadet_index_dict = {}, {}
    for soc in p['SOCs']:

        # Rated AFSCs for this SOC
        rated_afscs = determine_soc_rated_afscs(soc, all_rated_afscs)

        # Rated cadets for this SOC
        rated_cadets[soc] = p[cadets_dict[soc]]

        # Loop through SOC-specific rated AFSCs
        for afsc_index, afsc in enumerate(rated_afscs):

            # Need to re-normalize OM to make it fair across SOCs
            nonzero_indices_in_soc_dataset = np.where(p[dataset_dict[soc]][:, afsc_index] != 0)[0]
            num_eligible = len(nonzero_indices_in_soc_dataset)
            sorted_eligible_indices = np.argsort(p[dataset_dict[soc]][:, afsc_index])[::-1][:num_eligible]
            ordered_cadets = rated_cadets[soc][sorted_eligible_indices]
            combined_rated_om[afsc][ordered_cadets] = np.arange(1, num_eligible + 1)[::-1] / (num_eligible + 1)

    # Sort the OM to convert it into a 1-N
    for afsc in all_rated_afscs:
        j = np.where(p['afscs'] == afsc)[0][0]

        # Get AFSC preferences (list of cadet indices in order)
        ineligibles = np.where(combined_rated_om[afsc] == 0)[0]
        num_ineligible = len(ineligibles)  # Ineligibles are going to be at the bottom of the list (and we remove them)
        p['afsc_preferences'][j] = np.argsort(combined_rated_om[afsc])[::-1][:p['N'] - num_ineligible]

        # Reset this AFSC's "preference matrix" column
        p['a_pref_matrix'][:, j] = np.zeros(p['N'])

        # Since 'afsc_preferences' is an array of cadet indices, we can do this
        p['a_pref_matrix'][p['afsc_preferences'][j], j] = np.arange(1, len(p['afsc_preferences'][j]) + 1)

    return p  # Return updated parameters


def remove_ineligible_cadet_choices(parameters, printing=False):
    """
    This function removes ineligible choices from the cadets and from the AFSCs based on the qualification matrix
    """

    # Shorthand
    p = parameters

    # This is my final correction for preferences to make it all match up
    num_removals = 0
    lines = []
    for i in p['I']:
        for j in p['J']:
            afsc = p['afscs'][j]  # AFSC name

            # Cadet not eligible based on degree qualification matrix
            if i not in p['I^E'][j]:

                # AFSC is currently in the cadet's preference list
                if p['c_pref_matrix'][i, j] != 0:
                    p['c_pref_matrix'][i, j] = 0
                    num_removals += 1
                    lines.append('Edit ' + str(num_removals) + ': Cadet ' + str(i) + ' not eligible for ' + afsc +
                                 ' based on degree qualification matrix but the AFSC was in the cadet preference list. '
                                 'c_pref_matrix position (' + str(i) + ", " + str(j) + ') set to 0.')

                # Cadet is currently in the AFSC's preference list
                if p['a_pref_matrix'][i, j] != 0:
                    p['a_pref_matrix'][i, j] = 0
                    num_removals += 1
                    lines.append('Edit ' + str(num_removals) + ': Cadet ' + str(i) + ' not eligible for ' + afsc +
                                 ' based on degree qualification matrix but the cadet was in the AFSC preference list. '
                                 'a_pref_matrix position (' + str(i) + ", " + str(j) + ') set to 0.')

            # Cadet is currently eligible based on degree qualification matrix
            else:

                # If there's already an ineligible tier in this AFSC, we use it in case we need to adjust qual matrix
                if "I = 0" in p['Deg Tiers'][j]:
                    val = "I" + str(p['t_count'][j])
                else:
                    val = "I" + str(p['t_count'][j] + 1)

                # The cadet is not in the AFSC's preference list
                if p['a_pref_matrix'][i, j] == 0:

                    # The AFSC is in the cadet's preference list
                    if p['c_pref_matrix'][i, j] != 0:
                        p['c_pref_matrix'][i, j] = 0
                        num_removals += 1
                        lines.append('Edit ' + str(num_removals) + ': Cadet ' + str(i) + ' eligible for ' + afsc +
                                     ' based on degree qualification matrix but the cadet was not in the AFSC preference list. '
                                     'c_pref_matrix position (' + str(i) + ", " + str(j) +
                                     ') set to 0 and qual position adjusted to ' + val + ".")

                    # The AFSC is not in the cadet's preference list
                    else:
                        num_removals += 1
                        lines.append('Edit ' + str(num_removals) + ': Cadet ' + str(i) + ' eligible for ' + afsc +
                                     ' based on degree qualification matrix but the cadet/afsc pairing was in neither '
                                     'matrix (a_pref_matrix or c_pref_matrix). Both a_pref_matrix and c_pref_matrix '
                                     'position (' + str(i) + ", " + str(j) + ') were already 0 so only qual position was '
                                                                             'adjusted to ' + val + ".")

                    # Force ineligibility in the qual matrix as well
                    p['qual'][i, j] = val
                    p['eligible'][i, j] = 0
                    p['ineligible'][i, j] = 0

                # The AFSC is not in the cadet's preference list
                if p['c_pref_matrix'][i, j] == 0:

                    # The cadet is in the AFSC's preference list
                    if p['a_pref_matrix'][i, j] != 0:
                        p['a_pref_matrix'][i, j] = 0
                        num_removals += 1
                        lines.append('Edit ' + str(num_removals) + ': Cadet ' + str(i) + ' eligible for ' + afsc +
                                     ' based on degree qualification matrix but the AFSC was not in the cadet preference list. '
                                     'a_pref_matrix position (' + str(i) + ", " + str(j) +
                                     ') set to 0 and qual position adjusted to ' + val + ".")

                    # The cadet is not in the AFSC's preference list
                    else:
                        num_removals += 1
                        lines.append('Edit ' + str(num_removals) + ': Cadet ' + str(i) + ' eligible for ' + afsc +
                                     ' based on degree qualification matrix but the cadet/afsc pairing was in neither '
                                     'matrix (a_pref_matrix or c_pref_matrix). Both a_pref_matrix and c_pref_matrix '
                                     'position (' + str(i) + ", " + str(
                                         j) + ') were already 0 so only qual position was '
                                              'adjusted to ' + val + ".")

                    # Force ineligibility in the qual matrix as well
                    p['qual'][i, j] = val
                    p['eligible'][i, j] = 0
                    p['ineligible'][i, j] = 0

    # Print statement
    if printing:
        for line in lines:
            print(line)
        print(num_removals, "total adjustments.")
    return p  # Return parameters


def update_preference_matrices(parameters):
    """
    This method takes the preference arrays and re-creates the preference
    matrices based on the cadets/AFSCs on each list
    """
    # Shorthand
    p = parameters

    # Update the cadet preference matrix (c_pref_matrix)
    if 'cadet_preferences' in p:

        # Since 'cadet_preferences' is an array of AFSC indices, we can do this
        p['c_pref_matrix'] = np.zeros([p['N'], p['M']]).astype(int)
        for i in p['I']:

            # If this cadet does not have any preferences, we skip them (must be an OTS candidate)
            if len(p['cadet_preferences'][i]) == 0:
                continue
            p['c_pref_matrix'][i, p['cadet_preferences'][i]] = np.arange(1, len(p['cadet_preferences'][i]) + 1)

    # Update the AFSC preference matrix (a_pref_matrix)
    if 'afsc_preferences' in p:

        # Since 'afsc_preferences' is an array of cadet indices, we can do this
        p['a_pref_matrix'] = np.zeros([p['N'], p['M']]).astype(int)
        for j in p['J']:
            p['a_pref_matrix'][p['afsc_preferences'][j], j] = np.arange(1, len(p['afsc_preferences'][j]) + 1)

    return p


def update_cadet_utility_matrices(parameters):
    """
    This method takes in the "Util_1 -> Util_P" columns from Cadets.csv and updates the utility matrices
    accordingly
    """

    # Shorthand
    p = parameters

    # Simple error checking
    required_parameters = ['cadet_preferences', 'c_utilities']
    for param in required_parameters:
        if param not in p:
            raise ValueError("Error. Parameter '" + param + "' not in instance parameters. It is required.")

    # Initialize matrix (reported cadet utility)
    p['utility'] = np.zeros((p['N'], p['M'] + 1))  # Additional column for unmatched cadets

    # Loop through each cadet
    for i in p['I']:

        # If this cadet does not have any preferences, we skip them (must be an OTS candidate)
        if len(p['cadet_preferences'][i]) == 0:
            continue

        # List of ordered AFSC indices (by cadet preference) up to the number of utilities
        afsc_indices = p['cadet_preferences'][i][:p['num_util']]

        # Fill in the reported utilities
        p['utility'][i, afsc_indices] = p['c_utilities'][i, :len(afsc_indices)]

    # Create the "cadet_utility" matrix by re-calculating utility based on ordinal rankings
    if 'last_afsc' in p:
        p = create_final_cadet_utility_matrix_from_new_formula(p)
    else:
        p = create_new_cadet_utility_matrix(p)

    return p


def create_new_cadet_utility_matrix(parameters):
    """
    This function creates a new "cadet_utility" matrix using normalized preferences and
    the original "utility" matrix.
    """

    # Shorthand
    p = parameters

    # Initialize matrix
    p['cadet_utility'] = np.zeros([p['N'], p['M'] + 1])  # Additional column for unmatched cadets

    # Loop through each cadet
    for i in p['I']:

        # 1, 2, 3, 4, ..., N  (Pure rankings)
        rankings = np.arange(p['num_cadet_choices'][i]) + 1

        # 1, 0.8, 0.6, 0.4, ..., 1 / N  (Scale rankings to range from 1 to 0)
        normalized_rankings = 1 - (rankings / np.max(rankings)) + (1 / np.max(rankings))

        # 1, 0.75, 0, 0, etc. (From utility matrix)
        original_utilities = p['utility'][i, p['cadet_preferences'][i]]

        # "New" Utilities based on a weighted sum of normalized rankings and original utilities
        new_utilities = np.around(0.5 * normalized_rankings + 0.5 * original_utilities, 4)

        # Put these new utilities back into the utility matrix in the appropriate spots
        p['cadet_utility'][i, p['cadet_preferences'][i]] = new_utilities

    # Return parameters
    return p


def create_final_cadet_utility_matrix_from_new_formula(parameters):
    """
    This function creates a new "cadet_utility" matrix using normalized preferences and
    the original "utility" matrix and some other stuff.
    """

    # Shorthand
    p = parameters

    # Initialize matrix (0.1 for everyone by default as indifference)
    p['cadet_utility'] = np.ones([p['N'], p['M'] + 1]) * 0.1  # Additional column for unmatched cadets

    # Loop through each cadet
    for i in p['I']:

        # AFSCs the cadet is eligible for and selected (ordered appropriately)
        intersection = np.intersect1d(p['J^Selected'][i], p['cadet_preferences'][i])
        intersection = np.array([j for j in p['cadet_preferences'][i] if j in intersection])
        num_selected = len(intersection)

        # Skip this cadet if they don't have any eligible choices
        if num_selected == 0:
            continue

        # 1, 2, 3, 4, ..., N  (Pure rankings)
        rankings = np.arange(num_selected) + 1

        # 1, 0.8, 0.6, 0.4, ..., 1 / N  (Scale rankings to range from 1 to 0)
        normalized_rankings = 1 - (rankings / np.max(rankings)) + (1 / np.max(rankings))

        # Create dictionary of normalized ordinal rankings
        norm_ord_rankings_dict = {j: normalized_rankings[index] for index, j in enumerate(intersection)}

        # Loop through all AFSCs that the cadet is eligible for
        for j in p['cadet_preferences'][i]:

            # A: AFSC is NOT the LAST choice
            a = (j != p['J^Last Choice'][i]) * 1

            # B: AFSC is NOT in the bottom 3 choices
            b = ((j not in p['J^Bottom 2 Choices'][i]) and (j != p['J^Last Choice'][i])) * 1

            # C: AFSC was selected as a preference
            c = (j in p['J^Selected'][i]) * 1

            # D: AFSC was selected as a preference and has a utility assigned
            d = (p['utility'][i, j] > 0) * 1

            # X: Normalized ordinal ranking of the AFSC
            x = norm_ord_rankings_dict[j] if j in norm_ord_rankings_dict else 0

            # Y: Utility value the cadet assigned to the AFSC
            y = p['utility'][i, j]

            # Execute the formula and load it into the cadet utility matrix
            p['cadet_utility'][i, j] = 0.05*a + 0.05*b + 0.9*(0.3*c*x + 0.7*d*y)

    # Return parameters
    return p


def fill_remaining_preferences(parameters):
    """

    :param parameters:
    :return:
    """

    # Shorthand
    p = parameters

    # Loop through all cadets
    for i in p['I']:

        # We don't fill in remaining preferences for OTS!
        if 'I^OTS' in p:
            if i in p['I^OTS']:
                continue  # They are ineligible for anything they didn't select

        # Loop through all "indifferent" AFSCs that they are eligible for
        pref_num = len(p['cadet_preferences'][i]) + 1
        for j in p['J']:

            # The AFSC is not in the cadet's preferences and it's not in the bottom choices
            if j not in p['cadet_preferences'][i] and \
                    j not in p['J^Bottom 2 Choices'][i] and j != p['J^Last Choice'][i]:
                p['c_pref_matrix'][i, j] = pref_num
                pref_num += 1

        # Loop through bottom 2 choices
        for j in p['J^Bottom 2 Choices'][i]:
            p['c_pref_matrix'][i, j] = pref_num
            pref_num += 1

        # Set last choice preference if applicable
        if p['J^Last Choice'][i] != p['M']:
            p['c_pref_matrix'][i, p['J^Last Choice'][i]] = pref_num

    return p


def modify_rated_cadet_lists_based_on_eligibility(parameters, printing=True):

    # Shorthand
    p = parameters

    # At least one rated preference for rated eligible
    for soc in p['SOCs']:
        cadets_to_remove = []
        cadet_indices_in_matrix = []
        if soc in p['Rated Cadets']:
            for idx, i in enumerate(p['Rated Cadets'][soc]):
                if len(p['Rated Choices'][soc][i]) == 0:
                    cadets_to_remove.append(i)
                    cadet_indices_in_matrix.append(idx)

        # Remove cadets from set of rated cadets for this SOC
        cadets_to_remove = np.array(cadets_to_remove)
        p['Rated Cadets'][soc] = p['Rated Cadets'][soc][~np.isin(p['Rated Cadets'][soc], cadets_to_remove)]

        # Remove the cadet rows by position in the matrix
        cadet_indices_in_matrix = np.array(cadet_indices_in_matrix)
        if len(cadet_indices_in_matrix) > 0:
            parameter = f'{soc[0]}r_om_matrix'
            p[parameter] = np.delete(p[parameter], cadet_indices_in_matrix, axis=0)

            # Print results
            if printing:
                print_str = f"We removed {len(cadets_to_remove)} cadets from {soc.upper()}'s rated cadet list.\n" \
                            f"These were cadets {cadets_to_remove}.\nWe removed them from {parameter} as well."
                print(print_str)

    # Return modified parameters
    return p









