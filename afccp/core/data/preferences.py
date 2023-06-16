# Import libraries
import numpy as np
import pandas as pd
import copy
import afccp.core.globals


# "Fixed" Parameter Procedures
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


def convert_utility_matrices_preferences(parameters, cadets_as_well=False):
    """
    This function converts the cadet and AFSC utility matrices into the preference dataframes
    """
    p = parameters

    # Loop through each AFSC to get their preferences
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
            utilities = p["utility"][i, :p["M"]]
            sorted_indices = np.argsort(utilities)[::-1]
            preferences = np.argsort(
                sorted_indices) + 1  # Add 1 to change from python index (at 0) to rank (start at 1)
            p["c_pref_matrix"][i, :] = preferences
    return p


def generate_fake_afsc_preferences(parameters, value_parameters=None):
    """
    This function generates fake AFSC utilities/preferences using AFOCD, merit, cadet preferences etc.
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

    # Create AFSC Preference Matrix
    p["a_pref_matrix"] = np.zeros([p["N"], p["M"]]).astype(int)
    for j in p["J"]:

        # Sort the utilities to get the preference list
        utilities = p["afsc_utility"][:, j]
        nonzero_indices = np.where(utilities > 0)[0]
        sorted_indices = np.argsort(utilities[nonzero_indices])[::-1]
        for rank, util in enumerate(utilities[nonzero_indices][sorted_indices]):
            i = np.where(utilities == util)[0][0]
            p["a_pref_matrix"][i, j] = rank + 1  # Add one so the #1 guy isn't a zero!

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

    # # First weed out all those who are ineligible for each AFSC
    # p["afsc_utility"] = np.ones((p["N"], p["M"]))
    # for j in p["J"]:
    #
    #     p["afsc_utility"][:, j] = (p["num_eligible"][j] - p["a_pref_matrix"][:, j]) / p["num_eligible"][j]


    # All ineligible cadets are given percentiles of 0
    p["afsc_utility"] *= p["eligible"]

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
    dataset_dict = {'rotc': 'rr_om_matrix', 'usafa': 'ur_om_matrix'}
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
    rated_afscs = p['afscs_acc_grp']["Rated"]
    dataset_dict = {'rotc': 'rr_om_matrix', 'usafa': 'ur_om_matrix'}
    cadets_dict = {'rotc': 'rr_om_cadets', 'usafa': 'ur_om_cadets'}
    combined_rated_om = np.zeros([p['N'], len(rated_afscs)])  # Combined Rated OM from both SOCs

    # Loop through both sources of commissioning
    rated_cadets, rated_cadet_index_dict = {}, {}
    for soc in ['rotc', 'usafa']:

        # Rated cadets determined from OM dataset
        rated_cadets[soc] = p[cadets_dict[soc]]
        rated_cadet_index_dict[soc] = {i: cadet_index for cadet_index, i in enumerate(rated_cadets[soc])}
        for cadet_index, i in enumerate(rated_cadets[soc]):  # Index of cadet in OM dataset, "real" index of cadet

            # Loop through all rated AFSCs
            for afsc_index in range(len(rated_afscs)):
                combined_rated_om[i, afsc_index] = p[dataset_dict[soc]][cadet_index, afsc_index]

    # Sort the OM to convert it into a 1-N
    for afsc_index, afsc in enumerate(rated_afscs):
        j = np.where(p['afscs'] == afsc)[0][0]

        # Get AFSC preferences (list of cadet indices in order)
        ineligibles = np.where(combined_rated_om[:, afsc_index] == 0)[0]
        num_ineligible = len(ineligibles)  # Ineligibles are going to be at the bottom of the list (and we remove them)
        p['afsc_preferences'][j] = np.argsort(combined_rated_om[:, afsc_index])[::-1][:p['N'] - num_ineligible]

        # Reset this AFSC's "preference matrix" column
        p['a_pref_matrix'][:, j] = np.zeros(p['N'])

        # Since 'afsc_preferences' is an array of cadet indices, we can do this
        p['a_pref_matrix'][p['afsc_preferences'][j], j] = np.arange(1, len(p['afsc_preferences'][j]) + 1)

    return p  # Return updated parameters


def remove_ineligible_cadet_choices(parameters):
    """
    This function removes ineligible choices from the cadets and from the AFSCs based on the qualification matrix
    """

    # Shorthand
    p = parameters

    for i in p['I']:
        for j in p['J']:
            if i not in p['I^E'][j]:
                p['c_pref_matrix'][i, j] = 0
                p['a_pref_matrix'][i, j] = 0

    return p  # Return parameters








