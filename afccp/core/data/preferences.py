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
        sorted_indices = np.argsort(utilities)[::-1]
        preferences = np.argsort(sorted_indices)
        p["a_pref_matrix"][:, j] = preferences + 1  # Add one so the #1 guy isn't a zero!

    p["a_pref_matrix"] *= p["eligible"]  # They have to be eligible!

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





