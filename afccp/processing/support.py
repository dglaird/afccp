# Import libraries
import numpy as np
import pandas as pd


def clean_problem_instance_preferences_utilities(afscs, original_preferences, original_utilities=None,
                                                 year_2020=False, mapping=False):
    """
    This procedure takes the afscs that are matched this year, as well as the original preferences
    and optionally the original utilities and returns the cleaned preferences and utilities for
    the problem instance.
    :param year_2020: if this year is 2020 or not
    :param mapping: if we're showing how we change preferences
    :param afscs: original year afscs
    :param original_preferences: raw cadet preferences
    :param original_utilities: raw cadet utilities
    :return: cleaned preferences, cleaned utilities
    """

    N = len(original_preferences)
    if original_utilities is not None:
        original_utilities = original_utilities / 100

    # Reduce preferences to only include the AFSCs defined above
    unique_preferences = np.unique(original_preferences.astype(str))
    flattened_preferences = np.ndarray.flatten(original_preferences)
    for afsc in unique_preferences:
        indices = np.where(flattened_preferences == afsc)[0]
        if year_2020:
            if afsc not in afscs:
                np.put(flattened_preferences, indices, '')
        else:
            afsc = afsc.strip()

            # The AFSC directly matches what we would expect (rare)
            if afsc in afscs:
                np.put(flattened_preferences, indices, afsc)
            else:

                # Scenarios like "13NX" or "13N1" instead of "13N"
                if afsc[:-1] in afscs:
                    np.put(flattened_preferences, indices, afsc[:-1])

                else:

                    if afsc in ['17DX', ' 17DX', '17D1']:
                        np.put(flattened_preferences, indices, '17X')
                    else:

                        # Scenarios like "32E1E" instead of "32EXE"
                        if len(afsc) == 5:
                            test_afsc = afsc[:3] + 'X' + afsc[4]
                            if test_afsc in afscs:
                                np.put(flattened_preferences, indices, test_afsc)
                            else:

                                # Didn't find it
                                np.put(flattened_preferences, indices, '')
                        else:

                            # Scenarios where the AFSC is "NONE", "NOAF", etc.
                            np.put(flattened_preferences, indices, '')

                            # Clean up preferences and add utilities
    preferences = flattened_preferences.reshape(N, 6)
    new_preferences = np.array([[" " * 8 for _ in range(6)] for _ in range(N)])
    utility_vector = np.array([1, 0.50, 0.33, 0.25, 0.2, 0.17])
    utilities = np.zeros([N, 6])
    for i in range(N):

        # Get indices of year-specific AFSCs
        correct_indices = np.where(preferences[i, :] != '')[0]

        # Place preferences and utilities in the correct spots
        first_indices = np.arange(len(correct_indices))
        np.put(new_preferences[i, :], first_indices, preferences[i, correct_indices])
        if original_utilities is not None:
            np.put(utilities[i, :], first_indices, original_utilities[i, correct_indices])
        else:
            np.put(utilities[i, :], first_indices, utility_vector[correct_indices])

        # Reduce the "larger" blanks to the simple ''
        big_blank_indices = np.where(new_preferences[i, :] == " " * 8)[0]
        np.put(new_preferences[i, :], big_blank_indices, '')

        if mapping:
            print('\nCadet ' + str(i) + '. Original Preferences:', original_preferences[i, :],
                  ', Original Utilities:', original_utilities[i, :])
            print('Cadet ' + str(i) + '. Cleaned Preferences:', new_preferences[i, :], ', Cleaned Utilities:',
                  utilities[i, :])

    return new_preferences, utilities