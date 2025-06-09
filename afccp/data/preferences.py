"""
This module contains functions for constructing and modifying cadet and AFSC preference structures
within the Air Force Cadet Career Problem (AFCCP) model.

These utilities operate on the model's parameter dictionary to:
- Build preference matrices (`c_pref_matrix`, `a_pref_matrix`) from raw preference lists
- Convert preferences into normalized utilities and percentiles
- Apply eligibility filters to AFSC preferences
- Update final utility matrices for use in optimization

Functions in this module are essential for:
- Ensuring consistency between input data (Cadets.csv, AFSCs.csv) and the model structure
- Supporting preference-based evaluation and assignment logic
- Performing key preprocessing steps used during instance initialization or data generation
"""
import numpy as np
import pandas as pd
import copy


# _________________________________________DATA GENERATION SPECIFIC FUNCTIONS___________________________________________
def convert_utility_matrices_preferences(parameters, cadets_as_well=False):
    """
    Converts utility matrices into ordinal preference matrices.

    This function transforms the continuous utility values provided in the cadet and AFSC utility
    matrices into discrete preference rankings (ordinal preferences). These rankings are stored in
    `a_pref_matrix` and optionally `c_pref_matrix` within the `parameters` dictionary.

    Parameters
    ----------
    parameters : dict
        Dictionary of model parameters, including `afsc_utility` and optionally `cadet_utility`.

    cadets_as_well : bool, optional
        If `True`, the cadet utility matrix (`cadet_utility`) is also converted into a cadet
        preference matrix (`c_pref_matrix`). Defaults to `False`.

    Returns
    -------
    dict
        Updated `parameters` dictionary with added `a_pref_matrix` and optionally `c_pref_matrix`.
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
    Generate synthetic AFSC utility and preference matrices.

    This function constructs artificial utility scores and corresponding preference rankings for Air Force Specialty
    Codes (AFSCs) using merit, AFOCD tiers, and other known cadet attributes. It supports both weighted approaches
    using a provided set of value parameters or a default fixed weighting strategy. Preferences are automatically
    adjusted to ensure cadets and AFSCs only rank eligible options.

    Parameters
    ----------
    parameters : dict
        Dictionary containing fixed model parameters (cadet/AFSC eligibility, merit scores, utility matrices, etc.).

    value_parameters : dict, optional
        Value parameter dictionary containing weights and objectives to guide AFSC utility generation. If None,
        a default set of weights is used.

    fix_cadet_eligibility : bool, default=False
        If True, overrides cadet preferences to match eligibility criteria and recomputes rankings.

    Returns
    -------
    parameters : dict
        Updated parameter dictionary containing generated utility matrices and preference rankings:

        - `afsc_utility`: N x M matrix of cadet utility scores for each AFSC.
        - `a_pref_matrix`: AFSCs' preference rankings of cadets.
        - `c_pref_matrix`: Cadets' preference rankings of AFSCs.
        - `afsc_preferences`: Dict mapping each AFSC to its sorted list of cadet indices.
        - `cadet_preferences`: Dict mapping each cadet to their sorted list of AFSC indices.

    Examples
    --------
    ```python
    parameters = generate_fake_afsc_preferences(parameters)
    parameters = generate_fake_afsc_preferences(parameters, value_parameters=vp, fix_cadet_eligibility=True)
    ```
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


def generate_rated_data(parameters):
    """
    Generate Simulated Rated Interest and Order of Merit (OM) Data.

    This function generates ROTC-rated interest levels and USAFA/ROTC-rated Order of Merit (OM) scores for cadets
    eligible for rated AFSCs (e.g., Pilot, CSO, ABM, RPA). These scores are essential for modeling preferences and
    eligibility in rated board algorithms.

    Parameters
    ----------
    parameters : dict
        The main parameter dictionary for the cadet-AFSC assignment problem. It must contain:

        - `Rated Cadets`: Dictionary of rated cadets by commissioning source (`usafa`, `rotc`)
        - `afscs_acc_grp`: AFSCs categorized into assignment groups (must include 'Rated')
        - `SOCs`: List of commissioning source identifiers (e.g., `('usafa', 'Rated')`)
        - `afsc_preferences`: AFSCs’ ranked preferences over cadets
        - `I^E`: Cadet eligibility sets
        - `afscs`: Full list of AFSCs
        - `Rated Cadet Index Dict`: Lookup dict to convert cadet ID to matrix row index for each SOC

    Returns
    -------
    dict
    Updated parameter dictionary including:

    - `rr_interest_matrix`: ROTC cadets' self-assessed interest in rated AFSCs
    - `xr_om_matrix`, `ur_om_matrix`, etc.: Rated OM matrices for each SOC (generated if missing)

    Examples
    --------
    ```python
    parameters = generate_rated_data(parameters)
    ```

    This generates the following additions:

    - `parameters['rr_interest_matrix']` → random values like ['High', 'Med', 'Low', 'None']
    - `parameters['ur_om_matrix']` → OM percentiles for USAFA-rated cadets and AFSCs
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


# ____________________________________________MAIN DATA CORRECTION FUNCTIONS____________________________________________
def construct_rated_preferences_from_om_by_soc(parameters):
    """
    Construct AFSC Preferences for Rated Candidates Using OM Matrices.

    This function consolidates the Ordered Merit (OM) matrices from multiple Sources of Commissioning (SOCs)
    (e.g., USAFA and ROTC) and creates a unified AFSC preference list for Rated AFSCs. It normalizes OM rankings
    across SOCs, combines them into a single composite preference score, and updates both the `afsc_preferences`
    list and the `a_pref_matrix` for use in assignment modeling.

    Parameters:
    --------
    parameters (dict): Dictionary containing the model instance parameters, including:

    - `rr_om_matrix`, `ur_om_matrix`: Ordered merit matrices from ROTC and USAFA.
    - `or_om_matrix`: **Potentially** Ordered merit matrices from OTS.
    - `afsc_preferences`: Dictionary to update with new AFSC → cadet preference lists.
    - `a_pref_matrix`: Matrix representing cadet rankings from the AFSCs' perspective.
    - `SOCs`, `afscs_acc_grp`, and cadet lists for each SOC.

    Returns:
    --------
    dict: Updated `parameters` dictionary with modified `afsc_preferences` and `a_pref_matrix` reflecting
    normalized OM-based preference rankings for Rated AFSCs.

    Example:
    --------
    ```python
    parameters = construct_rated_preferences_from_om_by_soc(parameters)
    ```

    See Also:
    --------
    [`determine_soc_rated_afscs`](../../../afccp/reference/data/preferences/#data.preferences.determine_soc_rated_afscs):
        Identifies which AFSCs are rated within a given SOC.
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


def fill_remaining_preferences(parameters):
    """
    Fill in Remaining Cadet Preferences to Complete the Preference Matrix.

    This function ensures that each cadet has a complete preference list across all AFSCs. It fills in any
    unranked AFSCs (excluding bottom 2 and last choice) with incrementing ranks, followed by bottom 2 preferences,
    and finally the explicitly marked last choice if applicable.

    Parameters:
    --------
    parameters (dict): The problem instance parameters, containing:
        - `cadet_preferences`: Dictionary of AFSC preference orderings per cadet.
        - `c_pref_matrix`: Matrix of cadet preferences over AFSCs.
        - `J^Bottom 2 Choices`: Dictionary of each cadet's bottom two AFSCs.
        - `J^Last Choice`: Dictionary of each cadet's last AFSC choice.
        - `I`, `J`, `M`: Indexed sets of cadets, AFSCs, and unmatched AFSC index.

    Returns:
    --------
    dict: Updated parameters dictionary with a fully filled `c_pref_matrix`.

    Example:
    --------
    ```python
    parameters = fill_remaining_preferences(parameters)
    ```
    """

    # Shorthand
    p = parameters

    # Loop through all cadets
    for i in p['I']:

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


def remove_ineligible_cadet_choices(parameters, printing=False):
    """
    Clean Ineligible Cadet-AFSC Preference Pairings.

    This function audits and cleans the cadet-AFSC preference matrices by removing any inconsistent or ineligible pairings
    based on the qualification matrix. It ensures that both `c_pref_matrix` (cadet preferences) and `a_pref_matrix`
    (AFSC preferences) reflect only valid, eligible pairings. It also updates the qualification matrix to reflect enforced
    ineligibility for problematic pairs.

    Parameters:
    --------
    parameters (dict): Dictionary of the problem instance parameters.

    printing (bool, optional): If True, logs every change made. Default is False.

    Returns:
    --------
    dict: Updated parameters dictionary with cleaned preference matrices and enforced eligibility alignment.

    Example:
    --------
    ```python
    parameters = remove_ineligible_cadet_choices(parameters, printing=True)
    ```

    See Also:
    --------
    - [`fill_remaining_preferences`](../../../afccp/reference/data/preferences/#data.preferences.fill_remaining_preferences):
      Fills in arbitrary preferences for cadets, excluding bottom-ranked AFSCs.
    - [`parameter_sets_additions`](../../../afccp/reference/data/adjustments/#data.adjustments.parameter_sets_additions):
      Rebuilds indexed sets after modifying eligibility or preference matrices.
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
    Reconstructs cadet and AFSC preference matrices based on index-based preference lists.

    This function updates the `c_pref_matrix` (cadet preference matrix) and `a_pref_matrix`
    (AFSC preference matrix) using the indexed preference lists provided in
    `cadet_preferences` and `afsc_preferences`, respectively. Cadets and AFSCs with empty
    preferences are skipped.

    Parameters
    ----------
    parameters : dict
    A parameter dictionary containing:

    - `N` : int, total number of cadets
    - `M` : int, total number of AFSCs
    - `I` : list of cadet indices
    - `J` : list of AFSC indices
    - `cadet_preferences` : dictionary of lists; each list contains AFSC indices ranked by each cadet
    - `afsc_preferences` : dictionary of lists; each list contains cadet indices ranked by each AFSC

    Returns
    -------
    dict
    Updated parameter dictionary with:

    - `c_pref_matrix` : ndarray, shape (N, M)
        Preference matrix where entry (i, j) indicates cadet i's rank of AFSC j (0 if not ranked)
    - `a_pref_matrix` : ndarray, shape (N, M)
        Preference matrix where entry (i, j) indicates AFSC j's rank of cadet i (0 if not ranked)

    Examples
    --------
    ```python
    from afccp.data.preferences import update_preference_matrices
    parameters = update_preference_matrices(parameters)
    ```
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


def update_first_choice_cadet_utility_to_one(parameters, printing=True):
    """
    Fix First-Choice Cadet Utility to One.

    Updates the utility matrix so that each cadet's top-ranked AFSC is assigned a utility value of 1.0,
    indicating maximum preference. This only applies to cadets who have valid preference lists.

    Parameters
    ----------
    parameters : dict
    A parameter dictionary containing:

    - `I` : list of int
      Cadet indices.
    - `cadet_preferences` : dict of lists
      Each cadet's list of AFSC indices in ranked order of preference.
    - `utility` : ndarray of shape (N, M)
      Matrix of cadet utility values where entry (i, j) is cadet i's utility for AFSC j.

    printing : bool, optional
    If True, prints the number of cadets updated and their indices. Default is True.

    Returns
    -------
    ndarray:
    Updated utility matrix with each cadet's first choice AFSC set to a utility of 1.

    Examples
    --------
    ```python
    updated_utility = update_first_choice_cadet_utility_to_one(parameters)
    ```
    """

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


def convert_afsc_preferences_to_percentiles(parameters):
    """
    Convert AFSC Preferences to Percentile-Based Utilities.

    This method transforms each AFSC's preference list into a normalized percentile utility score
    for each cadet. Higher-ranked cadets receive higher percentile scores (closer to 1), while lower-ranked
    cadets receive lower scores (closer to 0). These scores are stored in a new matrix called `afsc_utility`.

    Parameters
    ----------
    parameters: dict
    A parameter dictionary containing:

    - `N` : int
      Total number of cadets
    - `M` : int
      Total number of AFSCs
    - `J` : list of AFSC indices
    - `afsc_preferences` : dict
      Dictionary where each key is an AFSC index and each value is a list of cadet indices ranked by that AFSC

    Returns
    -------
    parameters : dict
    Updated parameter dictionary with:

    - `afsc_utility`: ndarray of shape (N, M)
      Matrix of normalized percentile utility values for each cadet-AFSC pair. A value of 1.0 indicates
      top preference, and values decrease with lower preference.

    Examples
    --------
    ```python
    # Define AFSC preferences
    parameters = {
        'N': 3,
        'M': 2,
        'J': [0, 1],
        'afsc_preferences': {
            0: [2, 1, 0],
            1: [1, 0]
        }
    }

    # Convert preferences to percentiles
    updated = convert_afsc_preferences_to_percentiles(parameters)

    # View result
    print(updated['afsc_utility'])
    # Output:
    # array([[0.333, 0.5],
    #        [0.666, 1.0],
    #        [1.0,   0.0]])
    ```
    """

    # Shorthand
    p = parameters

    # Get normalized percentiles (Average of 0.5)
    p["afsc_utility"] = np.zeros([p['N'], p['M']])
    for j in p['J']:
        p['afsc_utility'][p['afsc_preferences'][j], j] = \
            np.arange(1, len(p['afsc_preferences'][j]) + 1)[::-1] / len(p['afsc_preferences'][j])

    return p


def update_cadet_columns_from_matrices(parameters):
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


def update_cadet_utility_matrices(parameters):
    """
    Update Cadet Utility Matrices from Reported Utilities.

    This method reads each cadet's self-reported utility values (from `c_utilities`) and updates the `utility` matrix accordingly.
    It also creates the normalized `cadet_utility` matrix based on ordinal preferences or a utility-based formula,
    depending on whether `last_afsc` is present in the parameters.

    Parameters
    ----------
    parameters : dict
    A parameter dictionary containing:

    - `N` : int, total number of cadets
    - `M` : int, total number of AFSCs
    - `I` : list of cadet indices
    - `cadet_preferences` : dict of lists; each list contains AFSC indices ranked by cadet
    - `c_utilities` : ndarray, shape (N, P); cadet-reported utilities aligned with their preferences
    - `num_util` : int; number of utility values reported per cadet
    - `last_afsc` (optional) : str; used to determine which utility processing function to apply

    Returns
    -------
    parameters : dict
    Updated parameter dictionary with:

    - `utility` : ndarray, shape (N, M+1)
        Utility matrix with cadet-reported values; last column represents unmatched utility
    - `cadet_utility` : ndarray
        Normalized utility matrix calculated from rankings or weighted formula

    Raises
    ------
    ValueError
        If required fields (`cadet_preferences`, `c_utilities`) are missing from the parameter dictionary.

    Examples
    --------
    ```python
    parameters = update_cadet_utility_matrices(parameters)
    ```

    See Also
    --------
    - [`create_final_cadet_utility_matrix_from_new_formula`](../../../../afccp/reference/data/preferences/#data.preferences.create_final_cadet_utility_matrix_from_new_formula)
    - [`create_new_cadet_utility_matrix`](../../../../afccp/reference/data/preferences/#data.preferences.create_new_cadet_utility_matrix)
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


def modify_rated_cadet_lists_based_on_eligibility(parameters, printing=True):
    """
    Remove Ineligible Rated Cadets from Rated Lists and Matrices.

    This method ensures that cadets in each Source of Commissioning (SOC)'s rated list are only included if they have at least
    one rated AFSC preference. Cadets without any rated preferences are removed from the rated cadet list for that SOC as well as
    the corresponding rated order-of-merit matrix (e.g., 'rr_om_matrix' for ROTC).

    Parameters
    ----------
    parameters : dict
    A dictionary of model parameters including:

    - `SOCs` : list of str
        List of Source of Commissioning identifiers (e.g., ['ROTC', 'USAFA', 'OTS'])
    - `Rated Cadets` : dict
        Dictionary mapping SOC names to arrays of rated cadet indices
    - `Rated Choices` : dict
        Dictionary mapping SOC names to dicts of cadet-rated-AFSC preferences
    - `rr_om_matrix`, `ur_om_matrix`, etc. : ndarray
        Matrices used in rated order-of-merit calculations by SOC

    printing : bool, optional
        If True (default), prints a summary of the cadets removed and the matrices updated.

    Returns
    -------
    dict
    The updated parameter dictionary with:

    - Rated cadet lists pruned of cadets lacking rated preferences
    - Rated order-of-merit matrices updated to exclude removed cadets

    Examples
    --------
    ```python
    parameters = modify_rated_cadet_lists_based_on_eligibility(parameters, printing=True)
    ```
    """

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


# ____________________________________________________AUXILIARY FUNCTIONS_______________________________________________
def determine_soc_rated_afscs(soc, all_rated_afscs):
    """
    Filter Rated AFSCs Based on Source of Commissioning (SOC).

    This function selects only the AFSCs relevant to the given SOC (e.g., USAFA, ROTC, OTS)
    by excluding AFSCs that are tagged for other SOCs using suffixes like `_U`, `_R`, or `_O`.

    Parameters:
    --------
    soc (str): The name of the source of commissioning (e.g., "usafa", "rotc").
    all_rated_afscs (List[str]): A list of all rated AFSC strings.

    Returns:
    --------
    List[str]: Filtered list of AFSCs associated with the provided SOC.

    Example:
    --------
    ```python
    determine_soc_rated_afscs("usafa", ["11XX_U", "11XX_R", "11XX_O", "12XX", "13B", "18X"])
    # Returns: ["11XX_U", "12XX", "13B", "18X"]
    ```

    Notes:
    --------
    The filtering logic assumes that the AFSC string may contain a SOC-specific suffix.
    - `_U` for USAFA
    - `_R` for ROTC
    - `_O` for OTS
    """

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


def create_new_cadet_utility_matrix(parameters):
    """
    Create New Cadet Utility Matrix from Rankings and Original Utilities.

    This function constructs a new "cadet_utility" matrix by blending each cadet's ordinal preferences
    (normalized rankings) with their original reported utility scores. The result is a weighted composite
    utility score for each cadet–AFSC pair.

    The final matrix is stored in `cadet_utility` and is used in the optimization models.

    Parameters
    ----------
    parameters : dict
    Parameter dictionary containing:

    - `N` : int, number of cadets
    - `M` : int, number of AFSCs
    - `I` : list of cadet indices
    - `utility` : ndarray, cadet-reported utility matrix
    - `cadet_preferences` : dict, cadet-to-AFSC preference lists
    - `num_cadet_choices` : dict, number of ranked AFSCs per cadet

    Returns
    -------
    dict :
    Updated parameter dictionary with:

    - `cadet_utility` : ndarray, shape (N, M+1)
        Blended utility matrix using normalized rankings and reported utilities.
        Last column is reserved for unmatched cadets.

    Examples
    --------
    ```python
    parameters = {
        'N': 2,
        'M': 3,
        'I': [0, 1],
        'utility': np.array([[1.0, 0.8, 0.6, 0.0], [0.5, 1.0, 0.0, 0.0]]),
        'cadet_preferences': {0: [0, 1, 2], 1: [1, 0]},
        'num_cadet_choices': {0: 3, 1: 2}
    }
    updated = create_new_cadet_utility_matrix(parameters)
    updated['cadet_utility']
    ```
    Output:
    ```
    array([[1.  , 0.8 , 0.6 , 0.  ],
           [1.  , 0.75, 0.  , 0.  ]])
    ```

    See Also
    --------
    - [`update_cadet_utility_matrices`](../../../afccp/reference/data/preferences/#data.preferences.update_cadet_utility_matrices)
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
    Create Final Cadet Utility Matrix Using Eligibility-Aware Scoring Formula.

    This function constructs a final `cadet_utility` matrix by integrating ordinal rankings, cadet-reported utilities,
    and least desired AFSC criteria. It uses a custom scoring formula that reflects cadet preferences, their
    eligibility, and how the AFSC ranks among their choices.

    The output matrix is used in the optimization models.

    Parameters
    ----------
    parameters : dict
    A dictionary containing cadet–AFSC preference and eligibility information:

    - `N` : int, number of cadets
    - `M` : int, number of AFSCs
    - `I` : list of cadet indices
    - `cadet_preferences` : dict, AFSC preference list per cadet
    - `utility` : ndarray, shape (N, M+1), cadet-reported utility values
    - `J^Selected` : dict, selected AFSCs for each cadet
    - `J^Bottom 2 Choices` : dict, bottom two AFSC preferences per cadet
    - `J^Last Choice` : dict, last-choice AFSC for each cadet

    Returns
    -------
    parameters : dict
    Updated parameter dictionary with:

    - `cadet_utility` : ndarray, shape (N, M+1)
        Weighted utility matrix accounting for ordinal preferences, eligibility, and cadet-reported utilities.

    Examples
    --------
    ```python
    p = {
        'N': 2,
        'M': 3,
        'I': [0, 1],
        'utility': np.array([[0.8, 1.0, 0.6, 0.0], [0.5, 0.0, 0.9, 0.0]]),
        'cadet_preferences': {0: [0, 1, 2], 1: [2, 0]},
        'J^Selected': {0: [0, 1], 1: [0, 2]},
        'J^Bottom 2 Choices': {0: [1, 2], 1: [0, 2]},
        'J^Last Choice': {0: 2, 1: 2}
    }
    p = create_final_cadet_utility_matrix_from_new_formula(p)
    print(p['cadet_utility'])
    ```

    See Also
    --------
    - [`create_new_cadet_utility_matrix`](../../../afccp/reference/data/preferences/#data.preferences.create_new_cadet_utility_matrix)
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











