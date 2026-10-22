"""
Data Adjustments Module for AFCCP
===========

This module contains utility functions that perform critical adjustments, validations, and transformations
on the `parameters` and `vp_dict` dictionaries that define each problem instance in the AFCCP model.

The functions here serve as post-processing or pre-processing steps to ensure internal consistency,
prepare data for model input, or apply specific business rules (such as OTS must-matches or degree tier qualification logic).
They are commonly called after loading data or before solving a model.

Functions:
--------
- `parameter_sanity_check(parameters)`
  Performs validation checks on parameters and value parameters to ensure modeling assumptions are satisfied.

- `parameter_sets_additions(parameters)`
  Updates derived parameter sets (like `I^OTS`, `I^USAF`, `J^Rated`) based on core problem inputs.

- `more_parameter_additions(parameters)`
  Adds further derived variables or flags used throughout the AFCCP model such as first-choice indicators.

- `base_training_parameter_additions(parameters)`
  Adds data structures needed to support Base Training assignments for cadets.

- `set_ots_must_matches(parameters)`
  Selects a subset of OTS cadets as "must-match" based on their merit and OTS accession targets.

- `gather_degree_tier_qual_matrix(cadets_df, parameters)`
  Determines the qualification matrix for AFSC eligibility based on degree tier requirements.

- `convert_instance_to_from_scrubbed(instance, new_letter=None, translation_dict=None, data_name='Unknown')`
  Converts instance AFSC names to "scrubbed" placeholders or restores them back to their original names for
  anonymized modeling and solution reproducibility.
"""
import numpy as np
import pandas as pd
import datetime
import copy
from functools import reduce

# afccp modules
import afccp.data.preferences
import afccp.data.support
import afccp.data.values


# ______________________________________________MAIN DATA ADJUSTMENTS___________________________________________________
def parameter_sets_additions(parameters, run_extra_again=False, printing=False):
    """
    Add Indexed Sets and Subsets to the Problem Instance Parameters.

    This function enhances the problem instance by creating indexed sets and subsets for both cadets and AFSCs,
    demographic filters, eligibility matrices, preference-related metadata, and readiness for optimization.
    It also validates eligibility constraints and appends additional calculated data fields.

    Parameters
    ----------
    parameters : dict
        The fixed model input parameters for a cadet-AFSC assignment instance, including eligibility matrices,
        cadet/AFSC attributes, utility matrices, and demographics.

    Returns
    -------
    Updated parameter dictionary with:

    - Indexed cadet and AFSC sets: ``I``, ``J``, ``J^E``, ``I^E``
    - Eligibility and preference counts: ``num_eligible``, ``Choice Count``
    - Demographic and qualification subsets: ``I^D``, ``I^USAFA``, ``I^Male``, ``I^Minority``, etc.
    - Assignment constraints: ``J^Fixed``, ``J^Reserved``
    - Cadet and AFSC preference mappings
    - Updated utility matrix with unmatched column

    Examples
    --------
    ```python
    from afccp.data.adjustments import parameter_sets_additions
    params = parameter_sets_additions(params)
    ```

    Notes
    -----
    - Automatically detects and processes USAFA/ROTC cadet splits based on `usafa` and `soc` columns.
    - Adds extra handling for cadets that are fixed to AFSCs via preassignments in `assigned`.
    - Includes support for rated cadets, STEM AFSCs, race/ethnicity filters, and eligibility-based breakouts.

    See Also
    --------
    - [`more_parameter_additions`](../../../afccp/reference/data/adjustments/#data.adjustments.more_parameter_additions):
      Adds enhanced logic for cadet/AFSC matching, preference flattening, and diversity tracking.
    - [`base_training_parameter_additions`](../../../afccp/reference/data/adjustments/#data.adjustments.base_training_parameter_additions):
      Adds base and training assignment structures to the parameters.
    """

    # Shorthand
    p = parameters

    # Cadet Indexed Sets
    p['I'] = np.arange(p['N'])
    p['J'] = np.arange(p['M'])
    p['J^E'] = [np.where(p['eligible'][i, :])[0] for i in p['I']]  # set of AFSCs that cadet i is eligible for

    # AFSC Indexed Sets
    p['I^E'] = [np.where(p['eligible'][:, j])[0] for j in p['J']]  # set of cadets that are eligible for AFSC j

    # Number of eligible cadets for each AFSC
    p["num_eligible"] = np.array([len(p['I^E'][j]) for j in p['J']])

    # More cadet preference sets if we have the "cadet preference columns"
    if "c_preferences" in p:
        p["I^Choice"] = {choice: [np.where(
            p["c_preferences"][:, choice] == afsc)[0] for afsc in p["afscs"][:p["M"]]] for choice in range(p["P"])}
        p["Choice Count"] = {choice: np.array(
            [len(p["I^Choice"][choice][j]) for j in p["J"]]) for choice in range(p["P"])}

    # Add demographic sets if they're included
    p['I^D'] = {}
    if 'usafa' in p:
        usafa = np.where(p['usafa'] == 1)[0]  # set of usafa cadets
        p['usafa_proportion'] = np.mean(p['usafa'])
        p['I^D']['USAFA Proportion'] = [np.intersect1d(p['I^E'][j], usafa) for j in p['J']]
    if 'mandatory' in p:  # Qual Type = "Old"
        p['I^D']['Mandatory'] = [np.where(p['mandatory'][:, j])[0] for j in p['J']]
        p['I^D']['Desired'] = [np.where(p['desired'][:, j])[0] for j in p['J']]
        p['I^D']['Permitted'] = [np.where(p['permitted'][:, j])[0] for j in p['J']]
    if "tier 1" in p:  # Qual Type = "Tiers"
        for t in ['1', '2', '3', '4']:
            p['I^D']['Tier ' + t] = [np.where(p['tier ' + t][:, j])[0] for j in p['J']]

        # Get arrays of unique degree tier values
        p['Deg Tier Values'] = {j: np.unique(p['qual'][:, j]) for j in p['J']}

    if 'male' in p:
        male = np.where(p['male'] == 1)[0]  # set of male cadets
        p['male_proportion'] = np.mean(p['male'])
        p['I^D']['Male'] = [np.intersect1d(p['I^E'][j], male) for j in p['J']]
    if 'minority' in p:
        minority = np.where(p['minority'] == 1)[0]  # set of minority cadets
        p['minority_proportion'] = np.mean(p['minority'])
        p['I^D']['Minority'] = [np.intersect1d(p['I^E'][j], minority) for j in p['J']]

    # Add an extra column to the utility matrix for cadets who are unmatched (if it hasn't already been added)
    zeros_vector = np.array([[0] for _ in range(p["N"])])
    if np.shape(p['utility']) == (p['N'], p['M']):
        p["utility"] = np.hstack((p["utility"], zeros_vector))

    # Add SOC column
    if "soc" not in p:
        p['soc'] = np.array(['USAFA' if p['usafa'][i] == 1 else 'ROTC' for i in p['I']])

    # Merit
    if 'merit' in p:
        p['sum_merit'] = p['merit'].sum()  # should be close to N/2

    # USAFA/ROTC/OTS cadets
    if 'SOCs' not in p:  # If it's just a "USAFA" column, we assume it's only USAFA/ROTC
        p['rotc'] = (p['usafa'] == 0) * 1
        p['usafa_cadets'] = np.where(p['usafa'])[0]
        p['rotc_cadets'] = np.where(p['rotc'])[0]
        p['usafa_eligible_count'] = np.array([len(np.intersect1d(p['I^E'][j], p['usafa_cadets'])) for j in p['J']])
        p['rotc_eligible_count'] = np.array([len(np.intersect1d(p['I^E'][j], p['rotc_cadets'])) for j in p['J']])
        p['SOCs'] = ['usafa', 'rotc']
    else:
        for soc in p['SOCs']:
            p[soc] = (p['soc'] == soc.upper()) * 1
            p[f'{soc}_cadets'] = np.where(p[soc])[0]
            p[f'{soc}_eligible_count'] = np.array(
                [len(np.intersect1d(p['I^E'][j], p[f'{soc}_cadets'])) for j in p['J']])

    # Initialize empty dictionaries of matched/reserved cadets and manually constrained cadets!
    p["J^Fixed"] = {}
    p["J^Reserved"] = {}
    if 'manually_constrained_afscs' not in p:
        p['manually_constrained_afscs'] = np.array([" " for _ in p['I']], dtype=object)

    # Cadet preference/rated cadet set additions
    p = more_parameter_additions(p)

    # Base/Training set additions
    if "bases" in p and ('I^B' not in p or run_extra_again):  # Avoid running this twice in one run
        p = base_training_parameter_additions(p, printing=printing)

    return p


def more_parameter_additions(parameters):
    """
    Add Additional Subsets and Parameter Structures to the Problem Instance.

    This function enhances the problem instance by appending numerous structured subsets and derived attributes
    based on cadet preferences, eligibility, accession groupings, demographics, and more. It enriches the input
    parameter dictionary in preparation for detailed analysis and optimization.

    Parameters
    ----------
    parameters : dict
        The initial problem instance dictionary, containing data on cadets, AFSCs, eligibility, utility matrices, etc.

    Returns
    -------
    dict
    The updated problem instance with additional fields, subsets, and derived variables including:

    - Cadet and AFSC preferences
    - Accessions group (Rated, USSF, NRL) AFSC indices
    - Rated-specific cadet groupings and OM mapping
    - Simpson index for race/ethnicity
    - Groupings by SOC (e.g., ROTC, USAFA), gender, and STEM designation
    - Subsets like `I^Must_Match`, `J^Bottom 2 Choices`, etc.

    Examples
    --------
    ```python
    parameters = more_parameter_additions(parameters)
    ```

    Notes
    -----
    The function performs a large number of conditional operations and appends dozens of new keys to `parameters`.
    These are used downstream in optimization and statistical evaluation of AFSC assignment plans.

    See Also
    --------
    - [`parameter_sets_additions`](../../../afccp/reference/data/adjustments/#data.adjustments.parameter_sets_additions):
      Related utility that adds indexed parameter sets post-preference generation.
    """

    # Shorthand
    p = parameters

    # Create Cadet preferences
    if 'c_pref_matrix' in p:
        p['cadet_preferences'] = {}
        p['num_cadet_choices'] = np.zeros(p['N'])
        for i in p['I']:

            # Sort the cadet preferences
            cadet_sorted_preferences = np.argsort(p['c_pref_matrix'][i, :])
            p['cadet_preferences'][i] = []

            # Loop through each AFSC in order of preference and add it to the cadet's list
            for j in cadet_sorted_preferences:

                # Only add AFSCs that the cadet is eligible for and expressed a preference for
                if 'last_afsc' not in p:
                    if j in p['J^E'][i] and p['c_pref_matrix'][i, j] != 0:
                        p['cadet_preferences'][i].append(j)

                # Only add AFSCs that the cadet expressed a preference for
                else:
                    if p['c_pref_matrix'][i, j] != 0:
                        p['cadet_preferences'][i].append(j)

            p['cadet_preferences'][i] = np.array(p['cadet_preferences'][i])  # Convert to numpy array
            p['num_cadet_choices'][i] = len(p['cadet_preferences'][i])

    # Create AFSC preferences
    if 'a_pref_matrix' in p:
        p['afsc_preferences'] = {}
        for j in p['J']:

            # Sort the AFSC preferences
            afsc_sorted_preferences = np.argsort(p['a_pref_matrix'][:, j])
            p['afsc_preferences'][j] = []

            # Loop through each cadet in order of preference and add them to the AFSC's list
            for i in afsc_sorted_preferences:

                # Only add cadets that are eligible for this AFSC and expressed a preference for it
                if i in p['I^E'][j] and p['a_pref_matrix'][i, j] != 0:
                    p['afsc_preferences'][j].append(i)

            p['afsc_preferences'][j] = np.array(p['afsc_preferences'][j])  # Convert to numpy array

    # Determine AFSCs by Accessions Group
    p['afscs_acc_grp'] = {}
    if 'acc_grp' in p:
        for acc_grp in ['Rated', 'USSF', 'NRL']:
            p['J^' + acc_grp] = np.where(p['acc_grp'] == acc_grp)[0]
            if len(p['J^' + acc_grp]) > 0:
                p['afscs_acc_grp'][acc_grp] = p['afscs'][p['J^' + acc_grp]]
    else:  # Previously, we've only assigned NRL cadets so we assume that's what we're dealing with here
        p['acc_grp'] = np.array(['NRL' for _ in p['J']])
        p['afscs_acc_grp']['NRL'] = p['afscs']
        p['J^NRL'] = p['J']

    # AF-wide career fields
    if 'USSF' in p['afscs_acc_grp']:
        p['J^USAF'] = np.array([j for j in p['J'] if j not in p['J^USSF']])
    else:
        p['J^USAF'] = p['J']

    # PGL Totals per SOC for USSF
    if 'USSF' in p['afscs_acc_grp']:
        for service in ['USSF', 'USAF']:
            for soc in p['SOCs']:
                p[f'{service.lower()}_{soc}_pgl'] = np.sum(p[f'{soc}_quota'][j] for j in p[f'J^{service}'])

    # Determine eligible Rated cadets for both SOCs (cadets that are considered by the board)
    cadets_dict = {soc: f'{soc[0]}r_om_cadets' for soc in p['SOCs']}
    p["Rated Cadets"] = {}
    p["Rated Cadet Index Dict"] = {}
    p['Rated Choices'] = {}  # Dictionary of Rated cadet choices (only Rated AFSCs) by SOC
    p['Num Rated Choices'] = {}  # Number of Rated cadet choices by SOC
    for soc in cadets_dict:

        # If we already have the array of cadets from the dataset
        if cadets_dict[soc] in p:
            p["Rated Cadets"][soc] = p[cadets_dict[soc]]
            p["Rated Cadet Index Dict"][soc] = {i: idx for idx, i in enumerate(p["Rated Cadets"][soc])}

        # If we don't have this dataset, we check to see if we have Rated AFSCs
        elif 'Rated' in p['afscs_acc_grp']:

            # Add Rated cadets in order to each SOC list
            p["Rated Cadets"][soc] = []
            for i in p[soc + '_cadets']:
                for j in p['J^Rated']:
                    if j in p['J^E'][i]:
                        p["Rated Cadets"][soc].append(i)
                        break

            # Convert to numpy array and get translation dictionary
            p["Rated Cadets"][soc] = np.array(p["Rated Cadets"][soc])
            p["Rated Cadet Index Dict"][soc] = {i: idx for idx, i in enumerate(p["Rated Cadets"][soc])}

        # Get Rated preferences (where we strip out all NRL/USSF choices
        if soc in p['Rated Cadets'] and 'cadet_preferences' in p:
            p['Rated Choices'][soc] = {}
            p['Num Rated Choices'][soc] = {i: 0 for i in p["Rated Cadets"][soc]}
            for i in p["Rated Cadets"][soc]:
                rated_order = []
                for j in p['cadet_preferences'][i]:
                    if j in p['J^Rated']:
                        rated_order.append(j)
                        p['Num Rated Choices'][soc][i] += 1
                p['Rated Choices'][soc][i] = np.array(rated_order)

    # If we haven't already created the "cadet_utility" matrix, we do that here (only one time)
    if 'cadet_utility' not in p:
        # Build out cadet_utility using cadet preferences
        p['cadet_utility'] = np.around(copy.deepcopy(p['utility']), 4)

    # set of AFSCs that cadet i has placed a preference for and is also eligible for
    non_zero_utils_j = [np.where(p['cadet_utility'][i, :] > 0)[0] for i in p['I']]
    p["J^P"] = [np.intersect1d(p['J^E'][i], non_zero_utils_j[i]) for i in p['I']]

    # set of cadets that have placed a preference for AFSC j and are eligible for AFSC j
    non_zero_utils_i = [np.where(p['cadet_utility'][:, j] > 0)[0] for j in p['J']]
    p["I^P"] = [np.intersect1d(p['I^E'][j], non_zero_utils_i[j]) for j in p['J']]

    # Race categories
    if 'race' in p:
        p['race_categories'] = np.unique(p['race'])
        for race in p['race_categories']:
            p['I^' + race] = np.where(p['race'] == race)[0]

        # Calculate simpson index for overall class as a baseline
        p['baseline_simpson_index'] = round(1 - np.sum([(len(
            p['I^' + race]) * (len(p['I^' + race]) - 1)) / (p['N'] * (p['N'] - 1)) for race in p['race_categories']]),
                                            2)

    # Ethnicity categories
    if 'ethnicity' in p:
        p['ethnicity_categories'] = np.unique(p['ethnicity'])
        for eth in p['ethnicity_categories']:
            p['I^' + eth] = np.where(p['ethnicity'] == eth)[0]

        # Calculate simpson index for overall class as a baseline
        p['baseline_simpson_index_eth'] = round(1 - np.sum([(len(
            p['I^' + eth]) * (len(p['I^' + eth]) - 1)) / (p['N'] * (p['N'] - 1)) for eth in
                                                            p['ethnicity_categories']]), 2)

    # SOC and Gender cadets standardized like above
    if 'SOCs' in p:
        for soc in p['SOCs']:
            p[f'I^{soc.upper()}'] = np.where(p[soc])[0]
    if 'male' in p:
        p['I^Male'] = np.where(p['male'])[0]
        p['I^Female'] = np.where(p['male'] == 0)[0]

    # STEM cadets
    if 'stem' in p:
        p['I^STEM'] = np.where(p['stem'])[0]

        if 'afscs_stem' in p:
            p['J^STEM'] = np.where(p['afscs_stem'] == 'Yes')[0]
            p['J^Not STEM'] = np.where(p['afscs_stem'] == 'No')[0]
            p['J^Hybrid'] = np.where(p['afscs_stem'] == 'Hybrid')[0]

    # Selected AFSCs
    if 'c_selected_matrix' in p:
        p['J^Selected'] = {}  # The cadet selected the AFSC as a preference
        p['J^Selected-E'] = {}  # The cadet selected the AFSC as a preference and is eligible for it
        for i in p['I']:
            p['J^Selected'][i] = np.where(p['c_selected_matrix'][i])[0]
            p['J^Selected-E'][i] = np.intersect1d(p['J^E'][i], p['J^Selected'][i])
        p['num_choices_selected'] = np.array(len(p['J^Selected'][i]) for i in p['I'])

    # Incorporate "Must Match" information for OTS
    if 'must_match' in p:
        p['I^Must_Match'] = np.where(p['must_match'] == 1)[0]

    # Last/Second to Last AFSCs
    if 'last_afsc' in p and 'second_to_last_afscs' in p:
        p['J^Bottom 2 Choices'] = {}
        p['J^Last Choice'] = {}

        # Loop through each cadet
        for i in p['I']:

            # Get a list of AFSC indices of the bottom 2 choices (but not last choice)
            if type(p['second_to_last_afscs'][i]) == str:
                afsc_list = p['second_to_last_afscs'][i].split(',')
            else:
                afsc_list = [""]

            p['J^Bottom 2 Choices'][i] = []
            for afsc in afsc_list:
                afsc = afsc.strip()
                if afsc in p['afscs']:
                    j = np.where(p['afscs'] == afsc)[0][0]
                    p['J^Bottom 2 Choices'][i].append(j)
            p['J^Bottom 2 Choices'][i] = np.array(p['J^Bottom 2 Choices'][i])

            # Get the last AFSC choice index
            p['J^Last Choice'][i] = p['M']  # Unmatched AFSC (*)
            if p['last_afsc'][i] in p['afscs']:
                p['J^Last Choice'][i] = np.where(p['afscs'] == p['last_afsc'][i])[0][0]

    # Add in ineligible AFSC information
    p['Ineligible AFSC Preferences'] = np.array(
        [', '.join([p['afscs'][j] for j in p['cadet_preferences'][i] if j not in p['J^E'][i]]) for i in p['I']])
    p['first_eligible_choice'] = np.array([
        next((index + 1 for index, j in enumerate(p['cadet_preferences'][i])
              if j in p['J^E'][i]), 0)
        for i in p['I']
    ])

    return p


def set_ots_must_matches(parameters):
    """
    Identify OTS Candidates Who Must Be Matched in the Assignment.

    This function determines which Officer Training School (OTS) cadets must be assigned (i.e., matched) by
    identifying the top candidates based on their Order of Merit (OM) scores. It updates the `must_match`
    array to indicate mandatory match requirements, and adds a new set `I^Must_Match` containing the indices
    of cadets who must be assigned a slot.

    If OTS is not a participating source of commissioning (SOC) in the instance, the function exits early
    with no modifications.

    Parameters:
    --------
    - parameters (dict): Dictionary of model parameters, including cadet index sets (`I`, `I^OTS`), merit scores,
      and SOC definitions.

    Returns:
    --------
    - dict: The updated parameters dictionary with the following changes:

        - `must_match`: N-length array with `1` for must-match cadets, `0` for others, and `NaN` for non-OTS cadets.
        - `I^Must_Match`: Set of cadet indices in `I^OTS` who are in the top ~99.5% of the OM distribution.

    Examples:
    --------
    ```python
    parameters = set_ots_must_matches(parameters)
    ```
    """

    # Shorthand
    p = parameters

    # Clear "must matches"
    p['must_match'] = np.array([np.nan for _ in p['I']])
    p['must_match'][p['I^OTS']] = 0

    # No OTS adjustments to be made
    if 'ots' not in p['SOCs']:
        print('OTS not included in this instance!! Nothing to do here.')
        return p

    # Sort OTS candidates by OM and take the top {ots_accessions} people
    sorted_by_merit = np.argsort(p['merit'])[::-1]
    ots_sorted = np.array([i for i in sorted_by_merit if i in p['I^OTS']])
    p['I^Must_Match'] = ots_sorted[:int(p['ots_accessions'] * 0.995)]
    p['must_match'][p['I^Must_Match']] = 1
    return p


def gather_degree_tier_qual_matrix(cadets_df, parameters):
    """
    Construct or Validate Degree Tier Qualification Matrix for Cadets.

    This function analyzes the provided `cadets_df` and `parameters` to determine if a valid degree
    qualification matrix (`qual`) exists. If not, or if the format differs from the expected
    "Tiers" structure, a new one is generated using CIP codes. It then computes a series of derived
    binary matrices (e.g., `eligible`, `mandatory`, `tier 1`, etc.) that describe cadet eligibility
    for each AFSC based on degree requirements.

    The degree tier qualification matrix is a critical part of the AFSC assignment model, influencing
    eligibility filtering, tier-based objective constraints, and value function evaluations.

    Parameters:
    --------
    - cadets_df (pd.DataFrame): The dataframe containing cadet qualification data. Must contain
      columns like `qual_AFSC` or CIP fields if generating the qualification matrix.
    - parameters (dict): Instance parameter dictionary (`p`) that includes AFSCs, CIP codes,
      qualification type, and degree tier expectations. This dictionary will be modified in place.

    Returns:
    --------
    - dict: Updated `parameters` dictionary with the following keys (if applicable):

        - `"qual"`: The constructed or validated NxM string matrix of qualification levels.
        - `"eligible"` / `"ineligible"`: Binary matrices indicating AFSC eligibility.
        - `"mandatory"` / `"desired"` / `"permitted"`: Binary matrices based on tier requirements.
        - `"tier 1"` to `"tier 4"`: Tier-specific binary matrices.
        - `"exception"`: Binary matrix marking cadets eligible through exception rules.
        - `"t_count"`: Array of number of degree tiers per AFSC.
        - `"t_proportion"`: Matrix with expected proportions for each tier per AFSC.
        - `"t_eq"` / `"t_geq"` / `"t_leq"`: Binary matrices specifying how tier requirements should be interpreted.

    Examples:
    --------
    ```python
    p = gather_degree_tier_qual_matrix(cadets_df, p)
    ```

    See Also:
    --------
    - [`cip_to_qual_tiers`](../../../afccp/reference/data/support/#data.support.cip_to_qual_tiers):
      Generates tier-based qualification matrix from CIP codes.
    """

    # Shorthand
    p = parameters

    # Determine if there is already a qualification matrix in the Cadets dataframe, and what "type" it is
    afsc_1, afsc_M = p["afscs"][0], p["afscs"][p["M"] - 1]
    current_qual_type = "None"  # If there is no qualification matrix, we'll have to generate it
    if cadets_df is not None:
        if "qual_" + afsc_1 in cadets_df:
            qual = np.array(cadets_df.loc[:, "qual_" + afsc_1: "qual_" + afsc_M]).astype(str)
            test_qual = str(qual[0, 0])  # Variable to determine if we need to alter the qualification matrix

            # Determine the type of qual matrix we *currently* have
            if len(test_qual) == 1:
                if test_qual in ["1", "0"]:
                    current_qual_type = "Binary"
                else:
                    current_qual_type = "Relaxed"
            else:
                current_qual_type = "Tiers"

    # If the current qualification matrix matches the one we want, then we don't need to do anything
    generate_qual_matrix = False
    if p["Qual Type"] != current_qual_type:

        # We don't have a qual matrix at all (We will generate the "Tiers" qual matrix!)
        if current_qual_type == "None":
            generate_qual_matrix = True

        # We have a qual matrix and have specified that we want to keep it the way it is (Don't need to generate it)
        elif p["Qual Type"] == "Consistent":
            p["Qual Type"] = current_qual_type

        # We have a qual matrix, but want to generate the "Tiers" qual matrix
        elif p["Qual Type"] == "Tiers":
            generate_qual_matrix = True

        else:
            generate_qual_matrix = True
            print("WARNING. The degree_qual_type parameter '" + p["Qual Type"] +
                  " specified but current qual matrix is of type '" + current_qual_type +
                  "'. We no longer generate that kind of qual matrix. We will generate a 'Tiers' qual matrix.")

    # If we're going to generate a qual matrix, it'll be a "Tiers" matrix
    if generate_qual_matrix:
        p["Qual Type"] = "Tiers"

        if "cip1" in p:
            if "cip2" in p:
                qual = afccp.data.support.cip_to_qual_tiers(
                    p["afscs"][:p["M"]], p['cip1'], cip2=p['cip2'])
            else:
                qual = afccp.data.support.cip_to_qual_tiers(
                    p["afscs"][:p["M"]], p['cip1'])
        else:
            raise ValueError("Error. Need to update the degree tier qualification matrix to include tiers "
                             "('M1' instead of 'M' for example) but don't have CIP codes. Please incorporate this.")

    # Determine the binary matrices for cadets based on their degree tiers and/or eligibility
    if p["Qual Type"] == "Tiers":

        # NxM matrices with various features
        p["ineligible"] = (np.core.defchararray.find(qual, "I") != -1) * 1
        p["eligible"] = (p["ineligible"] == 0) * 1
        for t in [1, 2, 3, 4]:
            p["tier " + str(t)] = (np.core.defchararray.find(qual, str(t)) != -1) * 1
        p["mandatory"] = (np.core.defchararray.find(qual, "M") != -1) * 1
        p["desired"] = (np.core.defchararray.find(qual, "D") != -1) * 1
        p["permitted"] = (np.core.defchararray.find(qual, "P") != -1) * 1

        # NEW: Exception to degree qualification based on CFM ranks
        p["exception"] = (np.core.defchararray.find(qual, "E") != -1) * 1

        # Error Handling
        if "Deg Tiers" not in p:
            raise ValueError("Error. Degree qualification matrix is 'Tiers' category ('M1' instead of 'M' for example)"
                             " and 'Deg Tier X' columns not found in 'AFSCs' dataframe. Please correct this issue.")

    elif p["Qual Type"] == "Relaxed":
        p['ineligible'] = (qual == 'I') * 1
        p['eligible'] = (p['ineligible'] == 0) * 1
        p['mandatory'] = (qual == 'M') * 1
        p['desired'] = (qual == 'D') * 1
        p['permitted'] = (qual == 'P') * 1

        # NEW: Exception to degree qualification based on CFM ranks
        p['exception'] = (qual == 'E') * 1

    else:  # 'Binary'
        p['ineligible'] = (qual == 0) * 1
        p['eligible'] = qual

    # Force string type!
    p['Deg Tiers'][pd.isnull(p["Deg Tiers"])] = ''
    p['Deg Tiers'] = p['Deg Tiers'].astype(str)

    # Load in Degree Tier information for each AFSC
    if p["Qual Type"] == "Tiers":

        # Initialize information for AFSC degree tiers
        p["t_count"] = np.zeros(p['M']).astype(int)
        p["t_proportion"] = np.zeros([p['M'], 4])
        p["t_leq"] = (np.core.defchararray.find(p["Deg Tiers"], "<") != -1) * 1
        p["t_geq"] = (np.core.defchararray.find(p["Deg Tiers"], ">") != -1) * 1
        p["t_eq"] = (np.core.defchararray.find(p["Deg Tiers"], "=") != -1) * 1
        p["t_mandatory"] = (np.core.defchararray.find(p["Deg Tiers"], "M") != -1) * 1
        p["t_desired"] = (np.core.defchararray.find(p["Deg Tiers"], "D") != -1) * 1
        p["t_permitted"] = (np.core.defchararray.find(p["Deg Tiers"], "P") != -1) * 1

        # Loop through each AFSC
        for j, afsc in enumerate(p["afscs"][:p['M']]):

            # Loop through each potential degree tier
            for t in range(4):
                val = p["Deg Tiers"][j, t]

                # Empty degree tier
                if val in ["nan", ""] or pd.isnull(val):
                    t -= 1
                    break

                # Degree Tier Proportion
                p["t_proportion"][j, t] = val.split(" ")[2]

            # Num tiers
            p["t_count"][j] = t + 1

    # Save qual matrix and return the instance p
    p["qual"] = qual
    return p


# __________________________________________CONSTRAINED AFSC RESOLUTIONS________________________________________________
def collect_individual_constraint_methods(parameters):
    """
    Collect individual cadet AFSC restrictions from all active constraint methods.

    Sources
    -------
    Manual:
        Imported from 'manually_constrained_afscs' as comma-delimited AFSC names.

    High OM:
        Imported from 'J^High OM Constrained AFSCs', populated by the
        High OM Receive Preference feature.

    Rated Matched:
        Imported from 'J^Fixed', containing cadet-to-AFSC assignments.

    Rated Reserved:
        Imported from 'J^Reserved', containing allowable AFSC indices.
        The final AFSC in the reservation list identifies the reserved slot.

    Each method is preserved independently in 'J^Constraint Methods'.
    No intersections or manual overrides are applied at this stage.

    Parameters
    ----------
    parameters : dict
        Problem parameters after feature resolution and rated incorporation.

    Returns
    -------
    dict
        Updated parameters containing 'J^Constraint Methods'.

    Notes
    -----
    - AFSC names are translated to integer indices using 'afscs'.
    - Invalid manual AFSC names are retained as validation issues.
    - Rated matched and reserved dictionaries are assumed to contain only
      rated-algorithm restrictions.
    - The collection is rebuilt from scratch on every invocation.
    """

    # ___________________________________________________INITIALIZATION_________________________________________________
    p = parameters
    methods = {}
    issues = []
    afsc_lookup = {str(afsc): int(j) for j, afsc in enumerate(p['afscs'][:p['M']])}

    # Helper to register individual constraint methods
    def add_method(i, method, afsc_indices):
        i = int(i)
        if i < 0 or i >= p['N']:
            issues.append(f"Invalid cadet index {i} for method '{method}'.")
            return
        methods.setdefault(i, {})[method] = np.asarray(afsc_indices, dtype=int).flatten()

    # __________________________________________________MANUAL CONSTRAINTS______________________________________________
    manual_values = p.get('manually_constrained_afscs', np.full(p['N'], '', dtype=object))
    if len(manual_values) != p['N']:
        raise ValueError("Manual AFSC constraint array length does not match N.")

    for i, value in enumerate(manual_values):

        # Skip empty values, including NaN values from CSV imports
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            continue

        afsc_names = [afsc.strip() for afsc in str(value).split(',') if afsc.strip()]

        if not afsc_names:
            continue

        # Preserve invalid AFSCs as issues rather than silently discarding them
        invalid_afscs = [afsc for afsc in afsc_names if afsc not in afsc_lookup]
        if invalid_afscs:
            issues.append(f"Cadet {i} - Manual: Unknown AFSC(s): {', '.join(invalid_afscs)}")
            continue

        add_method(i, 'Manual', [afsc_lookup[afsc] for afsc in afsc_names])

    # ______________________________________________HIGH OM CONSTRAINTS_________________________________________________
    if p.get('high_om_preference', False):

        preference_number = p.get('high_om_preference_number', 3)
        method_name = f'Top {preference_number} Choice'

        for i, afsc_indices in p.get('J^High OM Constrained AFSCs', {}).items():
            add_method(i, method_name, afsc_indices)

    # ___________________________________________RATED MATCHED CONSTRAINTS______________________________________________
    for i, j in p.get('J^Fixed', {}).items():

        j = int(j)

        if not 0 <= j < p['M']:
            issues.append(f"Cadet {i} - Rated Matched: Invalid AFSC index {j}.")
            continue

        method_name = f'Rated Matched to {p["afscs"][j]}'
        add_method(i, method_name, [j])

    # ___________________________________________RATED RESERVED CONSTRAINTS_____________________________________________
    for i, afsc_indices in p.get('J^Reserved', {}).items():

        afsc_indices = np.asarray(afsc_indices, dtype=int).flatten()

        if len(afsc_indices) == 0:
            issues.append(f"Cadet {i} - Rated Reserved: Empty reservation list.")
            continue

        # The last AFSC identifies the rated slot being reserved
        j_reserved = int(afsc_indices[-1])

        if not 0 <= j_reserved < p['M']:
            issues.append(f"Cadet {i} - Rated Reserved: Invalid reserved AFSC index {j_reserved}.")
            continue

        method_name = f'Rated Reserved for {p["afscs"][j_reserved]}'
        add_method(i, method_name, afsc_indices)

    # ________________________________________________UPDATE PARAMETERS_________________________________________________
    p['J^Constraint Methods'] = methods
    p['Individual Constraint Issues'] = issues
    return p


def resolve_individual_constraint_methods(parameters):
    """
    Resolve all collected individual cadet AFSC constraints.

    Resolution Rules
    ----------------
    1. Manual constraints are authoritative and supersede all other methods.
    2. Without Manual, all applicable method AFSC sets are intersected.
    3. Ineligible AFSCs are automatically removed from nonmanual restrictions.
    4. Ineligible manually constrained AFSCs are preserved and flagged during validation.
    5. Every applicable method is retained in the method description.
    6. Cadets without constraints are omitted from 'J^Constrained'.
    7. Empty intersections or eligibility-filtered sets are preserved for validation.

    Parameters
    ----------
    parameters : dict
        Problem parameters containing 'J^Constraint Methods'.

    Returns
    -------
    dict
        Updated parameters containing:
        - J^Constrained
        - constraint_methods
        - constrained_afscs
        - I^Constrained
        - I^Manual Override
        - Individual Constraint Filtered AFSCs

    Notes
    -----
    The function reconstructs all resolved constraints from their original
    method restrictions on every invocation.

    Manual restrictions are never modified by eligibility filtering. For
    nonmanual restrictions, only eligible AFSCs are retained in the final
    allowable AFSC set.
    """

    # ___________________________________________________INITIALIZATION_________________________________________________

    p = parameters
    methods = p['J^Constraint Methods']

    constrained = {}
    constraint_methods = np.full(p['N'], '', dtype=object)
    constrained_afscs = np.full(p['N'], '', dtype=object)
    manual_overrides = []
    filtered_afscs = {}

    # ______________________________________________RESOLVE INDIVIDUAL CADETS___________________________________________
    for i, cadet_methods in methods.items():

        # Preserve all applicable methods in their original collection order
        constraint_methods[i] = ', '.join(cadet_methods.keys())

        # Manual constraints supersede every other method
        if 'Manual' in cadet_methods:
            resolved = set(cadet_methods['Manual'])

            if len(cadet_methods) > 1:
                manual_overrides.append(i)

        # Otherwise, intersect all applicable method restrictions
        else:
            method_sets = [set(indices) for indices in cadet_methods.values()]
            resolved = set.intersection(*method_sets) if method_sets else set()

            # Remove ineligible AFSCs from nonmanual restrictions
            eligible = set(p['J^E'][i])
            ineligible = resolved - eligible

            if ineligible:
                filtered_afscs[i] = np.asarray(sorted(ineligible), dtype=int)

            resolved &= eligible

        # Preserve the cadet's original preference order where possible
        preference_order = list(p['cadet_preferences'][i])
        ordered = [j for j in preference_order if j in resolved]
        ordered.extend(sorted(resolved - set(ordered)))

        # Store final allowable AFSC indices
        constrained[i] = np.asarray(ordered, dtype=int)

        # Store CSV-ready AFSC names
        constrained_afscs[i] = ', '.join(str(p['afscs'][j]) for j in ordered if 0 <= j < p['M'])

    # ________________________________________________UPDATE PARAMETERS_________________________________________________
    p['J^Constrained'] = constrained
    p['constraint_methods'] = constraint_methods
    p['constrained_afscs'] = constrained_afscs

    p['I^Constrained'] = np.asarray(sorted(constrained), dtype=int)
    p['I^Manual Override'] = np.asarray(manual_overrides, dtype=int)
    p['Individual Constraint Filtered AFSCs'] = filtered_afscs

    return p


def validate_individual_cadet_constraints(parameters):
    """
    Validate resolved individual cadet AFSC restrictions and generate a report.

    Checks
    ------
    - Invalid AFSC names or indices from collected methods.
    - Empty allowable AFSC sets caused by conflicting restrictions.
    - Empty allowable AFSC sets after eligibility filtering.
    - Final AFSCs outside the model's valid AFSC index range.
    - Ineligible manually constrained AFSCs.

    Nonmanual ineligible AFSCs are removed during resolution and do not
    constitute validation errors unless no allowable AFSCs remain.

    Manual overrides are intentional and reported informationally.

    Parameters
    ----------
    parameters : dict
        Problem parameters containing resolved constraint structures.

    Returns
    -------
    str
        Standardized, color-coded constraint resolution report.

    Raises
    ------
    ValueError
        If one or more invalid constraints are detected.
    """

    # ___________________________________________________INITIALIZATION_________________________________________________

    p = parameters
    methods = p['J^Constraint Methods']
    constrained = p['J^Constrained']
    filtered_afscs = p.get('Individual Constraint Filtered AFSCs', {})
    issues = list(p.get('Individual Constraint Issues', []))

    colors = {'ACTIVE': '\033[92m', 'INACTIVE': '\033[90m',
              'ERROR': '\033[91m', 'VALID': '\033[92m'}
    color = lambda status: f"{colors.get(status, '')}{status}\033[0m"
    status_line = lambda label, status: f"    {label}: {color(status)}"

    # ___________________________________________________VALIDATE CONSTRAINTS_________________________________________________

    for i, afsc_indices in constrained.items():

        cadet = str(p['cadets'][i]) if 'cadets' in p else str(i)
        method_names = ', '.join(methods[i])

        # Empty allowable AFSC set
        if len(afsc_indices) == 0:

            if i in filtered_afscs:
                issues.append(
                    f"Cadet {cadet}: No eligible AFSCs remain after resolving "
                    f"[{method_names}] and applying eligibility restrictions."
                )
            else:
                issues.append(
                    f"Cadet {cadet}: No allowable AFSCs after resolving [{method_names}]."
                )

            continue

        # Invalid AFSC indices
        invalid = [int(j) for j in afsc_indices if not 0 <= int(j) < p['M']]

        if invalid:
            issues.append(f"Cadet {cadet}: Invalid AFSC indices: {invalid}.")
            continue

        # Manual AFSC eligibility conflicts are errors
        if 'Manual' in methods[i]:
            ineligible = [int(j) for j in afsc_indices if j not in p['J^E'][i]]

            if ineligible:
                afsc_names = ', '.join(str(p['afscs'][j]) for j in ineligible)
                issues.append(
                    f"Cadet {cadet}: Manually constrained to ineligible AFSC(s): {afsc_names}."
                )

        # Nonmanual restrictions should already have been eligibility-filtered
        else:
            ineligible = [int(j) for j in afsc_indices if j not in p['J^E'][i]]

            if ineligible:
                afsc_names = ', '.join(str(p['afscs'][j]) for j in ineligible)
                issues.append(
                    f"Cadet {cadet}: Ineligible AFSC(s) remain after filtering: {afsc_names}."
                )

    # ___________________________________________________CONSTRUCT REPORT_________________________________________________

    num_constrained = len(constrained)
    num_manual = sum('Manual' in m for m in methods.values())
    num_high_om = sum(any(name.startswith('Top ') and name.endswith(' Choice') for name in m) for m in methods.values())
    num_matched = sum(any(name.startswith('Rated Matched to ') for name in m) for m in methods.values())
    num_reserved = sum(any(name.startswith('Rated Reserved for ') for name in m) for m in methods.values())
    num_overrides = len(p['I^Manual Override'])

    num_filtered_cadets = len(filtered_afscs)
    num_filtered_afscs = sum(len(indices) for indices in filtered_afscs.values())

    resolution_status = 'ERROR' if issues else 'ACTIVE'

    report = [
        '[Individual Cadet Constraint Resolution]',
        '  Details:',
        f'    Total Cadets: {p["N"]:,}',
        f'    Constrained Cadets: {num_constrained:,}',
        f'    Manual Constraints: {num_manual:,}',
        f'    High OM Preference Constraints: {num_high_om:,}',
        f'    Rated Matched Constraints: {num_matched:,}',
        f'    Rated Reserved Constraints: {num_reserved:,}',
        f'    Manual Overrides: {num_overrides:,}',
        f'    Cadets with Ineligible AFSCs Filtered: {num_filtered_cadets:,}',
        f'    Total Ineligible AFSCs Removed: {num_filtered_afscs:,}',
        '  Resolution:',
        status_line('Status', resolution_status)
    ]

    if issues:
        report.append(f'    Validation failed with {len(issues)} issue(s).')
        report.extend(['  Issues:', *(f'    - {issue}' for issue in issues)])
    else:
        report.append(f'    Successfully validated {num_constrained:,} constrained cadets.')

    # ___________________________________________________FINAL VALIDATION_________________________________________________

    report_str = '\n'.join(report)

    if issues:
        raise ValueError(report_str)

    return report_str


# ______________________________________________BASE & IST ADJUSTMENTS__________________________________________________
def base_training_parameter_additions(parameters, printing=False):
    """
    Add all derived base- and training-related parameters required by the AFCCP model.

    This is the main orchestration function for preparing base assignment, training
    assignment, cadet state, and capacity data after the core parameter dictionary
    has already been created. It does not introduce new optimization logic; instead,
    it builds the derived sets and dictionaries that make the downstream optimization
    model easier to formulate.

    At a high level, this function performs three major tasks. First, it constructs
    the overall base-related structures, including which AFSCs participate in base
    and course assignment, which bases are available for each AFSC, and which bases
    are feasible for each cadet once cadet preferences and AFSC eligibility are
    combined. Second, it builds cadet-specific "states" that describe how different
    groups of AFSC outcomes should be evaluated based on the cadet's thresholds and
    which objectives matter in those regions of their preference list. Third, it
    computes course eligibility, course utility, and lower/upper assignment bounds
    for both bases and courses.

    The result is a parameter dictionary that contains the additional structure
    needed to support optimization formulations involving AFSC assignment, base
    assignment, and training-course assignment in a unified way.

    Parameters
    ----------
    - parameters : dict
        Dictionary of AFCCP model parameters. This dictionary is expected to already
        contain the core cadet, AFSC, preference, base, and training inputs used to
        derive the additional structures created here.

    Returns
    -------
    - dict
        The same parameter dictionary, augmented with derived base- and
        training-related sets, mappings, utilities, and assignment bounds.

    Examples
    --------
    ```python
    parameters = base_training_parameter_additions(parameters)
    ```

    See Also
    --------
    - [`_add_base_overall_parameters`](../../../reference/module/path/#module.path._add_base_overall_parameters)
    - [`_add_state_component_parameters`](../../../reference/module/path/#module.path._add_state_component_parameters)
    - [`_add_course_and_capacity_parameters`](../../../reference/module/path/#module.path._add_course_and_capacity_parameters)
    """
    if printing:
        print('[B/T Parameters] Adding base-state overall parameters...')
    parameters = _add_base_overall_parameters(p=parameters)
    if printing:
        print('[B/T Parameters] Adding state component parameters...')
    parameters = _add_state_component_parameters(p=parameters)
    if printing:
        print('[B/T Parameters] Adding base-state parameters...')
    parameters = _add_base_state_parameters(p=parameters)
    if printing:
        print('[B/T Parameters] Adding course-capacity parameters...')
    parameters = _add_course_and_capacity_parameters(p=parameters)
    return parameters


def _add_base_overall_parameters(p):
    """
    Build the overall base- and course-related feasibility structures for the model.

    This function creates the broad assignment framework that connects AFSCs, bases,
    courses, and cadet preferences before any cadet-specific state logic is applied.
    It defines the master set of bases and the course indices available for each
    AFSC, then categorizes AFSCs according to whether the optimization model may
    assign a base, a training course, both, or neither for that AFSC.

    It also determines which bases are available to each AFSC based on the base
    capacity inputs and converts each cadet's base preference matrix into an ordered
    list of ranked bases. From there, it combines cadet AFSC eligibility with AFSC
    base availability to determine the full set of bases each cadet could possibly
    receive, then intersects that set with the cadet's ranked bases to produce the
    final base-feasible set used by the optimization model.

    In short, this function answers three foundational questions:
    1. Which AFSCs participate in base and/or course assignment?
    2. Which bases can each AFSC assign?
    3. Which of those bases are actually feasible for each cadet once eligibility
       and preferences are both considered?

    Parameters
    ----------
    - p : dict
        Dictionary of AFCCP parameters. It must include the base/IST status for each
        AFSC, base capacity arrays, cadet base preference data, and AFSC eligibility
        information for each cadet.

    Returns
    -------
    - dict
        The parameter dictionary with additional overall base and course structures,
        including:
        - ``p['B']``: set of all bases
        - ``p['C']``: course indices for each AFSC
        - ``p['J^B']``: AFSCs that participate in base assignment
        - ``p['J^C']``: AFSCs that participate in course assignment
        - ``p['afsc_components']``: assignable components for each AFSC
        - ``p['B^A']``: bases available to each AFSC
        - ``p['B^P']``: ranked bases for each cadet
        - ``p['B^FULL']``: full base-feasible set for each cadet
        - ``p['B^E']``: optimization-feasible bases for each cadet

    Examples
    --------
    ```python
    p = _add_base_overall_parameters(p)
    ```

    See Also
    --------
    - [`_add_state_component_parameters`](../../../reference/module/path/#module.path._add_state_component_parameters)
    - [`base_training_parameter_additions`](../../../reference/module/path/#module.path.base_training_parameter_additions)
    """

    # Sets of bases and courses
    p['B'] = np.arange(p['S'])
    p['C'] = {t: np.arange(p['Q'][t]) for t in p['T']}

    # All unique course names
    p['all_courses'] = [course for t in p['T'] for course in p['courses'][t]]

    # Split up AFSCs into categories based on what the optimization model will assign
    p['J^B'] = np.array(  # Set of AFSCs that assign cadets to bases in the optimization model
        [j for j in p['J'] if p['base_ist_status'][j] in ['Opt Base, Opt IST', 'Opt Base, No IST']])
    p['J^T'] = np.array(  # Set of AFSCs that assign cadets to courses in the optimization model
        [j for j in p['J'] if p['base_ist_status'][j] in ['Opt Base, Opt IST', 'Fixed Base, Opt IST']])

    # Dictionary mapping each AFSC and the components they can match in the optimization model
    p['afsc_components'] = {}
    for j, status in enumerate(p['base_ist_status']):
        if status in ['No Base, No IST', 'UPT Base & IST']:
            p['afsc_components'][j] = ['afsc']  # These AFSCs don't match base/IST (at least in the opt. model)
        elif status == 'Opt Base, No IST':  # These AFSCs can match to a base, but not a training course
            p['afsc_components'][j] = ['afsc', 'base']
        elif status == 'Fixed Base, Opt IST':  # These AFSCs can match to a training course, but the base is fixed
            p['afsc_components'][j] = ['afsc', 'course']
        elif status == 'Opt Base, Opt IST':  # These AFSCs can do both IST and base assignments in the opt. model
            p['afsc_components'][j] = ['afsc', 'base', 'course']
        else:
            raise ValueError(f'Error. AFSC "{p["afscs"][j]}" Base/IST status "{status}" not valid.')

    # Set of bases that AFSC j may assign cadets to
    p['B^A'] = {j: np.where(p['base_max'][:, j] > 0)[0] for j in p['J']}

    # Ordered set of bases that each cadet has ranked in their preferences
    p['B^P'] = {}
    for i in p['I']:
        sorted_indices = np.argsort(p['b_pref_matrix'][i, :])
        p['B^P'][i] = np.array([j for j in sorted_indices if p['b_pref_matrix'][i, j] != 0])

    # Set of bases that cadet i may be assigned to (based on the union of all eligible bases from AFSCs in J^E_i)
    p['B^FULL'] = {}
    for i in p['I']:
        eligible_j = np.intersect1d(p['J^E'][i], p['J^B'])
        if len(eligible_j) == 0:
            p['B^FULL'][i] = np.array([])
        else:
            p['B^FULL'][i] = np.unique(
                np.concatenate([p['B^A'][j] for j in eligible_j])
            )

    # Set of bases that cadet i may be assigned to within the optimization model (had to preference it)
    p['B^E'] = {i: np.intersect1d(p['B^P'][i], p['B^FULL'][i]) for i in p['I']}  # Initially...will be reduced (states)

    # Load in base preferences
    p['base_preferences'] = {}
    for i in p['I']:
        prefs = p['b_pref_matrix'][i]

        # Bases the cadet ranked
        positive_idx = np.where(prefs > 0)[0]
        if len(positive_idx) == 0:
            p['base_preferences'][i] = np.array([])
        else:
            # Sort by preference value ASCENDING (1 = best)
            p['base_preferences'][i] = positive_idx[np.argsort(prefs[positive_idx])]

    return p


def _add_state_component_parameters(p):
    """
    Construct cadet-specific AFSC outcome states and the objective structure associated
    with each state.

    This function is one of the most important pieces of the base/training parameter
    setup because it translates a cadet's ordered AFSC preferences into a set of
    decision-relevant "states." These states are used to determine how the model
    should evaluate an assignment depending on where the assigned AFSC falls in the
    cadet's preference ordering and which additional assignment components still
    matter in that region of the ranking.

    The key idea is that not every AFSC outcome should be evaluated using the same
    combination of objectives. For example, for highly preferred AFSCs a cadet may
    only care about receiving the AFSC itself, while for lower-ranked AFSCs the cadet
    may also care about the assigned base, training course, or both. The cadet's
    base and training thresholds define where those evaluation regions begin and end.
    This function first uses those thresholds to partition the cadet's AFSC preference
    list into a small number of primary states. Each primary state is associated with
    a set of included objectives such as ``['afsc']``, ``['afsc', 'base']``,
    ``['afsc', 'course']``, or ``['afsc', 'base', 'course']``.

    The function then refines those primary states into final sub-states. This extra
    refinement is needed because AFSCs within the same primary region may not all
    support the same assignable components in the optimization model. Some AFSCs may
    permit base assignment, some may permit course assignment, some may permit both,
    and some may effectively lose a component for a specific cadet if that cadet did
    not rank any feasible base for that AFSC. To handle this, the function reduces
    each AFSC to the subset of components that both:
    1. are supported by the AFSC in the model, and
    2. matter to the cadet in the current primary state.

    AFSCs with the same reduced component structure are then grouped together into the
    same final state. For each final state, the function stores:
    - the AFSCs that belong to that state,
    - the objective components relevant to that state,
    - the normalized weights on AFSC, base, and course objectives within that state,
    - the state utility anchor based on the top AFSC in the primary state, and
    - the set of feasible bases associated with the AFSCs in that state.

    Conceptually, this function creates the bridge between raw cadet preferences and
    a state-based preference model that can be used cleanly in optimization. It tells
    the model not just *which AFSCs* a cadet likes, but *how assignments should be
    evaluated* when those AFSCs are reached in different regions of the cadet's
    preference ordering.

    Parameters
    ----------
    - p : dict
        Dictionary of AFCCP parameters. This dictionary is expected to already include
        cadet thresholds, AFSC preference orderings, AFSC component definitions, base
        feasibility sets, cadet utilities, and objective weights.

    Returns
    -------
    - dict
        The parameter dictionary with cadet-specific state structures added,
        including:
        - ``p['D']``: final states defined for each cadet
        - ``p['Cadet Objectives']``: objective-component label for each state
        - ``p['J^State']``: AFSCs belonging to each cadet-state
        - ``p['w^A']``: AFSC weight in each state
        - ``p['w^B']``: base weight in each state
        - ``p['w^C']``: course weight in each state
        - ``p['u^S']``: utility anchor for each state
        - ``p['B^State']``: feasible bases associated with each state

    Examples
    --------
    ```python
    p = _add_state_component_parameters(p)
    ```

    See Also
    --------
    - [`_add_base_overall_parameters`](../../../reference/module/path/#module.path._add_base_overall_parameters)
    - [`_add_course_and_capacity_parameters`](../../../reference/module/path/#module.path._add_course_and_capacity_parameters)
    - [`base_training_parameter_additions`](../../../reference/module/path/#module.path.base_training_parameter_additions)
    """

    # Sets/Parameters for AFSC outcome states for each cadet
    p['D'] = {}  # Set of all AFSC outcome states that cadet i has designated
    p['Cadet Objectives'] = {}  # Set of cadet objectives included for each cadet and each state
    p['J^State'] = {}  # Set of AFSCs that, if assigned, would put cadet i into state d
    p['w^A'] = {}  # the weight that cadet i places on AFSCs in state d
    p['w^B'] = {}  # the weight that cadet i places on bases in state d
    p['w^C'] = {}  # the weight that cadet i places on courses in state d
    p['u^S'] = {}  # the maximum utility that cadet i receives from state d (based on best AFSC)
    p['B^State'] = {}  # Set of bases that cadet i can be assigned to in state d (According to J^State_id)
    p['T^State'] = {}  # Set of training AFSCs that cadet i can be assigned to in state d (According to J^State_id)

    # Determine the "states" for each cadet based on the differences of AFSC outcomes
    for i in p['I']:

        # Base/Training Thresholds (Shorthand)
        bt, tt = p['base_threshold'][i], p['training_threshold'][i]

        # Determine "primary" set of AFSCs and states based on thresholds
        if bt < tt:  # Base Threshold is less than Training
            afscs = {1: p['cadet_preferences'][i][:bt],
                     2: p['cadet_preferences'][i][bt: tt],
                     3: p['cadet_preferences'][i][tt:]}
            included = {1: ['afsc'], 2: ['afsc', 'base'], 3: ['afsc', 'base', 'course']}
        elif tt < bt:  # Training Threshold is less than Base
            afscs = {1: p['cadet_preferences'][i][:tt],
                     2: p['cadet_preferences'][i][tt: bt],
                     3: p['cadet_preferences'][i][bt:]}
            included = {1: ['afsc'], 2: ['afsc', 'course'], 3: ['afsc', 'base', 'course']}
        else:  # They're equal!
            afscs = {1: p['cadet_preferences'][i][:bt],
                     2: p['cadet_preferences'][i][bt:]}
            included = {1: ['afsc'], 2: ['afsc', 'base', 'course']}

        # Ensure we are only dealing with AFSCs that the cadet is eligible!!
        for state, afsc_specifics in afscs.items():
            afscs[state] = np.array([j for j in afsc_specifics if j in p['J^E'][i]])

        # Sets/Parameters for AFSC outcome states for each cadet
        p['D'][i] = []  # Set of all AFSC outcome states that cadet i has designated
        p['Cadet Objectives'][i] = {}  # Set of cadet objectives included for each cadet and each state
        p['J^State'][i] = {}  # Set of AFSCs that, if assigned, would put cadet i into state d
        p['w^A'][i] = {}  # the weight that cadet i places on AFSCs in state d
        p['w^B'][i] = {}  # the weight that cadet i places on bases in state d
        p['w^C'][i] = {}  # the weight that cadet i places on courses in state d
        p['u^S'][i] = {}  # the maximum utility that cadet i receives from state d (based on best AFSC)
        p['B^State'][i] = {}  # Set of bases that cadet i can be assigned to in state d (According to J^State_id)
        p['T^State'][i] = {}  # Set of training AFSCs cadet i can be assigned to in state d (According to J^State_id)

        # Handle niche case where an AFSC is in a given state where bases matter to the cadet, and the AFSC...
        # ...can assign bases, but the cadet did not rank a base that the AFSC can assign them to. In this case,...
        # ...we treat that AFSC as if it can't assign a base (just for this specific cadet)
        afsc_components_for_cadet = copy.deepcopy(p['afsc_components'])
        for j, components in p['afsc_components'].items():

            # The number of bases that the AFSC can assign a cadet to (B^A) and that the cadet has ranked (B^P)
            num_eligible_bases = len(np.intersect1d(p['B^A'][j], p['B^P'][i]))

            # If the AFSC can assign bases, but they're not in the cadet's preferences, then we can't assign a base
            if 'base' in components and num_eligible_bases == 0:  # to the cadet in the optimization model
                afsc_components_for_cadet[j].remove('base')

        # Loop through each "primary" state to get "final" states (Split up states based on "extra" assignments)
        d = 1
        for state in included:

            # Empty state!
            if len(afscs[state]) == 0:
                continue

            # Reduce the AFSC's components down to what the cadet cares about in this state
            reduced_components = {j: [c for c in afsc_components_for_cadet[j] if c in included[state]] for j in
                                  afscs[state]}

            # Get unique reduced components (for breaking up primary state into sub-states)
            unique_comps_categories = []
            for j, reduced_comps in reduced_components.items():
                if reduced_comps not in unique_comps_categories:
                    unique_comps_categories.append(reduced_comps)

            # Little confusing, but we want to be able to handle the unique components as a list
            # and as a string (can't make a dictionary key a list)
            unique_comps_names = [  # Unique components List: [['afsc'], ['afsc', 'base'],
                # ['afsc', 'base', 'course']] -> ['afsc', 'afsc-base', 'afsc-base-course']
                '-'.join(unique_comps) for unique_comps in unique_comps_categories]

            # Dictionary of str: list {'afsc-base': ['afsc', 'base], 'afsc': ['afsc'], ...}
            unique_comps_dict = {component_type: unique_comps_categories[idx] for idx, component_type in
                                 enumerate(unique_comps_names)}

            # Now we can flip the reduced components idea and group AFSCs together based on what unique...
            components_afscs = {component_type: [] for component_type in unique_comps_names}  # ...component they're in
            # Create something like {'afsc': [5, 0], 'afsc-base': [2, 1], ...}
            for component_type, unique_comps in unique_comps_dict.items():
                components_afscs[component_type] = [j for j, comps in reduced_components.items() if
                                                    comps == unique_comps]

            # We can now loop through each sub-state
            for component_type, afscs_in_set in components_afscs.items():

                # Add information for this sub-state
                p['D'][i].append(d)
                p['Cadet Objectives'][i][d] = component_type
                p['J^State'][i][d] = afscs_in_set
                p['u^S'][i][d] = p['cadet_utility'][
                    i, afscs[state][0]]  # Utility of the top preferred AFSC in the primary state

                # Default zero-out weights, then re-balance weights based on included objectives
                p['w^A'][i][d], p['w^B'][i][d], p['w^C'][i][d] = 0, 0, 0
                for component in unique_comps_dict[component_type]:
                    letter = component[0].upper()
                    p[f'w^{letter}'][i][d] = p[f'weight_{component}'][i] / np.sum(
                        [p[f'weight_{obj}'][i] for obj in unique_comps_dict[component_type]])

                    if pd.isnull(p[f'w^{letter}'][i][d]):
                        print(f"Cadet {i} State {d} w^{letter} is null. "
                              f"AFSCs in State: {p['afscs'][p['J^State'][i][d]]}.\nFixing to 0.")
                        p[f'w^{letter}'][i][d] = 0

                # Determine the bases that can be assigned in this state
                if p['w^B'][i][d] > 0:  # Can only assign bases if we care about them!

                    # All bases that AFSCs in this state can assign
                    afsc_bases_in_state = reduce(np.union1d, (p['B^A'][j] for j in p['J^State'][i][d]))

                    # Intersection of bases from AFSCs in this state and AFSCs that the cadet has placed preferences for
                    p['B^State'][i][d] = np.intersect1d(afsc_bases_in_state, p['B^E'][i])

                else:  # If we don't care about bases, we can't be assigned any!
                    p['B^State'][i][d] = np.array([])

                # Determine the training AFSCs that can assign courses in this state
                if p['w^C'][i][d] > 0:  # Can only assign training afscs/course if we care about them!

                    # Just the unique mapping of training AFSCs from the regular AFSCs in this state
                    p['T^State'][i][d] = np.unique([p['tau'][j] for j in p['J^State'][i][d]])

                else:  # If we don't care about training, we can't be assigned any!
                    p['T^State'][i][d] = np.array([])

                # Next state
                d += 1

        # Print statement for specific cadet
        if i == 1 and False:  # Meant for debugging and sanity checking this logic!
            print('Cadet', i)
            print('Cadet AFSC Preferences', p['afscs'][p['cadet_preferences'][i]])
            print('Cadet Utility', p['cadet_utility'][i][p['cadet_preferences'][i]])
            for d in p['D'][i]:
                print('\n\n')
                print('State', d)
                print('Objectives', p['Cadet Objectives'][i][d])
                print('J^State', p['afscs'][p['J^State'][i][d]])
                if len(p['B^State'][i][d]) > 0:
                    print('B^State', p['bases'][p['B^State'][i][d]])
                else:
                    print('B^State', [])
                print('Weight (AFSC)', round(p['w^A'][i][d], 3))
                print('Weight (Base)', round(p['w^B'][i][d], 3))
                print('Weight (Course)', round(p['w^C'][i][d], 3))
                print('Utility (State)', round(p['u^S'][i][d], 3))

    return p


def _add_course_and_capacity_parameters(p, debug_i=None):

    import datetime

    # ------------------------------------------------------------
    # Helper Functions
    # ------------------------------------------------------------

    def to_date_str(dt):
        return dt.strftime("%Y-%m-%d")

    def to_date_safe(dt):
        if isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
            return dt
        if isinstance(dt, datetime.datetime):
            return dt.date()
        if isinstance(dt, np.datetime64):
            return pd.Timestamp(dt).to_pydatetime().date()
        if isinstance(dt, pd.Timestamp):
            return dt.to_pydatetime().date()
        if isinstance(dt, str):
            return _parse_training_date(dt)
        raise TypeError(f"Unsupported date type: {type(dt)}")

    def debug_print(i, msg=""):
        if debug_i is not None and i == debug_i:
            print(msg)

    # ------------------------------------------------------------
    # Reverse tau mapping
    # ------------------------------------------------------------
    p['J^G'] = {
        t: np.array([j for j in p['J^T'] if p['tau'][j] == t])
        for t in p['T']
    }

    # ------------------------------------------------------------
    # Eligible training AFSCs
    # ------------------------------------------------------------
    p['T^E'] = {
        i: np.unique([p['tau'][j] for j in np.intersect1d(p['J^T'], p['J^E'][i])])
        for i in p['I']
    }

    # ------------------------------------------------------------
    # Normalize Dates to datetime.date
    # ------------------------------------------------------------
    # course_start = copy.deepcopy(p['course_start'])
    for t in p['T']:
        p['course_start'][t] = np.array(
            [to_date_safe(dt) for dt in p['course_start'][t]],
            dtype=object
        )

    # training_start = np.array(
    #     [to_date_safe(dt) for dt in p['training_start']],
    #     dtype=object
    # )
    p['training_start'] = np.array(
        [to_date_safe(dt) for dt in p['training_start']],
        dtype=object
    )

    # ------------------------------------------------------------
    # Get pilot training AFSC
    # ------------------------------------------------------------
    pilot_j = [j for j in p['J'] if p['base_ist_status'][j] == 'UPT Base & IST']
    if len(pilot_j) > 0:  # Putting this fix in here for randomly generated data that doesn't use pilot
        pilot_t = p['tau'][pilot_j[0]]
    else:
        pilot_t = ''

    # ------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------
    p['C^E'] = {}
    p['course_days_cadet'] = {}
    p['course_utility'] = {}

    for i in p['I']:

        p['C^E'][i] = {}
        p['course_days_cadet'][i] = {}
        p['course_utility'][i] = {}

        cadet_start = p['training_start'][i]
        cadet_ord = cadet_start.toordinal()

        if debug_i == i:
            print("\n" + "="*75)
            print(f"CADET {i} TRAINING SUMMARY")
            print("="*75)
            print(f"Training Start Date : {to_date_str(cadet_start)}")
            print(f"Preference          : {p['training_preferences'][i]}")
            print()

        all_waits = []

        # Putting this fix in here for randomly generated data that doesn't use pilot
        if pilot_t != '':
            T_E_with_pilot = np.union1d(p['T^E'][i], np.array([pilot_t]))
        else:
            T_E_with_pilot = copy.deepcopy(p['T^E'][i])

        for t in T_E_with_pilot:

            courses = np.arange(len(p['course_start'][t]))
            starts_ord = np.fromiter(
                (d.toordinal() for d in p['course_start'][t]),
                dtype=int
            )

            waits = starts_ord - cadet_ord
            feasible_mask = waits >= 0

            feasible_courses = courses[feasible_mask]
            feasible_waits = waits[feasible_mask]

            p['C^E'][i][t] = feasible_courses
            p['course_days_cadet'][i][t] = {
                c: w for c, w in zip(feasible_courses, feasible_waits)
            }

            if len(feasible_waits) > 0:
                all_waits.extend(feasible_waits)

            # Debug AFSC Summary
            if debug_i == i and len(feasible_waits) > 0:
                earliest = min(feasible_waits)
                latest = max(feasible_waits)

                earliest_date = to_date_str(cadet_start + datetime.timedelta(days=int(earliest)))
                latest_date = to_date_str(cadet_start + datetime.timedelta(days=int(latest)))

                print(f"AFSC {t}")
                print(f"  # Feasible Courses : {len(feasible_courses)}")
                print(f"  Earliest Start     : {earliest_date} ({earliest} days)")
                print(f"  Latest Start       : {latest_date} ({latest} days)")
                print()

        # ------------------------------------------------------------
        # Wait Normalization
        # ------------------------------------------------------------
        if len(all_waits) > 0:
            min_wait = min(all_waits)
            max_wait = max(all_waits)
            denom = max_wait - min_wait if max_wait != min_wait else 1
        else:
            min_wait = max_wait = denom = 0

        # ------------------------------------------------------------
        # Utility Calculation
        # ------------------------------------------------------------
        for t in T_E_with_pilot:
            p['course_utility'][i][t] = {}

            if len(p['C^E'][i][t]) == 0:
                continue

            waits = np.array([
                p['course_days_cadet'][i][t][c]
                for c in p['C^E'][i][t]
            ])

            if p['training_preferences'][i] == 'Early':
                utilities = 1 - (waits - min_wait) / denom
            elif p['training_preferences'][i] == 'Late':
                utilities = (waits - min_wait) / denom
            else:
                utilities = np.zeros_like(waits, dtype=float)

            for c, u in zip(p['C^E'][i][t], utilities):
                p['course_utility'][i][t][c] = float(u)

    # ------------------------------------------------------------
    # Cadets who care about training
    # ------------------------------------------------------------
    p['I^T'] = np.array([
        i for i in p['I']
        if any(p['w^C'][i][d] > 0 for d in p['D'][i])
    ])

    # ------------------------------------------------------------
    # Build I^A and I^A-FULL
    # ------------------------------------------------------------
    p['I^A'] = {}
    p['I^A-FULL'] = {}

    cadets_by_t = {
        t: np.unique([i for j in p['J^G'][t] for i in p['I^E'][j]])
        for t in p['T']
    }

    I_T_set = set(p['I^T'])

    for t in p['T']:
        p['I^A'][t] = {}
        p['I^A-FULL'][t] = {}

        eligible_cadets = cadets_by_t[t]
        course_sets = {
            i: set(p['C^E'][i].get(t, []))
            for i in eligible_cadets
        }

        for c in p['C'][t]:
            feasible = [i for i in eligible_cadets if c in course_sets[i]]
            feasible_array = np.array(feasible)

            p['I^A-FULL'][t][c] = feasible_array
            p['I^A'][t][c] = np.array([i for i in feasible if i in I_T_set])

    # ------------------------------------------------------------
    # Capacity bounds
    # ------------------------------------------------------------
    p['lo^B'], p['hi^B'] = {}, {}

    for j in p['J']:
        p['lo^B'][j], p['hi^B'][j] = {}, {}
        for b in p['B']:
            p['lo^B'][j][b] = p['base_min'][b, j]
            p['hi^B'][j][b] = p['base_max'][b, j]

    p['lo^C'], p['hi^C'] = p['course_min'], p['course_max']

    return p


def _add_base_state_parameters(p):

    # ------------------------------------------------------------
    # 1) Cadets that care about bases
    # ------------------------------------------------------------
    p['I^B'] = np.array([
        i for i in p['I']
        if any(p['w^B'][i][d] > 0 for d in p['D'][i])
    ])

    # ------------------------------------------------------------
    # 2) AFSCs that can assign bases to cadet i
    # ------------------------------------------------------------
    p['J^BP'] = {}
    p['B^E'] = p.get('B^E', {})  # ensure exists

    for i in p['I^B']:

        # Collect relevant AFSC arrays only once
        relevant_states = [
            p['J^State'][i][d]
            for d in p['D'][i]
            if p['w^B'][i][d] > 0
        ]

        if not relevant_states:
            p['J^BP'][i] = np.array([])
            p['B^E'][i] = np.array([])
            continue

        # Faster union using set
        afscs = set().union(*relevant_states)
        p['J^BP'][i] = np.array(sorted(afscs))

        # Union of assignable bases
        bases = set().union(*(p['B^A'][j] for j in afscs))
        p['B^E'][i] = np.intersect1d(p['B^E'][i], np.array(list(bases)))

    # ------------------------------------------------------------
    # 3) Build reverse mapping efficiently
    #    I^BA[j][b] = cadets eligible
    # ------------------------------------------------------------

    # Precompute membership sets for fast lookup
    JBP_sets = {i: set(p['J^BP'][i]) for i in p['I^B']}
    BE_sets  = {i: set(p['B^E'][i]) for i in p['I^B']}

    p['I^BA'] = {j: {} for j in p['J^B']}

    for j in p['J^B']:
        for b in p['B^A'][j]:

            # Much faster membership using sets
            eligible = [
                i for i in p['I^B']
                if j in JBP_sets[i] and b in BE_sets[i]
            ]

            p['I^BA'][j][b] = np.array(eligible)

    # Helpful intersection of eligible bases for cadet i and bases that AFSC j can assign
    p['B^E-B^A'] = {i: {j: np.intersect1d(p['B^E'][i], p['B^A'][j]) for j in p['J^BP'][i]} for i in p['I^B']}

    return p


def _parse_training_date(date_str):
    """
    Add course-feasibility, course-utility, and assignment-capacity parameters.

    This function builds the training-side structures needed by the optimization
    model after base feasibility and cadet state logic have already been established.
    Its main purpose is to determine which training courses each cadet can actually
    attend, quantify how desirable those feasible courses are, and construct the
    capacity bounds needed for base and course assignment decisions.

    For each cadet and eligible AFSC, the function compares the cadet's training
    availability date to the start date of each course. Courses that begin before the
    cadet is available are excluded, while courses that begin on or after the cadet's
    start date are retained as feasible. The resulting feasible course sets are stored
    by cadet and AFSC.

    Once feasible courses are known, the function converts course wait times into
    normalized utilities based on the cadet's stated training preference. Cadets who
    prefer early training receive higher utility for shorter waits, while cadets who
    prefer late training receive higher utility for longer waits. Cadets with no
    timing preference receive neutral course utility. These values allow the training
    component of the model to reflect not just feasibility, but also relative
    desirability across course options.

    The function also reverses the perspective to determine, for each AFSC-course
    pair, which cadets are available to attend that course. Finally, it builds lower
    and upper assignment bounds for bases and courses from the provided minimum and
    maximum capacity data. Together, these structures supply the feasibility,
    preference, and capacity inputs needed for training-course assignment in the
    optimization model.

    Parameters
    ----------
    - p : dict
        Dictionary of AFCCP parameters. It must include training start dates, course
        start dates, cadet training preferences, AFSC-course structures, and base/course
        minimum and maximum capacity data.

    Returns
    -------
    - dict
        The parameter dictionary with additional training and capacity structures,
        including:
        - ``p['C^E']``: feasible courses for each cadet-AFSC pair
        - ``p['I^A']``: feasible cadets for each AFSC-course pair
        - ``p['course_days_cadet']``: wait times from cadet availability to course start
        - ``p['course_utility']``: normalized course utility values
        - ``p['lo^B']`` / ``p['hi^B']``: lower and upper base assignment bounds
        - ``p['lo^C']`` / ``p['hi^C']``: lower and upper course assignment bounds

    Examples
    --------
    ```python
    p = _add_course_and_capacity_parameters(p)
    ```

    See Also
    --------
    - [`_add_base_overall_parameters`](../../../reference/module/path/#module.path._add_base_overall_parameters)
    - [`_add_state_component_parameters`](../../../reference/module/path/#module.path._add_state_component_parameters)
    - [`base_training_parameter_additions`](../../../reference/module/path/#module.path.base_training_parameter_additions)
    """
    for fmt in ('%m/%d/%y', '%Y-%m-%d'):
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            pass
    return None

