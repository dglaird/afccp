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
from datetime import datetime
import copy
from functools import reduce

# afccp modules
import afccp.data.preferences
import afccp.data.support
import afccp.data.values


# __________________________________________________DATA VERIFICATION___________________________________________________
def parameter_sanity_check(instance):
    """
    Perform a Full Sanity Check on Problem Instance Parameters.

    This function performs a comprehensive audit of the problem instance's input data and value parameters to verify that all
    structures, matrices, and definitions are logically consistent and feasible. This includes checks on cadet eligibility,
    AFSC quotas, objective constraints, preference list coherence, utility monotonicity, and tiered qualification logic.

    The goal is to prevent downstream issues during optimization by catching data errors or logical mismatches in advance.
    All checks are printed with contextual explanations and will highlight both errors and warnings when inconsistencies are found.

    Parameters:
    --------
    - instance: `CadetCareerProblem` class instance
        An instantiated problem containing:

        - `parameters`: dictionaries and matrices representing cadets, AFSCs, preferences, and utility definitions.
        - `value_parameters`: constraints, objective targets, and value function metadata.

    Returns:
    --------
    - None: This function prints all identified issues to the console but does not return any values.
      It may raise a `ValueError` if `value_parameters` are not initialized.

    Examples:
    --------
    ```python
    from afccp.data.adjustments import parameter_sanity_check
    parameter_sanity_check(instance)
    ```

    This prints a series of diagnostics like:

    - "ISSUE: AFSC '15A' quota constraint invalid: 12 (min) > 10 (eligible)"
    - "WARNING: Cadet 41 has no preferences and is therefore eligible for nothing."
    - "ISSUE: Value function breakpoints for AFSC '17X' objective 'Tier 2' are misaligned."
    """

    # Shorthand
    p, vp = instance.parameters, instance.value_parameters

    if vp is None:
        raise ValueError("Cannot sanity check parameters without specifying which value_parameters to use.")

    # Initialization
    print("Sanity checking the instance parameters...")
    issue = 0

    # Check constraint type matrix (I discontinued "3"s and "4"s in favor of just doing "1"s and "2"s
    if 3 in vp['constraint_type'] or 4 in vp['constraint_type']:
        issue += 1
        print(issue, "ISSUE: 'constraint_type' matrix contains 3s and/or 4s instead of 1s and 2s. I discontinued the"
                     "use of the former in favor of the latter so please adjust it.")

    # Loop through each AFSC to check various elements
    for j, afsc in enumerate(p["afscs"][:p["M"]]):

        if p["num_eligible"][j] < p["quota_min"][j]:
            issue += 1
            print(issue, "ISSUE: AFSC '" + afsc + "' quota constraint invalid. " + str(p["quota_min"][j]) +
                  " (min) > " + str(p["num_eligible"][j]) + " (number of eligible cadets).")
        elif p["num_eligible"][j] == p["quota_min"][j]:
            issue += 1
            print(issue, "WARNING: AFSC '" + afsc +
                  "' has a lower quota that is the same as its number of eligible cadets (" +
                  str(p["quota_min"][j]) + "). All eligible cadets for this AFSC will be assigned to it.")

        if p["quota_min"][j] > p["quota_max"][j]:
            issue += 1
            print(issue, "ISSUE: AFSC '" + afsc + "' quota constraint invalid. " + str(p["quota_min"][j]) +
                  " (min) > " + str(p["quota_max"][j]) + " (max).")

        quota_k = np.where(vp["objectives"] == "Combined Quota")[0][0]
        if p["quota_d"][j] != vp["objective_target"][j, quota_k]:
            issue += 1
            print(issue, "ISSUE: AFSC '" + afsc + "' quota desired target of " + str(p["quota_d"][j]) +
                  " from AFSCs Fixed does not match its objective target (" + str(vp["objective_target"][j, quota_k]) +
                  ") in the value parameters.")

        if p["quota_d"][j] < p["quota_min"][j] or p["quota_d"][j] > p["quota_max"][j]:
            issue += 1
            print(issue, "ISSUE: AFSC '" + afsc + "' quota desired target of " + str(p["quota_d"][j]) +
                  " is outside the specified range on the number of cadets (" + str(p["quota_min"][j]) + ", " +
                  str(p["quota_max"][j]) + ").")

        # If we have the AFSC preference lists, we validate certain features
        if 'a_pref_matrix' in p:

            cfm_list = np.where(p['a_pref_matrix'][:, j])[0]  # Cadets on the AFSC preference list

            # Cadets that are both on the CFM preference list and are eligible for the AFSC (qual matrix)
            both_lists = np.intersect1d(cfm_list, p['I^E'][j])  # SHOULD contain the same cadets
            num_cfm, num_qual = len(cfm_list), len(p['I^E'][j])  # SHOULD be the same number of cadets

            # If the numbers aren't equal
            if len(both_lists) != num_qual:
                issue += 1
                cfm_not_qual = [cadet for cadet in cfm_list if cadet not in p['I^E'][j]]
                qual_not_cfm = [cadet for cadet in p['I^E'][j] if cadet not in cfm_list]
                print(issue, "ISSUE: AFSC '" + afsc + "' CFM preference list ('a_pref_matrix') does not match the qual"
                                                      "matrix. \nThere are " + str(num_cfm) +
                      " cadets that are on the preference list (non-zero ranks) but there are "
                      + str(num_qual) + " 'eligible' cadets (qual matrix). There are " + str(len(both_lists)) +
                      " cadets in both sets. \nCFM list but not qual cadets:", cfm_not_qual,
                      "\nQual but not CFM list cadets:", qual_not_cfm)

            # Make sure that all eligibility pairs line up
            if 'c_pref_matrix' in p:

                for i, cadet in enumerate(p['cadets']):

                    on_afsc_list = p['a_pref_matrix'][i, j] > 0
                    on_cadet_list = p['c_pref_matrix'][i, j] > 0

                    if on_cadet_list and not on_afsc_list:
                        issue += 1
                        print(issue, "ISSUE: AFSC '" + afsc + "' is on cadet '" + str(cadet) + "' (index=" +
                              str(i) + ")'s preference list (c_pref_matrix) but the cadet is not on their preference "
                                       "list (a_pref_matrix).")
                    elif on_afsc_list and not on_cadet_list:
                        issue += 1
                        print(issue, "ISSUE: Cadet '" + str(cadet) + "' (index=" + str(i) + ") is on AFSC '" + afsc +
                              "'s preference list (a_pref_matrix) but the AFSC is not on their preference list (c_pref_matrix).")

        # Validate AFOCD tier objectives
        for objective in ["Mandatory", "Desired", "Permitted", "Tier 1", "Tier 2", "Tier 3", "Tier 4"]:

            # Make sure this is a valid objective for this problem instance
            if objective not in vp["objectives"]:
                continue  # goes to the next objective

            # Get index
            k = np.where(vp["objectives"] == objective)[0][0]

            # Check if the AFSC is constraining this objective
            if k not in vp["K^C"][j]:
                continue

            # Make sure there are cadets that are in this degree tier
            if len(p["I^D"][objective][j]) == 0:
                issue += 1
                if "Tier" in objective:
                    print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective +
                          "' is empty. No cadets have degrees that fit in this tier for this class year.")
                else:
                    print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective +
                          "'-Tier is empty. No cadets have degrees that fit in this tier for this class year.")

            # Make sure objective has valid target
            if vp["objective_target"][j, k] == 0:
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective +
                      "'-Tier target cannot be 0 when it has a nonzero weight.")

        # Validate AFOCD Tier objectives
        levels = []
        for t, objective in enumerate(["Tier 1", "Tier 2", "Tier 3", "Tier 4"]):

            # Make sure this is a valid objective for this problem instance
            if objective not in vp["objectives"]:
                continue  # goes to the next objective

            # Get index
            k = np.where(vp["objectives"] == objective)[0][0]

            # Make sure that this is a valid tier for this AFSC
            if k not in vp['K^A'][j]:
                continue  # goes to the next objective

            level = "I" + str(t + 1)
            requirement_dict = {'t_mandatory': 'M', 't_desired': 'D', 't_permitted': 'P'}
            for r_level in requirement_dict:
                if p[r_level][j, t]:
                    level = requirement_dict[r_level] + str(t + 1)
            levels.append(level)

            # Make sure this requirement/qualification level is present with the cadets
            if level not in p['qual'][:, j]:
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective +
                      "' expected cadet qualification level is '" + level + "' but this is not in the qual matrix.")

        unique_levels = np.unique(p['qual'][:, j])
        for level in unique_levels:
            if level not in levels and 'E' not in level and 'I' not in level:
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' qualification level '" + level +
                      "' found within the cadet qual matrix but this is not defined within the value"
                      " parameters." )

        # Make sure all constrained objectives have appropriate constraints
        for k in vp["K^C"][j]:
            objective = vp["objectives"][k]

            # Check constraint type to see if something doesn't check out
            if vp["constraint_type"][j, k] == 1:

                # If the minimum is zero, we know this is an "at MOST" constraint (0 to 0.3, for example)
                if vp['objective_min'][j, k] == 0:
                    issue += 1
                    print(issue, "WARNING: AFSC '" + afsc + "' objective '" + objective +
                          "' has an 'at most' constraint of '" + vp['objective_value_min'][j, k] +
                          "'. The constraint_type is 1, indicating an approximate constraint but this is not recommended. "
                          "Instead, use the constraint_type '2' to indicate an exact constraint since this is the easiest"
                          " way to meet an 'at most' constraint.")

            # Make sure constrained objectives have valid constraint types
            if vp['constraint_type'][j, k] not in [1, 2]:
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective +
                      "' is in set of constrained objectives: vp['K^C'][j] but has a constraint_type of '" +
                      str(vp['constraint_type'][j, k]) + "'. This is not a valid active constraint.",
                      "Please update the set of value parameters using 'instance.update_value_parameters()'.")

            # Check valid 'objective_value_min' constraint range
            try:
                lb = float(vp["objective_value_min"][j, k].split(",")[0])
                ub = float(vp["objective_value_min"][j, k].split(",")[1])
                assert lb <= ub
            except:
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective +
                      "' constraint range (objective_value_min) '" + vp["objective_value_min"][j, k] +
                      "' is invalid. This constraint is currently activated.")

        # Make sure value functions are valid
        for k in vp["K^A"][j]:
            objective = vp["objectives"][k]
            vf_string_start = vp["value_functions"][j, k].split("|")[0]

            # VF String validation
            if vf_string_start not in ["Min Increasing", "Min Decreasing", "Balance", "Quota_Direct",
                                       "Quota_Normal"]:
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective + "' value function string '" +
                      vp["value_functions"][j, k] + "' is invalid.")

            # Validate number of breakpoints
            if vp["r"][j, k] == 0:
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective +
                      "' does not have any value function breakpoints. 'a':", vp["a"][j][k])
                continue

            # Value function should have same number of x and y coordinates
            if len(vp["a"][j][k]) != len(vp["f^hat"][j][k]):
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective +
                      "' value function breakpoint coordinates do not align. 'a' has length of " + len(vp["a"][j][k]) +
                      " while 'f^hat' has length of " + len(vp["f^hat"][j][k]) + ".")
                continue

            # Ensure that the breakpoint "x" coordinates are always getting bigger
            current_x = -1
            valid_x_bps = True
            for l in vp["L"][j][k]:
                if vp["a"][j][k][l] < current_x:
                    valid_x_bps = False
                    break
                else:
                    current_x = vp["a"][j][k][l]

            if not valid_x_bps:
                issue += 1
                print(issue, "ISSUE: AFSC '" + afsc + "' objective '" + objective +
                      "' value function x coordinates do not continuously increase along x-axis. 'a':", vp["a"][j][k],
                      "'vf_string':", vp["value_functions"][j, k])

        # Check all the objectives to see if the user missed something
        for k, objective in enumerate(vp['objectives']):

            if vp['constraint_type'][j, k] in [1, 2] and k not in vp['K^C'][j]:
                issue += 1
                print(issue, "WARNING: AFSC '" + afsc + "' objective '" + objective +
                      "' has a constraint_type of '" + str(vp['constraint_type'][j, k]) +
                      "' but is not in set of constrained objectives: vp['K^C'][j]. This is a mistake so",
                      "please update the set of value parameters using 'instance.update_value_parameters()'.")

    # Loop through each cadet to check preferences and utility values
    invalid_utility, invalid_cadet_utility = 0, 0
    invalid_utility_cadets, invalid_cadet_utility_cadets = [], []
    for i in p['I']:
        if 'c_preferences' in p and 'c_pref_matrix' in p:
            for choice in range(p['P']):
                afsc = p['c_preferences'][i, choice]
                if afsc in p['afscs']:
                    j = np.where(p['afscs'] == afsc)[0][0]
                    if p['c_pref_matrix'][i, j] != choice + 1:
                        issue += 1
                        print(issue, "ISSUE: Cadet", p['cadets'][i], "has AFSC '" + afsc + "' in position '"
                              + str(choice + 1) + "' in the Cadets.csv file, but it is ranked '" +
                              str(p['c_pref_matrix'][i, j]) + "' from the Cadets Preferences.csv file.")
                        break  # Don't need to check the rest of the cadet's preferences

            # If this cadet does not have any preferences, we skip them (must be an OTS candidate)
            if len(p['cadet_preferences'][i]) == 0:
                issue += 1
                print(issue, f"WARNING: Cadet {i} has no preferences and is therefore eligible for nothing.")
                continue

            # Make sure "utility" array is monotonically decreasing and the "cadet_utility" array is strictly decreasing
            arr_1 = p['utility'][i, p['cadet_preferences'][i]]
            arr_2 = p['cadet_utility'][i, p['cadet_preferences'][i]]
            if not all(arr_1[idx] >= arr_1[idx + 1] for idx in range(len(arr_1) - 1)):
                invalid_utility += 1
                invalid_utility_cadets.append(i)
            if not all(arr_2[idx] > arr_2[idx + 1] for idx in range(len(arr_2) - 1)):
                invalid_cadet_utility += 1
                invalid_cadet_utility_cadets.append(i)

    # Report issues with decreasing cadet utility values
    if invalid_utility > 0:
        issue += 1
        print(issue, "ISSUE: The cadet-reported utility matrix 'utility', located in 'Cadets Utility.csv'\nand in the "
                     "'Util' columns of 'Cadets.csv', does not incorporate monotonically\ndecreasing utility values for "
                     "" + str(invalid_utility) + " cadets. Please adjust.")
        if invalid_utility < 100:
            print('These are the cadets at indices', invalid_utility_cadets)
    if invalid_cadet_utility > 0 and 'last_afsc' not in p:  # There IS indifference with the new method of utilities
        issue += 1
        print(issue, "ISSUE: The constructed cadet utility matrix 'cadet_utility', located in 'Cadets Utility (Final)."
                     "csv',\ndoes not incorporate strictly decreasing utility values for "
                     "" + str(invalid_cadet_utility) + " cadets. Please adjust.")
        if invalid_cadet_utility < 40:
            print('These are the cadets at indices', invalid_cadet_utility_cadets)

    # Loop through each objective to see if there are any null values in the objective target array
    for k, objective in enumerate(vp["objectives"]):
        num_null = pd.isnull(vp["objective_target"][:, k]).sum()
        if num_null > 0:
            issue += 1
            print(issue, "ISSUE: Objective '" + objective + "' contains " +
                  str(num_null) + " null target values ('objective_target').")

    # USSF OM Constraint rules
    if instance.mdl_p['USSF OM'] is True and "USSF" not in p['afscs_acc_grp']:
        issue += 1
        print(issue, "ISSUE: Space Force OM constraint specified in controls (USSF OM = True) but no USSF"
                     " AFSCS found in the instance.")

    # At least one rated preference for rated eligible
    for soc in p['SOCs']:
        if soc in p['Rated Cadets']:
            for i in p['Rated Cadets'][soc]:
                if len(p['Rated Choices'][soc][i]) == 0:
                    issue += 1
                    print(issue,
                          "ISSUE: Cadet '" + str(p['cadets'][i]) + "' is on " + soc.upper() +
                          "'s Rated list (" + soc.upper() + " Rated OM.csv) but is not eligible for any Rated AFSCs. "
                                                            "You need to remove their row from the csv.")

    # Make sure all cadets eligible for at least one rated AFSC are in their SOC's rated OM list
    for soc in p['SOCs']:
        if 'J^Rated' in p:  # Make sure we have rated AFSCs

            # Loop through each cadet from this SOC
            for i in p[soc + '_cadets']:

                # Check if they're eligible for at least one rated AFSC
                if np.sum(p['eligible'][i][p['J^Rated']]) >= 1:

                    # If they're eligible for a Rated AFSC but aren't in the "Rated OM.csv" file, that's a problem
                    if i not in p['Rated Cadets'][soc]:
                        rated_afscs_eligible = p['afscs'][np.intersect1d(p['J^Rated'], p['J^E'][i])]
                        issue += 1
                        print(issue, "ISSUE: Cadet '" + str(p['cadets'][i]) + "' is not on " + soc.upper() +
                              "'s Rated list (" + soc.upper() + " Rated OM.csv), but is on the preference lists for",
                              rated_afscs_eligible, "Please add a row in 'Rated OM.csv' for this cadet reflecting their "
                                                    "OM.")

    # Validate that the "totals" for minimums/maximums work
    if np.sum(p['pgl']) > p['N']:
        issue += 1
        print(issue, "ISSUE: Total sum of PGL targets is", int(np.sum(p['pgl'])),
              " while 'N' is " + str(p['N']) + ". This is infeasible since we don't have enough cadets.")
    if np.sum(p['quota_min']) > p['N']:
        issue += 1
        print(issue, "ISSUE: Total sum of minimum constrained capacities (quota_min) is", int(np.sum(p['quota_min'])),
              " while 'N' is " + str(p['N']) + ". This is infeasible since we don't have enough cadets.")
    if (np.sum(p['quota_max']) < p['N']) and 'ots' not in p['SOCs']:  # OTS candidates can go unmatched
        issue += 1
        print(issue, "ISSUE: Maximum constrained capacities (quota_max) is", int(np.sum(p['quota_max'])),
              " while 'N' is " + str(p['N']) + ". This is infeasible; we don't have enough positions for cadets to fill.")

    # Print statement
    print('Done,', issue, "issues found.")


# _____________________________________________________DATA ADJUSTMENTS_________________________________________________
def parameter_sets_additions(parameters):
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
            p[f'{soc}_eligible_count'] = np.array([len(np.intersect1d(p['I^E'][j], p[f'{soc}_cadets'])) for j in p['J']])

    # Initialize empty dictionaries of matched/reserved cadets
    p["J^Fixed"] = {}
    p["J^Reserved"] = {}

    # If we have the "Assigned" column in Cadets.csv, we can check to see if anyone is "fixed" in this solution
    if "assigned" in p:

        for i, afsc in enumerate(p["assigned"]):
            j = np.where(p["afscs"] == afsc)[0]  # AFSC index

            # Check if the cadet is actually assigned an AFSC already (it's not blank)
            if len(j) != 0:
                j = int(j[0])  # Actual index

                # Check if the cadet is assigned to an AFSC they're not eligible for
                if j not in p["J^E"][i]:
                    cadet = str(p['cadets'][i])
                    raise ValueError("Cadet " + cadet + " assigned to '" +
                                     afsc + "' but is not eligible for it. Adjust the qualification matrix!")
                else:
                    p["J^Fixed"][i] = j

    # Cadet preference/rated cadet set additions
    p = more_parameter_additions(p)

    # Base/Training set additions
    if "bases" in p:
        p = base_training_parameter_additions(p)

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
            p['afscs_acc_grp'][acc_grp] = p['afscs'][p['J^' + acc_grp]]
    else:  # Previously, we've only assigned NRL cadets so we assume that's what we're dealing with here
        p['acc_grp'] = np.array(['NRL' for _ in p['J']])
        p['afscs_acc_grp']['NRL'] = p['afscs']
        p['J^NRL'] = p['J']

    # If we have the "Accessions Group" column in Cadets.csv, we can check to see if anyone is fixed to a group here
    if 'acc_grp_constraint' in p:

        # Loop through each Accession group and get the cadets that are constrained to be in this "group"
        for acc_grp in p['afscs_acc_grp']:  # This should really only ever apply to USSF, but we're generalizing it

            # Constrained cadets for each Accession group (don't confuse with I^*acc_grp* in the "solutions" dictionary!)
            p['I^' + acc_grp] = np.where(p['acc_grp_constraint'] == acc_grp)[0]

    # PGL Totals per SOC for USSF
    if 'USSF' in p['afscs_acc_grp']:
        p['ussf_usafa_pgl'] = np.sum(p['usafa_quota'][j] for j in p['J^USSF'])
        p['ussf_rotc_pgl'] = np.sum(p['rotc_quota'][j] for j in p['J^USSF'])

    # We already have "J^USSF" defined above; now we want one for USAF (NRL + Rated)
    if 'USSF' in p['afscs_acc_grp']:
        p['J^USAF'] = np.array([j for j in p['J'] if j not in p['J^USSF']])

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
            p['I^' + race]) * (len(p['I^' + race]) - 1)) / (p['N'] * (p['N'] - 1)) for race in p['race_categories']]), 2)

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

    return p


def base_training_parameter_additions(parameters):
    """
    Add Base and Training Parameters to the Problem Instance.

    This function extends the parameter dictionary with the data structures required to support base assignments and
    training course scheduling within the CASTLE Base/Training optimization model. Each cadet is categorized into
    preference-based "states" depending on their AFSC priorities and base/course interest.

    The function also calculates cadet-course availability, utility of wait times, and assignment eligibility across bases
    and courses. This enables simultaneous modeling of AFSC matches, base assignments, and training timelines.

    Parameters
    ----------
    parameters : dict
        The problem instance parameters, including cadet preferences, AFSC eligibility, utility scores, training
        thresholds, and configuration flags for base/course logic.

    Returns
    -------
    dict
    Updated parameter dictionary with additional sets and matrices such as:

    - `D`, `Cadet Objectives`, `J^State`, `w^A`, `w^B`, `w^C`, `u^S`: cadet state structures.
    - `B^A`, `B^E`, `B^State`: base assignment eligibility mappings.
    - `C^E`, `I^A`, `course_days_cadet`, `course_utility`: training availability and utility values.
    - `lo^B`, `hi^B`, `lo^C`, `hi^C`: quantity constraints on base/course assignments.

    Examples
    --------
    ```python
    from afccp.data.adjustments import base_training_parameter_additions
    parameters = base_training_parameter_additions(parameters)
    ```

    Notes
    -----
    - Cadet states are built using `base_threshold` and `training_threshold`, which split cadet preferences into
      AFSCs only, AFSC + base, and AFSC + base + course states.
    - Utility from training courses is based on cadet preferences (`Early`, `Late`, `None`) and normalized start dates.
    - Course utility is scaled from 0 to 1, with utility decreasing/increasing with wait time as appropriate.
    - This logic assumes all relevant arrays like `training_start`, `course_start`, `afsc_assign_base`, etc., exist and
      are preloaded in the parameter dictionary.

    See Also
    --------
    - [`parameter_sets_additions`](../../../afccp/reference/data/adjustments/#data.adjustments.parameter_sets_additions):
      Adds foundational indexed sets and preference structures used prior to base/training expansion.
    """

    # Helpful function to extract the datetime object from a specific string containing date information
    def parse_date(date_str):
        for fmt in ('%m/%d/%y', '%Y-%m-%d'):
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                pass
        return None  # Return None if no format matches

    # Shorthand
    p = parameters

    # Sets of bases and courses
    p['B'] = np.arange(p['S'])
    p['C'] = {j: np.arange(p['T'][j]) for j in p['J']}

    # Set of AFSCs that assign cadets to bases
    p['J^B'] = np.where(p['afsc_assign_base'])[0]

    # Set of bases that AFSC j may assign cadets to
    p['B^A'] = {j: np.where(p['base_max'][:, j] > 0)[0] for j in p['J']}

    # Set of bases that cadet i may be assigned to (based on the union of all eligible bases from AFSCs in J^E_i)
    p['B^E'] = {i: reduce(np.union1d, (p['B^A'][j] for j in np.intersect1d(p['J^E'][i], p['J^B']))) for i in p['I']}

    # Sets/Parameters for AFSC outcome states for each cadet
    p['D'] = {}  # Set of all AFSC outcome states that cadet i has designated
    p['Cadet Objectives'] = {}  # Set of cadet objectives included for each cadet and each state
    p['J^State'] = {}  # Set of AFSCs that, if assigned, would put cadet i into state d
    p['w^A'] = {}  # the weight that cadet i places on AFSCs in state d
    p['w^B'] = {}  # the weight that cadet i places on bases in state d
    p['w^C'] = {}  # the weight that cadet i places on courses in state d
    p['u^S'] = {}  # the maximum utility that cadet i receives from state d (based on best AFSC)
    p['B^State'] = {}  # Set of bases that cadet i can be assigned to in state d (According to J^State_id)

    # Determine the "states" for each cadet based on the differences of AFSC outcomes
    for i in p['I']:

        # Base/Training Thresholds (Shorthand)
        bt, tt = p['base_threshold'][i], p['training_threshold'][i]

        # Determine "primary" set of AFSCs and states based on thresholds
        if bt < tt:
            afscs = {1: p['cadet_preferences'][i][:bt],
                      2: p['cadet_preferences'][i][bt: tt],
                      3: p['cadet_preferences'][i][tt:]}
            included = {1: ['afsc'], 2: ['afsc', 'base'], 3: ['afsc', 'base', 'course']}
        elif tt < bt:
            afscs = {1: p['cadet_preferences'][i][:tt],
                      2: p['cadet_preferences'][i][tt: bt],
                      3: p['cadet_preferences'][i][bt:]}
            included = {1: ['afsc'], 2: ['afsc', 'course'], 3: ['afsc', 'base', 'course']}
        else:  # They're equal!
            afscs = {1: p['cadet_preferences'][i][:bt],
                      2: p['cadet_preferences'][i][bt:]}
            included = {1: ['afsc'], 2: ['afsc', 'base', 'course']}

        # Sets/Parameters for AFSC outcome states for each cadet
        p['D'][i] = []  # Set of all AFSC outcome states that cadet i has designated
        p['Cadet Objectives'][i] = {}  # Set of cadet objectives included for each cadet and each state
        p['J^State'][i] = {}  # Set of AFSCs that, if assigned, would put cadet i into state d
        p['w^A'][i] = {}  # the weight that cadet i places on AFSCs in state d
        p['w^B'][i] = {}  # the weight that cadet i places on bases in state d
        p['w^C'][i] = {}  # the weight that cadet i places on courses in state d
        p['u^S'][i] = {}  # the maximum utility that cadet i receives from state d (based on best AFSC)
        p['B^State'][i] = {}  # Set of bases that cadet i can be assigned to in state d (According to J^State_id)

        # Loop through each "primary" state to get "final" states (Split up states based on base assignment AFSCs)
        d = 1
        for state in included:

            # Empty state!
            if len(afscs[state]) == 0:
                continue

            # Split up the AFSCs into two groups if they assign cadets to bases or not
            sets = {'Assigned': np.intersect1d(p['J^B'], afscs[state]),
                    'Not Assigned': np.array([j for j in afscs[state] if j not in p['J^B']])}

            # Loop through both sets and create a new state if the set contains AFSCs
            for set_name, afscs_in_set in sets.items():
                if len(afscs_in_set) != 0:

                    # Add information to this state
                    p['D'][i].append(d)
                    p['Cadet Objectives'][i][d] = included[state]
                    p['J^State'][i][d] = afscs_in_set
                    p['u^S'][i][d] = p['cadet_utility'][i, afscs[state][0]]  # Utility of the top preferred AFSC

                    # Weights and set of bases are differentiated by if this is a set containing J^B AFSCs or not
                    if set_name == "Assigned":

                        # Re-scale weights based on the objectives included in this state
                        p['w^A'][i][d] = p['weight_afsc'][i] / sum(p['weight_' + obj][i] for obj in included[state])
                        p['w^C'][i][d] = p['weight_course'][i] / sum(p['weight_' + obj][i] for obj in included[state]) \
                                         * ('course' in included[state])
                        p['w^B'][i][d] = p['weight_base'][i] / sum(p['weight_' + obj][i] for obj in included[state]) \
                                         * ('base' in included[state])

                        # Union of bases that this cadet could be assigned to in this state according to J^State_id
                        p['B^State'][i][d] = reduce(np.union1d, (p['B^A'][j] for j in p['J^State'][i][d]))

                    else:

                        # Re-scale weights based on the objectives included in this state
                        p['w^A'][i][d] = p['weight_afsc'][i] / \
                                         sum(p['weight_' + obj][i] for obj in included[state] if obj != "base")
                        p['w^C'][i][d] = ('course' in included[state]) * p['weight_course'][i] / \
                                         sum(p['weight_' + obj][i] for obj in included[state] if obj != "base")
                        p['w^B'][i][d] = 0
                        p['B^State'][i][d] = np.array([])  # Empty array (no bases)

                    # Next state
                    d += 1

        # Print statement for specific cadet
        if i == 10 and False:  # Meant for debugging and sanity checking this logic!
            print('Cadet', i)
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

    # Adjust AFSC, base, course weights to give slight bump to ensure all are considered in each applicable state
    max_afsc_weight = max([p['w^A'][i][d] for i in p['I'] for d in p['D'][i] if p['w^A'][i][d] != 1])
    max_afsc_weight += (1 - max_afsc_weight) / 2
    for i in p['I']:
        for d in p['D'][i]:

            # Take some weight from AFSCs
            p['w^A'][i][d] = p['w^A'][i][d] * max_afsc_weight

            # Redistribute the weight to Base/Courses depending on existence of J^B AFSCs
            if len(p['B^State'][i][d]) > 0:
                p['w^B'][i][d] = p['w^B'][i][d] * max_afsc_weight + (1 - max_afsc_weight) / 2
                p['w^C'][i][d] = p['w^C'][i][d] * max_afsc_weight + (1 - max_afsc_weight) / 2
            else:
                p['w^C'][i][d] = p['w^C'][i][d] * max_afsc_weight + (1 - max_afsc_weight)
                
    # Sets pertaining to courses for each AFSC
    p['C^E'] = {}  # Set of courses that cadet i is available to take with AFSC j
    p['I^A'] = {}  # Set of cadets that are available to take course c with AFSC j

    # Calculate course utility for each cadet, AFSC, course tuple
    p['course_days_cadet'] = {}
    p['course_utility'] = {}
    for i in p['I']:

        # Initialize information for this cadet
        p['course_days_cadet'][i] = {}
        p['course_utility'][i] = {}
        p['C^E'][i] = {}

        # Loop through each AFSC and course to determine days between cadet start and course start
        for j in p['J^E'][i]:
            p['course_days_cadet'][i][j] = {}
            for c in p['C'][j]:

                # Convert str format to datetime format if necessary
                if type(p['course_start'][j][c]) == str:
                    course_start = parse_date(p['course_start'][j][c])
                    cadet_start = parse_date(p['training_start'][i])
                else:
                    course_start = p['course_start'][j][c]
                    cadet_start = p['training_start'][i]

                # Calculate days between
                days_between = (course_start - cadet_start).days
                if days_between >= 0:  # If the cadet is available to take the course before it starts
                    p['course_days_cadet'][i][j][c] = days_between

            # Get subset of courses that this cadet can take for this AFSC
            p['C^E'][i][j] = np.array([c for c in p['course_days_cadet'][i][j]])

        # Get course wait times and determine min and max
        course_waits = [p['course_days_cadet'][i][j][c] for j in p['J^E'][i] for c in p['C^E'][i][j]]
        max_wait, min_wait = max(course_waits), min(course_waits)

        # Loop through each AFSC and course again to calculate utility (normalize the wait times)
        for j in p['J^E'][i]:
            p['course_utility'][i][j] = {}
            for c in p['C^E'][i][j]:
                if p['training_preferences'][i] == 'Early':
                    p['course_utility'][i][j][c] = 1 - (p['course_days_cadet'][i][j][c] - min_wait) / \
                                                   (max_wait - min_wait)
                elif p['training_preferences'][i] == 'Late':
                    p['course_utility'][i][j][c] = (p['course_days_cadet'][i][j][c] - min_wait) / \
                                                   (max_wait - min_wait)
                else:  # No preference
                    p['course_utility'][i][j][c] = 0

                if i == 0 and False:  # Meant for debugging and sanity checking this logic!
                    print(i, j, c, "Dates", course_start, cadet_start, "DAYS", p['course_days_cadet'][i][j][c],
                          "Utility", p['course_utility'][i][j][c])

    # Determine set of cadets that are available to take course c with AFSC j
    for j in p['J']:
        p['I^A'][j] = {}
        for c in p['C'][j]:
            p['I^A'][j][c] = np.array([i for i in p['I^E'][j] if c in p['C^E'][i][j]])

    # Get minimum and maximum quantities for bases
    p['lo^B'], p['hi^B'] = {}, {}
    for j in p['J^B']:
        p['lo^B'][j], p['hi^B'][j] = {}, {}
        for b in p['B^A'][j]:
            p['lo^B'][j][b], p['hi^B'][j][b] = p['base_min'][b, j], p['base_max'][b, j]

    # Get minimum and maximum quantities for courses
    p['lo^C'], p['hi^C'] = p['course_min'], p['course_max']

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


# ______________________________________________________MISC FUNCTIONS__________________________________________________
def convert_instance_to_from_scrubbed(instance, new_letter=None, translation_dict=None, data_name='Unknown'):
    """
    Convert Between Original and Scrubbed AFSC Names Based on PGL Sorting.

    This function transforms a problem instance by reordering or restoring AFSC names based on their PGL targets.
    It is used to anonymize (scrub) AFSCs for publication or experimentation by replacing real AFSC identifiers
    with generic labels (e.g., "X1", "X2", ...) while preserving order. If a translation dictionary is provided,
    it performs the inverse operation—restoring original AFSC names from their scrubbed versions.

    The function updates all relevant AFSC-indexed matrices, arrays, and value parameters in the instance.
    It also modifies the solution dictionary (`instance.solutions`) and preference matrices to maintain consistency.

    Parameters:
    --------
    - instance (`CadetCareerProblem`): The full problem instance containing parameter and solution data.
    - new_letter (str, optional): A single letter (e.g., `"X"`) to use as the prefix for scrubbed AFSC names.
      If provided, performs a *forward* conversion (real → scrubbed).
    - translation_dict (dict, optional): A mapping from real to scrubbed AFSC names. If provided and
      `new_letter` is None, performs a *reverse* conversion (scrubbed → real).
    - data_name (str, optional): A custom label to attach to the instance's `data_name` attribute.

    Returns:
    --------
    - tuple:
        - `instance` (`CadetCareerProblem`): The updated instance with renamed AFSCs and adjusted internal data.
        - `translation_dict` (dict): The mapping used for conversion (real → scrubbed).

    Examples:
    --------
    ```python
    # Forward conversion (scrubbing AFSC names)
    new_instance, afsc_mapping = convert_instance_to_from_scrubbed(instance, new_letter="X")

    # Reverse conversion (restoring AFSC names)
    original_instance, _ = convert_instance_to_from_scrubbed(new_instance, translation_dict=afsc_mapping)
    ```
    """

    # Load parameters
    p = copy.deepcopy(instance.parameters)

    # Initialize AFSC information
    current_afscs_unsorted = p["afscs"][:p["M"]]
    new_p = copy.deepcopy(p)

    # We're going from original to scrubbed
    if new_letter is not None:
        data_name = new_letter

        # Sort current list of AFSCs by PGL
        t_indices = np.argsort(p["pgl"])[::-1]  # Indices that word sort the list -> used a lot below!
        current_afscs = copy.deepcopy(current_afscs_unsorted[t_indices])

        # Construct new list of AFSCS
        new_p['afscs'] = np.array([' ' * 10 for _ in p['J']])
        for j, afsc in enumerate(current_afscs):
            new_p['afscs'][j] = new_letter + str(j + 1)

            # Adjust new AFSC by adding "_U" or "_R" extension if necessary
            for ext in ["_R", "_U"]:
                if ext in afsc:
                    new_p['afscs'][j] += ext
                    break

        # Translate AFSCs to the new list
        translation_dict = {}
        for afsc in current_afscs_unsorted:
            j = np.where(current_afscs == afsc)[0][0]
            translation_dict[afsc] = new_p['afscs'][j]  # Save this AFSC to the translation dictionary
        new_p["afscs"] = np.hstack((new_p["afscs"], "*"))  # Add "unmatched" AFSC

    # We're going from scrubbed to original
    else:

        # Translate AFSCs (Really weird sorting going on...sorry)
        new_p["afscs"] = np.array(list(translation_dict.keys()))
        new_p["afscs"] = np.hstack((new_p["afscs"], "*"))  # Add "unmatched" AFSC
        flipped_translation_dict = {translation_dict[afsc]: afsc for afsc in translation_dict}
        real_order_scrubbed_afscs = np.array(list(flipped_translation_dict.keys()))
        scrubbed_order_indices = np.array(
            [np.where(real_order_scrubbed_afscs==afsc)[0][0] for afsc in current_afscs_unsorted])
        scrubbed_order_real_afscs = new_p['afscs'][scrubbed_order_indices]
        current_afscs = real_order_scrubbed_afscs

        # Get sorted indices
        t_indices = np.array([np.where(scrubbed_order_real_afscs==afsc)[0][0] for afsc in new_p["afscs"][:p["M"]]])

    # Loop through each key in the parameter dictionary to translate it
    for key in p:

        # If it's a one dimensional array of length M, we translate it accordingly
        if np.shape(p[key]) == (p["M"], ) and "^" not in key:  # Sets/Subsets will be adjusted later
            new_p[key] = p[key][t_indices]

        # If it's a two-dimensional array of shape Mx4, we translate it accordingly
        elif np.shape(p[key]) == (p["M"], 4):
            new_p[key] = p[key][t_indices, :]

        # If it's a two-dimensional array of shape (NxM), we translate it accordingly
        elif np.shape(p[key]) == (p["N"], p["M"]) and key not in ['c_preferences', 'c_utilities']:
            new_p[key] = p[key][:, t_indices]

        # If it's a two-dimensional array of shape (NxM+1), we translate it accordingly (leave unmatched AFSC alone)
        elif np.shape(p[key]) == (p["N"], p["M"] + 1):
            new_p[key] = copy.deepcopy(p[key])
            new_p[key][:, :p['M']] = p[key][:, t_indices]

    # Get assigned AFSC vector
    for i, real_afsc in enumerate(p["assigned"]):
        if real_afsc in current_afscs:
            j = np.where(current_afscs == real_afsc)[0][0]
            new_p["assigned"][i] = new_p["afscs"][j]

    # Set additions, and add to the instance
    instance.parameters = parameter_sets_additions(new_p)

    # Translate value parameters
    if instance.vp_dict is not None:
        new_vp_dict = {}
        for vp_name in instance.vp_dict:
            vp = copy.deepcopy(instance.vp_dict[vp_name])
            new_vp = copy.deepcopy(vp)

            for key in vp:

                # If it's a one dimensional array of length M, we translate it accordingly
                if np.shape(vp[key]) == (p["M"],):
                    new_vp[key] = vp[key][t_indices]

                # If it's a two-dimensional array of shape (NxM), we translate it accordingly
                elif np.shape(vp[key]) == (p["N"], p["M"]):
                    new_vp[key] = vp[key][:, t_indices]

                # If it's a two-dimensional array of shape (MxO), we translate it accordingly
                elif np.shape(vp[key]) == (vp["M"], vp["O"]) and key not in ["a", "f^hat"]:
                    new_vp[key] = vp[key][t_indices, :]

            for j, old_j in enumerate(t_indices):
                for k in vp["K"]:
                    for key in ["a", "f^hat"]:
                        new_vp[key][j][k] = vp[key][old_j][k]

            # Set value parameters to dict
            new_vp_dict[vp_name] = new_vp

        # Set it to the instance
        instance.vp_dict = new_vp_dict

        # Loop through each set of value parameters again
        for vp_name in instance.vp_dict:

            # Set additions
            instance.vp_dict[vp_name] = \
                afccp.data.values.value_parameters_sets_additions(instance.parameters, instance.vp_dict[vp_name])

    else:
        instance.vp_dict = None

    # Translate solutions
    if instance.solutions is not None:
        new_solutions_dict = {}

        # Loop through each solution
        for solution_name in instance.solutions:
            real_solution = copy.deepcopy(instance.solutions[solution_name])
            new_solutions_dict[solution_name] = copy.deepcopy(real_solution)

            # Loop through each assigned AFSC for the cadets
            for i, j in enumerate(real_solution['j_array']):
                if j != p["M"]:
                    real_afsc = p["afscs"][j]
                    j = np.where(current_afscs == real_afsc)[0][0]
                    new_solutions_dict[solution_name]['j_array'][i] = j

        # Save solutions dictionary
        instance.solutions = new_solutions_dict

    else:
        instance.solutions = None

    # Convert "c_preferences" array
    if "c_preferences" in p:
        for i in p["I"]:
            for pref in range(p["P"]):
                real_afsc = p["c_preferences"][i, pref]
                if real_afsc in current_afscs:
                    j = np.where(current_afscs == real_afsc)[0][0]
                    new_p["c_preferences"][i, pref] = new_p["afscs"][j]

    # Instance Attributes
    instance.data_name, instance.data_version = data_name, "Default"
    instance.import_paths, instance.export_paths = None, None

    return instance, translation_dict






