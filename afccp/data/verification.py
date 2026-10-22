import numpy as np
import pandas as pd
import datetime
import copy
from functools import reduce


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

            # Make sure that all eligibility pairs line up (if we care about cadet preferences matching)
            if 'c_pref_matrix' in p and instance.remove_cadet_choices_eligibility:

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
                      " parameters.")

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
                print(issue, f"WARNING: Cadet {i} has no preferences.")
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
                              rated_afscs_eligible,
                              "Please add a row in 'Rated OM.csv' for this cadet reflecting their "
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
              " while 'N' is " + str(
                  p['N']) + ". This is infeasible; we don't have enough positions for cadets to fill.")

    # Check rated holistic properties (there should be enough rated folks from both SOCs to fill total rated quotas
    for soc in p['SOCs']:
        targets = np.sum([p[f'{soc.lower()}_quota'][j] for j in p['J^Rated']])
        num_cadets = len(p['Rated Cadets'][soc])
        if targets > num_cadets:
            issue += 1
            print(issue, f"ISSUE: Total number of {soc.upper()} rated targets is {targets}. "
                         f"Total number of {soc.upper()} rated cadets is {num_cadets}.\n"
                         f"This will be infeasible (assuming standard rated/SOC algorithms) since "
                         f"we do not have enough {soc.upper()} rated cadets to fill slots.")

    # Run through the base/training checks
    if "bases" in p:
        issue = base_training_checks(p, issue)

    # Print statement
    print('Done,', issue, "issues found.")


def base_training_checks(p, issue):

    # Check if we have UPT preferences for all the pilot-qualified cadets
    pilot_j = [j for j in p['J'] if p['base_ist_status'][j] == 'UPT Base & IST']
    cadets = np.array([])
    for j in pilot_j:
        cadets = np.union1d(cadets, p['I^E'][j])
    upt_prefs_cadets = np.array(p['upt_preferences_df']['Cadet'])
    cadets_without_prefs = [int(i) for i in cadets if i not in upt_prefs_cadets]
    if len(cadets_without_prefs) > 0:
        issue += 1
        print(issue, f"ISSUE: Missing UPT preferences for {len(cadets_without_prefs)} pilot-qualified cadets. "
                     f"These are cadets: {cadets_without_prefs}")
    return issue