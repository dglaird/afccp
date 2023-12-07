import numpy as np
import pandas as pd
from datetime import datetime
import copy
from functools import reduce

# afccp modules
import afccp.core.data.preferences
import afccp.core.data.support
import afccp.core.data.values


# Data Adjustments
def gather_degree_tier_qual_matrix(cadets_df, parameters):
    """
    This function takes in the cadets dataframe and the dictionary of parameters (p) to determine what kind of degree
    tier qual matrix we'll be dealing with and if we have to generate one ourselves, we will do that here. This function
    handles most degree tier qualification information.
    :return: instance parameters
    """

    # Shorthand
    p = parameters

    # Determine if there is already a qualification matrix in the Cadets dataframe, and what "type" it is
    afsc_1, afsc_M = p["afscs"][0], p["afscs"][p["M"] - 1]
    current_qual_type = "None"  # If there is no qualification matrix, we'll have to generate it
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
                qual = afccp.core.data.support.cip_to_qual_tiers(
                    p["afscs"][:p["M"]], p['cip1'], cip2=p['cip2'])
            else:
                qual = afccp.core.data.support.cip_to_qual_tiers(
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


def parameter_sets_additions(parameters):
    """
    Add Indexed Sets and Subsets to the Problem Instance Parameters.

    This function enhances the problem instance parameters by creating indexed sets and subsets for both AFSCs and cadets.
    It helps organize the data for efficient processing and optimization. These indexed sets and subsets include:

    - Cadet Indexed Sets: `I`, `J`, `J^E`, `I^Choice`, `Choice Count`, and specific demographic subsets.
    - AFSC Indexed Sets: `I^E` and counts of eligible cadets for each AFSC.
    - Demographic Sets: Sets related to specific demographics such as USAFA cadets, minority cadets, male cadets, and
      associated proportions.

    Additionally, it handles other tasks like adjusting the utility matrix for unmatched cadets, calculating the sum of
    cadet merits, differentiating USAFA and ROTC cadets, identifying fixed and reserved cadets, and managing cadet preferences
    and rated cadets.

    Args:
        parameters: The problem instance parameters.

    Returns:
        The updated problem instance parameters with added indexed sets and subsets.

    Example:
    ```python
    import your_module

    # Create a problem instance
    parameters = your_module.create_instance()

    # Add indexed sets and subsets
    updated_parameters = your_module.parameter_sets_additions(parameters)
    ```

    """

    # Shorthand
    p = parameters

    # if 'eligible' not in p:
    #     print("No eligibility matrix here yet. Using dummy matrix for now.")
    #     p['eligible'] = np.ones((p['N'], p['M']))

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

    # USAFA/ROTC cadets
    if 'usafa' in p:
        p['rotc'] = (p['usafa'] == 0) * 1
        p['usafa_cadets'] = np.where(p['usafa'])[0]
        p['rotc_cadets'] = np.where(p['rotc'])[0]

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
    Enhance the problem instance parameters by adding additional parameter sets and subsets.

    This function extends the 'parameters' dictionary by adding various parameter sets and subsets, improving the organization
    of data for optimization. These additions include:

    - Cadet Preferences: Sets cadet preferences and counts the number of cadet choices.
    - AFSC Preferences: Sets AFSC preferences.
    - AFSCs by Accessions Group: Categorizes AFSCs into accessions groups such as Rated, USSF, and NRL.
    - Constrained Cadets: Identifies cadets constrained to specific accessions groups.
    - PGL Totals for USSF: Calculates totals for USAFA and ROTC PGL (Projected Gain/Loss) within USSF.
    - Rated Cadets: Identifies rated cadets, sets their preferences, and counts their choices.
    - Cadet Utility Matrix: Constructs a utility matrix based on cadet preferences.
    - Sets for cadets who have preferences and are eligible for specific AFSCs, and vice versa.
    - Race and Ethnicity Categories: Organizes cadets based on race and ethnicity categories.
    - SOC and Gender Categories: Organizes cadets into categories like USAFA, ROTC, male, female, etc.
    - STEM Cadets: Identifies STEM cadets and related AFSCs.

    Args:
        parameters: The problem instance parameters.

    Returns:
        The updated problem instance parameters with added parameter sets and subsets.
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
                if j in p['J^E'][i] and p['c_pref_matrix'][i, j] != 0:
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
    cadets_dict = {'rotc': 'rr_om_cadets', 'usafa': 'ur_om_cadets'}
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
        if 'cadet_preferences' in p:
            p = afccp.core.data.preferences.create_new_cadet_utility_matrix(p)
        else:
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
    if 'usafa' in p:
        p['I^USAFA'] = np.where(p['usafa'])[0]
        p['I^ROTC'] = np.where(p['usafa'] == 0)[0]
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

    return p


def base_training_parameter_additions(parameters):
    """
    This function takes in our set of parameters and adds more components to address the base/training model
    components as an "expansion" of the afccp functionality. This model performs base assignments and schedules
    cadets for training courses simultaneously.
    """

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
                    course_start = datetime.strptime(p['course_start'][j][c], '%Y-%m-%d').date()
                    cadet_start = datetime.strptime(p['training_start'][i], '%Y-%m-%d').date()
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


# Instance changes
def convert_instance_to_from_scrubbed(instance, new_letter=None, translation_dict=None, data_name='Unknown'):
    """
    This function takes in a problem instance and scrubs the AFSC names by sorting them by their PGL targets.
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

            # USAFA-constrained AFSCs
            if vp["J^USAFA"] is not None:
                usafa_afscs = vp["USAFA-Constrained AFSCs"].split(", ")
                new_str = ""
                for index, real_afsc in enumerate(usafa_afscs):
                    real_afsc = str(real_afsc.strip())
                    j = np.where(current_afscs == real_afsc)[0][0]
                    usafa_afscs[index] = new_p["afscs"][j]
                    if index == len(usafa_afscs) - 1:
                        new_str += usafa_afscs[index]
                    else:
                        new_str += usafa_afscs[index] + ", "

                new_vp["USAFA-Constrained AFSCs"] = new_str

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
                afccp.core.data.values.value_parameters_sets_additions(instance.parameters, instance.vp_dict[vp_name])

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


# Data Verification
def parameter_sanity_check(instance):
    """
    Perform a Sanity Check on Problem Instance Parameters.

    This function rigorously checks the validity of various parameters and configurations within the given problem instance.
    It's an essential step to ensure the consistency and feasibility of the problem definition before running any optimization.

    Args:
        instance: The problem instance to be checked.

    Raises:
        ValueError: If the provided instance doesn't have value parameters (vp).

    The function examines a range of parameters and configurations within the problem instance. It checks for issues in the
    following categories:

    1. **Constraint Type**: It ensures that the 'constraint_type' matrix doesn't contain deprecated values (3 or 4).
    2. **AFSC Quota Constraints**: Verifies the validity of quota constraints, such as minimum and maximum quotas for AFSCs.
    3. **Objective Targets**: Validates that the 'quota_d' value matches the objective target for Combined Quota.
    4. **Degree Tiers**: Checks if objectives related to degree tiers have a proper number of eligible cadets.
    5. **Qualification Levels**: Ensures that qualification levels specified in the 'qual' matrix are coherent with value parameters.
    6. **Constrained Objectives**: Verifies that constrained objectives have appropriate constraints defined.
    7. **Value Functions**: Validates value functions, including the format of value function strings and breakpoints.
    8. **Cadet Preferences**: Ensures that cadet preferences align with preference matrices (c_preferences and c_pref_matrix).
    9. **Monotonically Decreasing Utility**: Checks that the cadet-reported utility matrix 'utility' is monotonically decreasing.
    10. **Strictly Decreasing Cadet Utility**: Verifies that the constructed cadet utility matrix 'cadet_utility' is strictly decreasing.
    11. **Objective Targets Null Values**: Checks for null values in the objective target array.
    12. **USSF OM Constraint**: Ensures that the USSF OM constraint is not set if no USSF AFSCs are defined.
    13. **Rated Preferences**: Verifies that rated cadets have at least one rated preference.
    14. **Total Minimum and Maximum Capacities**: Checks that the total sum of minimum and maximum capacities is feasible.

    This function provides detailed information about issues, if any, found within the problem instance. It is a crucial step
    in guaranteeing the reliability and accuracy of the optimization process.
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

            # Make sure "utility" array is monotonically decreasing and the "cadet_utility" array is strictly decreasing
            arr_1 = p['utility'][i, p['cadet_preferences'][i]]
            arr_2 = p['cadet_utility'][i, p['cadet_preferences'][i]]
            if not all(arr_1[i] >= arr_1[i + 1] for i in range(len(arr_1) - 1)):
                invalid_utility += 1
                invalid_utility_cadets.append(i)
            if not all(arr_2[i] > arr_2[i + 1] for i in range(len(arr_2) - 1)):
                invalid_cadet_utility += 1
                invalid_cadet_utility_cadets.append(i)

    # Report issues with decreasing cadet utility values
    if invalid_utility > 0:
        issue += 1
        print(issue, "ISSUE: The cadet-reported utility matrix 'utility', located in 'Cadets Utility.csv'\nand in the "
                     "'Util' columns of 'Cadets.csv', does not incorporate monotonically\ndecreasing utility values for "
                     "" + str(invalid_utility) + " cadets. Please adjust.")
        if invalid_utility < 40:
            print('These are the cadets at indices', invalid_utility_cadets)
    if invalid_cadet_utility > 0:
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
    if vp['USSF OM'] is True and "USSF" not in p['afscs_acc_grp']:
        issue += 1
        print(issue, "ISSUE: Space Force OM constraint specified in value parameters (USSF OM = True) but no USSF"
                     " AFSCS found in the instance.")

    # At least one rated preference for rated eligible
    for soc in ['usafa', 'rotc']:
        if soc in p['Rated Cadets']:
            for i in p['Rated Cadets'][soc]:
                if len(p['Rated Choices'][soc][i]) == 0:
                    issue += 1
                    print(issue,
                          "ISSUE: Cadet '" + str(p['cadets'][i]) + "' is on " + soc.upper() +
                          "'s Rated list (" + soc.upper() + " Rated OM.csv) but is not eligible for any Rated AFSCs. "
                                                            "You need to remove their row from the csv.")

    # Make sure all cadets eligible for at least one rated AFSC are in their SOC's rated OM list
    for soc in ['usafa', 'rotc']:
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
    if np.sum(p['quota_max']) < p['N']:
        issue += 1
        print(issue, "ISSUE: Maximum constrained capacities (quota_max) is", int(np.sum(p['quota_max'])),
              " while 'N' is " + str(p['N']) + ". This is infeasible; we don't have enough positions for cadets to fill.")

    # Print statement
    print('Done,', issue, "issues found.")



