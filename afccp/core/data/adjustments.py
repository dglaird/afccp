import numpy as np
import pandas as pd
import copy

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
    Creates indexed sets and subsets for both the AFSCs and the cadets
    :return instance parameters
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
        p['usafa_cadets'] = np.where(p['usafa'])[0]
        p['rotc_cadets'] = np.where(p['usafa'] == 0)[0]

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

    return p


def more_parameter_additions(parameters):
    """
    This function adds even more parameter sets to "parameters"
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
    This function runs through all the different parameters and sanity checks them to make sure that they make
    sense and don't break the model
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



