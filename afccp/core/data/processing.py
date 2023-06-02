# Import libraries
import os
import numpy as np
import pandas as pd
import copy
import afccp.core.globals
import afccp.core.data.preprocessing
import afccp.core.data.values


# File & Parameter Handling
def initialize_file_information(data_name: str, data_version: str):
    """
    Returns the file paths for importing and exporting the data files for a given data instance.

    Parameters:
    data_name (str): The name of the data instance.
    data_version (str): The version of the data instance.

    Returns:
    Tuple[Dict[str, str], Dict[str, str]]: A tuple containing two dictionaries. The first dictionary contains
    the file paths for importing the necessary data files, and the second dictionary contains the file paths
    for exporting the data files.

    The function checks for the existence of the data instance folder in the 'instances' directory. If the
    folder does not exist, it is created. The function also creates any sub-folders that do not already exist.

    The file paths are determined by checking the sub-folders for the necessary data files. If the data version
    is 'Default', the function imports and exports the files from the default sub-folders. If the data version is
    specified, the function checks if the sub-folder with that version exists. If it does not, the function
    imports the files from the default sub-folders and exports them to the specified version sub-folder. If the
    sub-folder with the specified version already exists, the function imports and exports the files from that
    sub-folder.

    The function returns two dictionaries containing the file paths for importing and exporting the data files.
    The keys in the dictionaries are the names of the data files, and the values are the corresponding file paths.
    """

    # If we don't already have the instance folder, we make it now
    instance_path = "instances/" + data_name + "/"
    if data_name not in afccp.core.globals.instances_available:
        os.mkdir(instance_path)
    instance_folder = np.array(os.listdir(instance_path))

    # Valid files/folders
    sub_folders = ["Original & Supplemental", "Combined Data", "CFMs", "Model Input", "Analysis & Results"]
    sub_folder_files = {"Model Input": ["Cadets", "Cadets Preferences", "Cadets Utility", "Cadets Utility Constraints",
                                        "AFSCs", "AFSCs Preferences", "AFSCs Utility", "Value Parameters",
                                        "Goal Programming"],
                        "Analysis & Results": ["Solutions"]}

    # Loop through each sub-folder in the above list and determine the filepaths for the various files
    import_filepaths = {}
    export_filepaths = {}
    for i, sub_folder in enumerate(sub_folders):

        # Sub-Folder with the number: "4. Model Input" for example
        numbered_sub_folder = str(i + 1) + ". " + sub_folder

        # All the sub-folders that have this numbered sub-folder
        if len(instance_folder) != 0:
            indices = np.flatnonzero(np.core.defchararray.find(instance_folder, numbered_sub_folder) != -1)
            sub_folder_individuals = instance_folder[indices]
        else:
            sub_folder_individuals = []

        # If this is the "default version", we already know what the sub-folder has to be
        if data_version == "Default":
            import_sub_folder = numbered_sub_folder
            export_sub_folder = numbered_sub_folder

            # If we don't currently have this sub-folder, we make it (New instance file)
            if numbered_sub_folder not in sub_folder_individuals:
                os.mkdir(instance_path + numbered_sub_folder + "/")

        # If the data version was specified, we have to check if it has the specific folder or not
        else:

            # If the version folder is not there, we import from the default but will export to the data version folder
            version_indices = np.flatnonzero(np.core.defchararray.find(sub_folder_individuals, data_version) != -1)
            if len(version_indices) == 0:
                import_sub_folder = numbered_sub_folder
                export_sub_folder = numbered_sub_folder + " (" + data_version + ")"

                # We will only ever export specific version data to these sub-folders
                if sub_folder in ["Model Input", "Analysis & Results"]:
                    os.mkdir(instance_path + export_sub_folder + "/")

            # We already have the version folder
            else:
                import_sub_folder = sub_folder_individuals[version_indices[0]]
                export_sub_folder = sub_folder_individuals[version_indices[0]]

        # If this is one of the sub-folders we can import/export to/from
        if sub_folder in sub_folder_files:

            # Get sub folder paths
            import_sub_folder_path = instance_path + import_sub_folder + "/"
            export_sub_folder_path = instance_path + export_sub_folder + "/"

            # Add generic file-paths for this sub-folder
            export_filepaths[sub_folder] = export_sub_folder_path
            import_filepaths[sub_folder] = import_sub_folder_path

            # Loop through each file listed above in the "sub_folder_files" for this sub-folder
            sub_folder_files_available = os.listdir(import_sub_folder_path)
            for file in sub_folder_files[sub_folder]:

                # Create the name of the file
                if data_version == "Default":
                    filename = data_name + " " + file + ".csv"
                else:
                    filename = data_name + " " + file + " (" + data_version + ").csv"

                # Get the path that we would export this file to
                export_filepaths[file] = export_sub_folder_path + filename

                # If we already have this file in the "import path", we add it to that filepath dictionary
                if filename in sub_folder_files_available:
                    import_filepaths[file] = import_sub_folder_path + filename
                elif data_version != "Default" and data_name + " " + file + ".csv" in sub_folder_files_available:
                    import_filepaths[file] = import_sub_folder_path + data_name + " " + file + ".csv"

    # If we don't have one of the Analysis & Results "sub-sub folders", we make it
    for sub_sub_folder in ["Data Charts", "Results Charts"]:
        if sub_sub_folder not in os.listdir(export_filepaths["Analysis & Results"]):
            os.mkdir(export_filepaths["Analysis & Results"] + sub_sub_folder + "/")

    # Return the information
    return import_filepaths, export_filepaths


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
                qual = afccp.core.handling.preprocessing.cip_to_qual_tiers(
                    p["afscs"][:p["M"]], p['cip1'], cip2=p['cip2'])
            else:
                qual = afccp.core.handling.preprocessing.cip_to_qual_tiers(
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

    # Cadet Indexed Sets
    p['I'] = np.arange(p['N'])
    p['J'] = np.arange(p['M'])
    p['J^E'] = [np.where(p['eligible'][i, :])[0] for i in p['I']]  # set of AFSCs that cadet i is eligible for

    # set of AFSCs that cadet i has placed a preference for and is also eligible for
    non_zero_utils_j = [np.where(p['utility'][i, :] > 0)[0] for i in p['I']]
    p["J^P"] = [np.intersect1d(p['J^E'][i], non_zero_utils_j[i]) for i in p['I']]

    # AFSC Indexed Sets
    p['I^E'] = [np.where(p['eligible'][:, j])[0] for j in p['J']]  # set of cadets that are eligible for AFSC j

    # Number of eligible cadets for each AFSC
    p["num_eligible"] = np.array([len(p['I^E'][j]) for j in p['J']])

    # set of cadets that have placed a preference for AFSC j and are eligible for AFSC j
    non_zero_utils_i = [np.where(p['utility'][:, j] > 0)[0] for j in p['J']]
    p["I^P"] = [np.intersect1d(p['I^E'][j], non_zero_utils_i[j]) for j in p['J']]

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
    if 'male' in p:
        male = np.where(p['male'] == 1)[0]  # set of male cadets
        p['male_proportion'] = np.mean(p['male'])
        p['I^D']['Male'] = [np.intersect1d(p['I^E'][j], male) for j in p['J']]
    if 'minority' in p:
        minority = np.where(p['minority'] == 1)[0]  # set of minority cadets
        p['minority_proportion'] = np.mean(p['minority'])
        p['I^D']['Minority'] = [np.intersect1d(p['I^E'][j], minority) for j in p['J']]

    # Add an extra column to the utility matrix for cadets who are unmatched
    zeros_vector = np.array([[0] for _ in range(p["N"])])
    p["utility"] = np.hstack((p["utility"], zeros_vector))

    # Merit
    if 'merit' in p:
        p['sum_merit'] = p['merit'].sum()  # should be close to N/2

    # Already Assigned cadets
    if "assigned" in p:
        p["J^Fixed"] = {}

        for i, afsc in enumerate(p["assigned"]):
            j = np.where(p["afscs"] == afsc)[0]  # AFSC index

            # Check if the cadet is actually assigned an AFSC already (it's not blank)
            if len(j) != 0:
                j = j[0]  # Actual index

                # Check if the cadet is assigned to an AFSC they're not eligible for
                if j not in p["J^E"][i]:
                    cadet = str(p['cadets'][i])
                    raise ValueError("Cadet " + cadet + " assigned to '" +
                                     afsc + "' but is not eligible for it. Adjust the qualification matrix!")
                else:
                    p["J^Fixed"][i] = j

    return p


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

    # Loop through each objective to see if there are any null values in the objective target array
    for k, objective in enumerate(vp["objectives"]):
        num_null = pd.isnull(vp["objective_target"][:, k]).sum()
        if num_null > 0:
            issue += 1
            print(issue, "ISSUE: Objective '" + objective + "' contains " +
                  str(num_null) + " null target values ('objective_target').")

    print('Done,', issue, "issues found.")


# Data Imports
def import_afscs_data(import_filepaths: dict, parameters: dict) -> dict:
    """
    Imports the 'AFSCs' csv file and updates the instance parameters dictionary with the values from the file.

    Args:
        import_filepaths (dict): A dictionary of filepaths containing the location of the 'AFSCs' csv file.
        parameters (dict): A dictionary of instance parameters to update.

    Returns:
        dict: The updated instance parameters.
    """

    # Shorthand
    p = parameters

    # Import 'AFSCs' dataframe
    afscs_df = afccp.core.globals.import_csv_data(import_filepaths["AFSCs"])

    # Remove "nans"
    afscs_df = afscs_df.replace('nan', '')
    afscs_df = afscs_df.fillna('')

    # Initialize dictionary translating 'AFSCs' df columns to their parameter counterparts
    afsc_columns_to_parameters = {"AFSC": "afscs", "Accessions Group": "acc_grp", "USAFA Target": "usafa_quota",
                                  "ROTC Target": "rotc_quota", "PGL Target": "pgl", "Estimated": "quota_e",
                                  "Desired": "quota_d", "Min": "quota_min", "Max": "quota_max"}

    # Loop through each column in the 'AFSCs' dataframe to put it into the p dictionary
    for col in afscs_df.columns:

        # If the column is an instance parameter, we load it in as a numpy array
        if col in afsc_columns_to_parameters:
            p_name = afsc_columns_to_parameters[col]
            p[p_name] = np.array(afscs_df.loc[:, col])

    # Number of AFSCs
    p["M"] = len(p["afscs"])

    # Add an "*" to the list of AFSCs to be considered the "Unmatched AFSC"
    p["afscs"] = np.hstack((p["afscs"], "*"))

    # Get the degree tier information from the AFSCs
    if "Deg Tier 1" in afscs_df:
        p["Deg Tiers"] = np.array(afscs_df.loc[:, "Deg Tier 1": "Deg Tier 4"]).astype(str)

    # Return parameters dictionary
    return p


def import_cadets_data(import_filepaths, parameters):
    """
    Imports data from the "Cadets" csv file and updates the instance parameters dictionary with relevant information.

    Args:
    import_filepaths (dict): A dictionary with the names and paths of the csv files to import. The "Cadets" file should be included.
    parameters (dict): A dictionary of instance parameters to be updated.

    Returns:
    dict: The updated instance parameters dictionary.
    """

    # Shorthand
    p = parameters

    # Import 'Cadets' dataframe
    cadets_df = afccp.core.globals.import_csv_data(import_filepaths["Cadets"])

    # Initialize dictionary translating 'AFSCs' df columns to their parameter counterparts
    cadet_columns_to_parameters = {"Cadet": "cadets", 'Male': 'male', 'Minority': 'minority', 'Race': 'race',
                                   "Ethnicity": "ethnicity", 'USAFA': 'usafa', 'ASC1': 'asc1', 'ASC2': 'asc2',
                                   'CIP1': 'cip1', 'CIP2': 'cip2', 'Merit': 'merit', 'Real Merit': 'merit_all',
                                   "Assigned": "assigned"}

    # Loop through each column in the 'Cadets' dataframe to put it into the p dictionary
    for col in cadets_df.columns:

        # Weird characters showing up
        if "ï»¿" in col:
            col_name = col.replace("ï»¿", "")
        else:
            col_name = col

        # If the column is an instance parameter, we load it in as a numpy array
        if col_name in cadet_columns_to_parameters:
            p_name = cadet_columns_to_parameters[col_name]
            p[p_name] = np.array(cadets_df.loc[:, col])

    # Number of Cadets
    p["N"] = len(p["cadets"])

    # Get qual matrix information
    p = gather_degree_tier_qual_matrix(cadets_df, p)

    # Number of cadet preference choices available and number of utilities available
    p["P"] = len([col for col in cadets_df.columns if 'Pref_' in col])
    p["num_util"] = min(10, p["P"])
    if p["P"] != 0:
        # Get the preferences and utilities columns from the cadets dataframe
        p["c_preferences"] = np.array(cadets_df.loc[:, "Pref_1": "Pref_" + str(p['P'])])
        p["c_utilities"] = np.array(cadets_df.loc[:, "Util_1": "Util_" + str(p["num_util"])])

    # Return parameters dictionary
    return p


def import_preferences_data(import_filepaths, parameters):
    """
    Imports additional preference data (if available) for cadets and AFSCs, and adds the relevant information to the
    input dictionary of parameters.

    Parameters:
    import_filepaths (dict): A dictionary containing the filepaths of the csv files to be imported.
        - Required keys: 'Cadets Utility', 'Cadets Preferences', 'AFSCs Utility', 'AFSCs Preferences' (if available)
    parameters (dict): A dictionary containing the initial input parameters for the model.
        - Required keys: 'afscs', 'N', 'M', 'num_util', 'P'

    Returns:
    dict: The updated dictionary of parameters.

    Raises:
    ValueError: If there is no cadet utility data provided, which is required.

    Note:
    The function expects that the 'AFSCs' and 'Cadets' csv files have already been imported.
    """

    # Shorthand
    p = parameters

    # Loop through the potential additional dataframes and import them if we have them
    datasets = {}
    for dataset in ["Cadets Utility", "Cadets Preferences", "AFSCs Utility", "AFSCs Preferences"]:

        # If we have the dataset, import it
        if dataset in import_filepaths:
            datasets[dataset] = afccp.core.globals.import_csv_data(import_filepaths[dataset])

    # Determine how we incorporate the cadets' utility matrix
    afsc_1, afsc_M = p["afscs"][0], p["afscs"][p["M"] - 1]
    if "Cadets Utility" in datasets:  # Load in the matrix directly
        p["utility"] = np.array(datasets["Cadets Utility"].loc[:, afsc_1: afsc_M])
    elif "c_utilities" in p:  # Create the matrix using the columns

        # Create utility matrix (numpy array NxM) from the utility/preference column information
        p["utility"] = np.zeros([p["N"], p["M"]])
        for i in range(p["N"]):
            for util in range(p['num_util']):
                j = np.where(p["c_preferences"][i, util] == p["afscs"])[0]
                if len(j) != 0:
                    p['utility'][i, j[0]] = p["c_utilities"][i, util]
    else:
        raise ValueError("Error. No cadet utility data provided which is required.")

    # Determine how we incorporate the cadets' preferences dataframe
    if "Cadets Preferences" in datasets:  # Load in the preferences dataframe directly
        p["c_pref_matrix"] = np.array(datasets["Cadets Preferences"].loc[:, afsc_1: afsc_M])
    elif "c_preferences" in p:  # Create the preferences dataframe using the columns

        # Create cadet preferences dataframe (numpy array NxM) from the preference column information
        p["c_pref_matrix"] = np.zeros([p["N"], p["M"]]).astype(int)
        for i in range(p["N"]):
            for util in range(p['P']):
                j = np.where(p["c_preferences"][i, util] == p["afscs"])[0]
                if len(j) != 0:
                    p['c_pref_matrix'][i, j[0]] = util + 1  # 1 is first choice (NOT 0)

    # AFSC preferences and utilities are not required initial data elements (Depending on how we solve, they may be)
    if "AFSCs Utility" in datasets:  # Load in the AFSC utility matrix
        p["afsc_utility"] = np.array(datasets["AFSCs Utility"].loc[:, afsc_1: afsc_M])
    if "AFSCs Preferences" in datasets:  # Load in the AFSC preferences dataframe
        p["a_pref_matrix"] = np.array(datasets["AFSCs Preferences"].loc[:, afsc_1: afsc_M])

    # Return dictionary of parameters
    return p


def import_value_parameters_data(import_filepaths, parameters, num_breakpoints=24):
    """
    Imports the data pertaining to the value parameters of the model.

    Args:
        import_filepaths (dict): A dictionary of file paths to import.
            Required keys: "Model Input", "Value Parameters".
            Optional key: "Cadets Utility Constraints".
        parameters (dict): A dictionary of parameters for the model.
        num_breakpoints (int): The number of breakpoints to use for value functions.

    Returns:
        dict: A dictionary containing the value parameters for the model.

    Raises:
        FileNotFoundError: If a file path in import_filepaths is invalid.

    The "value parameters" refer to the weights, values, and constraints applied to the VFT model that by in large
    the analyst determines. The function loads the following data:
        - A dataframe of cadet utility constraints (if present).
        - A dataframe of value parameters that describes sets of weights, values, and constraints for the VFT model.
        - A dataframe for each set of value parameters, specifying weights, values, and constraints for the VFT model.
    The function returns a dictionary of value parameters, containing the following keys:
        - 'O': The number of AFSC objectives.
        - 'afscs_overall_weight': The weight for the AFSCs component.
        - 'cadets_overall_weight': The weight for the cadets component.
        - 'cadet_weight_function': The type of function used to calculate cadet weights.
        - 'afsc_weight_function': The type of function used to calculate AFSC weights.
        - 'cadets_overall_value_min': The minimum value for the cadets component.
        - 'afscs_overall_value_min': The minimum value for the AFSCs component.
        - 'M': The number of AFSCs.
        - 'afsc_value_min': An array of minimum AFSC values.
        - 'cadet_value_min': An array of minimum cadet values.
        - 'objective_value_min': A 2D array of minimum objective values.
        - 'value_functions': A 2D array of value functions for each AFSC objective.
        - 'constraint_type': A 2D array specifying the type of constraint for each objective and AFSC.
        - 'a': A nested list of breakpoints for each AFSC objective.
        - 'objective_target': A 2D array of target values for each objective and AFSC.
        - 'f^hat': A nested list of breakpoint values for each objective and AFSC.
        - 'objective_weight': A 2D array of weights for each objective and AFSC.
        - 'afsc_weight': An array of weights for each AFSC.
        - 'objectives': An array of AFSC objectives.
        - 'K^A': A dictionary of weights for the cadets component.
    """

    # Shorthand
    p = parameters
    afccp_vp = afccp.core.handling.values  # Reduce the module name so it fits on one line

    # Import the cadets utility constraints dataframe if we have it.
    if "Cadets Utility Constraints" in import_filepaths:
        vp_cadet_df = afccp.core.globals.import_csv_data(import_filepaths["Cadets Utility Constraints"])
    else:
        vp_cadet_df = None

    # Import the "Value Parameters" dataframe if we have it. If we don't, the "vp_dict" will be "None"
    if "Value Parameters" in import_filepaths:
        overall_vp_df = afccp.core.globals.import_csv_data(import_filepaths["Value Parameters"])
    else:
        return None  # Nothing more we can do now (No "Value Parameters" determined yet for this instance)

    # Information about the sets of value parameters listed in the "Value Parameters" dataframe
    vp_names = np.array(overall_vp_df['VP Name'])
    num_vps = len(vp_names)
    vp_weights = np.ones(num_vps) * 100  # Initially all weighted at 100%
    if 'VP Weight' in overall_vp_df:
        vp_weights = np.array(overall_vp_df['VP Weight'])

    # Determine the filenames for the sets of value parameters (VP, VP_2, etc.)
    vp_files = {}
    for file in os.listdir(import_filepaths["Model Input"]):
        if ".csv" not in file:
            continue
        check_vp = file.split(" ")[1].replace(".csv", "")
        if check_vp in vp_names:
            vp_files[check_vp] = file

    # Loop through each set of value parameters and load it into the dictionary
    vp_dict = {}
    for v, vp_name in enumerate(vp_names):

        # Only load in the set of value parameters if we have it
        if vp_name in vp_files:
            filepath = import_filepaths["Model Input"] + vp_files[vp_name]
        else:
            print("WARNING. Value Parameter set '" + vp_name + "' listed in the 'Value Parameters' dataframe but does"
                                                               " not have its own dataframe (.csv) in 'Model Inputs'.")
            continue  # Skip this set of value parameters

        # Load value parameter set dataframe
        vp_df = afccp.core.globals.import_csv_data(filepath)
        M, O = p['M'], int(len(vp_df) / p['M'])  # Number of AFSCs (M) and number of AFSC objectives (O)

        # Initialize value parameters dictionary
        value_parameters = {'O': O, "afscs_overall_weight": np.array(overall_vp_df['AFSCs Weight'])[v],
                            "cadets_overall_weight": np.array(overall_vp_df['Cadets Weight'])[v],
                            "cadet_weight_function": np.array(overall_vp_df['Cadet Weight Function'])[v],
                            "afsc_weight_function": np.array(overall_vp_df['AFSC Weight Function'])[v],
                            "cadets_overall_value_min": np.array(overall_vp_df['Cadets Min Value'])[v],
                            "afscs_overall_value_min": np.array(overall_vp_df['AFSCs Min Value'])[v], "M": M,
                            "afsc_value_min": np.zeros(M), 'cadet_value_min': np.zeros(p['N']),
                            "objective_value_min": np.array([[" " * 20 for _ in range(O)] for _ in range(M)]),
                            "value_functions": np.array([[" " * 200 for _ in range(O)] for _ in range(M)]),
                            "constraint_type": np.zeros([M, O]), 'a': [[[] for _ in range(O)] for _ in range(M)],
                            "objective_target": np.zeros([M, O]), 'f^hat': [[[] for _ in range(O)] for _ in range(M)],
                            "objective_weight": np.zeros([M, O]), "afsc_weight": np.zeros(M),
                            'objectives': np.array(vp_df.loc[:int(len(vp_df) / M - 1), 'Objective']), "K^A": {},
                            'num_breakpoints': num_breakpoints}

        # If we have constraints specified for cadet utility
        if vp_cadet_df is not None:
            value_parameters["cadet_value_min"] = np.array(vp_cadet_df[vp_name]).astype(float)

        # Check if other columns are present (phasing these in)
        more_vp_columns = ["USAFA-Constrained AFSCs", "Cadets Top 3 Constraint"]
        for col in more_vp_columns:
            if col in overall_vp_df:
                element = str(np.array(overall_vp_df[col])[v])
                if element == "nan":
                    element = ""
                value_parameters[col] = element
            else:
                value_parameters[col] = ""

        # Determine weights on cadets
        if 'merit_all' in parameters:
            value_parameters['cadet_weight'] = afccp_vp.cadet_weight_function(
                parameters['merit_all'], func=value_parameters['cadet_weight_function'])
        else:
            value_parameters['cadet_weight'] = afccp_vp.cadet_weight_function(
                parameters['merit'], func=value_parameters['cadet_weight_function'])

        # Load in value parameter data for each AFSC
        for j in p["J"]:  # These are Os (Ohs) not 0s (zeros)
            value_parameters["objective_target"][j, :] = np.array(vp_df.loc[j * O:(j * O + O - 1), 'Objective Target'])

            # Force objective weights to sum to 1. K^A is the set of objectives that have non-zero weights for each AFSC
            objective_weights = np.array(vp_df.loc[j * O:(j * O + O - 1), 'Objective Weight'])
            value_parameters["objective_weight"][j, :] = objective_weights / sum(objective_weights)
            value_parameters['K^A'][j] = np.where(value_parameters['objective_weight'][j, :] > 0)[0].astype(int)

            value_parameters["objective_value_min"][j, :] = np.array(vp_df.loc[j * O:(j * O + O - 1),
                                                                     'Min Objective Value'])
            value_parameters["constraint_type"][j, :] = np.array(vp_df.loc[j * O:(j * O + O - 1), 'Constraint Type'])
            value_parameters["value_functions"][j, :] = np.array(vp_df.loc[j * O:(j * O + O - 1), 'Value Functions'])
            value_parameters["afsc_weight"][j] = vp_df.loc[j * O, "AFSC Weight"]
            value_parameters["afsc_value_min"][j] = vp_df.loc[j * O, "Min Value"]
            cadets = parameters['I^E'][j]  # Indices of cadets that are eligible for this AFSC

            # Loop through each objective for this AFSC
            for k, objective in enumerate(value_parameters['objectives']):

                # Refactored column names
                if 'Function Breakpoints' in vp_df:
                    measure_col_name = 'Function Breakpoints'
                    value_col_name = 'Function Breakpoint Values'
                else:
                    measure_col_name = 'Function Breakpoint Measures (a)'
                    value_col_name = 'Function Breakpoint Values (f^hat)'

                # We import the functions directly from the breakpoints
                if num_breakpoints is None:
                    a_string = vp_df.loc[j * O + k, measure_col_name]
                    if type(a_string) == str:
                        value_parameters['a'][j][k] = [float(x) for x in a_string.split(",")]
                    fhat_string = vp_df.loc[j * O + k, value_col_name]
                    if type(fhat_string) == str:
                        value_parameters['f^hat'][j][k] = [float(x) for x in fhat_string.split(",")]

                # Recreate the functions from the vf strings
                else:
                    vf_string = value_parameters["value_functions"][j, k]
                    if vf_string != 'None':
                        if objective == 'Merit':
                            actual = np.mean(parameters['merit'][cadets])
                        elif objective == 'USAFA Proportion':
                            actual = np.mean(parameters['usafa'][cadets])
                        else:
                            actual = None

                        # Adjust target information for the "Combined Quota" objective
                        if objective == 'Combined Quota':
                            minimum, maximum = p['quota_min'][j], p['quota_max'][j]
                            target = p['quota_d'][j]  # Desired number of cadets
                        else:
                            minimum, maximum, target = None, None, value_parameters['objective_target'][j, k]

                        # Construct the value function (get the breakpoint coordinates)
                        segment_dict = afccp_vp.create_segment_dict_from_string(
                            vf_string, target, actual=actual, maximum=maximum, minimum=minimum)
                        value_parameters['a'][j][k], value_parameters['f^hat'][j][k] = afccp_vp.value_function_builder(
                            segment_dict, num_breakpoints=num_breakpoints)

        # Force AFSC weights to sum to 1
        value_parameters["afsc_weight"] = value_parameters["afsc_weight"] / sum(value_parameters["afsc_weight"])

        # "Condense" the value functions by removing unnecessary zeros
        value_parameters = afccp_vp.condense_value_functions(p, value_parameters)

        # Add indexed sets and subsets of AFSCs and AFSC objectives
        value_parameters = afccp_vp.value_parameters_sets_additions(p, value_parameters)

        # Save the value parameters to the dictionary
        vp_dict[vp_name] = copy.deepcopy(value_parameters)
        vp_dict[vp_name]['vp_weight'] = vp_weights[v]
        vp_dict[vp_name]['vp_local_weight'] = vp_weights[v] / sum(vp_weights)

    # Return the dictionary of value parameter sets
    return vp_dict


def import_solutions_data(import_filepaths, parameters):
    """
    This function takes in the names of the files (and paths) to import, as well as a dictionary of parameters (p)
    and then imports the "Solutions" dataframe. A dictionary of solutions is then returned.

    Args:
        import_filepaths (dict): A dictionary of file names (and paths) to import.
        parameters (dict): A dictionary of parameters needed for the function to run.

    Returns:
        dict: A dictionary of solutions for the given parameters.
    """

    # Shorthand
    p = parameters

    # Import the "Solutions" dataframe if we have it. If we don't, the "solutions_dict" will be "None"
    if "Solutions" in import_filepaths:
        solutions_df = afccp.core.globals.import_csv_data(import_filepaths["Solutions"])
    else:
        return None  # Nothing more we can do now (No solutions determined yet for this instance)

    # Get list of solution names
    solution_names = list(solutions_df.keys())[1:]

    # Loop through each solution, convert to a numpy array of AFSC indices, and then add it to the dictionary
    solution_dict = {}
    for solution_name in solution_names:
        # Convert solution of AFSC names to indices and then save it to the dictionary
        afsc_solution = np.array(solutions_df[solution_name])  # ["15A", "14N", "17X", ...]
        solution = np.array([np.where(p['afscs'] == afsc)[0][0] for afsc in afsc_solution])  # [3, 2, 5, ...]
        solution_dict[solution_name] = solution

    # Return the dictionary of solutions
    return solution_dict


# Data Exports
def export_afscs_data(instance):
    """
    This function takes an Instance object as an argument, which contains the AFSC data to be exported.
    It creates a dictionary of columns to be included in the "AFSCs" csv file by translating AFSC parameters
    to their corresponding column names in the file. The function then creates a dataframe using these columns
    and exports it as a csv file to the path specified in the export_paths attribute of the Instance object.

    Args:
        instance (Instance): An Instance object containing the AFSC data to export.

    Returns:
        None
    """

    # Shorthand
    p = instance.parameters

    # Initialize dictionary translating AFSC parameters to their "AFSCs" df column counterparts
    afsc_parameters_to_columns = {"afscs": "AFSC", "acc_grp": "Accessions Group", "usafa_quota": "USAFA Target",
                                  "rotc_quota": "ROTC Target", "pgl": "PGL Target", "quota_e": "Estimated",
                                  "quota_d": "Desired", "quota_min": "Min", "quota_max": "Max"}

    # Loop through each parameter in the translation dictionary to create dictionary of "AFSCs" columns
    afscs_columns = {}
    for parameter in afsc_parameters_to_columns:

        # If we have the parameter, create the column
        if parameter in p:
            col_name = afsc_parameters_to_columns[parameter]
            afscs_columns[col_name] = p[parameter][:p["M"]]  # Don't want to include the *!

    # Create the degree tier columns
    if "Deg Tiers" in p:
        for t in range(4):
            afscs_columns["Deg Tier " + str(t + 1)] = p["Deg Tiers"][:, t]

    # Create dataframe
    afscs_df = pd.DataFrame(afscs_columns)

    # Export 'AFSCs' dataframe
    afscs_df.to_csv(instance.export_paths["AFSCs"], index=False)


def export_cadets_data(instance):
    """
    Export the "Cadets" csv by taking in the names of the files (and paths) to export, as well as a dictionary of
    parameters (p). The function first translates 'AFSCs' df columns to their parameter counterparts and creates
    the corresponding 'Cadets' dataframe column. If cadet preference/utility columns and qualification matrix
    exist in the input parameters, they will be added to the 'Cadets' dataframe. Finally, the function exports the
    'Cadets' dataframe to a csv file.

    Args:
    - instance: an instance of the CadetCareerProblem class

    Returns: None
    """

    # Shorthand
    p = instance.parameters

    # Initialize dictionary translating 'AFSCs' df columns to their parameter counterparts
    cadet_parameters_to_columns = {"cadets": "Cadet", "assigned": "Assigned", 'usafa': 'USAFA', 'male': 'Male',
                                   'minority': 'Minority', 'race': 'Race', "ethnicity": "Ethnicity",
                                   'asc1': 'ASC1', 'asc2': 'ASC2', 'cip1': 'CIP1', 'cip2': 'CIP2',
                                   'merit': 'Merit', 'merit_all': 'Real Merit'}

    # Loop through each parameter in the translation dictionary to get "Cadets" dataframe column counterpart
    cadets_columns = {}
    for parameter in cadet_parameters_to_columns:

        # If we have the parameter, we create its column
        if parameter in p:
            col_name = cadet_parameters_to_columns[parameter]
            cadets_columns[col_name] = p[parameter]

    # If we had the cadet preference/utility columns before, we'll add them back in
    if "c_preferences" in p:

        # Add utility columns
        for c in range(p["num_util"]):
            cadets_columns["Util_" + str(c + 1)] = p["c_utilities"][:, c]

        # Add preference columns
        for c in range(p["P"]):
            cadets_columns["Pref_" + str(c + 1)] = p["c_preferences"][:, c]

    # If we have the qual matrix, we add that here
    if "qual" in p:

        for j, afsc in enumerate(p["afscs"][:p["M"]]):
            cadets_columns["qual_" + afsc] = p['qual'][:, j]

    # Create dataframe
    cadets_df = pd.DataFrame(cadets_columns)

    # Export 'Cadets' dataframe
    cadets_df.to_csv(instance.export_paths["Cadets"], index=False)


def export_preferences_data(instance):
    """
    Exports the preferences dataframes if they exist in the instance's parameters dictionary.

    Parameters:
    instance (Instance): An Instance object containing the parameters dictionary and export paths.

    Returns:
    None

    For each potential dataset to export, if it exists in the instance's parameters dictionary, a new dataframe is constructed
    and exported to the corresponding file path. The potential datasets are:
    - "Cadets Utility": A matrix of the utility values that each cadet assigns to each AFSC.
    - "Cadets Preferences": A matrix of the preferences of each cadet for each AFSC, derived from the utility values.
    - "AFSCs Utility": A matrix of the utility values that each AFSC assigns to each cadet.
    - "AFSCs Preferences": A matrix of the preferences of each AFSC for each cadet, derived from the utility values.

    Each dataframe has the cadets in the first column and the AFSCs in the remaining columns. The values in each cell
    correspond to the utility or preference value of the cadet or AFSC for that particular AFSC or cadet.
    """

    # Shorthand
    p = instance.parameters

    # Dataset name translations
    parameter_trans_dict = {"utility": "Cadets Utility", "c_pref_matrix": "Cadets Preferences",
                            "afsc_utility": "AFSCs Utility", "a_pref_matrix": "AFSCs Preferences"}

    # Loop through each potential dataset to export
    for parameter in parameter_trans_dict:

        # If we have this dataset, we export it
        if parameter in p:
            dataset = parameter_trans_dict[parameter]

            # Construct the dataframe
            pref_df = pd.DataFrame({"Cadet": p["cadets"]})

            # Add the AFSC columns
            for j, afsc in enumerate(p["afscs"][:p["M"]]):
                pref_df[afsc] = p[parameter][:, j]

            # Export the dataset
            pref_df.to_csv(instance.export_paths[dataset], index=False)


def export_value_parameters_data(instance):
    """
    This function takes in a problem instance object and then exports the various value parameter datasets to their csvs
    """

    # Shorthand
    p = instance.parameters

    # Error data
    if instance.vp_dict is None:
        raise ValueError("Error. No value parameters to export.")

    # Determine how we're going to show merit for context in the Cadet Constraints dataframe
    merit_col = "merit"
    if "merit_all" in p:
        merit_col = "merit_all"

    # Initialize dataframes
    vp_cadet_df = pd.DataFrame({"Cadet": p["cadets"], "Merit": p[merit_col]})
    vp_overall_df = pd.DataFrame({})

    # Loop through each set of value parameters
    for v, vp_name in enumerate(list(instance.vp_dict.keys())):
        vp = instance.vp_dict[vp_name]

        # Initialize Value Function breakpoint arrays
        a_strings = np.array([[" " * 400 for _ in vp["K"]] for _ in p["J"]])
        fhat_strings = np.array([[" " * 400 for _ in vp["K"]] for _ in p["J"]])
        for j, afsc in enumerate(p['afscs'][:p["M"]]):
            for k, objective in enumerate(vp['objectives']):
                a_string_list = [str(x) for x in vp['a'][j][k]]
                a_strings[j, k] = ",".join(a_string_list)
                fhat_strings_list = [str(x) for x in vp['f^hat'][j][k]]
                fhat_strings[j, k] = ",".join(fhat_strings_list)

        # Flatten the 2-d arrays to convert them into one long list that is sorted by AFSC and then by objective
        objective_value_min = np.ndarray.flatten(vp['objective_value_min'])
        constraint_type = np.ndarray.flatten(vp['constraint_type'])
        objective_target = np.ndarray.flatten(vp['objective_target'])
        value_functions = np.ndarray.flatten(vp['value_functions'])
        breakpoint_a = np.ndarray.flatten(a_strings)
        breakpoint_fhat = np.ndarray.flatten(fhat_strings)

        # AFSC objective swing weights
        max_weights = np.max(vp['objective_weight'], axis=1)
        ow = np.array([[vp['objective_weight'][j, k] / max_weights[j] for k in vp["K"]] for j in p["J"]])
        objective_weight = np.ndarray.flatten(np.around(ow * 100, 3))

        # Repeating objectives
        objectives = np.tile(vp['objectives'], p['M'])

        # Repeating AFSCs
        afscs = np.ndarray.flatten(np.array(list(np.repeat(p['afscs'][j], vp['O']) for j in p["J"])))
        afsc_value_min = np.ndarray.flatten(np.array(list(np.repeat(vp['afsc_value_min'][j], vp['O']) for j in p["J"])))

        # AFSC swing weights
        afsc_weight = np.around(vp['afsc_weight'] / np.max(vp['afsc_weight']) * 100, 3)
        afsc_weight = np.repeat(afsc_weight, vp['O'])

        # Create the "vp_df"
        vp_df = pd.DataFrame({'AFSC': afscs, 'Objective': objectives, 'Objective Weight': objective_weight,
                              'Objective Target': objective_target, 'AFSC Weight': afsc_weight,
                              'Min Value': afsc_value_min, 'Min Objective Value': objective_value_min,
                              'Constraint Type': constraint_type, 'Function Breakpoint Measures (a)': breakpoint_a,
                              'Function Breakpoint Values (f^hat)': breakpoint_fhat,
                              'Value Functions': value_functions})

        # Add the minimum value column for this set of value parameters to the cadets df
        vp_cadet_df[vp_name] = vp["cadet_value_min"]

        # Initialize overall vp column dictionary
        overall_vp_columns = {'VP Name': vp_name,
                              'Cadets Weight': vp['cadets_overall_weight'],
                              'AFSCs Weight': vp['afscs_overall_weight'],
                              'Cadets Min Value': vp['cadets_overall_value_min'],
                              'AFSCs Min Value': vp['afscs_overall_value_min'],
                              'Cadet Weight Function': vp['cadet_weight_function'],
                              'AFSC Weight Function': vp['afsc_weight_function'],
                              "USAFA-Constrained AFSCs": vp["USAFA-Constrained AFSCs"],
                              "Cadets Top 3 Constraint": vp["Cadets Top 3 Constraint"]}

        # Add the row for this set of value parameters to the overall df
        for col in overall_vp_columns:
            vp_overall_df.loc[v, col] = overall_vp_columns[col]

        # Determine filename of VP file
        if instance.data_version == "Default":
            filename = instance.data_name + " " + vp_name + ".csv"
        else:
            filename = instance.data_name + " " + vp_name + " (" + instance.data_version + ").csv"

        # Export 'VP' dataframe
        vp_df.to_csv(instance.export_paths["Model Input"] + filename, index=False)

    # Export 'Value Parameters' dataframe
    vp_overall_df.to_csv(instance.export_paths["Value Parameters"], index=False)

    # Export 'Cadets Utility Constraints' dataframe
    vp_cadet_df.to_csv(instance.export_paths["Cadets Utility Constraints"], index=False)


def export_solutions_data(instance):
    """
    This function takes in a problem instance object and then exports the "Solutions" dataframe to a csv
    """

    # Shorthand
    p = instance.parameters

    # Error data
    if instance.solution_dict is None:
        raise ValueError("Error. No solutions to export.")

    # Initialize solutions dataframe
    solutions_df = pd.DataFrame({"Cadet": p["cadets"]})

    # Loop through each solution and add it to the dataframe
    for solution_name in instance.solution_dict:
        # Convert solution of indices to AFSC names and then save it to the dictionary
        solution = instance.solution_dict[solution_name]
        afsc_solution = [p["afscs"][j] for j in solution]
        solutions_df[solution_name] = afsc_solution

    # Export 'Solutions' dataframe
    solutions_df.to_csv(instance.export_paths["Solutions"], index=False)
