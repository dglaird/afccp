import os
import numpy as np
import pandas as pd
import xlsxwriter
import copy

# afccp modules
import afccp.core.globals
import afccp.core.data.values
import afccp.core.data.adjustments

# File Handling
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
                                        "Cadets Utility (Final)", "AFSCs", "AFSCs Preferences", "AFSCs Utility",
                                        "Value Parameters", "Goal Programming", "ROTC Rated Interest",
                                        "ROTC Rated OM", "USAFA Rated OM"],
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
    for sub_sub_folder in ["Data Charts", "Results Charts", "Cadet Board", 'Value Functions']:
        if sub_sub_folder not in os.listdir(export_filepaths["Analysis & Results"]):
            os.mkdir(export_filepaths["Analysis & Results"] + sub_sub_folder + "/")

    # Return the information
    return import_filepaths, export_filepaths


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
    p = afccp.core.data.adjustments.gather_degree_tier_qual_matrix(cadets_df, p)

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
    for dataset in ["Cadets Utility", "Cadets Preferences", "AFSCs Utility", "AFSCs Preferences",
                    "ROTC Rated Interest", "ROTC Rated OM", "USAFA Rated OM", 'Cadets Utility (Final)']:

        # If we have the dataset, import it
        if dataset in import_filepaths:
            datasets[dataset] = afccp.core.globals.import_csv_data(import_filepaths[dataset])

    # First and last AFSC (for collecting matrices from dataframes)
    afsc_1, afsc_M = p["afscs"][0], p["afscs"][p["M"] - 1]

    # Determine how we incorporate the original cadets' utility matrix
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

    # Cadets "Real" Utility (after aggregating it with their ordinal rankings)
    if 'Cadets Utility (Final)' in datasets:  # Load in the cadet utility matrix
        p['cadet_utility'] = np.array(datasets["Cadets Utility (Final)"].loc[:, afsc_1: afsc_M])

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

    # All USAFA Cadets
    p['usafa_cadets'] = np.where(p['usafa'])[0]

    # Rated dataframes
    if "ROTC Rated Interest" in datasets:
        r_afscs = list(datasets['ROTC Rated Interest'].columns[1:])
        p['rr_interest_matrix'] = np.array(datasets['ROTC Rated Interest'].loc[:, r_afscs[0]:r_afscs[len(r_afscs) - 1]])
    if "ROTC Rated OM" in datasets:
        r_afscs = list(datasets['ROTC Rated OM'].columns[1:])
        p['rr_om_matrix'] = np.array(datasets['ROTC Rated OM'].loc[:, r_afscs[0]:r_afscs[len(r_afscs) - 1]])
        p['rr_om_cadets'] = np.array(datasets['ROTC Rated OM']['Cadet'])
    if "USAFA Rated OM" in datasets:
        r_afscs = list(datasets['USAFA Rated OM'].columns[1:])
        p['ur_om_matrix'] = np.array(datasets['USAFA Rated OM'].loc[:, r_afscs[0]:r_afscs[len(r_afscs) - 1]])
        p['ur_om_cadets'] = np.array(datasets['USAFA Rated OM']['Cadet'])

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
    afccp_vp = afccp.core.data.values  # Reduce the module name so it fits on one line

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

    # Degree Tier Counts
    if "Tier 1" in p['I^D']:
        for t in ['1', '2', '3', '4']:
            afscs_columns['Deg Tier ' + t + ' Count'] = [len(p['I^D']['Tier ' + t][j]) for j in p['J']]

    # Preference Counts
    if 'Choice Count' in p:
        for choice in p['Choice Count']:
            afscs_columns['Choice ' + str(choice + 1)] = p['Choice Count'][choice]

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
    p, vp = instance.parameters, instance.value_parameters

    # Dataset name translations
    parameter_trans_dict = {"utility": "Cadets Utility", "c_pref_matrix": "Cadets Preferences",
                            "afsc_utility": "AFSCs Utility", "a_pref_matrix": "AFSCs Preferences",
                            "rr_interest_matrix": "ROTC Rated Interest", "rr_om_matrix": "ROTC Rated OM",
                            'ur_om_matrix': 'USAFA Rated OM', 'cadet_utility': 'Cadets Utility (Final)'}

    # Loop through each potential dataset to export
    for parameter in parameter_trans_dict:

        # If we have this dataset, we export it
        if parameter in p:
            dataset = parameter_trans_dict[parameter]

            # Construct the dataframe
            if 'ROTC' in dataset:
                cadet_indices = p["Rated Cadets"]['rotc']
                pref_df = pd.DataFrame({"Cadet": p['cadets'][cadet_indices]})
                afscs = [afsc for afsc in p['afscs_acc_grp']['Rated'] if '_U' not in afsc]
            elif 'USAFA' in dataset:
                cadet_indices = p["Rated Cadets"]['usafa']
                pref_df = pd.DataFrame({"Cadet": p['cadets'][cadet_indices]})
                afscs = [afsc for afsc in p['afscs_acc_grp']['Rated'] if '_R' not in afsc]
            else:
                pref_df = pd.DataFrame({"Cadet": p["cadets"]})
                afscs = p["afscs"][:p["M"]]

            # Add the AFSC columns
            for j, afsc in enumerate(afscs):
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
        return None  # No value parameters to export!

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

        # Determine extension of VP file (in case it's a different version of data)
        if instance.data_version == "Default":
            extension = ".csv"
        else:
            extension = " (" + instance.data_version + ").csv"

        # Export 'VP' dataframe
        vp_df.to_csv(instance.export_paths["Model Input"] + instance.data_name + " " + vp_name + extension, index=False)

        # Create "Global Utility" dataframe if it's in the value parameters
        if "global_utility" in vp:
            gu_df = pd.DataFrame({'Cadet': p['cadets']})
            for j, afsc in enumerate(p['afscs'][:p['M']]):
                gu_df[afsc] = vp['global_utility'][:, j]

            # Export "Global Utility" dataframe
            filename = instance.data_name + " " + vp_name + " Global Utility" + extension
            gu_df.to_csv(instance.export_paths["Model Input"] + filename, index=False)

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
        return None  # No solutions to export!

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


# Solution Results excel file
def export_solution_results_excel(instance, filepath):
    """
    This function exports the metrics for one solution back to excel for review
    """

    # Shorthand
    p, vp, metrics = instance.parameters, instance.value_parameters, instance.metrics

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter')

    # Get the xlsxwriter objects from the dataframe writer object.
    workbook = writer.book
    worksheet = workbook.add_worksheet("Main")

    # Make the background white initially
    white_format = workbook.add_format({'bold': False, 'font_color': 'black', 'bg_color': 'white',
                                       'font_size': 14, 'font_name': 'Calibri'})
    for r in range(200):
        for c in range(50):
            worksheet.write(r, c, '', white_format)

    # Merge cells
    merge_format = workbook.add_format({'bold': True, 'font_color': 'black', 'bg_color': 'white',
                                       'font_size': 14, 'font_name': 'Calibri', 'align': 'center',
                                        'valign': 'vcenter', 'border_color': 'black', 'border': 1})
    worksheet.merge_range("B2:D2", "VFT Overall Metrics", merge_format)
    worksheet.merge_range("F2:G2", "Additional Metrics", merge_format)
    worksheet.write('I2', 'Preference', merge_format)
    worksheet.write('J2', 'Count', merge_format)
    worksheet.write('K2', 'Proportion', merge_format)

    # Objective Value
    obj_format = workbook.add_format({'bold': True, 'font_color': 'black', 'bg_color': 'yellow',
                                      'font_size': 14, 'font_name': 'Calibri', 'align': 'center',
                                      'valign': 'vcenter', 'border_color': 'black', 'border': 1})
    worksheet.merge_range("C6:D6", round(metrics['z'], 4), obj_format)

    # Other cells
    cell_format = workbook.add_format({'bold': False, 'font_color': 'black', 'bg_color': 'white',
                                       'font_size': 14, 'font_name': 'Calibri', 'align': 'center',
                                        'valign': 'vcenter', 'border_color': 'black', 'border': 1})
    worksheet.write('B3', 'VFT', cell_format)
    worksheet.write('B4', 'Cadets', cell_format)
    worksheet.write('B5', 'AFSCs', cell_format)
    worksheet.write('B6', 'Z', cell_format)
    worksheet.write('C3', 'Value', cell_format)
    worksheet.write('D3', 'Weight', cell_format)

    # Basic format
    cell_format = workbook.add_format({'bold': False, 'font_color': 'black', 'bg_color': 'white',
                                        'font_size': 14, 'font_name': 'Calibri', 'border_color': 'black', 'border': 1})

    # Choice Counts
    choice_dict = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth", 6: "Sixth", 7: "Seventh",
                   8: "Eighth", 9: "Ninth", 10: "Tenth"}
    for choice in choice_dict:
        worksheet.write("J" + str(2 + choice), int(metrics['cadet_choice_counts'][choice]), cell_format)
        worksheet.write("I" + str(2 + choice), choice_dict[choice], cell_format)
        worksheet.write("K" + str(2 + choice), round(metrics['cadet_choice_counts'][choice] / p['N'], 3), cell_format)
    worksheet.write("I" + str(3 + choice), "All Others", cell_format)
    worksheet.write("J" + str(3 + choice), int(metrics['cadet_choice_counts']['All Others']), cell_format)
    worksheet.write("K" + str(3 + choice), round(metrics['cadet_choice_counts']["All Others"] / p['N'], 3), cell_format)

    # Additional metrics
    worksheet.write('F3', 'Blocking Pairs', cell_format)
    worksheet.write('G3', metrics['num_blocking_pairs'], cell_format)
    worksheet.write('F4', 'Ineligible Cadets', cell_format)
    worksheet.write('G4', metrics['num_ineligible'], cell_format)
    worksheet.write('F5', 'Unmatched Cadets', cell_format)
    worksheet.write('G5', metrics['num_unmatched'], cell_format)
    worksheet.write('F6', 'Average Cadet Choice', cell_format)
    worksheet.write('G6', metrics['average_cadet_choice'], cell_format)
    worksheet.write('F7', 'Average Normalized AFSC Score', cell_format)
    worksheet.write('G7', metrics['weighted_average_afsc_score'], cell_format)
    worksheet.write('F8', 'Failed Constraints', cell_format)
    worksheet.write('G8', metrics['total_failed_constraints'], cell_format)

    # VFT Metrics
    worksheet.write('C4', round(metrics['cadets_overall_value'], 4), cell_format)
    worksheet.write('C5', round(metrics['afscs_overall_value'], 4), cell_format)
    worksheet.write('D4', round(vp['cadets_overall_weight'], 4), cell_format)
    worksheet.write('D5', round(vp['afscs_overall_weight'], 4), cell_format)

    # Draw bigger borders
    draw_frame_border_outside(workbook, worksheet, 1, 1, 5, 3, color='black', width=2)
    draw_frame_border_outside(workbook, worksheet, 1, 5, 7, 2, color='black', width=2)
    draw_frame_border_outside(workbook, worksheet, 1, 8, 12, 3, color='black', width=2)

    # Adjust Column Widths
    column_widths = {0: 1.50, 4: 1.50, 5: 31, 7: 1.50, 8: 14, 10: 12}
    for c in column_widths:
        worksheet.set_column(c, c, column_widths[c])

    def export_results_dfs():
        """
        This nested function is here to export all other dataframes
        """

        # AFSC Objective measures dataframe
        df = pd.DataFrame({'AFSC': p['afscs'][:p['M']]})
        for k, objective in enumerate(vp['objectives']):
            df[objective] = np.around(metrics['objective_measure'][:, k], 2)

        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Objective Measures', index=False)

        # AFSC Constraint Fail dataframe
        df = pd.DataFrame({'AFSC': p['afscs'][:p['M']]})
        for k, objective in enumerate(vp['objectives']):
            df[objective] = metrics['objective_constraint_fail'][:, k]

        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Constraint Fails', index=False)

        # AFSC Objective values dataframe
        df = pd.DataFrame({'AFSC': p['afscs'][:p['M']]})
        for k, objective in enumerate(vp['objectives']):
            values = np.empty((p['M']))
            values[:] = np.nan
            np.put(values, vp['J^A'][k], np.around(metrics['objective_value'][vp['J^A'][k], k], 2))

            df[objective] = values

        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Objective Values', index=False)

        # Solution Dataframe
        df = pd.DataFrame({'Cadet': p['cadets']})
        df['USAFA'] = p['usafa']
        df['Merit'] = p['merit']
        df["Matched"] = metrics['afsc_solution']
        df['Cadet Choice'] = metrics['cadet_choice']
        df['AFSC Choice'] = metrics['afsc_choice']
        df['Cadet Utility'] = metrics['cadet_utility_achieved']
        df['AFSC Utility'] = metrics['afsc_utility_achieved']
        df['Global Utility'] = metrics['global_utility_achieved']

        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Solution', index=False)

        # Solution/X Matrix
        if instance.x is not None:
            df = pd.DataFrame({'Cadet': p['cadets']})
            df[instance.solution_name] = instance.metrics['afsc_solution']
            for j, afsc in enumerate(p['afscs'][:p['M']]):
                df[afsc] = instance.x[:, j]

            # Convert the dataframe to an XlsxWriter Excel object.
            df.to_excel(writer, sheet_name='X', index=False)

        # Save the workbook (writer object)
        writer.save()
    export_results_dfs()


def draw_frame_border_outside(workbook, worksheet, first_row, first_col, rows_count, cols_count,
                              color='#0000FF', width=2):

    # verify type of data passed in
    if first_row <= 0:
        first_row = 1
    if first_col <= 0:
        first_col = 1
    cols_count = abs(cols_count)
    rows_count = abs(rows_count)

    # top left corner
    worksheet.conditional_format(first_row - 1, first_col,
                                 first_row - 1, first_col,
                                 {'type': 'formula', 'criteria': 'True',
                                  'format': workbook.add_format({'bottom': width, 'border_color': color})})
    worksheet.conditional_format(first_row, first_col - 1,
                                 first_row, first_col - 1,
                                 {'type': 'formula', 'criteria': 'True',
                                  'format': workbook.add_format({'right': width, 'border_color': color})})
    # top right corner
    worksheet.conditional_format(first_row - 1, first_col + cols_count - 1,
                                 first_row - 1, first_col + cols_count - 1,
                                 {'type': 'formula', 'criteria': 'True',
                                  'format': workbook.add_format({'bottom': width, 'border_color': color})})
    worksheet.conditional_format(first_row, first_col + cols_count,
                                 first_row, first_col + cols_count,
                                 {'type': 'formula', 'criteria': 'True',
                                  'format': workbook.add_format({'left': width, 'border_color': color})})
    # bottom left corner
    worksheet.conditional_format(first_row + rows_count - 1, first_col - 1,
                                 first_row + rows_count - 1, first_col - 1,
                                 {'type': 'formula', 'criteria': 'True',
                                  'format': workbook.add_format({'right': width, 'border_color': color})})
    worksheet.conditional_format(first_row + rows_count, first_col,
                                 first_row + rows_count, first_col,
                                 {'type': 'formula', 'criteria': 'True',
                                  'format': workbook.add_format({'top': width, 'border_color': color})})
    # bottom right corner
    worksheet.conditional_format(first_row + rows_count - 1, first_col + cols_count,
                                 first_row + rows_count - 1, first_col + cols_count,
                                 {'type': 'formula', 'criteria': 'True',
                                  'format': workbook.add_format({'left': width, 'border_color': color})})
    worksheet.conditional_format(first_row + rows_count, first_col + cols_count - 1,
                                 first_row + rows_count, first_col + cols_count - 1,
                                 {'type': 'formula', 'criteria': 'True',
                                  'format': workbook.add_format({'top': width, 'border_color': color})})
    # top
    worksheet.conditional_format(first_row - 1, first_col + 1,
                                 first_row - 1, first_col + cols_count - 2,
                                 {'type': 'formula', 'criteria': 'True',
                                  'format': workbook.add_format({'bottom': width, 'border_color': color})})
    # left
    worksheet.conditional_format(first_row + 1, first_col - 1,
                                 first_row + rows_count - 2, first_col - 1,
                                 {'type': 'formula', 'criteria': 'True',
                                  'format': workbook.add_format({'right': width, 'border_color': color})})
    # bottom
    worksheet.conditional_format(first_row + rows_count, first_col + 1,
                                 first_row + rows_count, first_col + cols_count - 2,
                                 {'type': 'formula', 'criteria': 'True',
                                  'format': workbook.add_format({'top': width, 'border_color': color})})
    # right
    worksheet.conditional_format(first_row + 1, first_col + cols_count,
                                 first_row + rows_count - 2, first_col + cols_count,
                                 {'type': 'formula', 'criteria': 'True',
                                  'format': workbook.add_format({'left': width, 'border_color': color})})





