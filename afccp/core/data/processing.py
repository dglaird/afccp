import os
import numpy as np
import pandas as pd
import xlsxwriter
import copy
import string

# afccp modules
import afccp.core.globals
import afccp.core.data.values
import afccp.core.data.adjustments
import afccp.core.data.preferences

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
                                        "ROTC Rated OM", "USAFA Rated OM", "OTS Rated OM",
                                        "Bases", "Bases Preferences",
                                        "Bases Utility", "Courses", "Cadets Selected", "AFSCs Buckets",
                                        'Castle Input'],
                        "Analysis & Results": ["Solutions", "Base Solutions", "Course Solutions"]}

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
    afsc_columns_to_parameters = {"AFSC": "afscs", "Accessions Group": "acc_grp", "STEM": 'afscs_stem',
                                  "USAFA Target": "usafa_quota",
                                  "ROTC Target": "rotc_quota",
                                  "OTS Target": "ots_quota",
                                  "PGL Target": "pgl", "Estimated": "quota_e",
                                  "Desired": "quota_d", "Min": "quota_min", "Max": "quota_max",
                                  "Assign Base": 'afsc_assign_base', 'Num Courses': 'T'}

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
                                   "Ethnicity": "ethnicity", 'USAFA': 'usafa', 'SOC': 'soc',
                                   'ASC1': 'asc1', 'ASC2': 'asc2',
                                   'CIP1': 'cip1', 'CIP2': 'cip2', 'Merit': 'merit', 'Real Merit': 'merit_all',
                                   "Assigned": "assigned", "STEM": "stem", "Accessions Group": "acc_grp_constraint",
                                   "SF OM": "sf_om", 'Start Date': 'training_start', 'Start Pref': 'training_preferences',
                                   'Base Threshold': 'base_threshold', 'Course Threshold': 'training_threshold',
                                   'AFSC Weight': 'weight_afsc', 'Base Weight': 'weight_base',
                                   'Course Weight': 'weight_course', 'Least Desired AFSC': 'last_afsc',
                                   'Second Least Desired AFSCs': 'second_to_last_afscs'}

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

    # Determine which SOCs are in this instance
    if 'soc' in p:
        unique_socs = np.unique(p['soc'])  # Get unique list of SOCs

        # This just gets the SOCs in the right order
        soc_options = ['USAFA', 'ROTC', 'OTS']
        p['SOCs'] = np.array([soc.lower() for soc in soc_options if soc in unique_socs])

        for soc in unique_socs:
            if soc not in soc_options:
                raise ValueError(f'SOC {soc} not recognized as valid SOC option! At least one cadet has it.')

    # Return parameters dictionary
    return p


def import_afsc_cadet_matrices_data(import_filepaths, parameters):
    """
    Imports additional  data (if available) for cadets and AFSCs, and adds the relevant information to the
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
                    "ROTC Rated Interest", "ROTC Rated OM", "USAFA Rated OM", "OTS Rated OM", 'Cadets Utility (Final)',
                    "Cadets Selected", "AFSCs Buckets"]:

        # If we have the dataset, import it
        if dataset in import_filepaths:
            datasets[dataset] = afccp.core.globals.import_csv_data(import_filepaths[dataset])

    # First and last AFSC (for collecting matrices from dataframes)
    afsc_1, afsc_M = p["afscs"][0], p["afscs"][p["M"] - 1]

    # Load in extra dataframes
    for dataset, param in {'Cadets Selected': 'c_selected_matrix', 'AFSCs Buckets': 'a_bucket_matrix'}.items():
        if dataset in datasets:
            p[param] = np.array(datasets[dataset].loc[:, afsc_1: afsc_M])

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
    if "OTS Rated OM" in datasets:
        r_afscs = list(datasets['OTS Rated OM'].columns[1:])
        p['or_om_matrix'] = np.array(datasets['OTS Rated OM'].loc[:, r_afscs[0]:r_afscs[len(r_afscs) - 1]])
        p['or_om_cadets'] = np.array(datasets['OTS Rated OM']['Cadet'])

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
        if check_vp in vp_names and "Utility" not in file:  # Don't want VP Global Utility included here
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
    solutions = {}
    for solution_name in solution_names:

        # Convert solution of AFSC names to indices and then save it to the dictionary
        afsc_solution = np.array(solutions_df[solution_name])  # ["15A", "14N", "17X", ...]
        solution = {'j_array': np.array([np.where(p['afscs'] == afsc)[0][0] for afsc in afsc_solution]), # [3, 2, 5, ...]
                    'name': solution_name, 'afsc_array': afsc_solution}
        solutions[solution_name] = copy.deepcopy(solution)

    # If we have this extra component
    if 'Base Solutions' in import_filepaths:
        solutions_df = afccp.core.globals.import_csv_data(import_filepaths['Base Solutions'])

        # Get list of solution names
        solution_names = list(solutions_df.keys())[1:]

        # Loop through each solution in this dataframe and add the base assignments
        for solution_name in solution_names:
            base_solution = np.array(solutions_df[solution_name])
            solutions[solution_name]['base_array'] = base_solution
            solutions[solution_name]['b_array'] = np.array(
                [np.where(p['bases'] == base)[0][0] if base in base_solution else p['S'] for base in base_solution])

    # If we have this extra component
    if 'Course Solutions' in import_filepaths:
        solutions_df = afccp.core.globals.import_csv_data(import_filepaths['Course Solutions'])

        # Get list of solution names
        solution_names = list(solutions_df.keys())[1:]

        # Loop through each solution in this dataframe and add the course assignments
        for solution_name in solution_names:
            course_solution = np.array(solutions_df[solution_name])
            solutions[solution_name]['course_array'] = course_solution
            c_array = []
            for i, course in enumerate(course_solution):
                found = False
                for j in range(p['M']):
                    if course in p['courses'][j]:
                        c = np.where(p['courses'][j] == course)[0][0]
                        c_array.append((j, c))
                        found = True
                        break

                # This shouldn't happen!
                if not found:
                    print("Course '" + str(course) + "' not valid for cadet", i)
                    c_array.append((0, 0))
            solutions[solution_name]['c_array'] = np.array(c_array)

    # Return the dictionary of solutions
    return solutions


def import_additional_data(import_filepaths, parameters):
    """
    Imports additional csv files and updates the instance parameters dictionary with the values from the file.
        Extra datasets: "Bases.csv"
    Args:
        import_filepaths (dict): A dictionary of filepaths containing the location of the additional csv files.
        parameters (dict): A dictionary of instance parameters to update.

    Returns:
        dict: The updated instance parameters.
    """

    # Shorthand
    p = parameters

    # Loop through the potential additional dataframes and import them if we have them
    datasets = {}
    for dataset in ["Bases", "Bases Preferences", "Bases Utility", "Courses", "Castle Input"]:

        # If we have the dataset, import it
        if dataset in import_filepaths:
            datasets[dataset] = afccp.core.globals.import_csv_data(import_filepaths[dataset])

    # First and last AFSC (for collecting matrices from dataframes)
    afsc_1, afsc_M = p["afscs"][0], p["afscs"][p["M"] - 1]

    # Extract data from "Bases.csv" if applicable
    if "Bases" in datasets:
        p['bases'] = np.array(datasets["Bases"]["Base"])  # Set of bases (names)
        p['S'] = len(p['bases'])  # Number of bases (S for "Station")
        p['base_min'] = np.array(datasets['Bases'].loc[:, afsc_1 + " Min": afsc_M + " Min"])  # Minimum base # by AFSC
        p['base_max'] = np.array(datasets['Bases'].loc[:, afsc_1 + " Max": afsc_M + " Max"])  # Maximum base # by AFSC

    # Extract data from "Base Preferences.csv" and "Base Utility.csv" if applicable
    for parameter, dataset in {'b_pref_matrix': 'Bases Preferences', 'base_utility': 'Bases Utility'}.items():
        if dataset in datasets:
            base_1, base_S = p['bases'][0], p['bases'][p['S'] - 1]
            p[parameter] = np.array(datasets[dataset].loc[:, base_1: base_S])

    # Extract data from "Courses.csv" if applicable
    if "Courses" in datasets:

        # Need a dictionary of indices of courses that apply to each AFSC
        afscs = np.array(datasets["Courses"]["AFSC"])
        afsc_courses = {j: np.where(afscs == p['afscs'][j])[0] for j in range(p['M'])}

        # Dictionary to translate parameter names to column names
        column_translation = {"Course": 'courses', 'Start Date': 'course_start', 'Min': 'course_min',
                              'Max': 'course_max'}

        # Get each parameter from the columns of this dataset
        for col, param in column_translation.items():
            arr = np.array(datasets['Courses'][col])  # Convert dataframe column to numpy array
            p[param] = {j: arr[afsc_courses[j]] for j in range(p['M'])}

    # Extract data from "Castle Input.csv" if applicable
    if "Castle Input" in datasets:

        # Load in AFSC arrays
        castle_afscs = np.array(datasets['Castle Input']['CASTLE AFSC'])
        afpc_afscs = np.array(datasets['Castle Input']['AFPC AFSC'])
        p['castle_afscs_arr'], p['afpc_afscs_arr'] = castle_afscs, afpc_afscs

        # Create dictionary of CASTLE AFSCs -> AFPC AFSCs (account for groupings)
        p['castle_afscs'], p['J^CASTLE'] = {}, {}
        for castle_afsc in np.unique(castle_afscs):
            indices = np.where(castle_afscs == castle_afsc)[0]
            p['castle_afscs'][castle_afsc] = afpc_afscs[indices]
            p['J^CASTLE'][castle_afsc] = np.array([np.where(p['afscs'] == afsc)[0][0] for afsc in afpc_afscs[indices]])

        # Initialize OTS counts and optimal policy dictionary
        p['ots_counts'], p['optimal_policy'] = {}, {}

        # Load in "q" dictionary information if it exists
        df = datasets['Castle Input']  # Shorthand
        if 'a' in df.columns:
            q = {'a': {}, 'f^hat': {}, 'r': {}, 'L': {}}
            for afsc in np.unique(castle_afscs):
                row = df.loc[df['CASTLE AFSC'] == afsc].head(1).iloc[0]

                # Add OTS count information and optimal policy information for this AFSC
                p['ots_counts'][afsc] = row['OTS Count']
                p['optimal_policy'][afsc] = row['Optimal']

                # Load breakpoint coordinates into q dictionary
                a_str, f_hat_str = str(row['a']), str(row['f^hat'])
                q['a'][afsc] = np.array([float(x) for x in a_str.split(",")])
                q['f^hat'][afsc] = np.array([float(x) for x in f_hat_str.split(",")])

                # Save additional information to q dictionary
                q['r'][afsc], q['L'][afsc] = len(q['a'][afsc]), np.arange(len(q['a'][afsc]))
                p['castle_q'] = q  # Save to parameters dictionary

    # Return parameters dictionary
    return p


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
    afsc_parameters_to_columns = {"afscs": "AFSC", "acc_grp": "Accessions Group", "afscs_stem": "STEM",
                                  "usafa_quota": "USAFA Target", "rotc_quota": "ROTC Target", 'ots_quota': 'OTS Target',
                                  "pgl": "PGL Target", "quota_e": "Estimated",
                                  "quota_d": "Desired", "quota_min": "Min", "quota_max": "Max",
                                  "afsc_assign_base": 'Assign Base', 'T': 'Num Courses',
                                  'usafa_eligible_count': 'USAFA Eligible', 'rotc_eligible_count': 'ROTC Eligible',
                                  'ots_eligible_count': "OTS Eligible"}

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
    cadet_parameters_to_columns = {"cadets": "Cadet", "assigned": "Assigned", "acc_grp_constraint": "Accessions Group",
                                   'training_start': 'Start Date', 'training_preferences': 'Start Pref',
                                   'base_threshold': 'Base Threshold', 'training_threshold': 'Course Threshold',
                                   'weight_afsc': 'AFSC Weight', 'weight_base': 'Base Weight',
                                   'weight_course': 'Course Weight',
                                   "sf_om": "SF OM", 'usafa': 'USAFA', 'soc': 'SOC',
                                   'male': 'Male', 'minority': 'Minority',
                                   'race': 'Race', "ethnicity": "Ethnicity", 'asc1': 'ASC1', 'asc2': 'ASC2',
                                   'stem': 'STEM', 'cip1': 'CIP1', 'cip2': 'CIP2', 'merit': 'Merit',
                                   'merit_all': 'Real Merit', 'last_afsc': 'Least Desired AFSC',
                                   'second_to_last_afscs': 'Second Least Desired AFSCs'}

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


def export_afsc_cadet_matrices_data(instance):
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
                            'ur_om_matrix': 'USAFA Rated OM', 'or_om_matrix': 'OTS Rated OM',
                           'cadet_utility': 'Cadets Utility (Final)',
                            'c_selected_matrix': 'Cadets Selected', 'a_bucket_matrix': 'AFSCs Buckets'
                            }

    # Get all rated AFSCs
    all_rated_afscs = p['afscs'][p['J^Rated']]

    # Loop through each potential dataset to export
    for parameter in parameter_trans_dict:

        # If we have this dataset, we export it
        if parameter in p:
            dataset = parameter_trans_dict[parameter]

            # Construct the dataframe
            if 'ROTC' in dataset:
                cadet_indices = p["Rated Cadets"]['rotc']
                pref_df = pd.DataFrame({"Cadet": p['cadets'][cadet_indices]})
                afscs = afccp.core.data.preferences.determine_soc_rated_afscs(
                    soc='rotc', all_rated_afscs=all_rated_afscs)
            elif 'USAFA' in dataset:
                cadet_indices = p["Rated Cadets"]['usafa']
                pref_df = pd.DataFrame({"Cadet": p['cadets'][cadet_indices]})
                afscs = afccp.core.data.preferences.determine_soc_rated_afscs(
                    soc='usafa', all_rated_afscs=all_rated_afscs)
            elif 'OTS' in dataset:
                cadet_indices = p["Rated Cadets"]['ots']
                pref_df = pd.DataFrame({"Cadet": p['cadets'][cadet_indices]})
                afscs = afccp.core.data.preferences.determine_soc_rated_afscs(
                    soc='ots', all_rated_afscs=all_rated_afscs)
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
    Export value parameter datasets and related information to CSV files for analysis.

    This function exports various value parameter datasets to separate CSV files. It provides detailed information
    about value parameters and constraint data for further analysis.

    Args:
        instance (object): An object containing problem instance and value parameter data.

    Returns:
        None

    Details:
    - Value parameters (VP) define optimization settings and constraints. This function extracts and exports different
      aspects of the VP data.

    - Value Parameters CSVs:
        - For each set of value parameters, this function exports a CSV file containing detailed information about
          objective weights, objectives, breakpoint measures (a), breakpoint values (f^hat), and value functions.
        - These CSVs provide insights into the parameters used in the optimization process.

    - Global Utility CSVs (if applicable):
        - If global utility data is included in the value parameters, this function exports a CSV containing global
          utility values for each cadet and AFSC.
        - This additional dataset is useful for assessing the global utility aspect of the optimization solution.

    - Cadet Utility Constraints CSV:
        - This CSV file contains information about minimum value settings for cadets.
        - It helps understand the constraints and preferences applied to individual cadets.

    - Overall Value Parameters CSV:
        - A summary CSV is generated, providing an overview of overall value parameters for different VP sets.
        - It includes information on weights, minimum values, weight functions, and other key parameters.

    The exported CSV files are saved with informative names and extensions to distinguish between different versions
    of data. The function assumes that the 'instance' object contains the required data and file paths for export.
    ```
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
                              'AFSC Weight Function': vp['afsc_weight_function']}

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
    Export solutions data to a CSV file for analysis.

    This function exports the solutions generated by the optimization process to a CSV file. Each solution is
    represented as a column in the CSV, providing detailed information about the assignment of cadets to AFSCs
    for different scenarios or optimization runs.

    Args:
        instance (object): An object containing problem instance and solution data.

    Returns:
        None

    Details:
    - This function creates a solutions dataframe with each column representing a different solution. It includes
      data about the assignment of cadets to AFSCs for each scenario.

    - The exported CSV file is saved with an informative name, which typically corresponds to the specific scenario
      or optimization run being documented.

    - The function assumes that the 'instance' object contains the required data and file paths for export.
    ```
    """

    # Shorthand
    p = instance.parameters

    # Error data
    if instance.solutions is None:
        return None  # No solutions to export!

    # Initialize solutions dataframe
    solutions_df = pd.DataFrame({"Cadet": p["cadets"]})

    # Loop through each solution and add it to the dataframe
    for solution_name in instance.solutions:
        solutions_df[solution_name] = instance.solutions[solution_name]['afsc_array']

    # Export 'Solutions' dataframe
    solutions_df.to_csv(instance.export_paths["Solutions"], index=False)

    # Initialize solutions dataframe
    extra_dict = {"Base Solutions": "base_array", "Course Solutions": "course_array"}
    for df_name, key in extra_dict.items():

        # Get list of solutions that have this extra component
        solution_names_found = []
        for solution_name in instance.solutions:
            if key in instance.solutions[solution_name]:
                solution_names_found.append(solution_name)

        # If we have at least one solution with this base or course component, we create this dataframe
        if len(solution_names_found) > 0:
            solutions_df = pd.DataFrame({"Cadet": p["cadets"]})

            # Loop through each solution with this component and add it to the dataframe
            for solution_name in solution_names_found:
                solutions_df[solution_name] = instance.solutions[solution_name][key]

            # Export extra 'Solutions' dataframe
            solutions_df.to_csv(instance.export_paths[df_name], index=False)


def export_additional_data(instance):
    """
    This function takes an Instance object as an argument, which contains the additional data to be exported.
    It creates a dictionary of columns to be included in the additional csv files by translating the parameters
    to their corresponding column names in the file. The function then creates a dataframe using these columns
    and exports it as a csv file to the path specified in the export_paths attribute of the Instance object.

    Args:
        instance (Instance): An Instance object containing the additional data to export.

    Returns:
        None
    """

    # Shorthand
    p = instance.parameters

    # See if we can export the "Bases" csv file
    if "bases" in p:

        # Initialize dataframe
        df = pd.DataFrame({"Base": p['bases']})

        # Add "minimum" # to assign to each base by AFSC
        for j in p['J']:
            afsc = p['afscs'][j]
            df[afsc + ' Min'] = p['base_min'][:, j]

        # Add "maximum" # to assign to each base by AFSC
        for j in p['J']:
            afsc = p['afscs'][j]
            df[afsc + ' Max'] = p['base_max'][:, j]

        # Export the dataset
        df.to_csv(instance.export_paths["Bases"], index=False)

    # Export base preferences/utility if applicable
    for parameter, dataset in {'b_pref_matrix': 'Bases Preferences', 'base_utility': 'Bases Utility'}.items():
        if parameter in p:

            # Initialize dataframe
            df = pd.DataFrame({"Cadet": p['cadets']})

            # Add base columns
            for b, base in enumerate(p['bases']):
                df[base] = p[parameter][:, b]

            # Export the dataset
            df.to_csv(instance.export_paths[dataset], index=False)

    # Export training course data if applicable
    if 'courses' in p:

        # Dictionary to translate parameter names to column names
        column_translation = {"Course": 'courses', 'Start Date': 'course_start', 'Min': 'course_min', 'Max': 'course_max'}

        # Initialize dictionary of new columns
        new_cols = {'AFSC': [p['afscs'][j] for j in p['J'] for _ in range(p['T'][j])]}

        # Create each column for this dataset
        for col, param in column_translation.items():
            new_cols[col] = [p[param][j][c] for j in p['J'] for c in range(p['T'][j])]

        # Create dataframe
        df = pd.DataFrame(new_cols)

        # Export the dataframe
        df.to_csv(instance.export_paths['Courses'], index=False)

    # Export Castle AFSCs data
    if 'castle_afscs_arr' in p:

        # Create dataframe
        df = pd.DataFrame({'AFPC AFSC': p['afpc_afscs_arr'],
                           'CASTLE AFSC': p['castle_afscs_arr']})

        # Add in value curve data
        if 'castle_q' in p:
            df['a'] = [', '.join(np.around(p['castle_q']['a'][afsc], 3).astype(str)) for afsc in p['castle_afscs_arr']]
            df['f^hat'] = \
                [', '.join(np.around(p['castle_q']['f^hat'][afsc], 3).astype(str)) for afsc in p['castle_afscs_arr']]
            df['Optimal'] = [p['optimal_policy'][afsc] for afsc in p['castle_afscs_arr']]
            df['OTS Count'] = [p['ots_counts'][afsc] for afsc in p['castle_afscs_arr']]

        # Export the dataframe
        df.to_csv(instance.export_paths['Castle Input'], index=False)


# Solution Results excel file
def export_solution_results_excel(instance, filepath):
    """
    Export a solution and associated metrics to an Excel file for detailed analysis.

    This function exports a comprehensive set of solution metrics, objective values, and constraints, along with detailed
    solution information, to an Excel file. The exported file serves as a valuable resource for in-depth analysis of
    optimization results.

    Args:
        instance (object): An object containing problem instance and solution data.
        filepath (str): The path where the Excel file will be saved.

    Returns:
        None

    Details:
    - The exported Excel file contains multiple sheets, each providing specific information about the optimization
      solution.

    - "Main" Sheet:
        - Displays overall metrics such as the value of the objective function (z).
        - Lists key metrics related to cadet preferences and choices, including choice counts and proportions.
        - Provides additional metrics relevant to the optimization problem, constraints, and objectives.

    - "Objective Measures" Sheet:
        - Presents objective measures for each AFSC (Air Force Specialty Code) in the optimization problem.
        - Helps in evaluating how well the solution aligns with each objective.

    - "Constraint Fails" Sheet:
        - Displays the number of constraint violations for each AFSC based on various optimization objectives.
        - Identifies where constraints are not met.

    - "Objective Values" Sheet:
        - Lists the values achieved for each AFSC for all objectives.
        - Highlights the performance of AFSCs with conditional formatting.

    - "Solution" Sheet:
        - Provides detailed information on each cadet's assignment, including matched AFSCs and preferences.
        - Shows merit, cadet choice, AFSC choice, cadet utility, and AFSC utility.
        - Includes information about rated cadet assignments and AFSC rankings.

    - "X" Sheet (Optional):
        - Displays the assignment matrix (X matrix) for cadets and AFSCs.
        - Shows the assignment status for each combination.

    The function expects an 'instance' object to contain relevant data, including problem parameters, solution data,
    and value parameters. The Excel file generated by this function serves as a valuable tool for studying and assessing
    optimization results.

    Note:
    This function assumes that the 'instance' object conforms to the specific structure expected for your optimization
    problem. Ensure that the 'instance' is correctly configured before using this function for export.
    ```
    """

    # Shorthand
    p, vp, solution = instance.parameters, instance.value_parameters, instance.solution
    mdl_p = instance.mdl_p

    # Get list of excel columns in order ("A", "AB", etc.)
    alphabet = list(string.ascii_uppercase)
    excel_columns = copy.deepcopy(alphabet)
    for letter in alphabet:
        for letter_2 in alphabet:
            excel_columns.append(letter + letter_2)

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
    worksheet.merge_range("C6:D6", round(solution['z'], 4), obj_format)

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
        worksheet.write("J" + str(2 + choice), int(solution['cadet_choice_counts'][choice]), cell_format)
        worksheet.write("I" + str(2 + choice), choice_dict[choice], cell_format)
        worksheet.write("K" + str(2 + choice), round(solution['cadet_choice_counts'][choice] / p['N^Match'], 3),
                        cell_format)
    worksheet.write("I" + str(3 + choice), "All Others", cell_format)
    worksheet.write("J" + str(3 + choice), int(solution['cadet_choice_counts']['All Others']), cell_format)
    worksheet.write("K" + str(3 + choice), round(solution['cadet_choice_counts']["All Others"] / p['N^Match'], 3),
                    cell_format)

    # Additional solution metrics
    name_metric_dict = {'Blocking Pairs': 'num_blocking_pairs', 'Ineligible Cadets': 'num_ineligible',
                        'Unmatched Cadets': 'num_unmatched',
                        'Top 3 Choices (Proportion) for USSF': 'top_3_ussf_count',
                        'Top 3 Choices (Proportion) for USAF': 'top_3_usaf_count',
                        'Top 3 Choices (Proportion) for USAFA': 'top_3_usafa_count',
                        'Top 3 Choices (Proportion) for ROTC': 'top_3_rotc_count',
                        'Top 3 Choices (Proportion) for OTS': 'top_3_ots_count',
                        'Average Cadet Choice': 'average_cadet_choice',
                        'Average Normalized AFSC Score': 'weighted_average_afsc_score',
                        'Average NRL Normalized AFSC Score': 'weighted_average_nrl_afsc_score',
                        'Failed Constraints': 'total_failed_constraints', 'USSF OM': 'ussf_om',
                        'Global Utility': 'z^gu', 'Cadet Utility': 'cadet_utility_overall',
                        'z^CASTLE': 'z^CASTLE', 'z^CASTLE (Values)': 'z^CASTLE (Values)',
                        'AFSC Utility': 'afsc_utility_overall', 'USAFA Cadet Utility': 'usafa_cadet_utility',
                        'ROTC Cadet Utility': 'rotc_cadet_utility', 'OTS Cadet Utility': 'ots_cadet_utility',
                        'USSF Cadet Utility': 'ussf_cadet_utility',
                        'USAF Cadet Utility': 'usaf_cadet_utility', 'USSF AFSC Utility': 'ussf_afsc_utility',
                        'USAF AFSC Utility': 'usaf_afsc_utility',
                        'Average Normalized AFSC Score (USSF)': 'weighted_average_ussf_afsc_score',
                        'Average Normalized AFSC Score (USAF)': 'weighted_average_usaf_afsc_score',
                        'USAFA USSF Cadets / USAFA USSF PGL': 'ussf_usafa_pgl_target',
                        'ROTC USSF Cadets / ROTC USSF PGL': 'ussf_rotc_pgl_target',
                        'Cadets Successfully Constrained to Accessions Group / Total Fixed Accession Group Slots':
                            'constrained_acc_grp_target',
                        "Cadets Successfully Constrained to AFSC / Total Fixed AFSC Slots":
                            'cadets_fixed_correctly',
                        "Cadets Successfully Reserved to AFSC / Total Reserved AFSC Slots":
                            'cadets_reserved_correctly',
                        "Successful Alternate List Scenarios / Total Possible Alternate List Scenarios":
                            'alternate_list_metric'}
    for acc_grp in p['afscs_acc_grp']:
        name_metric_dict[acc_grp + " Racial Simpson Index"] = 'simpson_index_' + acc_grp

    # Add these metrics into excel
    for r, name in enumerate(list(name_metric_dict.keys())):
        if name_metric_dict[name] in solution:

            # Sometimes we can't write in a value for a solution metric that is incomplete
            try:
                worksheet.write('F' + str(3 + r), name, cell_format)
                worksheet.write('G' + str(3 + r), solution[name_metric_dict[name]], cell_format)
            except:
                pass

    # VFT Metrics
    worksheet.write('C4', round(solution['cadets_overall_value'], 4), cell_format)
    worksheet.write('C5', round(solution['afscs_overall_value'], 4), cell_format)
    worksheet.write('D4', round(vp['cadets_overall_weight'], 4), cell_format)
    worksheet.write('D5', round(vp['afscs_overall_weight'], 4), cell_format)

    # Draw bigger borders
    draw_frame_border_outside(workbook, worksheet, 1, 1, 5, 3, color='black', width=2)
    draw_frame_border_outside(workbook, worksheet, 1, 5, len(name_metric_dict.keys()) + 1, 2, color='black', width=2)
    draw_frame_border_outside(workbook, worksheet, 1, 8, 12, 3, color='black', width=2)

    # Adjust Column Widths
    column_widths = {0: 1.50, 4: 1.50, 5: 31, 7: 1.50, 8: 14, 10: 12}
    for c in column_widths:
        worksheet.set_column(c, c, column_widths[c])

    def export_results_dfs():
        """
        This nested function is here to export all other dataframes
        """

        # Get the xlsxwriter worksheet object.
        workbook = writer.book

        # AFSC Objective measures dataframe
        df = pd.DataFrame({'AFSC': p['afscs'][:p['M']]})
        for k, objective in enumerate(vp['objectives']):
            df[objective] = np.around(solution['objective_measure'][:, k], 2)

        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Objective Measures', index=False)

        # Add Castle Data if it exists
        if 'castle_q' in p:
            castle_afscs = [afsc for afsc, _ in p['J^CASTLE'].items()]
            castle_counts = [solution['castle_counts'][afsc] for afsc in castle_afscs]
            castle_values = [solution['castle_v'][afsc] for afsc in castle_afscs]
            df = pd.DataFrame({'AFSC': castle_afscs, 'Count': castle_counts, 'Value': castle_values})
            df.to_excel(writer, sheet_name='Castle Metrics', index=False)

        # AFSC Constraint Fail dataframe
        df = pd.DataFrame({'AFSC': p['afscs'][:p['M']]})
        for k, objective in enumerate(vp['objectives']):
            df[objective] = solution['objective_constraint_fail'][:, k]

        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Constraint Fails', index=False)

        # AFSC Objective values dataframe
        df = pd.DataFrame({'AFSC': p['afscs'][:p['M']]})
        for k, objective in enumerate(vp['objectives']):
            values = np.empty((p['M']))
            values[:] = np.nan
            np.put(values, vp['J^A'][k], np.around(solution['objective_value'][vp['J^A'][k], k], 2))

            df[objective] = values

        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Objective Values', index=False)

        # Get the xlsxwriter worksheet object.
        worksheet = writer.sheets["Objective Values"]

        # Add a percent number format
        percent_format = workbook.add_format({"num_format": "0.0%"})
        worksheet.set_column(1, vp['O'] + 1, None, percent_format)

        # Conditional formatting to highlight which objectives were met/not met
        worksheet.conditional_format("B2:" + excel_columns[vp['O']] + str(p['M'] + 1), {"type": "3_color_scale"})

        # Solution Dataframe
        df = pd.DataFrame({'Cadet': p['cadets']})
        df['USAFA'] = p['usafa']
        df['Merit'] = p['merit']
        df["Matched AFSC"] = solution['afsc_array']
        if "base_array" in solution:
            df["Matched Base"] = solution['base_array']
            df["Matched Course"] = solution['course_array']
            col_dict = {'Base Choice': 'base_choice', 'Cadet State': 'cadet_state_achieved',
                        'State Utility': 'state_utility_used', 'Cadet Value': 'cadet_value_achieved',
                        'AFSC Weight': 'afsc_weight_used', 'Base Weight': 'base_weight_used',
                        'Course Weight': 'course_weight_used', 'Base Utility': 'base_utility_achieved',
                        'Course Utility': 'course_utility_achieved'}
            for key, val in col_dict.items():
                if val in solution:
                    df[key] = solution[val]
        df['Cadet Choice'] = solution['cadet_choice']
        df['AFSC Choice'] = solution['afsc_choice']
        df['Cadet Utility'] = solution['cadet_utility_achieved']
        df['AFSC Utility'] = solution['afsc_utility_achieved']
        df['Global Utility'] = solution['global_utility_achieved']
        df['Matched Deg Tier'] = [  # "U" for unmatched cadets
            p['qual'][i, solution['j_array'][i]] if solution['j_array'][i] in p['J'] else 'U' for i in p['I']]

        # Rated Reserves/Matches/Alternates
        for s_name in ['Rated Matches', 'Rated Reserves', 'Rated Alternates (Hard)', 'Rated Alternates (Soft)']:
            if s_name in instance.solutions:
                df[s_name] = instance.solutions[s_name]['afsc_array']
            else:
                df[s_name] = ["*" for _ in p['I']]

        # Rated Rankings
        for j in p['J^Rated']:
            df[p['afscs'][j] + " Rank"] = p['a_pref_matrix'][:, j]

        # Add the cadet's top 10 choices for more information!
        for choice in range(min(p['P'], 10)):
            df['Choice ' + str(choice + 1)] = p['c_preferences'][:, choice]

        # Capture preference columns
        preference_columns = excel_columns[len(df.columns) - min(p['P'], 10): len(df.columns)]

        # Add the cadet's top 10 utilities for more information!
        for choice in range(min(p['P'], 10)):
            df['Utility ' + str(choice + 1)] = np.zeros(p['N'])
            for i in p['I']:

                # Might run out of utilities
                if len(p['cadet_preferences'][i]) > choice:
                    j = p['cadet_preferences'][i][choice]
                    df['Utility ' + str(choice + 1)][i] = p['cadet_utility'][i, j]

        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Solution', index=False, startrow=1, header=False)

        # Get the xlsxwriter worksheet object.
        worksheet = writer.sheets["Solution"]

        # Add additional formatting to top row
        header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'align': 'center', 'valign': 'vcenter'})
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        # Get numpy array of df columns
        df_columns = np.array(df.columns)

        # Conditional Formatting on preferences
        cadet_choice_col = excel_columns[np.where(df_columns == 'Cadet Choice')[0][0]]
        for col, c in enumerate(preference_columns):
            format1 = workbook.add_format({'bg_color': mdl_p['choice_colors'][col + 1]})
            worksheet.conditional_format(c + "2:" + c + str(p['N'] + 1),
                                         {'type': 'cell', 'value': 'D2',
                                          'criteria': '=',
                                          'format': format1
                                          })
            # Cadet Choice Column
            worksheet.conditional_format(cadet_choice_col + "2:" + cadet_choice_col + str(p['N'] + 1),
                                         {'type': 'cell', 'value': col + 1,
                                          'criteria': '=',
                                          'format': format1
                                          })

        # Freeze top row
        worksheet.freeze_panes(1, 0)

        # Add filter to headers
        worksheet.autofilter('A1:' + str(excel_columns[len(df.columns) - 1]) + str(p['N'] + 1))

        # Replace unassigned base values
        if 'Base Choice' in df:
            cell_format = workbook.add_format({})
            for i in p['I']:
                if df.loc[i, 'Base Choice'] == 0:
                    letters = [excel_columns[np.where(df_columns == col)[0][0]] for col in
                               ['Base Weight', 'Base Choice', 'Base Utility']]
                    for letter in letters:
                        worksheet.write(letter + str(i + 2), '', cell_format)

        # Small values good (1, 2, 3, 4, ...) Conditional Formatting
        sv_excel_cols = []
        for col in ['AFSC Choice', 'Base Choice', 'Cadet State']:
            if col in df:
                sv_excel_cols.append(excel_columns[np.where(df_columns == col)[0][0]])
        for c in sv_excel_cols:
            worksheet.conditional_format(
                c + "2:" + c + str(p['N'] + 1), {'type': '3_color_scale', 'min_color': '#63be7b',
                                                 'mid_color': '#ffeb84', 'max_color': '#f8696b'})

        # Large values good (1, ...,  0) Conditional Formatting
        lv_excel_cols = []
        for col in ['Merit', 'Cadet Utility', 'AFSC Utility', 'Global Utility', 'Base Utility', 'Course Utility',
                    'State Utility', 'Cadet Value', 'Cadet Value (Pyomo)']:
            if col in df:
                lv_excel_cols.append(excel_columns[np.where(df_columns == col)[0][0]])
        for c in lv_excel_cols:
            worksheet.conditional_format(c + "2:" + c + str(p['N'] + 1), {"type": "3_color_scale"})

        # "All Others" for the choice column
        format1 = workbook.add_format({'bg_color': mdl_p['all_other_choice_colors']})
        worksheet.conditional_format(cadet_choice_col + "2:" + cadet_choice_col + str(p['N'] + 1),
                                     {'type': 'cell', 'value': 10, 'criteria': '>', 'format': format1})

        # Add a percent number format to certain columns
        percent_cols = ['Merit', 'State Utility', 'Cadet Value', 'AFSC Weight', 'Base Weight', 'Course Weight',
                        'Base Utility', 'Course Utility', 'Cadet Utility', 'AFSC Utility', 'Global Utility']
        percent_format = workbook.add_format({"num_format": "0.0%"})
        for col in percent_cols:
            if col in df_columns:
                c_num = np.where(df_columns == col)[0][0]
                worksheet.set_column(c_num, c_num, None, percent_format)

        # Solution/X Matrix
        if 'x' in solution:
            df = pd.DataFrame({'Cadet': p['cadets']})
            df[instance.solution_name] = instance.solution['afsc_array']
            for j, afsc in enumerate(p['afscs'][:p['M']]):
                df[afsc] = instance.solution['x'][:, j]

            # Convert the dataframe to an XlsxWriter Excel object.
            df.to_excel(writer, sheet_name='X', index=False)

        # Base Matrix
        if 'v' in solution:
            df = pd.DataFrame({'Cadet': p['cadets']})
            df[instance.solution_name] = instance.solution['base_array']
            for b, base in enumerate(p['bases'][:p['S']]):
                df[base] = instance.solution['v'][:, b]

            # Convert the dataframe to an XlsxWriter Excel object.
            df.to_excel(writer, sheet_name='V', index=False)

        # Training Matrix
        if 'q' in solution:

            df = pd.DataFrame({'Cadet': p['cadets']})
            df[instance.solution_name] = instance.solution['course_array']
            for j, afsc in enumerate(p['afscs'][:p['M']]):
                for c, course in enumerate(p['courses'][j]):
                    df[afsc + "-" + course] = instance.solution['q'][:, j, c]

            # Convert the dataframe to an XlsxWriter Excel object.
            df.to_excel(writer, sheet_name='Q', index=False)

        # Value Function Matrices
        if 'lambda' in solution:

            afsc_arr = [afsc for afsc in p['afscs'][:p['M']] for _ in vp['K']]
            objective_arr = [objective for _ in p['J'] for objective in vp['objectives']]

            df = pd.DataFrame({'AFSC': afsc_arr, 'Objective': objective_arr})
            for l in range(solution['r^max']):
                arr = [solution['lambda'][j, k, l] for j in p['J'] for k in vp['K']]
                df['l_' + str(l + 1)] =  arr

            # Convert the dataframe to an XlsxWriter Excel object.
            df.to_excel(writer, sheet_name='Lambda', index=False)

            df = pd.DataFrame({'AFSC': afsc_arr, 'Objective': objective_arr})
            for l in range(solution['r^max']):
                arr = [solution['y'][j, k, l] for j in p['J'] for k in vp['K']]
                df['y_' + str(l + 1)] = arr

            # Convert the dataframe to an XlsxWriter Excel object.
            df.to_excel(writer, sheet_name='Y', index=False)

        # Blocking Pairs
        if 'blocking_pairs' in solution:

            # Create "blocking pairs" dataframe and export it to excel
            blocking_cadets = [tuple[0] for tuple in solution['blocking_pairs']]
            blocking_afscs = [p['afscs'][tuple[1]] for tuple in solution['blocking_pairs']]
            df = pd.DataFrame({'Blocking Cadet': blocking_cadets, 'Blocking AFSC': blocking_afscs})
            df.to_excel(writer, sheet_name='Blocking Pairs', index=False)

        # Close the workbook
        writer.close()
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





