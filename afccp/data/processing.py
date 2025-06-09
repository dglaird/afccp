"""
Data Processing Module
===============

This module handles all input/output operations for the AFCCP modeling pipeline.

It supports importing problem instance data (AFSCs, cadets, preferences, value functions, etc.)
and exporting solutions and diagnostics to CSV and Excel formats for analysis. It also
initializes the directory structure for versioned data instances.

Key Capabilities
----------------
- Organizes input/output folders and paths for a data instance
- Imports cleaned data required for AFCCP optimization
- Exports results and supporting data for evaluation and visualization

Primary Functions
-----------------

- **Initialization**

    - `initialize_file_information`: Sets up import/export folder paths for a given data name and version

- **Import Functions**

    - `import_afscs_data`: Loads AFSCs and related structural data
    - `import_cadets_data`: Loads cadet records and attributes
    - `import_afsc_cadet_matrices_data`: Loads cadet preference matrices and qualification matrices
    - `import_value_parameters_data`: Loads objective weights and breakpoints for value functions
    - `import_solutions_data`: Imports previously solved cadet-to-AFSC assignments
    - `import_additional_data`: Loads auxiliary data like base assignments and course info

- **Export Functions**

    - `export_afscs_data`: Saves AFSC-related data to CSV
    - `export_cadets_data`: Saves cadet-related data to CSV
    - `export_afsc_cadet_matrices_data`: Saves preference and qualification matrices to CSV
    - `export_value_parameters_data`: Saves value function breakpoints and weights to CSV
    - `export_solutions_data`: Saves one or more solutions in compact CSV format
    - `export_additional_data`: Saves supporting data (base preferences, utility matrices, courses)
    - `export_solution_results_excel`: Writes detailed solution metrics and diagnostics to an Excel workbook

Notes
-----
All functions assume access to an `Instance` object that contains parameters, value structures,
solution data, and export path information. The module is used during both the data preparation
and results analysis stages of the AFCCP workflow.
"""
import os
import numpy as np
import pandas as pd
import copy
import string

# afccp modules
import afccp.globals
import afccp.data.values
import afccp.data.adjustments
import afccp.data.preferences


# ______________________________________________INITIALIZATION__________________________________________________________
def initialize_file_information(data_name: str, data_version: str):
    """
    Initialize filepaths for an AFCCP data instance.

    This function constructs and returns import/export file path dictionaries for a given AFCCP
    data instance identified by `data_name` and `data_version`. It ensures the required directory
    structure exists under `instances/` and dynamically builds file paths for reading and writing
    instance-specific data and results.

    It is primarily used to manage file I/O consistently across different data experiments or
    scenario versions within the AFCCP modeling system.

    Parameters
    ----------
    - data_name : str
      The name of the data instance (e.g., `"2025"`, `"Baseline"`, `"TestRun01"`). This defines the
      subdirectory under `instances/` where the data is stored.

    - data_version : str
      The version label for the run (e.g., `"Default"`, `"V1"`). Used to separate multiple experimental
      runs under the same data instance name, enabling controlled versioning of model input and output files.

    Returns
    -------
    - Tuple[Dict[str, str], Dict[str, str]]
      A tuple of two dictionaries:
        - `import_paths`: maps each input data type (e.g., `"Cadets"`, `"AFSCs"`) to its CSV file path.
        - `export_paths`: maps each output destination (e.g., `"Solutions"`, `"Results Charts"`) to its folder or file path.

    Directory Behavior
    ------------------
    - Creates base folder `instances/<data_name>/` if it doesn't exist.
    - Creates version-specific folders for `"Model Input"` and `"Analysis & Results"`:
        - e.g., `"4. Model Input (V1)"`, `"5. Analysis & Results (V1)"`
    - Also creates subfolders under `"Analysis & Results"` such as:
        - `"Data Charts"`, `"Results Charts"`, `"Cadet Board"`, `"Value Functions"`
    - If version-specific input files are not found, it defaults to shared or base files when appropriate.

    Examples
    --------
    ```python
    from afccp.data.processing import initialize_file_information

    import_paths, export_paths = initialize_file_information("2025", "V1")
    afsc_path = import_paths["AFSCs"]
    solution_folder = export_paths["Analysis & Results"]
    ```
    """

    # If we don't already have the instance folder, we make it now
    instance_path = "instances/" + data_name + "/"
    if data_name not in afccp.globals.instances_available:
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


# ___________________________________________IMPORT DATA FUNCTIONS______________________________________________________
def import_afscs_data(import_filepaths: dict, parameters: dict) -> dict:
    """
    Imports AFSC-level model input data from CSV and populates the instance parameter dictionary.

    This function reads the `"AFSCs"` input file (provided via `import_filepaths`) and updates the supplied
    `parameters` dictionary with structured information for each Air Force Specialty Code (AFSC). These inputs
    are essential for AFCCP modeling and include AFSC quotas, groupings, tiered degree requirements, and other
    structural attributes.

    The function handles type conversion, fills missing entries, and appends a special unmatched AFSC ("*") for
    use in optimization logic.

    Parameters
    ----------
    - import_filepaths : dict
      A dictionary of import paths keyed by label (e.g., `"AFSCs"`). Must contain the key `"AFSCs"` pointing to
      the location of the AFSCs input CSV file.

    - parameters : dict
      A dictionary of instance-wide input parameters. This will be updated in-place with the AFSC-specific
      parameter values.

    Returns
    -------
    - dict
    The updated `parameters` dictionary, now containing keys such as:

    - `"afscs"`: List of AFSC names (plus an unmatched AFSC "*")
    - `"acc_grp"`: Accession group categories
    - `"afscs_stem"`: STEM-designation indicator for each AFSC
    - `"quota_d"`, `"quota_e"`, `"quota_min"`, `"quota_max"`: Target and constraint bounds
    - `"pgl"`: Projected graduation levels for each AFSC
    - `"Deg Tiers"`: Tiered degree qualification matrix (if present)

    Notes
    -----
    - All values are loaded as NumPy arrays to facilitate vectorized modeling.
    - The unmatched AFSC `"*"` is appended to `"afscs"` for modeling unmatched cadets.
    - NaN entries and string "nan" values in the CSV are sanitized to empty strings before processing.
    - Degree tiers are only added if the `"Deg Tier 1"` column is present in the CSV.
    ```

    See Also
    --------
    - [`initialize_file_information`](../../../afccp/reference/data/processing/#data.processing.initialize_file_information)
    - [`import_csv_data`](../../../afccp/reference/globals/#globals.import_csv_data)
    """

    # Shorthand
    p = parameters

    # Import 'AFSCs' dataframe
    afscs_df = afccp.globals.import_csv_data(import_filepaths["AFSCs"])

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
                                  'Max (Bubbles)': 'max_bubbles',
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
    Imports Cadet-level model input data from CSV and populates the instance parameter dictionary.

    This function reads the `"Cadets"` CSV file specified in `import_filepaths` and extracts relevant demographic,
    qualification, preference, training, and weighting information for each cadet. It populates the provided
    `parameters` dictionary with this structured data, including derived quantities like total cadet count,
    preference matrix dimensions, and accession source types (SOCs).

    Parameters
    ----------
    - import_filepaths : dict
      Dictionary of import paths keyed by label. Must contain the key `"Cadets"` pointing to the cadet input CSV path.

    - parameters : dict
      Dictionary of instance-level parameters. This dictionary will be updated in-place with cadet-related entries.

    Returns
    -------
    dict
    The updated `parameters` dictionary, now containing cadet-specific fields such as:

    - `"cadets"`, `"merit"`, `"assigned"`, `"asc1"`/`"asc2"`, `"cip1"`/`"cip2"` (basic identifiers)
    - `"usafa"`, `"soc"`, `"minority"`, `"race"`, `"ethnicity"` (demographic data)
    - `"must_match"` (AFSCs that must be assigned)
    - `"c_preferences"`: Preference matrix (N x P) where N = cadets, P = preference slots
    - `"c_utilities"`: Utility matrix (N x U), where U = min(P, 10)
    - `"SOCs"`: List of accession sources present in this instance (e.g., `["usafa", "rotc"]`)
    - `"training_start"`, `"training_preferences"`, `"training_threshold"` (training pipeline values)
    - `"weight_afsc"`, `"weight_base"`, `"weight_course"` (objective weights)

    Notes
    -----
    - NaN or string 'nan' entries in the CSV are automatically sanitized.
    - Extra care is taken to remove BOM characters (e.g., `"ï»¿"`) in CSV headers.
    - Preferences are detected using any column starting with `"Pref_"`, and corresponding utilities from `"Util_1"` onward.
    - SOCs must be one of `"USAFA"`, `"ROTC"`, or `"OTS"`; any other value will raise an error.
    - This function calls `gather_degree_tier_qual_matrix()` to supplement qualification mappings.

    See Also
    --------
    - [`initialize_file_information`](../../../afccp/reference/data/processing/#data.processing.initialize_file_information)
    - [`gather_degree_tier_qual_matrix`](../../../afccp/reference/data/adjustments/#data.adjustments.gather_degree_tier_qual_matrix)
    """

    # Shorthand
    p = parameters

    # Import 'Cadets' dataframe
    cadets_df = afccp.globals.import_csv_data(import_filepaths["Cadets"])

    # Initialize dictionary translating 'AFSCs' df columns to their parameter counterparts
    cadet_columns_to_parameters = {"Cadet": "cadets", 'Male': 'male', 'Minority': 'minority', 'Race': 'race',
                                   "Ethnicity": "ethnicity", 'USAFA': 'usafa', 'SOC': 'soc',
                                   "Must Match": "must_match",
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
    p = afccp.data.adjustments.gather_degree_tier_qual_matrix(cadets_df, p)

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
    Imports optional AFSC-cadet interaction matrices and updates the instance parameter dictionary accordingly.

    This function augments the core cadet and AFSC input data by importing preference matrices, utility matrices,
    and supplemental rated-selection files (if available). The imported data enables more advanced modeling of
    cadet-AFSC interactions including two-sided preferences and selection boards.

    Parameters
    ----------
    - import_filepaths : dict
        Dictionary of filepaths keyed by dataset name. Recognized keys include:

        - `"Cadets Utility"`
        - `"Cadets Preferences"`
        - `"AFSCs Utility"`
        - `"AFSCs Preferences"`
        - `"Cadets Utility (Final)"`
        - `"Cadets Selected"`
        - `"AFSCs Buckets"`
        - `"ROTC Rated Interest"`, `"ROTC Rated OM"`, `"USAFA Rated OM"`, `"OTS Rated OM"`

    - parameters : dict
        Dictionary of model instance parameters. Must contain:

        - `"afscs"`: array of AFSC names
        - `"N"`: number of cadets
        - `"M"`: number of AFSCs
        - `"num_util"`: number of utility entries per cadet
        - `"P"`: number of preferences

    Returns
    -------
    dict
    Updated parameter dictionary with new keys (if data was provided), including:

    - `"utility"`: cadet utility matrix (N x M)
    - `"c_pref_matrix"`: cadet preference rankings (N x M, integer-valued, 1 = top choice)
    - `"afsc_utility"`: AFSC utility matrix (M x M)
    - `"a_pref_matrix"`: AFSC preference rankings (M x M)
    - `"cadet_utility"`: final cadet utility matrix (N x M)
    - `"c_selected_matrix"`: matrix of cadets selected by AFSCs (N x M)
    - `"a_bucket_matrix"`: bucketing of AFSCs for visualization or selection (M x M)
    - `"rr_interest_matrix"`, `"rr_om_matrix"`, `"rr_om_cadets"`: ROTC board data
    - `"ur_om_matrix"`, `"ur_om_cadets"`: USAFA board data
    - `"or_om_matrix"`, `"or_om_cadets"`: OTS board data
    - `"usafa_cadets"`: indices of USAFA cadets in the instance

    Raises
    ------
    ValueError
      If neither `"Cadets Utility"` nor `"c_utilities"` are provided in the inputs, since cadet utility data is required.

    Notes
    -----
    - This function assumes the `"Cadets"` and `"AFSCs"` CSVs have already been processed.
    - If raw preferences/utilities are not explicitly imported, they are reconstructed from `c_preferences` and `c_utilities`.
    - Preference ranks use integers where `1` is most preferred (not `0`).
    - Utility matrices are aligned by the order of `p["afscs"]` and not by the file column order alone.
    - AFSC utility and preference data are optional but support two-sided matching or board processes.
    - Rated OM/Interest files enable specialty-specific board logic for each SOC.
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
            datasets[dataset] = afccp.globals.import_csv_data(import_filepaths[dataset])

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
    Imports and constructs value parameter sets for the model based on CSV definitions and analyst-defined breakpoints.

    This function reads and compiles all information associated with value-based decision modeling, including
    objective functions, breakpoints, weights, and value constraints for both cadets and AFSCs. It supports multiple
    sets of value parameters, each potentially with different assumptions or constraints.

    Parameters
    ----------
    - import_filepaths : dict
      Dictionary containing paths to relevant files. Required keys:

        - `"Model Input"`: folder path containing individual VP set CSVs.
        - `"Value Parameters"`: CSV listing metadata about each VP set.
        Optional keys:
        - `"Cadets Utility Constraints"`: file containing minimum cadet utility constraints by VP set.

    - parameters : dict
      Instance parameter dictionary already populated with `"M"`, `"N"`, and AFSC/cadet-level arrays.

    - num_breakpoints : int, optional (default=24)
      Number of breakpoints to discretize each value function unless exact breakpoints are provided.

    Returns
    -------
    dict
    A dictionary keyed by VP set names. Each value is a dictionary of value parameters containing:

    - Objective definitions and weights
    - Breakpoints (`a`) and values (`f^hat`)
    - Minimum utility constraints
    - AFSC objective indexed sets (`K^A`)
    - `cadet_weight`, `afsc_weight`, and associated metadata
    - VP set weights and local weights for combination logic

    Raises
    ------
    - FileNotFoundError
      If required files are missing from the provided import paths.

    - ValueError
      If the `"Value Parameters"` file is missing or empty.

    Notes
    -----
    - All AFSC objectives are indexed across `O` objectives per AFSC.
    - The function automatically parses objective weight strings and reconstructs value functions if needed.
    - For each VP set listed in the `"Value Parameters"` file, the function expects a matching CSV file named
      `"VP <VP Name>.csv"` (e.g., `"VP Baseline.csv"`).
    - Supports optional use of `"Cadets Utility Constraints"` to impose per-cadet minimums.
    - If `num_breakpoints` is `None`, raw a/f^hat arrays are expected instead of constructing from strings.
    - Value functions are compressed after loading to remove redundant zero segments for performance.

    See Also
    --------
    - [`value_function_builder`](../../../afccp/reference/data/values/#data.values.value_function_builder)
    - [`create_segment_dict_from_string`](../../../afccp/reference/data/values/#data.values.create_segment_dict_from_string)
    - [`condense_value_functions`](../../../afccp/reference/data/values/#data.values.condense_value_functions)
    - [`value_parameters_sets_additions`](../../../afccp/reference/data/values/#data.values.value_parameters_sets_additions)
    """

    # Shorthand
    p = parameters
    afccp_vp = afccp.data.values  # Reduce the module name so it fits on one line

    # Import the cadets utility constraints dataframe if we have it.
    if "Cadets Utility Constraints" in import_filepaths:
        vp_cadet_df = afccp.globals.import_csv_data(import_filepaths["Cadets Utility Constraints"])
    else:
        vp_cadet_df = None

    # Import the "Value Parameters" dataframe if we have it. If we don't, the "vp_dict" will be "None"
    if "Value Parameters" in import_filepaths:
        overall_vp_df = afccp.globals.import_csv_data(import_filepaths["Value Parameters"])
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
        vp_df = afccp.globals.import_csv_data(filepath)
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
    Imports and assembles cadet assignment solutions from a saved output file (`<data_name> Solutions.csv`). Optionally,
    more files are read in for Base/Training Course solutions.

    This function reads solution files containing AFSC, base, and course assignments for each cadet,
    converts string labels to indexed arrays, and returns a structured dictionary containing all
    available solution configurations.

    Parameters
    ----------
    - import_filepaths : dict
    Dictionary containing filepaths to solution files. Expected keys:

        - `"Solutions"`: required CSV file with AFSC assignments (one column per solution).
        - `"Base Solutions"`: optional CSV with base assignments (same column names as above).
        - `"Course Solutions"`: optional CSV with course assignments (same column names as above).

    - parameters : dict
    Dictionary of instance parameters. Must contain:

        - `'afscs'`: array of valid AFSC names.
        - `'bases'`: array of valid base names (if `"Base Solutions"` is provided).
        - `'courses'`: list of valid course arrays for each AFSC (if `"Course Solutions"` is provided).
        - `'S'`: sentinel index value for unmatched bases.

    Returns
    -------
    dict
    Dictionary mapping solution names to their data. Each solution entry contains:

    - `'name'`: name of the solution (from CSV column header)
    - `'afsc_array'`: array of assigned AFSC strings
    - `'j_array'`: array of assigned AFSC indices (matching `parameters['afscs']`)
    - `'base_array'` (optional): array of base names (if base data is present)
    - `'b_array'` (optional): array of base indices (or sentinel `S` if unmatched)
    - `'course_array'` (optional): array of course names (if course data is present)
    - `'c_array'` (optional): array of `(j, c)` tuples representing AFSC/course index pairs

    Raises
    ------
    - FileNotFoundError
      If the required `"Solutions"` file is not present in `import_filepaths`.

    Notes
    -----
    - If a course assignment is not found within any AFSC’s course list, a fallback value of `(0, 0)` is added
      and a warning is printed.
    - Assumes that all solution files share the same cadet ordering and column headers for consistent mapping.

    See Also
    --------
    - [`import_csv_data`](../../../afccp/reference/globals/#globals.import_csv_data)
    - [`initialize_file_information`](../../../afccp/reference/data/processing/#data.processing.initialize_file_information)
    """

    # Shorthand
    p = parameters

    # Import the "Solutions" dataframe if we have it. If we don't, the "solutions_dict" will be "None"
    if "Solutions" in import_filepaths:
        solutions_df = afccp.globals.import_csv_data(import_filepaths["Solutions"])
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
        solutions_df = afccp.globals.import_csv_data(import_filepaths['Base Solutions'])

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
        solutions_df = afccp.globals.import_csv_data(import_filepaths['Course Solutions'])

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
    Imports supplemental data files (if present) and updates the instance parameters dictionary.

    This function loads optional model extensions including base and course assignments, preference matrices,
    and CASTLE-specific AFSC data. These components are not required for basic operation but enhance downstream
    modeling functionality (e.g., course scheduling, base optimization, CASTLE implementation).

    Parameters
    ----------
    - import_filepaths : dict
      Dictionary containing filepaths to additional optional data files. Expected keys include:

        - "Bases", "Bases Preferences", "Bases Utility"
        - "Courses", "Castle Input"

    - parameters : dict
      Dictionary of core model parameters. This dictionary will be updated with any new fields derived from imported files.

    Returns
    -------
    dict
    Updated parameter dictionary with the following optional fields added if available:

    - `'bases'`: Array of base names
    - `'S'`: Number of bases
    - `'base_min'` / `'base_max'`: Base assignment bounds by AFSC
    - `'b_pref_matrix'`: Cadet base preference matrix
    - `'base_utility'`: Cadet base utility matrix
    - `'courses'`: Dict of course options by AFSC
    - `'course_start'`, `'course_min'`, `'course_max'`: Dicts with course metadata by AFSC
    - `'castle_afscs_arr'`, `'afpc_afscs_arr'`: Raw CASTLE vs AFPC AFSC labels
    - `'castle_afscs'`: Mapping of CASTLE AFSCs → AFPC AFSCs
    - `'J^CASTLE'`: CASTLE AFSCs mapped to indices in the AFPC AFSC array
    - `'ots_counts'`: OTS accession counts for CASTLE AFSCs
    - `'optimal_policy'`: Policy toggle per CASTLE AFSC
    - `'castle_q'`: Dictionary of breakpoint-based value functions:

        - `'a'`, `'f^hat'`: Breakpoints and values
        - `'r'`: Number of breakpoints
        - `'L'`: Breakpoint indices

    Notes
    -----
    - Breakpoint information from `"Castle Input"` is stored under `castle_q`.
    - Course and base matrices are assumed to be properly aligned with the cadet and AFSC indices already in memory.
    - All newly imported data is optional and loaded only if the corresponding files are provided.

    See Also
    --------
    - [`initialize_file_information`](../../../afccp/reference/data/processing/#data.processing.initialize_file_information)
    - [`import_csv_data`](../../../afccp/reference/globals/#globals.import_csv_data)
    """

    # Shorthand
    p = parameters

    # Loop through the potential additional dataframes and import them if we have them
    datasets = {}
    for dataset in ["Bases", "Bases Preferences", "Bases Utility", "Courses", "Castle Input"]:

        # If we have the dataset, import it
        if dataset in import_filepaths:
            datasets[dataset] = afccp.globals.import_csv_data(import_filepaths[dataset])

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


# ___________________________________________EXPORT DATA FUNCTIONS______________________________________________________
def export_afscs_data(instance):
    """
    Exports AFSC-level data from the given `Instance` object to a CSV file.

    This function collects Air Force Specialty Code (AFSC) parameters stored in the instance,
    organizes them into a structured dataframe, and writes the result to disk at the location
    specified by `instance.export_paths["AFSCs"]`.

    Parameters
    ----------
    - instance : Instance
      A fully initialized `Instance` object with a populated `parameters` dictionary and `export_paths` mapping.
      The instance must include AFSC data such as quotas, eligibility, preference counts, and any
      derived degree tier breakdowns.

    Returns
    -------
    - None
      The function writes the output to disk and does not return a value.

    Notes
    -----
    - The function dynamically detects and exports the following fields if present:

        - Core AFSC descriptors: name, accession group, STEM tag, base assignment
        - Quota targets: Desired, Estimated, Min, Max, PGL, commissioning source quotas
        - Course counts (`T`) and bubble caps (`max_bubbles`)
        - Eligibility counts per commissioning source
        - Degree tier distributions and tier counts (if `"Deg Tiers"` and `"I^D"` are available)
        - Cadet preference counts per AFSC (if `"Choice Count"` is present)

    - Only the first `p["M"]` AFSCs are included in the output. Any padding elements (e.g., "*") are excluded.

    - The output file is named `"AFSCs.csv"` and stored in the directory determined by `instance.export_paths`.

    See Also
    --------
    - [`import_afscs_data`](../../../afccp/reference/data/processing/#data.processing.import_afscs_data)
    - [`initialize_file_information`](../../../afccp/reference/data/processing/#data.processing.initialize_file_information)
    """

    # Shorthand
    p = instance.parameters

    # Initialize dictionary translating AFSC parameters to their "AFSCs" df column counterparts
    afsc_parameters_to_columns = {"afscs": "AFSC", "acc_grp": "Accessions Group", "afscs_stem": "STEM",
                                  "usafa_quota": "USAFA Target", "rotc_quota": "ROTC Target", 'ots_quota': 'OTS Target',
                                  "pgl": "PGL Target", "quota_e": "Estimated",
                                  "quota_d": "Desired", "quota_min": "Min", "quota_max": "Max",
                                  'max_bubbles': 'Max (Bubbles)',
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
    Exports cadet-level data from the given `Instance` object to a CSV file.

    This function builds the "Cadets" dataframe from internal model parameters stored in the instance,
    capturing individual cadet characteristics, preferences, and qualification data (if available).
    The output is saved to disk at the location specified by `instance.export_paths["Cadets"]`.

    Parameters
    ----------
    - instance : Instance
      A fully initialized `Instance` object containing model parameters (`parameters`) and a configured
      export path for the "Cadets" CSV file.

    Returns
    -------
    - None
      The function writes the cadet-level data to disk and does not return a value.

    Notes
    -----
    - The following cadet-level attributes will be included if present in `parameters`:

        - Basic profile: Cadet ID, gender, race, ethnicity, accession group, STEM tag, ASC codes
        - Assignment metadata: must-match flags, base/course preferences, assigned AFSC
        - Merit metrics: raw merit, real merit
        - Training data: start date, preference rankings, course/base weights and thresholds
        - Utility and preference columns: if present, the full `c_utilities` and `c_preferences` matrices will be exported
        - Qualification data: if a `qual` matrix is present, columns are added for each AFSC (e.g., `qual_17X`, `qual_21R`)

    - The preference and utility columns are labeled as `Pref_1`, `Pref_2`, ..., `Util_1`, `Util_2`, etc.
    - The output is saved as `"Cadets.csv"` under the directory given by `instance.export_paths`.

    See Also
    --------
    - [`import_cadets_data`](../../../afccp/reference/data/processing/#data.processing.import_cadets_data)
    - [`initialize_file_information`](../../../afccp/reference/data/processing/#data.processing.initialize_file_information)
    """

    # Shorthand
    p = instance.parameters

    # Initialize dictionary translating 'AFSCs' df columns to their parameter counterparts
    cadet_parameters_to_columns = {"cadets": "Cadet", "must_match": "Must Match",
                                   "assigned": "Assigned", "acc_grp_constraint": "Accessions Group",
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

    # If we had the cadet pafccp/reference/utility columns before, we'll add them back in
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
    Exports cadet-AFSC utility and preference matrices from the given `Instance` object to CSV files.

    This function checks for the presence of known matrix-style parameters in the model (e.g., utility values,
    preference rankings, interest scores) and exports them to disk. Each matrix is stored as a CSV with cadets
    as rows and AFSCs as columns (or vice versa), depending on the context.

    Parameters
    ----------
    - instance : Instance
      A fully initialized `Instance` object containing model parameters (`parameters`),
      value parameters (`value_parameters`), and export paths.

    Returns
    -------
    - None
      The function writes one or more matrix-style datasets to disk if they exist.

    Notes
    -----
    - The following parameters will be exported if present:

        - `utility`: Cadet utilities over all AFSCs → `"Cadets Utility"`
        - `c_pref_matrix`: Cadet preferences over all AFSCs → `"Cadets Preferences"`
        - `afsc_utility`: AFSC utilities over all cadets → `"AFSCs Utility"`
        - `a_pref_matrix`: AFSC preferences over all cadets → `"AFSCs Preferences"`
        - `rr_interest_matrix`: ROTC-rated interest scores → `"ROTC Rated Interest"`
        - `rr_om_matrix`: ROTC OM values → `"ROTC Rated OM"`
        - `ur_om_matrix`: USAFA OM values → `"USAFA Rated OM"`
        - `or_om_matrix`: OTS OM values → `"OTS Rated OM"`
        - `cadet_utility`: Finalized cadet utility values → `"Cadets Utility (Final)"`
        - `c_selected_matrix`: Final cadet selection matrix → `"Cadets Selected"`
        - `a_bucket_matrix`: AFSC bucket matrix → `"AFSCs Buckets"`

    - Each exported dataframe will have a `"Cadet"` column followed by one column per AFSC in the relevant set.
      The set of AFSCs may vary depending on whether the data is specific to a commissioning source (SOC).

    - Datasets related to specific SOCs (e.g., `"ROTC Rated OM"`) use filtered cadet subsets and
      AFSCs determined by `determine_soc_rated_afscs()`.

    See Also
    --------
    - [`determine_soc_rated_afscs`](../../../afccp/reference/data/preferences/#data.preferences.determine_soc_rated_afscs)
    - [`import_afsc_cadet_matrices_data`](../../../afccp/reference/data/processing/#data.processing.import_afsc_cadet_matrices_data)
    - [`initialize_file_information`](../../../afccp/reference/data/processing/#data.processing.initialize_file_information)
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
                afscs = afccp.data.preferences.determine_soc_rated_afscs(
                    soc='rotc', all_rated_afscs=all_rated_afscs)
            elif 'USAFA' in dataset:
                cadet_indices = p["Rated Cadets"]['usafa']
                pref_df = pd.DataFrame({"Cadet": p['cadets'][cadet_indices]})
                afscs = afccp.data.preferences.determine_soc_rated_afscs(
                    soc='usafa', all_rated_afscs=all_rated_afscs)
            elif 'OTS' in dataset:
                cadet_indices = p["Rated Cadets"]['ots']
                pref_df = pd.DataFrame({"Cadet": p['cadets'][cadet_indices]})
                afscs = afccp.data.preferences.determine_soc_rated_afscs(
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

    This function extracts and exports optimization value parameter sets, global utility matrices (if available),
    and cadet-specific constraints. It supports multiple value parameter configurations by exporting separate
    files per set. This facilitates analysis, debugging, or visualization of value-based multi-objective optimization.

    Parameters
    ----------
    - instance : Instance
      A configured instance of the CadetCareerProblem class, including:

        - `vp_dict`: Dictionary of value parameter sets
        - `value_parameters`: Active value parameter configuration
        - `parameters`: General instance parameters (e.g., AFSCs, cadets)
        - `export_paths`: File paths for saving exports

    Returns
    -------
    None
      The function writes multiple CSV files to disk for each available dataset.

    This command will generate:
    - A separate CSV file for each set of value parameters (e.g., weights, targets, value functions)
    - An overall summary CSV file of value parameter metadata
    - A cadet-level constraints CSV file (min values per cadet per VP set)
    - A global utility matrix CSV (if `global_utility` is present in a VP set)

    Notes
    -----
    - Value function breakpoints `a` and values `f^hat` are stored as comma-separated strings for readability.
    - Objective weights are scaled to a 0–100 range and normalized per AFSC.
    - The output files use naming conventions like:

        - `{data_name} {vp_name}.csv`
        - `{data_name} {vp_name} Global Utility.csv`
    - These files are versioned if the instance's data version is not `"Default"`.

    See Also
    --------
    - [`import_value_parameters_data`](../../../afccp/reference/data/processing/#data.processing.import_value_parameters_data)
    - [`initialize_file_information`](../../../afccp/reference/data/processing/#data.processing.initialize_file_information)
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
    Export cadet-to-AFSC solution assignments to CSV files.

    This function exports all available cadet solution assignments (including AFSC, base, and course solutions)
    to CSV files for downstream analysis, visualization, or comparison. Each solution is saved as a column in
    the exported file, enabling side-by-side comparison of multiple optimization outcomes.

    Parameters
    ----------
    - instance : Instance
        The problem instance containing:

        - `parameters` – model parameter dictionary
        - `solutions` – dictionary of cadet-to-AFSC assignments by solution name
        - `export_paths` – dictionary of destination paths for saving outputs

    Returns
    -------
    None
        The function writes 1 to 3 CSV files to disk, depending on available solution components.

    This exports:
    - `Solutions.csv`: Main cadet-to-AFSC assignment matrix
    - `Base Solutions.csv`: Optional cadet-to-base assignments, if present
    - `Course Solutions.csv`: Optional cadet-to-course assignments, if present

    Notes
    -----
    - Each file contains cadets in the first column and one or more solution columns following.
    - Solution names (keys from `instance.solutions`) define the column headers.
    - The function safely skips missing data (e.g., base or course assignments are only exported if they exist).
    - Used primarily to track scenario-based solution outputs from multi-run experiments.

    See Also
    --------
    - [`import_solutions_data`](../../../afccp/reference/data/processing/#data.processing.import_solutions_data)
    - [`initialize_file_information`](../../../afccp/reference/data/processing/#data.processing.initialize_file_information)
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
    Export additional configuration and metadata to CSV files.

    This function exports all supplementary datasets associated with the problem instance, including base
    assignments, base preferences, utility scores, training course data, and CASTLE-AFSC mappings. These
    datasets are derived from the `instance.parameters` dictionary and written to disk using paths from
    `instance.export_paths`.

    Parameters
    ----------
    - instance : Instance
        The problem instance containing:

        - `parameters` – dictionary of model inputs and outputs
        - `export_paths` – dictionary of file paths for each dataset

    Returns
    -------
    None
        Outputs are written to CSV files; no value is returned.

    This generates the following (if applicable):
    - `Bases.csv`: Min/max cadet assignments per AFSC at each base
    - `Bases Preferences.csv`: Cadet preferences over bases
    - `Bases Utility.csv`: Cadet utility scores for each base
    - `Courses.csv`: Course-level details per AFSC (min/max/start)
    - `Castle Input.csv`: CASTLE-to-AFPC AFSC mappings with optional value curves

    Notes
    -----
    - The export is conditional: datasets are only written if their associated parameters exist in `instance.parameters`.
    - CASTLE-related data (`castle_q`, `castle_afscs_arr`, etc.) must be present to trigger `Castle Input.csv` export.
    - Base utility and preference matrices are assumed to be cadet-by-base numpy arrays.

    See Also
    --------
    - [`import_additional_data`](../../../afccp/reference/data/processing/#data.processing.import_additional_data)
    - [`initialize_file_information`](../../../afccp/reference/data/processing/#data.processing.initialize_file_information)
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


# __________________________________________SOLUTION RESULTS HANDLING___________________________________________________
def export_solution_results_excel(instance, filepath):
    """
    Export a comprehensive Excel workbook of solution results.

    This function generates an Excel file containing detailed outputs from a solved cadet-AFSC assignment instance,
    including objective values, cadet assignments, constraint violations, and other performance metrics. The resulting
    Excel workbook supports deep post-solution analysis and includes conditional formatting for visual clarity.

    Parameters
    ----------
    - instance : Instance
      An object representing the solved assignment problem. Must contain:

      - `parameters` (dict): Problem data
      - `value_parameters` (dict): Objective metadata and weights
      - `solution` (dict): Final solution output (e.g., assignments, utilities, choice rankings)
      - `mdl_p` (dict): Metadata including formatting options
    - filepath : str
      Full path where the Excel file will be saved (e.g., `"output/solution_results.xlsx"`)

    Returns
    -------
    None
      Writes an `.xlsx` file to disk containing multiple sheets of structured solution data.

    Excel Output Includes
    ---------------------
    - Main: High-level metrics, objective value, choice counts, and performance indicators
    - Objective Measures: AFSC scores for each weighted objective
    - Constraint Fails: Constraint violations by AFSC
    - Objective Values: Weighted performance per AFSC with visual scoring heatmaps
    - Solution: Per-cadet assignment breakdown with preferences, utilities, base/course matches
    - X, V, Q (optional): Assignment matrices for AFSCs, bases, and training courses
    - Lambda, Y (optional): Value function parameters per AFSC and objective
    - Castle Metrics (if applicable): Metrics for CASTLE-mode AFSCs
    - Blocking Pairs (if present): Cadet-AFSC blocking violations

    Notes
    -----
    - Conditional formatting is applied to highlight preference rankings, merit scores, and match quality.
    - The function handles presence or absence of optional components (e.g., base matching, training courses).
    - Top 10 cadet choices and utilities are shown in the Solution tab for deeper preference analysis.

    See Also
    --------
    - [`draw_frame_border_outside`](../../../afccp/reference/data/processing/#data.processing.draw_frame_border_outside)
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
                        'AFSC Utility': 'afsc_utility_overall',
                        'USAFA Cadet Utility': 'usafa_cadet_utility',
                        'ROTC Cadet Utility': 'rotc_cadet_utility',
                        'OTS Cadet Utility': 'ots_cadet_utility',
                        'USSF Cadet Utility': 'ussf_cadet_utility',
                        'USAF Cadet Utility': 'usaf_cadet_utility',
                        'USSF AFSC Utility': 'ussf_afsc_utility',
                        'USAF AFSC Utility': 'usaf_afsc_utility',
                        'OTS Average Cadet Utility': 'OTS Average Cadet Utility',
                        'OTS Average AFSC Utility': 'OTS Average AFSC Utility',
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
                            'alternate_list_metric',
                        'OTS "Must Match" Candidates / Total Matched OTS Candidates': 'matched_out_of_must_match',
                        'OTS Candidates receiving an AFSC they selected ': 'OTS Selected Pref Count'}
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
    """
    Draws an outer border around a rectangular cell range using conditional formatting.

    - Applies a frame to the specified region starting at (first_row, first_col) with size (rows_count x cols_count)
    - Border color and width are customizable
    - Assumes 0-based indexing and adjusts if row/column values are less than 1

    Parameters
    ----------
    - workbook : xlsxwriter.Workbook
    - worksheet : xlsxwriter.Worksheet
    - first_row : int
    - first_col : int
    - rows_count : int
    - cols_count : int
    - color : str, default '#0000FF'
    - width : int, default 2
    """

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





