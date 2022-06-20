# Import Libraries
import numpy as np
from math import *
from afccp.core.globals import *


# Value Parameter Procedures
def model_value_parameters_from_excel(parameters, filepath, num_breakpoints=None, use_actual=True, printing=False):
    """
    This procedure imports weight and value parameters from a file and converts them to the model weight and
    value parameters
    :param use_actual: if we want to incorporate the set of eligible cadets for each AFSCs to generate their objective
    weights and value functions
    :param parameters: Model fixed parameters
    :param num_breakpoints: if this is specified, we recreate the functions from the strings
    :param printing: Whether the procedure should print something
    :param filepath: input filepath containing weight/value parameter data
    :return: weight/value parameters
    """
    if printing:
        print('Importing value parameters from excel...')

    # Import Data sets
    afsc_weights = import_data(filepath, sheet_name="AFSC Weights")
    cadet_weights = import_data(filepath, sheet_name="Cadet Weights")
    overall_weights = import_data(filepath, sheet_name="Overall Weights")

    # Load into the value_parameters dictionary
    M = len(np.unique(afsc_weights['AFSC']))
    O = int(len(afsc_weights) / M)
    value_parameters = {'O': O, "afscs_overall_weight": np.array(overall_weights['AFSCs Weight'])[0],
                        "cadets_overall_weight": np.array(overall_weights['Cadets Weight'])[0],
                        "cadet_weight_function": np.array(overall_weights['Cadet Weight Function'])[0],
                        "afsc_weight_function": np.array(overall_weights['AFSC Weight Function'])[0],
                        "cadets_overall_value_min": np.array(overall_weights['Cadets Min Value'])[0],
                        "afscs_overall_value_min": np.array(overall_weights['AFSCs Min Value'])[0],
                        "cadet_weight": np.array(cadet_weights["Weight"]), "M": M,
                        "cadet_value_min": np.array(cadet_weights["Min Value"]),
                        "afsc_value_min": np.zeros(M),
                        "objective_value_min": np.array([[" " * 20 for _ in range(O)] for _ in range(M)]),
                        "value_functions": np.array([[" " * 200 for _ in range(O)] for _ in range(M)]),
                        "constraint_type": np.zeros([M, O]), 'a': [[[] for _ in range(O)] for _ in range(M)],
                        "objective_target": np.zeros([M, O]), 'f^hat': [[[] for _ in range(O)] for _ in range(M)],
                        "objective_weight": np.zeros([M, O]), "afsc_weight": np.zeros(M),
                        'objectives': np.array(afsc_weights.loc[:int(len(afsc_weights) / M - 1), 'Objective'])}

    # Load in value parameter data for each AFSC
    for j in range(M):  # These are Os (Ohs) not 0s (zeros)
        value_parameters["objective_target"][j, :] = np.array(afsc_weights.loc[j * O:(j * O + O - 1),
                                                              'Objective Target'])
        objective_weights = np.array(afsc_weights.loc[j * O:(j * O + O - 1), 'Objective Weight'])
        value_parameters["objective_weight"][j, :] = objective_weights / sum(objective_weights)
        value_parameters["objective_value_min"][j, :] = np.array(afsc_weights.loc[j * O:(j * O + O - 1),
                                                                 'Min Objective Value'])
        value_parameters["constraint_type"][j, :] = np.array(afsc_weights.loc[j * O:(j * O + O - 1),
                                                             'Constraint Type'])
        value_parameters["value_functions"][j, :] = np.array(afsc_weights.loc[j * O:(j * O + O - 1),
                                                             'Value Functions'])
        value_parameters["afsc_weight"][j] = afsc_weights.loc[j * O, "AFSC Weight"]
        value_parameters["afsc_value_min"][j] = afsc_weights.loc[j * O, "Min Value"]
        cadets = parameters['I^E'][j]

        # Loop through each objective for this AFSC
        for k, objective in enumerate(value_parameters['objectives']):

            # Value Function string
            vf_string = value_parameters["value_functions"][j, k]

            # We import the function directly from the breakpoints
            if num_breakpoints is None or 'Quota_Over' in vf_string:
                string = afsc_weights.loc[j * O + k, 'Function Breakpoints']
                if type(string) == str:
                    value_parameters['a'][j][k] = [float(x) for x in string.split(",")]
                string = afsc_weights.loc[j * O + k, 'Function Breakpoint Values']
                if type(string) == str:
                    value_parameters['f^hat'][j][k] = [float(x) for x in string.split(",")]

            # We recreate the function from the vf string
            else:
                if vf_string != 'None':
                    target = value_parameters['objective_target'][j, k]
                    actual = None
                    maximum = None
                    minimum = None

                    if use_actual:
                        if objective == 'Merit':
                            actual = np.mean(parameters['merit'][cadets])
                        elif objective == 'USAFA Proportion':
                            actual = np.mean(parameters['usafa'][cadets])

                    if objective == 'Combined Quota':

                        # Get bounds
                        minimum, maximum = parameters['quota_min'][j], parameters['quota_max'][j]
                        target = parameters['quota'][j]

                    segment_dict = create_segment_dict_from_string(vf_string, target, actual=actual,
                                                                   maximum=maximum, minimum=minimum)
                    value_parameters['a'][j][k], value_parameters['f^hat'][j][k] = value_function_builder(
                        segment_dict, num_breakpoints=num_breakpoints)

    # Force AFSC weights to sum to 1
    value_parameters["afsc_weight"] = value_parameters["afsc_weight"] / sum(value_parameters["afsc_weight"])
    return value_parameters


def model_value_parameter_data_frame_from_parameters(parameters, value_parameters):
    """
    This procedure takes a set of parameters and value_parameters and constructs the value parameter data frames
    :param parameters: fixed model parameters
    :param value_parameters: weight and value parameters
    :return: dataframes
    """
    N, M = parameters['N'], parameters['M']
    O = len(value_parameters['objectives'])
    a_strings = np.array([[" " * 400 for _ in range(O)] for _ in range(M)])
    fhat_strings = np.array([[" " * 400 for _ in range(O)] for _ in range(M)])
    for j, afsc in enumerate(parameters['afsc_vector']):
        for k, objective in enumerate(value_parameters['objectives']):
            string_list = [str(x) for x in value_parameters['a'][j][k]]
            a_strings[j, k] = ",".join(string_list)
            string_list = [str(x) for x in value_parameters['f^hat'][j][k]]
            fhat_strings[j, k] = ",".join(string_list)

    afsc_objective_min_values = np.ndarray.flatten(value_parameters['objective_value_min'])
    afsc_objective_convex_constraints = np.ndarray.flatten(value_parameters['constraint_type'])
    afsc_objective_targets = np.ndarray.flatten(value_parameters['objective_target'])
    afsc_objectives = np.tile(value_parameters['objectives'], parameters['M'])

    # Get AFSC objective swing weights
    ow = value_parameters['objective_weight']
    max_weights = np.max(value_parameters['objective_weight'], axis=1)
    ow = np.array([[ow[j, k] / max_weights[j] for k in range(O)] for j in range(M)])
    afsc_objective_weights = np.ndarray.flatten(np.around(ow * 100, 3))

    # More stuff
    afsc_value_functions = np.ndarray.flatten(value_parameters['value_functions'])
    afscs = np.ndarray.flatten(np.array(list(np.repeat(parameters['afsc_vector'][j],
                                                       value_parameters['O']) for j in range(parameters['M']))))

    # AFSC swing weights
    value_parameters['afsc_weight'] = value_parameters['afsc_weight'] / np.max(value_parameters['afsc_weight'])
    value_parameters['afsc_weight'] = np.around(value_parameters['afsc_weight'] * 100, 3)
    afsc_weights = np.ndarray.flatten(np.array(list(np.repeat(value_parameters['afsc_weight'][j],
                                                              value_parameters['O']) for j in
                                                    range(parameters['M']))))
    afsc_min_values = np.ndarray.flatten(np.array(list(np.repeat(value_parameters['afsc_value_min'][j],
                                                                 value_parameters['O']) for j in
                                                       range(parameters['M']))))
    afsc_weights_df = pd.DataFrame({'AFSC': afscs, 'Objective': afsc_objectives,
                                    'Objective Weight': afsc_objective_weights,
                                    'Objective Target': afsc_objective_targets, 'AFSC Weight': afsc_weights,
                                    'Min Value': afsc_min_values, 'Min Objective Value': afsc_objective_min_values,
                                    'Constraint Type': afsc_objective_convex_constraints,
                                    'Function Breakpoint Measures (a)': np.ndarray.flatten(a_strings),
                                    'Function Breakpoint Values (f^hat)': np.ndarray.flatten(fhat_strings),
                                    'Value Functions': afsc_value_functions})

    cadet_weights_df = pd.DataFrame({'Cadet': parameters['ID'],
                                     'Weight': value_parameters['cadet_weight'],
                                     'Min Value': value_parameters['cadet_value_min']})

    overall_weights_df = pd.DataFrame({'Cadets Weight': [value_parameters['cadets_overall_weight']],
                                       'AFSCs Weight': [value_parameters['afscs_overall_weight']],
                                       'Cadets Min Value': [value_parameters['cadets_overall_value_min']],
                                       'AFSCs Min Value': [value_parameters['afscs_overall_value_min']],
                                       'AFSC Weight Function': [value_parameters['afsc_weight_function']],
                                       'Cadet Weight Function': [value_parameters['cadet_weight_function']]})

    # Check if other columns are present (phasing these in)
    more_vp_columns = ["USAFA-Constrained AFSCs", "Similarity Constraint"]
    for col in more_vp_columns:
        if col in value_parameters:
            overall_weights_df[col] = [value_parameters[col]]
        else:
            overall_weights_df[col] = [""]

    return overall_weights_df, cadet_weights_df, afsc_weights_df


def model_value_parameters_set_additions(parameters, value_parameters, printing=False):
    """
    Creates subsets for AFSCs and objectives to distinguish which AFSCs care about which objectives so that we don't
    have to calculate every value only to multiply them by zero.
    :param parameters: model fixed parameters
    :param value_parameters: model weight/value parameters
    :param printing: whether the procedure should print something
    :return: updated value parameters with sets
    """
    if printing:
        print('Adding AFSC and objective subsets to value parameters...')

    # Grab number of AFSCs and objectives
    M = value_parameters['M']
    O = value_parameters['O']
    value_parameters['K'] = np.arange(O)

    # Set of objectives for each AFSC
    value_parameters['K^A'] = {}  # objectives
    value_parameters['K^C'] = {}  # constrained objectives
    for j in range(M):
        value_parameters['K^A'][j] = np.where(
            value_parameters['objective_weight'][j, :] > 0)[0].astype(int)
        value_parameters['K^C'][j] = np.where(
            value_parameters['constraint_type'][j, :] > 0)[0].astype(int)

    # 5% of total USAFA graduating class set
    value_parameters["J^USAFA"] = None

    # Add the AFSC indices to the set
    if "USAFA-Constrained AFSCs" in value_parameters:
        if "," in value_parameters["USAFA-Constrained AFSCs"]:
            value_parameters["J^USAFA"] = np.array([])
            usafa_afscs = value_parameters["USAFA-Constrained AFSCs"].split(",")
            for afsc in usafa_afscs:
                afsc = afsc.strip()
                j = np.where(parameters["afsc_vector"] == afsc)[0]
                if len(j) == 0:
                    print("WARNING: Something is wrong with the USAFA-Constrained AFSCs! "
                          "'" + afsc + "' is not in the list of AFSCs.")
                else:
                    value_parameters["J^USAFA"] = np.hstack((value_parameters["J^USAFA"], j))
            value_parameters["J^USAFA"] = value_parameters["J^USAFA"].astype(int)

    # Set of objectives that seek to balance some cadet demographic
    value_parameters['K^D'] = ['USAFA Proportion', 'Mandatory', 'Desired', 'Permitted', 'Male', 'Minority']

    # Set of AFSCs for each objective:
    value_parameters['J^A'] = {}
    for k in range(O):
        value_parameters['J^A'][k] = np.where(value_parameters['objective_weight'][:, k] > 0)[0].astype(int)

    # Cadet Value Constraint Set
    value_parameters['I^C'] = np.where(value_parameters['cadet_value_min'] > 0)[0]

    # AFSC Value Constraint Set
    value_parameters['J^C'] = np.where(value_parameters['afsc_value_min'] > 0)[0]

    # number of breakpoints
    value_parameters['r'] = np.array([[len(value_parameters['a'][j][k]) for k in range(O)] for j in range(M)])

    # set of breakpoints
    value_parameters['L'] = np.array([[np.arange(value_parameters['r'][j, k]) for k in range(O)] for j in range(M)],
                                     dtype=object)

    # Round weights
    value_parameters['objective_weight'] = np.around(value_parameters['objective_weight'], 8)
    value_parameters['afsc_weight'] = np.around(value_parameters['afsc_weight'], 8)
    value_parameters['cadet_weight'] = np.around(value_parameters['cadet_weight'], 8)

    return value_parameters


def model_value_parameters_to_defaults(parameters, value_parameters, filepath=None,
                                       printing=False):
    """
    This procedure takes the user parameters and exports them to the default user parameter excel sheet where they
    can be used as defaults for a new problem later. Assumes all 32 AFSCs and at least > 10 cadets
    :param filepath: optional filepath instead of the standard defaults file
    :param parameters: Model Fixed parameters, used only for AFSC vector
    :param value_parameters: model value parameters
    :param printing: Whether the procedure should print something
    :return: None.
    """
    if printing:
        print('Exporting value parameters as defaults to excel...')

    if filepath is None:
        if len(parameters['afsc_vector'][0]) == 2:  # Letter + "1"  (A1, B1, etc.)
            folder_out = 'scrubbed'
        else:
            folder_out = 'real'

        # Will have to change the name to the specific scrubbed letter if applicable
        filepath = support_paths[folder_out] + 'Value_Parameters_Default_New.xlsx'

    overall_weights_df = pd.DataFrame({'Cadets Overall': [value_parameters['cadets_overall_weight']],
                                       'AFSCs Overall': [value_parameters['afscs_overall_weight']],
                                       'AFSCs Min Value': [value_parameters['afscs_overall_value_min']],
                                       'Cadets Min Value': [value_parameters['cadets_overall_value_min']],
                                       'Cadet Weight Function': [value_parameters['cadet_weight_function']],
                                       'AFSC Weight Function': [value_parameters['afsc_weight_function']]})

    afsc_weights_df = pd.DataFrame({'AFSC': parameters['afsc_vector'],
                                    'AFSC Swing Weight': np.around((value_parameters['afsc_weight'] /
                                                                    max(value_parameters['afsc_weight'])) * 100, 2),
                                    'AFSC Min Value': value_parameters['afsc_value_min']})
    afsc_objective_weights_df = pd.DataFrame({'AFSC': parameters['afsc_vector']})
    afsc_objective_targets_df = pd.DataFrame({'AFSC': parameters['afsc_vector']})
    afsc_objective_value_min_df = pd.DataFrame({'AFSC': parameters['afsc_vector']})
    afsc_objective_convex_constraint_df = pd.DataFrame({'AFSC': parameters['afsc_vector']})
    afsc_objective_value_functions_df = pd.DataFrame({'AFSC': parameters['afsc_vector']})

    # Turn local weights into swing weights
    weights = value_parameters['objective_weight']
    for j in range(parameters['M']):
        max_weight = max(weights[j, :])
        weights[j, :] = (weights[j, :] / max_weight) * 100

    for k, objective in enumerate(value_parameters['objectives']):
        afsc_objective_weights_df[objective] = weights[:, k]
        afsc_objective_targets_df[objective] = value_parameters['objective_target'][:, k]
        afsc_objective_value_min_df[objective] = value_parameters['objective_value_min'][:, k]
        afsc_objective_convex_constraint_df[objective] = value_parameters['constraint_type'][:, k]
        afsc_objective_value_functions_df[objective] = value_parameters['value_functions'][:, k]

    with pd.ExcelWriter(filepath) as writer:  # Export to excel
        overall_weights_df.to_excel(writer, sheet_name="Overall Weights", index=False)
        afsc_weights_df.to_excel(writer, sheet_name="AFSC Weights", index=False)
        afsc_objective_weights_df.to_excel(writer, sheet_name="AFSC Objective Weights", index=False)
        afsc_objective_targets_df.to_excel(writer, sheet_name="AFSC Objective Targets", index=False)
        afsc_objective_value_min_df.to_excel(writer, sheet_name='AFSC Objective Min Value', index=False)
        afsc_objective_convex_constraint_df.to_excel(writer, sheet_name='Constraint Type', index=False)
        afsc_objective_value_functions_df.to_excel(writer, sheet_name='Value Functions', index=False)


def default_value_parameters_from_excel(filepath, num_breakpoints=24, printing=False):
    """
    Imports the "factory defaults" for value parameters
    :param num_breakpoints: number of breakpoints to use for value functions
    :param filepath: Filepath to import from
    :param printing: Whether the function should print something
    :return: default user parameters
    """
    if printing:
        print('Importing default value parameters...')

    # Get dataframes
    overall_weights_df = import_data(filepath, sheet_name="Overall Weights")
    afsc_weights_df = import_data(filepath, sheet_name="AFSC Weights")
    afsc_objective_weights_df = import_data(filepath, sheet_name="AFSC Objective Weights")
    afsc_objective_targets_df = import_data(filepath, sheet_name="AFSC Objective Targets")
    afsc_objective_value_min_df = import_data(filepath, sheet_name="AFSC Objective Min Value")
    afsc_objective_convex_constraints_df = import_data(filepath, sheet_name="Constraint Type")
    afsc_value_functions_df = import_data(filepath, sheet_name="Value Functions")
    objectives = np.array(afsc_objective_weights_df.keys()[1:])
    default_value_parameters = {'cadet_weight_function': overall_weights_df['Cadet Weight Function'][0],
                                'afsc_weight_function': overall_weights_df['AFSC Weight Function'][0],
                                'cadets_overall_weight': overall_weights_df['Cadets Overall'][0],
                                'afscs_overall_weight': overall_weights_df['AFSCs Overall'][0],
                                'afsc_weight': np.array(afsc_weights_df['AFSC Swing Weight']),
                                'objective_weight': np.array(afsc_objective_weights_df.iloc[:,
                                                             1:(len(objectives) + 1)]),
                                'objective_target': np.array(
                                    afsc_objective_targets_df.iloc[:, 1:(len(objectives) + 1)]),
                                'objective_value_min': np.array(
                                    afsc_objective_value_min_df.iloc[:, 1:(len(objectives) + 1)]),
                                'constraint_type': np.array(
                                    afsc_objective_convex_constraints_df.iloc[:, 1:(len(objectives) + 1)]),
                                'value_functions': np.array(afsc_value_functions_df.iloc[:, 1:(len(objectives) + 1)]),
                                'cadets_overall_value_min': overall_weights_df['Cadets Min Value'][0],
                                'afscs_overall_value_min': overall_weights_df['AFSCs Min Value'][0],
                                'afsc_value_min': np.array(afsc_weights_df['AFSC Min Value']),
                                'objectives': objectives,
                                'complete_afsc_vector': np.array(afsc_weights_df['AFSC']),
                                'num_breakpoints': num_breakpoints}

    # Check if other columns are present (phasing these in)
    more_vp_columns = ["USAFA-Constrained AFSCs", "Similarity Constraint", "Cadets Top 3 Constraint"]
    for col in more_vp_columns:
        if col in overall_weights_df:
            element = str(np.array(overall_weights_df[col])[0])
            if element == "nan":
                element = ""
            default_value_parameters[col] = element
        else:
            default_value_parameters[col] = ""
    return default_value_parameters


def generate_value_parameters_from_defaults(parameters, default_value_parameters, generate_afsc_weights=True,
                                            num_breakpoints=None, printing=False):
    """
    Generates value parameters from the defaults for a specified problem
    :param generate_afsc_weights: if we should be generating AFSC weights
    :param num_breakpoints: number of breakpoints to use for each value function
    :param parameters: model fixed parameters
    :param default_value_parameters: default generalised parameters
    :param printing: Whether the procedure should print something
    :return: model value parameters
    """
    if printing:
        print('Generating value parameters from defaults...')
    M = len(parameters['afsc_vector'])
    N = parameters['N']

    # Add the AFSC objectives that are included in this instance
    objective_lookups = {'Merit': 'merit', 'USAFA Proportion': 'usafa', 'Combined Quota': 'quota',
                         'USAFA Quota': 'usafa_quota', 'ROTC Quota': 'rotc_quota', 'Mandatory': 'mandatory',
                         'Desired': 'desired', 'Permitted': 'permitted', 'Utility': 'utility', 'Male': 'male',
                         'Minority': 'minority'}
    objectives = []
    objective_indices = []
    for k, objective in enumerate(list(objective_lookups.keys())):
        if objective_lookups[objective] in parameters:
            objectives.append(objective)
            objective_indices.append(k)

    objectives = np.array(objectives)
    objective_indices = np.array(objective_indices)
    afsc_indices = np.array([np.where(
        default_value_parameters['complete_afsc_vector'] == parameters['afsc_vector'][j])[0][0] for j in range(M)])
    O = len(objectives)

    value_parameters = {'cadets_overall_weight': default_value_parameters['cadets_overall_weight'],
                        'afscs_overall_weight': default_value_parameters['afscs_overall_weight'],
                        'cadet_weight_function': default_value_parameters['cadet_weight_function'],
                        'afsc_weight_function': default_value_parameters['afsc_weight_function'],
                        'cadets_overall_value_min': default_value_parameters['cadets_overall_value_min'],
                        'afscs_overall_value_min': default_value_parameters['afscs_overall_value_min'],
                        'afsc_value_min': np.zeros(M), 'cadet_value_min': np.zeros(N),
                        'objective_weight': np.zeros([M, O]), 'afsc_weight': np.zeros(M), "M": M,
                        'objective_target': np.zeros([M, O]), 'objectives': objectives, 'O': O,
                        'objective_value_min': np.array([[" " * 20 for _ in range(O)] for _ in range(M)]),
                        'constraint_type': np.zeros([M, O]).astype(int),
                        "USAFA-Constrained AFSCs": default_value_parameters["USAFA-Constrained AFSCs"],
                        "Similarity Constraint": default_value_parameters["Similarity Constraint"]}

    if num_breakpoints is None:
        num_breakpoints = default_value_parameters['num_breakpoints']

    value_parameters['num_breakpoints'] = num_breakpoints

    # Determine weights on cadets
    if 'merit_all' in parameters:
        value_parameters['cadet_weight'] = cadet_weight_function(parameters['merit_all'],
                                                                 func=value_parameters['cadet_weight_function'])
    else:
        value_parameters['cadet_weight'] = cadet_weight_function(parameters['merit'],
                                                                 func=value_parameters['cadet_weight_function'])

    # Determine weights on AFSCs
    if generate_afsc_weights:
        func = value_parameters['afsc_weight_function']
        if func == 'Custom':  # We take the AFSC weights directly from excel
            generate_afsc_weights = False
        else:
            if "pgl" in parameters:
                quota = parameters["pgl"]
            else:
                quota = parameters["quota"]
            value_parameters['afsc_weight'] = afsc_weight_function(quota, func)

    # Initialize breakpoints
    value_parameters['a'] = [[[] for _ in range(O)] for _ in range(M)]
    value_parameters['f^hat'] = [[[] for _ in range(O)] for _ in range(M)]

    # Load value function strings
    value_functions = default_value_parameters['value_functions'][:, objective_indices]
    value_functions = value_functions[afsc_indices, :]
    value_parameters['value_functions'] = value_functions

    for j, afsc in enumerate(parameters['afsc_vector']):

        # Get location of afsc in the default value parameters (matters if this set of afscs does not match)
        loc = np.where(default_value_parameters['complete_afsc_vector'] == afsc)[0][0]

        # Initially assign all default weights, targets, etc.
        value_parameters['objective_weight'][j, :] = default_value_parameters['objective_weight'][loc,
                                                                                                  objective_indices]
        value_parameters['objective_target'][j] = default_value_parameters['objective_target'][loc,
                                                                                               objective_indices]
        value_parameters['objective_value_min'][j] = default_value_parameters['objective_value_min'][loc,
                                                                                                     objective_indices]
        value_parameters['afsc_value_min'][j] = default_value_parameters['afsc_value_min'][loc]
        value_parameters['constraint_type'][j] = \
            default_value_parameters['constraint_type'][loc, objective_indices]

        # If we're not generating afsc weights using the specified weight function...
        if not generate_afsc_weights:  # Also, if the weight function is "Custom"
            value_parameters['afsc_weight'][j] = default_value_parameters['afsc_weight'][loc]

        # Loop through each objective to load their targets
        for k, objective in enumerate(value_parameters['objectives']):

            maximum, minimum, actual = None, None, None
            if objective == 'Merit' and value_parameters['objective_weight'][j, k] != 0:
                value_parameters['objective_target'][j, k] = parameters['sum_merit'] / parameters['N']
                actual = np.mean(parameters['merit'][parameters['I^E'][j]])

            elif objective == 'USAFA Proportion' and value_parameters['objective_weight'][j, k] != 0:
                if parameters['usafa_quota'][j] == 0:
                    value_parameters['objective_target'][j, k] = 0

                elif parameters['usafa_quota'][j] == parameters['quota'][j]:
                    value_parameters['objective_target'][j, k] = 1
                else:
                    value_parameters['objective_target'][j, k] = parameters['usafa_proportion']
                actual = len(parameters['I^D'][objective][j]) / len(parameters['I^E'][j])

            elif objective == 'Combined Quota' and value_parameters['objective_weight'][j, k] != 0:
                value_parameters['objective_target'][j, k] = parameters['quota'][j]

                # Get bounds
                minimum, maximum = parameters['quota_min'][j], parameters['quota_max'][j]
                value_parameters['objective_value_min'][j, k] = str(int(minimum)) + ", " + str(int(maximum))

            elif objective == 'USAFA Quota' and value_parameters['objective_weight'][j, k] != 0:
                value_parameters['objective_target'][j, k] = parameters['usafa_quota'][j]
                value_parameters['objective_value_min'][j, k] = str(int(parameters['usafa_quota'][j])) + ", " + \
                                                                str(int(parameters['quota_max'][j]))

            elif objective == 'ROTC Quota' and value_parameters['objective_weight'][j, k] != 0:
                value_parameters['objective_target'][j, k] = parameters['quota'][j] - parameters['usafa_quota'][j]
                value_parameters['objective_value_min'][j, k] = str(int(parameters['rotc_quota'][j])) + ", " + \
                                                                str(int(parameters['quota_max'][j]))

            elif objective == 'Male' and value_parameters['objective_weight'][j, k] != 0:
                value_parameters['objective_target'][j, k] = parameters['male_proportion']
                actual = len(parameters['I^D'][objective][j]) / len(parameters['I^E'][j])

            elif objective == 'Minority' and value_parameters['objective_weight'][j, k] != 0:
                value_parameters['objective_target'][j, k] = parameters['minority_proportion']
                actual = len(parameters['I^D'][objective][j]) / len(parameters['I^E'][j])

            # If we care about this objective, we load in its value function breakpoints
            if value_parameters['objective_weight'][j, k] != 0:

                # Create the non-linear piecewise exponential segment dictionary
                segment_dict = create_segment_dict_from_string(value_functions[j, k],
                                                               value_parameters['objective_target'][j, k],
                                                               minimum=minimum, maximum=maximum, actual=actual)

                # Linearize the non-linear function using the specified number of breakpoints
                value_parameters['a'][j][k], value_parameters['f^hat'][j][k] = value_function_builder(
                    segment_dict, num_breakpoints=num_breakpoints)

        # Scale the weights for this AFSC, so they sum to 1
        value_parameters['objective_weight'][j] = value_parameters['objective_weight'][j] / \
                                                  sum(value_parameters['objective_weight'][j])
    # Scale the weights across all AFSCs, so they sum to 1
    if not generate_afsc_weights:
        value_parameters['afsc_weight'] = value_parameters['afsc_weight'] / \
                                          sum(value_parameters['afsc_weight'])

    return value_parameters


def compare_value_parameters(parameters, vp1, vp2, printing=False):
    """
    Compares two sets of value parameters to see if they are identical
    :param printing: if we should print how the two sets are similar
    :param parameters: set of fixed cadet/AFSC parameters
    :param vp1: set 1
    :param vp2: set 2
    :return: True if they're identical, False otherwise
    """

    # Assume identical until proven otherwise
    identical = True

    # Loop through each value parameter key
    for key in vp1:

        if key in ['afscs_overall_weight', 'cadets_overall_weight', 'cadet_weight_function', 'afsc_weight_function',
                   'cadets_overall_value_min', 'afscs_overall_value_min']:
            if vp1[key] != vp2[key]:
                if printing:
                    print('VPs not the same. ' + key + ' is different.')
                identical = False
                break

        elif key in ['afsc_value_min', 'cadet_value_min', 'objective_value_min', 'value_functions', 'objective_target',
                     'objective_weight', 'afsc_weight', 'cadet_weight', 'I^C', 'J^C', 'value_functions']:
            if key not in ['objective_value_min', 'value_functions']:
                vp_1_arr, vp_2_arr = np.ravel(np.around(vp1[key], 4)), np.ravel(np.around(vp2[key], 4))
            else:
                vp_1_arr, vp_2_arr = np.ravel(vp1[key]), np.ravel(vp2[key])
            diff_arr = np.array([vp_1_arr[i] != vp_2_arr[i] for i in range(len(vp_1_arr))])
            if sum(diff_arr) != 0 and (vp1[key] != [] or vp2[key] != []):
                if printing:
                    print('VPs not the same. ' + key + ' is different.')
                identical = False
                break

        elif key == 'a':  # Check the breakpoints

            for j in parameters['J']:
                for k in vp1['K^A'][j]:
                    for l in vp1['L'][j][k]:
                        try:
                            if vp1[key][j][k][l] != vp2[key][j][k][l]:
                                identical = False
                                if printing:
                                    print('VPs not the same. Breakpoints are different for AFSC ' + str(j) +
                                          ' Objective ' + str(k) + '.')
                                break
                        except:  # If there was a range error, then the breakpoints are not the same
                            identical = False
                            if printing:
                                print('VPs not the same. Breakpoints are different for AFSC ' + str(j) +
                                      ' Objective ' + str(k) + '.')
                            break
                    if not identical:
                        break
                if not identical:
                    break
            if not identical:
                break

    if identical and printing:
        print('VPs are the same.')

    return identical


def cadet_weight_function(merit, func="Curve_1"):
    """
    Take in a merit array and generate cadet weights depending on function specified
    """

    # Number of Cadets
    N = len(merit)

    # Generate Swing Weights based on function
    if func == 'Linear':
        swing_weights = np.array([1 + (2 * x) for x in merit])
    elif func == "Direct":
        swing_weights = merit
    elif func == 'Curve_1':
        swing_weights = np.array([1 + 2 / (1 + exp(-10 * (x - 0.5))) for x in merit])
    elif func == 'Curve_2':
        swing_weights = np.array([1 + 2 / (1 + exp(-12 * (x - 0.7))) for x in merit])
    elif func == 'Equal':
        swing_weights = np.ones(N)
    else:  # Exponential Function
        rho = -0.3
        swing_weights = np.array([(1 - exp(-x / rho)) / (1 - exp(-1 / rho)) for x in merit])

    # Normalize weights and return them
    weights = swing_weights / sum(swing_weights)
    return weights


def afsc_weight_function(quota, func="Curve"):
    """
    Take in an AFSC quota array and generate AFSC weights depending on function specified
    """

    # Number of AFSCs
    M = len(quota)

    # Scale quota to be 0-1 (referencing biggest AFSC)
    quota_scale = quota / np.max(quota)

    # Generate Swing Weights based on function
    if func == 'Linear':
        swing_weights = np.array([1 + (10 * x) for x in quota_scale])
    elif func in ["Direct", "Size"]:  # Direct relationship between size and importance
        swing_weights = quota
    elif func == "Piece":
        swing_weights = np.zeros(M)
        for j, x in enumerate(quota):
            if x >= 200:
                swing_weights[j] = 1
            elif 150 <= x < 200:
                swing_weights[j] = 0.9
            elif 100 <= x < 150:
                swing_weights[j] = 0.8
            elif 50 <= x < 100:
                swing_weights[j] = 0.7
            elif 25 <= x < 50:
                swing_weights[j] = 0.6
            else:
                swing_weights[j] = 0.5
    elif func == 'Curve_1':  # Sigmoid Function
        swing_weights = np.array([1 + 10 / (1 + exp(-5 * (x - 0.5))) for x in quota_scale])
    elif func == 'Curve_2':  # Sigmoid Function
        swing_weights = np.array([1 + 12 / (1 + exp(-20 * (x - 0.5))) for x in quota_scale])
    elif func == 'Equal':  # They're all the same
        swing_weights = np.ones(M)
    else:  # Exponential Function
        rho = -0.3
        swing_weights = np.array([(1 - exp(-x / rho)) / (1 - exp(-1 / rho)) for x in quota_scale])

    # Scale weights and return them
    weights = swing_weights / sum(swing_weights)
    return weights


# Value Function Construction
def create_segment_dict_from_string(vf_string, target=None, maximum=None, actual=None, multiplier=False, minimum=None):
    """
    This function takes a value function string and converts it into the segment
    dictionary which can then be used to generate the function breakpoints
    :param minimum: minimum objective measure (optional)
    :param multiplier: if we're multiplying the target values by some scalar for the quota objectives or not
    :param actual: proportion of eligible cadets
    :param vf_string: value function string
    :param target: target objective measure (optional)
    :param maximum: maximum objective measure (optional)
    :return: segment_dict
    """

    # Collect the kind of function we're creating
    split_list = vf_string.split('|')
    f_type = split_list[0]

    if f_type == 'Balance':

        # Receive values from string
        f_param_list = split_list[1]
        split_list = f_param_list.split(',')
        left_bm = float(split_list[0].strip())
        right_bm = float(split_list[1].strip())
        rho1 = float(split_list[2].strip())
        rho2 = float(split_list[3].strip())
        rho3 = float(split_list[4].strip())
        rho4 = float(split_list[5].strip())
        buffer_y = float(split_list[6].strip())

        if actual < target:
            left_margin = (target - actual) / 4 + left_bm
            right_margin = right_bm
        elif actual > target:
            left_margin = left_bm
            right_margin = (actual - target) / 4 + right_bm
        else:
            left_margin = left_bm
            right_margin = right_bm

        # Build segments
        segment_dict = {1: {'x1': 0, 'y1': 0, 'x2': round(target - left_margin, 3), 'y2': buffer_y, 'rho': -rho1},
                        2: {'x1': round(target - left_margin, 3), 'y1': buffer_y, 'x2': target, 'y2': 1, 'rho': rho2},
                        3: {'x1': target, 'y1': 1, 'x2': round(target + right_margin, 3), 'y2': buffer_y, 'rho': rho3},
                        4: {'x1': round(target + right_margin, 3), 'y1': buffer_y, 'x2': 1, 'y2': 0, 'rho': -rho4}}

    elif f_type == 'Quota_Normal':  # "Method 1" as described in thesis

        # Receive values from string
        f_param_list = split_list[1]
        split_list = f_param_list.split(',')
        domain_max = float(split_list[0].strip())
        rho1 = float(split_list[1].strip()) * target
        rho2 = float(split_list[2].strip()) * target
        if multiplier:
            maximum = int(target * maximum)
        real_max = max(int(target + (maximum - target) + target * domain_max), maximum + 1)

        # Build segments
        segment_dict = {1: {'x1': 0, 'y1': 0, 'x2': target, 'y2': 1, 'rho': -rho1},
                        2: {'x1': maximum, 'y1': 1, 'x2': real_max, 'y2': 0, 'rho': -rho2}}

    elif f_type == 'Quota_Over':  # "Method 2" as described in thesis

        # Receive values from string
        f_param_list = split_list[1]
        split_list = f_param_list.split(',')
        domain_max = float(split_list[0].strip())
        rho1 = float(split_list[1].strip()) * target
        rho2 = float(split_list[2].strip()) * target
        rho3 = float(split_list[3].strip()) * target
        buffer_y = float(split_list[4].strip())
        if multiplier:
            maximum = int(target * maximum)
            actual = int(target * actual)
        real_max = max(int(target + (actual - target) + target * domain_max), actual + 1)

        # Build segments
        segment_dict = {1: {'x1': 0, 'y1': 0, 'x2': target, 'y2': 1, 'rho': -rho1},
                        2: {'x1': maximum, 'y1': 1, 'x2': actual, 'y2': buffer_y, 'rho': rho2},
                        3: {'x1': actual, 'y1': buffer_y, 'x2': real_max, 'y2': 0, 'rho': -rho3}}

    elif f_type == 'Quota_Direct':  # The benefit of value functions is here! Captures ambiguity of PGL/constraints

        # Receive values from string
        f_param_list = split_list[1]
        split_list = f_param_list.split(',')
        domain_max = 1.05  # Arbitrary max (this doesn't really matter since we constrain quota anyway)
        rho1 = float(split_list[0].strip())
        rho2 = float(split_list[1].strip())
        rho3 = float(split_list[2].strip())
        rho4 = float(split_list[3].strip())
        y1 = float(split_list[4].strip())
        y2 = float(split_list[5].strip())

        if len(split_list) == 7:
            pref_target = int(split_list[6].strip())
        else:
            pref_target = target

        # Build segments
        if pref_target == minimum:
            if pref_target == maximum:
                segment_dict = {1: {'x1': 0, 'y1': 0, 'x2': pref_target, 'y2': 1, 'rho': -(rho1 * pref_target)},
                                2: {'x1': maximum, 'y1': 1, 'x2': domain_max * maximum, 'y2': 0,
                                    'rho': -(rho4 * (maximum - pref_target))}}
            else:
                segment_dict = {1: {'x1': 0, 'y1': 0, 'x2': pref_target, 'y2': 1, 'rho': -(rho1 * pref_target)},
                                2: {'x1': pref_target, 'y1': 1, 'x2': maximum, 'y2': y2,
                                    'rho': (rho3 * (maximum - pref_target))},
                                3: {'x1': maximum, 'y1': y2, 'x2': domain_max * maximum, 'y2': 0,
                                    'rho': -(rho4 * (domain_max * maximum - maximum))}}

        else:
            if pref_target == maximum:
                segment_dict = {1: {'x1': 0, 'y1': 0, 'x2': minimum, 'y2': y1, 'rho': -(rho1 * (minimum - 0))},
                                2: {'x1': minimum, 'y1': y1, 'x2': pref_target, 'y2': 1,
                                    'rho': (rho2 * (pref_target - minimum))},
                                3: {'x1': pref_target, 'y1': 1, 'x2': domain_max * maximum, 'y2': 0,
                                    'rho': -(rho4 * (domain_max * maximum - maximum))}}
            else:
                segment_dict = {1: {'x1': 0, 'y1': 0, 'x2': minimum, 'y2': y1, 'rho': -(rho1 * (minimum - 0))},
                                2: {'x1': minimum, 'y1': y1, 'x2': pref_target, 'y2': 1,
                                    'rho': (rho2 * (pref_target - minimum))},
                                3: {'x1': pref_target, 'y1': 1, 'x2': maximum, 'y2': y2,
                                    'rho': (rho3 * (maximum - pref_target))},
                                4: {'x1': maximum, 'y1': y2, 'x2': domain_max * maximum, 'y2': 0,
                                    'rho': -(rho4 * (domain_max * maximum - maximum))}}

    else:  # Must be a "Min Increasing/Decreasing" function

        # Receive values from string
        f_param_list = split_list[1]
        split_list = f_param_list.split(',')
        rho = float(split_list[0].strip())

        # Build segments
        if f_type == 'Min Increasing':
            segment_dict = {1: {'x1': 0, 'y1': 0, 'x2': target, 'y2': 1, 'rho': -rho}}
        else:
            segment_dict = {1: {'x1': target, 'y1': 1, 'x2': 1, 'y2': 0, 'rho': -rho}}

    return segment_dict


def value_function_builder(segment_dict=None, num_breakpoints=None, derivative_locations=False):
    """
    This procedure takes in a dictionary of exponential segments and returns the breakpoints (measures and values) for
    that value function.
    :param derivative_locations: if we want to place breakpoints at locations where derivative increases
    by some interval
    :param segment_dict: (x1, y1, x2, y2, rho, and optional "r": number of breakpoints per segment)
    :param num_breakpoints: if num_breakpoints is not specified within segment array, we have a
    general number of breakpoints to go off of
    :return: a, f^hat  (breakpoint measures and values)
    """
    if segment_dict is None:
        segment_dict = {1: {'x1': 0, 'y1': 0, 'x2': 0.2, 'y2': 0.8, 'rho': -0.2, 'r': 10},
                        2: {'x1': 0.2, 'y1': 0.8, 'x2': 0.3, 'y2': 1, 'rho': 0.2, 'r': 10},
                        3: {'x1': 0.5, 'y1': 1, 'x2': 0.7, 'y2': 0.8, 'rho': 0.2, 'r': 10},
                        4: {'x1': 0.7, 'y1': 0.8, 'x2': 1, 'y2': 0, 'rho': -0.2, 'r': 10}}

    # Collect segments
    segments = list(segment_dict.keys())
    num_segments = len(segments)

    # We need number of breakpoints for each exponential segment
    if num_breakpoints is None:
        num_breakpoints = 0
        for segment in segments:
            if 'r' in segment_dict[segment].keys():
                segment_dict[segment]['r'] = segment_dict[segment]['r']
            else:
                segment_dict[segment]['r'] = 10

            num_breakpoints += segment_dict[segment]['r']

    else:
        new_num_breakpoints = 0
        for segment in segments:
            if 'r' in segment_dict[segment].keys():
                segment_dict[segment]['r'] = max(int(num_breakpoints / num_segments), segment_dict[segment]['r'])
            else:
                segment_dict[segment]['r'] = int(num_breakpoints / num_segments)
            new_num_breakpoints += segment_dict[segment]['r']
        num_breakpoints = new_num_breakpoints

    # Number of breakpoints are determined based on which kinds of exponential segments we're using
    add_bp = False
    extra = False
    insert = False
    if (num_segments == 4 and segment_dict[2]['x2'] != segment_dict[3]['x1']) or num_segments == 2:
        a = np.zeros(num_breakpoints + 2)
        fhat = np.zeros(num_breakpoints + 2)
        extra = True
    elif num_segments == 1 and segment_dict[1]['x2'] < 1:
        a = np.zeros(num_breakpoints + 2)
        fhat = np.zeros(num_breakpoints + 2)
        add_bp = True
    elif num_segments == 1 and segment_dict[1]['x1'] != 0:
        a = np.zeros(num_breakpoints + 2)
        fhat = np.zeros(num_breakpoints + 2)
        fhat[0] = 1
        insert = True
    else:
        a = np.zeros(num_breakpoints + 1)
        fhat = np.zeros(num_breakpoints + 1)

    # Loop through each exponential segment
    i = 1
    for segment in segments:

        # Load variables
        x1 = segment_dict[segment]['x1']
        y1 = segment_dict[segment]['y1']
        x2 = segment_dict[segment]['x2']
        y2 = segment_dict[segment]['y2']
        rho = segment_dict[segment]['rho']
        r = segment_dict[segment]['r']

        # Necessary operations
        x_diff = x2 - x1
        y_diff = y2 - y1
        x_over_y = abs(x_diff / y_diff)
        if y_diff < 0:
            positive = False
        else:
            positive = True

        if derivative_locations:  # We place x breakpoints at fixed intervals based on derivative
            y_prime_i = round(derivative_function(0, x_over_y, rho, positive), 2)
            y_prime_f = round(derivative_function(x_over_y, x_over_y, rho, positive), 2)
            y_prime_step = (y_prime_f - y_prime_i) / r
            y_prime_arr = np.array([y_prime_i + y_prime_step * i for i in range(1, r + 1)])
            x_arr = np.array([
                inverse_derivative_function(y_prime, x_over_y, rho, positive) for y_prime in y_prime_arr])

        else:  # We place them at fixed intervals based on x
            x_arr = (np.arange(1, r + 1) / r) * x_over_y

        # Get the y-values of the corresponding x-values
        vals = np.array([exponential_function(x, 0, x_over_y, rho, positive) for x in x_arr])

        # If we need to add extra breakpoints
        if y1 == 1 and extra:
            a[i] = x1
            fhat[i] = 1
            i += 1

        if insert:
            a[1] = x1
            fhat[1] = 1
            a[2:2 + r] = x1 + x_arr * abs(y_diff)
            fhat[2:2 + r] = y2 + vals * abs(y_diff)
        else:
            a[i:i + r] = x1 + x_arr * abs(y_diff)
            if positive:
                fhat[i:i + r] = y1 + vals * y_diff
            else:
                fhat[i:i + r] = y2 + vals * abs(y_diff)
            i += r

            if add_bp:
                a[r + 1] = 1
                fhat[r + 1] = 1

    # Return breakpoint measures and values used in value function
    a = np.around(a, 5)
    fhat = np.around(fhat, 5)
    return a, fhat


def exponential_function(x, x_i, x_f, rho, positive):
    """
    This function returns the value obtained from the specified exponential value function
    :param x: current x
    :param x_i: initial x from segment
    :param x_f: final x from segment
    :param rho: rho parameter
    :param positive: if we have an increasing function or not
    :return: current y
    """
    if positive:
        y = (1 - exp(-(x - x_i) / rho)) / \
            (1 - exp(-(x_f - x_i) / rho))
    else:

        y = (1 - exp(-(x_f - x) / rho)) / \
            (1 - exp(-(x_f - x_i) / rho))

    return y


def derivative_function(x, x_f, rho, positive):
    """
    This function calculates the derivative of x for some point
    along a line segment
    :param x: some x
    :param x_f: final x from segment
    :param rho: rho parameter
    :param positive: if we have an increasing function or not
    :return: y_prime
    """
    if positive:
        y_prime = exp(-x / rho) / (rho * (1 - exp(-x_f / rho)))
    else:
        y_prime = -exp(-(x_f - x) / rho) / (rho * (1 - exp(-x_f / rho)))
    return y_prime


def inverse_derivative_function(y_prime, x_f, rho, positive):
    """
    This function calculates the position of x based on its derivative
    :param y_prime: first derivative of x
    :param x_f: final x from segment
    :param rho: rho parameter
    :param positive: if we have an increasing function or not
    :return: x
    """
    if positive:
        x = -rho * log(rho * (1 - exp(-x_f / rho)) * y_prime)
    else:
        x = x_f + rho * log(-rho * (1 - exp(-x_f / rho)) * y_prime)

    return x


def condense_value_functions(parameters, value_parameters):
    """
    This procedure takes an instances' value functions and removes all unnecessary zeros in the values
    :param parameters: fixed cadet parameters
    :param value_parameters: weight and value parameters
    :return: value parameters with cleaned value functions
    """
    for j in parameters['J']:
        for k in value_parameters['K^A'][j]:
            a = np.array(value_parameters['a'][j][k])
            fhat = np.array(value_parameters['f^hat'][j][k])

            # Find unnecessary zeros
            zero_indices = np.where(fhat == 0)[0]
            last_i = len(a) - 1
            removals = []
            for i in zero_indices:
                if i + 1 in zero_indices and i + 1 != last_i and i != 0:
                    removals.append(i)

            # Remove unnecessary zeros
            value_parameters['a'][j][k] = np.delete(a, removals)
            value_parameters['f^hat'][j][k] = np.delete(fhat, removals)

    return value_parameters


# Rebecca's Model Parameter Translation
def translate_vft_to_gp_parameters(parameters, value_parameters, gp_df=None, use_gp_df=True, printing=False):
    """
    This function translates the VFT parameters to Rebecca's model's parameters
    :param use_gp_df: if we want to obtain the rewards and penalties for this instance
    :param printing: Whether or not to print a status update
    :param gp_df: dataframe of parameters used in Rebecca's model
    :param parameters: fixed cadet/AFSC parameters
    :param value_parameters: weight and value parameters
    :return: gp_parameters
    """
    if printing:
        print('Translating VFT model parameters to Goal Programming Model parameters...')

    if use_gp_df:
        if gp_df is None:
            filepath = support_paths['scrubbed'] + 'gp_parameters.xlsx'
            gp_df = import_data(filepath=filepath, sheet_name='Weights and Scaling')

    # Shorthand
    p = parameters
    vp = value_parameters
    objectives = vp['objectives']

    # Other parameters
    large_afscs = np.where(p['quota'] >= 40)[0]  # set of large AFSCs
    mand_k = np.where(objectives == 'Mandatory')[0][0]  # mandatory objective index
    des_k = np.where(objectives == 'Desired')[0][0]  # desired objective index
    perm_k = np.where(objectives == 'Permitted')[0][0]  # permitted objective index
    usafa_k = np.where(objectives == 'USAFA Proportion')[0][0]  # USAFA proportion objective index

    # Initialize "gp" dictionary (Goal Programming Model Parameters)
    gp = {}

    # Main sets
    A = np.arange(p['M'])
    C = np.arange(p['N'])
    gp['A'] = A  # AFSCs
    gp['C'] = C  # Cadets

    # List of constraints
    gp['con'] = ['T',  # Target constraint
                 'F',  # Over-classification constraint
                 'M',  # Mandatory education constraint
                 'D_under',  # Desired education constraint (lower bound)
                 'D_over',  # Desired education constraint (upper bound)
                 'P',  # Permitted education constraint
                 'U_under',  # USAFA proportion constraint (lower bound)
                 'U_over',  # USAFA proportion constraint (upper bound)
                 'R_under',  # Percentile constraint (lower bound)
                 'R_over',  # Percentile constraint (upper bound)
                 'W']  # Cadet preference constraint

    # Subsets of AFSCs that pertain to each constraint (1 dimensional arrays)
    gp['A^'] = {'T': A,  # Subset of AFSCs with a minimum target quota: assumed all AFSCs
                'F': A,  # Subset of AFSCs with over-classification limits: assumed all AFSCs
                'M': np.array([a for a in A if 'Mandatory' in objectives[vp['K^A'][a]]]),  # Mandatory AFSCs
                'D_under': np.array([a for a in A if 'Increasing' in vp['value_functions'][a, des_k]]),  # Desired AFSCs
                'D_over': np.array([a for a in A if 'Decreasing' in vp['value_functions'][a, des_k]]),  # Desired AFSCs
                'P': np.array([a for a in A if 'Permitted' in objectives[vp['K^A'][a]]]),  # Permitted AFSCs
                'U_under': large_afscs,  # USAFA Proportion constrained AFSCs
                'U_over': large_afscs,  # USAFA Proportion constrained AFSCs
                'R_under': large_afscs,  # Percentile constrained AFSCs
                'R_over': large_afscs,  # Percentile constrained AFSCs
                'W': A}  # Subset of AFSCs with a cadet preference constraint: assumed all AFSCs

    # Subset of AFSCs for which each cadet is eligible (Replaced A | A^I)
    gp['A^']['E'] = p['J^E']

    # Subset of AFSCs that each cadet has placed a preference for
    A_Utility = [np.where(p['utility'][c, :] > 0)[0] for c in C]

    # Subset of AFSCs that each cadet has placed a preference for and is also eligible for
    gp['A^']['W^E'] = [np.intersect1d(A_Utility[c], gp['A^']['E'][c]) for c in C]

    # Subset of AFSCs which have an upper bound on the number of USAFA cadets
    gp['A^']['U_lim'] = np.array([a for a in A if ',' in vp['objective_value_min'][a, usafa_k]])

    # Set of cadets that have placed preferences on each of the AFSCs
    C_Utility = [np.where(p['utility'][:, a] > 0)[0] for a in A]

    # Subsets of Cadets that pertain to each constraint  (2 dimensional arrays)
    gp['C^'] = {'T': p['I^E'],  # Eligible Cadets for each AFSC
                'F': p['I^E'],  # Eligible Cadets for each AFSC
                'M': p['I^D']['Mandatory'],  # Cadets that have mandatory degrees for each AFSC
                'D_under': p['I^D']['Desired'],  # Cadets that have desired degrees for each AFSC
                'D_over': p['I^D']['Desired'],  # Cadets that have desired degrees for each AFSC
                'P': p['I^D']['Permitted'],  # Cadets that have permitted degrees for each AFSC
                'U_under': p['I^D']['USAFA Proportion'],  # Eligible USAFA Cadets for each AFSC
                'U_over': p['I^D']['USAFA Proportion'],  # Eligible USAFA Cadets for each AFSC
                'R_under': p['I^E'],  # Eligible Cadets for each AFSC
                'R_over': p['I^E'],  # Eligible Cadets for each AFSC

                # Eligible Cadets that have placed preferences for each AFSC
                'W': [np.intersect1d(C_Utility[a], p['I^E'][a]) for a in A]}

    # Subset of eligible cadets for each AFSC
    gp['C^']['E'] = p['I^E']

    # Subset of eligible usafa cadets for each AFSC
    gp['C^']['U'] = p['I^D']['USAFA Proportion']

    # Parameters for each of the constraints (1 dimensional arrays)
    gp['param'] = {'T': p['quota'],  # Target quotas
                   'F': p['quota_max'],  # Over-classification amounts
                   'M': vp['objective_target'][:, mand_k],  # Mandatory targets
                   'D_under': vp['objective_target'][:, des_k],  # Desired targets
                   'D_over': vp['objective_target'][:, des_k],  # Desired targets
                   'P': vp['objective_target'][:, perm_k],  # Permitted targets
                   'U_under': np.repeat(0.2, p['M']),  # USAFA Proportion lower bound
                   'U_over': np.repeat(0.4, p['M']),  # USAFA Proportion upper bound
                   'R_under': np.repeat(0.35, p['M']),  # Percentile lower bound
                   'R_over': np.repeat(0.65, p['M']),  # Percentile upper bound
                   'W': np.repeat(0.5, p['M'])}  # Cadet preference lower bound

    # Other parameters
    gp['utility'] = p['utility']  # utility matrix (replaced "w" since "w" was already a parameter in her model)
    gp['Big_M'] = 2000  # sufficiently large number
    gp['u_limit'] = 0.05  # limit on number of USAFA cadets for certain AFSCs
    gp['merit'] = p['merit']  # cadet percentiles

    # Penalty and Reward parameters
    if use_gp_df:
        columns = ['Normalized Penalty', 'Normalized Reward', 'Run Penalty', 'Run Reward']
        column_dict = {column: np.array(gp_df[column]) for column in columns}

        # actual reward parameters
        reward = column_dict['Normalized Reward'] * column_dict['Run Reward']

        # actual penalty parameters
        penalty = column_dict['Normalized Penalty'] * column_dict['Run Penalty']

        # mu parameters (Penalties)
        gp['mu^'] = {con: penalty[index] for index, con in enumerate(gp['con'])}

        # lambda parameters (Rewards)
        gp['lam^'] = {con: reward[index] for index, con in enumerate(gp['con'])}
        gp['lam^']['S'] = reward[len(gp['con'])]  # extra reward for preference in order of merit

    if printing:
        print('Translated.')

    return gp

