import numpy as np
import pandas as pd
from math import *

# afccp modules
import afccp.core.globals


# Value Parameter Procedures
def value_parameters_sets_additions(parameters, value_parameters, printing=False):
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

    # Shorthand
    p, vp = parameters, value_parameters

    # Temporary manual adjustment of constraint_type matrix
    indices_3 = np.where(vp['constraint_type'] == 3)
    indices_4 = np.where(vp['constraint_type'] == 4)
    vp['constraint_type'][indices_3] = 1
    vp['constraint_type'][indices_4] = 2

    # Set of Objectives
    vp['K'] = np.arange(vp["O"])

    # Set of objectives for each AFSC
    vp['K^A'] = {}  # objectives
    vp['K^C'] = {}  # constrained objectives
    for j in p["J"]:
        vp['K^A'][j] = np.where(vp['objective_weight'][j, :] > 0)[0].astype(int)
        vp['K^C'][j] = np.where(vp['constraint_type'][j, :] > 0)[0].astype(int)

    # 5% of total USAFA graduating class set
    vp["J^USAFA"] = None

    # Add the AFSC indices to the set
    if "USAFA-Constrained AFSCs" in vp:
        if "," in vp["USAFA-Constrained AFSCs"]:
            vp["J^USAFA"] = np.array([])
            usafa_afscs = vp["USAFA-Constrained AFSCs"].split(",")
            for afsc in usafa_afscs:
                afsc = afsc.strip()
                j = np.where(p["afscs"] == afsc)[0]
                if len(j) == 0:
                    print("WARNING: Something is wrong with the USAFA-Constrained AFSCs! "
                          "'" + afsc + "' is not in the list of AFSCs.")
                else:
                    vp["J^USAFA"] = np.hstack((vp["J^USAFA"], j))
            vp["J^USAFA"] = vp["J^USAFA"].astype(int)

    # USSF OM Constraint
    if vp['USSF OM'] == 'True':
        vp['USSF OM'] = True
    else:
        vp['USSF OM'] = False

    # Set of objectives that seek to balance some cadet demographic
    vp['K^D'] = ['USAFA Proportion', 'Mandatory', 'Desired', 'Permitted', 'Male', 'Minority',
                 'Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']

    # Set of AFSCs for each objective:
    vp['J^A'] = {}
    for k in range(vp["O"]):
        vp['J^A'][k] = np.where(vp['objective_weight'][:, k] > 0)[0].astype(int)

    # Cadet Value Constraint Set
    vp['I^C'] = np.where(vp['cadet_value_min'] > 0)[0]

    # AFSC Value Constraint Set
    vp['J^C'] = np.where(vp['afsc_value_min'] > 0)[0]

    # number of breakpoints
    vp['r'] = np.array([[len(vp['a'][j][k]) for k in vp["K"]] for j in p["J"]])

    # set of breakpoints
    vp['L'] = np.array([[np.arange(vp['r'][j, k]) for k in range(vp["O"])] for j in range(p["M"])], dtype=object)

    # Round weights
    vp['objective_weight'] = np.around(vp['objective_weight'], 8)
    vp['afsc_weight'] = np.around(vp['afsc_weight'], 8)
    vp['cadet_weight'] = np.around(vp['cadet_weight'], 8)

    # Extract AFSC objective min/max measures
    vp["objective_min"], vp["objective_max"] = np.zeros([p['M'], vp['O']]), np.zeros([p["M"], vp["O"]])
    for j in p['J']:
        for k in vp['K^C'][j]:
            value_list = vp['objective_value_min'][j, k].split(",")
            vp["objective_min"][j, k] = float(value_list[0].strip())
            vp["objective_max"][j, k] = float(value_list[1].strip())

    # "Global Utility" matrix
    if "afsc_utility" in p and "cadet_utility" in p:
        vp['global_utility'] = np.zeros([p['N'], p['M'] + 1])
        for j in p['J']:
            vp['global_utility'][:, j] = vp['cadets_overall_weight'] * p['cadet_utility'][:, j] + \
                                         vp['afscs_overall_weight'] * p['afsc_utility'][:, j]

    return vp


def model_value_parameters_to_defaults(instance, filepath, printing=False):
    """
    This function takes the current instance value parameters and exports them as defaults back to excel
    """
    if printing:
        print('Exporting value parameters as defaults to excel...')

    # Shorthand
    p, vp = instance.parameters, instance.value_parameters

    # Create "Overall Weights" dataframe
    overall_weights_df = pd.DataFrame({'Cadets Weight': [vp['cadets_overall_weight']],
                                       'AFSCs Weight': [vp['afscs_overall_weight']],
                                       'Cadets Min Value': [vp['cadets_overall_value_min']],
                                       'AFSCs Min Value': [vp['afscs_overall_value_min']],
                                       'Cadet Weight Function': [vp['cadet_weight_function']],
                                       'AFSC Weight Function': [vp['afsc_weight_function']],
                                       "USAFA-Constrained AFSCs": vp["USAFA-Constrained AFSCs"],
                                       "Cadets Top 3 Constraint": vp["Cadets Top 3 Constraint"],
                                       "USSF OM": vp["USSF OM"]})

    # Construct other dataframes
    afsc_weights_df = pd.DataFrame({'AFSC': p['afscs'][:p["M"]],
                                    'AFSC Swing Weight': np.around((vp['afsc_weight'] / max(vp['afsc_weight'])) * 100, 2),
                                    'AFSC Min Value': vp['afsc_value_min']})

    # AFSC Objective Components Translations Dictionary
    ao_trans_dict = {'AFSC Objective Weights': 'objective_weight', 'AFSC Objective Targets': 'objective_target',
                     'AFSC Objective Min Value': 'objective_value_min', 'Constraint Type': 'constraint_type',
                     'Value Functions': 'value_functions'}

    # Create the AFSC Objective Components DataFrames
    ao_dfs = {component: pd.DataFrame({'AFSC': p['afscs'][:p["M"]]}) for component in ao_trans_dict}
    for component in ao_dfs:

        # Scale AFSC Objective Weights so that the largest is set to 100 rather than forcing them all to sum to 1
        if component == "AFSC Objective Weights":
            comp_arr = np.array([(vp["objective_weight"][j] / max(vp["objective_weight"][j])) * 100 for j in p["J"]])
        else:
            comp_arr = vp[ao_trans_dict[component]]  # "Component Array", vp["objective_target"] for example

        # Load columns of the dataframe
        for k, objective in enumerate(vp['objectives']):
            ao_dfs[component][objective] = comp_arr[:, k]

    # Export to excel
    with pd.ExcelWriter(filepath) as writer:
        overall_weights_df.to_excel(writer, sheet_name="Overall Weights", index=False)
        afsc_weights_df.to_excel(writer, sheet_name="AFSC Weights", index=False)
        for component in ao_dfs:
            ao_dfs[component].to_excel(writer, sheet_name=component, index=False)


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
    overall_weights_df = afccp.core.globals.import_data(filepath, sheet_name="Overall Weights")
    afsc_weights_df = afccp.core.globals.import_data(filepath, sheet_name="AFSC Weights")
    afsc_objective_weights_df = afccp.core.globals.import_data(filepath, sheet_name="AFSC Objective Weights")
    afsc_objective_targets_df = afccp.core.globals.import_data(filepath, sheet_name="AFSC Objective Targets")
    afsc_objective_value_min_df = afccp.core.globals.import_data(filepath, sheet_name="AFSC Objective Min Value")
    afsc_objective_convex_constraints_df = afccp.core.globals.import_data(filepath, sheet_name="Constraint Type")
    afsc_value_functions_df = afccp.core.globals.import_data(filepath, sheet_name="Value Functions")
    objectives = np.array(afsc_objective_weights_df.keys()[1:])
    default_value_parameters = {'cadet_weight_function': overall_weights_df['Cadet Weight Function'][0],
                                'afsc_weight_function': overall_weights_df['AFSC Weight Function'][0],
                                'cadets_overall_weight': overall_weights_df['Cadets Weight'][0],
                                'afscs_overall_weight': overall_weights_df['AFSCs Weight'][0],
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
                                'complete_afscs': np.array(afsc_weights_df['AFSC']),
                                'num_breakpoints': num_breakpoints, "M": len(afsc_weights_df)}

    # Check if other columns are present (phasing these in)
    more_vp_columns = ["USAFA-Constrained AFSCs", "Cadets Top 3 Constraint", "USSF OM"]
    for col in more_vp_columns:
        if col in overall_weights_df:
            default_value_parameters[col] = str(overall_weights_df.loc[0, col]).replace("nan", "")
        else:
            default_value_parameters[col] = ""
    return default_value_parameters


def generate_afocd_value_parameters(parameters, default_value_parameters):
    """
    Function to add in AFOCD value parameters
    """
    p, vp = parameters, default_value_parameters

    # AFOCD Objective Indices
    tier_objectives = np.array([np.where(vp["objectives"] == "Tier " + t)[0][0] for t in ["1", "2", "3", "4"]])

    # Tier multipliers
    tm = {1: 1, 2: 0.8, 3: 0.6, 4: 0.4}

    # Loop through each AFSC
    for j in p["J"]:

        # Loop through each AFOCD tier
        for t, k in enumerate(tier_objectives):
            tier = t + 1

            # We only do this for valid AFOCD tiers
            if tier <= p["t_count"][j]:

                # Objective Weight
                if p["t_mandatory"][j, t] == 1:
                    vp["objective_weight"][j, k] = round(90 * tm[tier], 2)
                elif p["t_desired"][j, t] == 1:
                    vp["objective_weight"][j, k] = round(70 * tm[tier], 2)
                elif p["t_permitted"][j, t] == 1:
                    vp["objective_weight"][j, k] = round(50 * tm[tier], 2)
                else:
                    vp["objective_weight"][j, k] = 0  # Tier doesn't exist or is an ineligible tier

                # Objective Target
                vp["objective_target"][j, k] = p["t_proportion"][j, t]

                # Objective Min Value
                if p["t_leq"][j, t] == 1:
                    vp["objective_value_min"][j, k] = "0, " + str(p["t_proportion"][j, t])
                else:  # <= OR ==
                    vp["objective_value_min"][j, k] = str(p["t_proportion"][j, t]) + ", 5"

                # Constraint Type (Default to turning all constraints on. It's easier to make them zeros later...)
                if p["t_mandatory"][j, t] == 1:
                    vp["constraint_type"][j, k] = 1  # Easier to meet "at least" constraint (M >/= x) based on PGL
                elif p["t_desired"][j, t] == 1:
                    if p["t_leq"][j, t] == 1:
                        vp["constraint_type"][j, k] = 2  # Easier to meet "at most" constraint (D < x) based on proportion
                    elif p["t_geq"][j, t] == 1:
                        vp["constraint_type"][j, k] = 1  # Easier to meet "at least" constraint (D > x) based on PGL
                elif p["t_permitted"][j, t] == 1:
                    vp["constraint_type"][j, k] = 2  # Easier to meet "at most" constraint (P < x) based on proportion

                # Value Functions
                if p['t_leq'][j, t] == 1:
                    vp['value_functions'][j, k] = "Min Decreasing|0.3"
                else:  # <= OR ==
                    vp['value_functions'][j, k] = "Min Increasing|0.3"

    return vp


def generate_value_parameters_from_defaults(parameters, default_value_parameters, generate_afsc_weights=True,
                                            num_breakpoints=None, printing=False):
    """
    Generates value parameters from the defaults for a specified problem
    """
    if printing:
        print('Generating value parameters from defaults...')

    # Shorthand
    p, dvp = parameters, default_value_parameters

    # Variable to control if we load in AFOCD value parameters from the default excel workbook or create them here
    init_afocd = False

    # Manipulate AFOCD objectives based on what "system" we're using (Tiers or "Old")
    if p["Qual Type"] == "Tiers":

        # Weight "Old" AFOCD Objectives at zero (but keep them in because we can select them if needed)
        for objective in ["Mandatory", "Desired", "Permitted"]:
            if objective in dvp["objectives"]:
                k = np.where(dvp["objectives"] == objective)[0][0]
                dvp["objective_weight"][:, k] = np.zeros(len(dvp["objective_weight"]))

        # Add in "Tier" AFOCD Objectives
        for t in ["1", "2", "3", "4"]:
            objective = "Tier " + t
            if objective not in dvp["objectives"]:
                init_afocd = True  # We're going to initialize the value parameters for AFOCD objectives
                dvp["objectives"] = np.hstack([dvp["objectives"], objective])

                # Add these columns in the other arrays as well
                for vp_key in ["objective_weight", "objective_value_min", "constraint_type", "objective_target",
                               "value_functions"]:
                    if vp_key in ["objective_weight", "constraint_type", "objective_target"]:
                        new_column = np.array([[0] for _ in p["J"]])
                    else:
                        new_column = np.array([["0"] for _ in p["J"]])
                    dvp[vp_key] = np.hstack((dvp[vp_key], new_column))

    elif p["Qual Type"] == "Relaxed":

        # Remove "Tier" AFOCD Objectives
        for t in ["1", "2", "3", "4"]:
            objective = "Tier " + t
            if objective in dvp["objectives"]:
                k = np.where(dvp["objectives"] == objective)[0][0]
                dvp["objectives"] = np.delete(dvp["objectives"], k)

    # Generate AFOCD value parameters if necessary
    if init_afocd:
        dvp = generate_afocd_value_parameters(p, dvp)

    # Objective to parameters lookup dictionary (if the parameter is in "p", we include the objective)
    objective_lookups = {'Norm Score': 'a_pref_matrix', 'Merit': 'merit', 'USAFA Proportion': 'usafa',
                         'Combined Quota': 'quota_d', 'USAFA Quota': 'usafa_quota', 'ROTC Quota': 'rotc_quota',
                         'Mandatory': 'mandatory', 'Desired': 'desired', 'Permitted': 'permitted',
                         'Utility': 'utility', 'Male': 'male', 'Minority': 'minority'}
    for t in ["1", "2", "3", "4"]:  # Add in AFOCD Degree tiers
        objective_lookups["Tier " + t] = "tier " + t

    # Add the AFSC objectives that are included in this instance (check corresponding parameters using dict above)
    objectives = []
    objective_indices = []
    for k, objective in enumerate(dvp["objectives"]):
        if objective_lookups[objective] in p:
            objectives.append(objective)
            objective_indices.append(k)

    # Convert to numpy arrays
    objectives = np.array(objectives)
    objective_indices = np.array(objective_indices)

    # Additional information
    afsc_indices = np.array([np.where(dvp['complete_afscs'] == p['afscs'][j])[0][0] for j in p["J"]])
    O = len(objectives)
    if num_breakpoints is None:
        num_breakpoints = dvp['num_breakpoints']

    # Initialize set of value p
    vp = {'cadets_overall_weight': dvp['cadets_overall_weight'],
          'afscs_overall_weight': dvp['afscs_overall_weight'],
          'cadet_weight_function': dvp['cadet_weight_function'],
          'afsc_weight_function': dvp['afsc_weight_function'],
          'cadets_overall_value_min': dvp['cadets_overall_value_min'],
          'afscs_overall_value_min': dvp['afscs_overall_value_min'],
          'afsc_value_min': np.zeros(p["M"]), 'cadet_value_min': np.zeros(p["N"]),
          'objective_weight': np.zeros([p["M"], O]), 'afsc_weight': np.zeros(p["M"]), "M": p["M"],
          'objective_target': np.zeros([p["M"], O]), 'objectives': objectives, 'O': O,
          'objective_value_min': np.array([[" " * 20 for _ in range(O)] for _ in p["J"]]),
          'constraint_type': np.zeros([p["M"], O]).astype(int),
          "USAFA-Constrained AFSCs": dvp["USAFA-Constrained AFSCs"],
          "Cadets Top 3 Constraint": dvp["Cadets Top 3 Constraint"],
          "USSF OM": dvp["USSF OM"],
          'num_breakpoints': num_breakpoints}

    # Determine weights on cadets
    if 'merit_all' in p:
        vp['cadet_weight'] = cadet_weight_function(p['merit_all'], func=vp['cadet_weight_function'])
    else:
        vp['cadet_weight'] = cadet_weight_function(p['merit'], func=vp['cadet_weight_function'])

    # Determine weights on AFSCs
    if generate_afsc_weights:
        func = vp['afsc_weight_function']
        if func == 'Custom':  # We take the AFSC weights directly from the defaults
            generate_afsc_weights = False
        else:
            vp['afsc_weight'] = afsc_weight_function(p["pgl"], func)

    # Initialize breakpoints
    vp['a'] = [[[] for _ in range(O)] for _ in p["J"]]
    vp['f^hat'] = [[[] for _ in range(O)] for _ in p["J"]]

    # Load value function strings
    value_functions = dvp['value_functions'][:, objective_indices]
    value_functions = value_functions[afsc_indices, :]
    vp['value_functions'] = value_functions

    # Initialize objective set
    vp['K^A'] = {}

    # Loop through each AFSC to load in their value parameters
    for j, afsc in enumerate(p['afscs']):

        if afsc != "*":

            # Get location of afsc in the default value parameters (matters if this set of afscs does not match)
            loc = np.where(dvp['complete_afscs'] == afsc)[0][0]

            # Initially assign all default weights, targets, etc.
            vp['objective_weight'][j, :] = dvp['objective_weight'][loc, objective_indices]
            vp['objective_target'][j] = dvp['objective_target'][loc, objective_indices]
            vp['objective_value_min'][j] = dvp['objective_value_min'][loc, objective_indices]
            vp['afsc_value_min'][j] = dvp['afsc_value_min'][loc]
            vp['constraint_type'][j] = dvp['constraint_type'][loc, objective_indices]
            vp['K^A'][j] = np.where(vp['objective_weight'][j, :] > 0)[0].astype(int)

            # If we're not generating afsc weights using the specified weight function...
            if not generate_afsc_weights:  # Also, if the weight function is "Custom"
                vp['afsc_weight'][j] = dvp['afsc_weight'][loc]

            # Loop through each objective to load their targets
            for k, objective in enumerate(vp['objectives']):

                maximum, minimum, actual = None, None, None
                if objective == 'Merit':
                    vp['objective_target'][j, k] = p['sum_merit'] / p['N']
                    actual = np.mean(p['merit'][p['I^E'][j]])

                elif objective == 'USAFA Proportion':
                    vp['objective_target'][j, k] = p['usafa_proportion']
                    if len(p['I^E'][j]) == 0:  # In case there are no eligible USAFA cadets
                        actual = 0
                    else:
                        actual = len(p['I^D'][objective][j]) / len(p['I^E'][j])

                elif objective == 'Combined Quota':
                    vp['objective_target'][j, k] = p['quota_d'][j]

                    # Get bounds
                    minimum, maximum = p['quota_min'][j], p['quota_max'][j]
                    vp['objective_value_min'][j, k] = str(int(minimum)) + ", " + str(int(maximum))

                elif objective == 'USAFA Quota':
                    vp['objective_target'][j, k] = p['usafa_quota'][j]
                    vp['objective_value_min'][j, k] = str(int(p['usafa_quota'][j])) + ", " + \
                                                                    str(int(p['quota_max'][j]))

                elif objective == 'ROTC Quota':
                    vp['objective_target'][j, k] = p['pgl'][j] - p['usafa_quota'][j]
                    vp['objective_value_min'][j, k] = str(int(p['rotc_quota'][j])) + ", " + \
                                                                    str(int(p['quota_max'][j]))

                elif objective == 'Male':
                    vp['objective_target'][j, k] = p['male_proportion']
                    if len(p['I^E'][j]) == 0:  # In case there are no eligible male cadets
                        actual = 0
                    else:
                        actual = len(p['I^D'][objective][j]) / len(p['I^E'][j])

                elif objective == 'Minority':
                    vp['objective_target'][j, k] = p['minority_proportion']
                    if len(p['I^E'][j]) == 0:  # In case there are no eligible minority cadets
                        actual = 0
                    else:
                        actual = len(p['I^D'][objective][j]) / len(p['I^E'][j])

                # If we care about this objective, we load in its value function breakpoints
                if vp['objective_weight'][j, k] != 0:

                    # Create the non-linear piecewise exponential segment dictionary
                    segment_dict = create_segment_dict_from_string(value_functions[j, k],
                                                                   vp['objective_target'][j, k],
                                                                   minimum=minimum, maximum=maximum, actual=actual)

                    # Linearize the non-linear function using the specified number of breakpoints
                    vp['a'][j][k], vp['f^hat'][j][k] = value_function_builder(
                        segment_dict, num_breakpoints=num_breakpoints)

            # Scale the objective weights for this AFSC, so they sum to 1
            vp['objective_weight'][j] = vp['objective_weight'][j] / sum(vp['objective_weight'][j])

    # Scale the weights across all AFSCs, so they sum to 1
    vp['afsc_weight'] = vp['afsc_weight'] / sum(vp['afsc_weight'])

    return vp


def update_value_and_weight_functions(instance, num_breakpoints=None):
    """
    This function takes in a problem instance and updates the set of value parameters in case the user
    makes a change to some aspect of them. Note: Only works for the current set of objectives in the value
    parameters. This does not add/subtract from the objectives. If you want to add a new one in that will
    have to be re-generated. This works for when the user wants to update an objective weight, weight/value function,
    etc.
    """

    # Shorthand
    p, vp = instance.parameters, instance.value_parameters

    # Determine weights on cadets
    if 'merit_all' in p:
        vp['cadet_weight'] = cadet_weight_function(p['merit_all'], func=vp['cadet_weight_function'])
    else:
        vp['cadet_weight'] = cadet_weight_function(p['merit'], func=vp['cadet_weight_function'])

    # Determine weights on AFSCs
    if vp['afsc_weight_function'] != 'Custom':  # If the AFSC weight function is not "custom", we regenerate the weights
        vp['afsc_weight'] = afsc_weight_function(p["pgl"], vp['afsc_weight_function'])

    # Initialize breakpoints
    vp['a'] = [[[] for _ in range(vp['O'])] for _ in p["J"]]
    vp['f^hat'] = [[[] for _ in range(vp['O'])] for _ in p["J"]]

    # Loop through each AFSC
    for j, afsc in enumerate(p['afscs'][:p['M']]):  # Skip the "unmatched AFSC": '*'

        # Loop through each objective
        for k, objective in enumerate(vp['objectives']):

            # Value Function specific parameters
            actual, minimum, maximum = None, None, None
            if objective == 'Merit':
                actual = np.mean(p['merit'][p['I^E'][j]])
            if objective in ['USAFA Proportion', 'Male', 'Minority']:
                actual = len(p['I^D'][objective][j]) / len(p['I^E'][j])
            if objective in ['Combined Quota', 'ROTC Quota', 'USAFA Quota']:

                # Dictionaries for getting the right value for the specific quota objective
                min_dict = {"Combined Quota": 'quota_min', 'ROTC Quota': 'rotc_quota',
                            'USAFA Quota': 'usafa_quota'}
                target_dict = {"Combined Quota": 'quota_d', 'ROTC Quota': 'rotc_quota',
                               'USAFA Quota': 'usafa_quota'}
                minimum, maximum = int(p[min_dict[objective]][j]), int(p['quota_max'][j])

                # Update minimum values for combined quota objective
                vp['objective_value_min'][j, k] = str(minimum) + ', ' + str(maximum)
                vp['objective_min'][j, k], vp['objective_max'][j, k] = minimum, maximum
                vp['objective_target'][j, k] = p[target_dict[objective]][j]

            # If we care about this objective, we load in its value function breakpoints
            if vp['objective_weight'][j, k] != 0:

                # Create the non-linear piecewise exponential segment dictionary
                segment_dict = create_segment_dict_from_string(vp['value_functions'][j, k],
                                                               vp['objective_target'][j, k],
                                                               minimum=minimum, maximum=maximum, actual=actual)

                # Linearize the non-linear function using the specified number of breakpoints
                vp['a'][j][k], vp['f^hat'][j][k] = value_function_builder(
                    segment_dict, num_breakpoints=num_breakpoints)

        # Scale the objective weights for this AFSC, so they sum to 1
        vp['objective_weight'][j] = vp['objective_weight'][j] / sum(vp['objective_weight'][j])

    # Scale the weights across all AFSCs, so they sum to 1
    vp['afsc_weight'] = vp['afsc_weight'] / sum(vp['afsc_weight'])

    return vp  # Return set of value parameters


def compare_value_parameters(parameters, vp1, vp2, vp1name, vp2name, printing=False):
    """
    Compares two sets of value parameters to see if they are identical
    :return: True if they're identical, False otherwise
    """
    # Shorthand
    p = parameters

    # Assume identical until proven otherwise
    identical = True

    # Loop through each value parameter key
    for key in vp1:

        if np.shape(vp1[key]) != np.shape(vp2[key]):
            if printing:
                print(vp1name + ' and ' + vp2name + ' not the same. ' + key + ' is a different size.')
            identical = False
            break

        if key in ['afscs_overall_weight', 'cadets_overall_weight', 'cadet_weight_function', 'afsc_weight_function',
                   'cadets_overall_value_min', 'afscs_overall_value_min']:
            if vp1[key] != vp2[key]:
                if printing:
                    print(vp1name + ' and ' + vp2name + ' not the same. ' + key + ' is different.')
                identical = False
                break

        elif key in ['afsc_value_min', 'cadet_value_min', 'objective_value_min', 'value_functions', 'objective_target',
                     'objective_weight', 'afsc_weight', 'cadet_weight', 'I^C', 'J^C', 'value_functions',
                     'constraint_type']:
            if key not in ['objective_value_min', 'value_functions']:
                vp_1_arr, vp_2_arr = np.ravel(np.around(vp1[key], 4)), np.ravel(np.around(vp2[key], 4))
            else:
                vp_1_arr, vp_2_arr = np.ravel(vp1[key]), np.ravel(vp2[key])

            diff_arr = np.array([vp_1_arr[i] != vp_2_arr[i] for i in range(len(vp_1_arr))])
            if sum(diff_arr) != 0 and (vp1[key] != [] or vp2[key] != []):
                if printing:
                    print(vp1name + ' and ' + vp2name + ' not the same. ' + key + ' is different.')
                identical = False
                break

        elif key == 'a':  # Check the breakpoints

            for j in p['J']:
                afsc = p["afscs"][j]
                for k in vp1['K^A'][j]:
                    objective = vp1["objectives"][k]
                    for l in vp1['L'][j][k]:
                        try:
                            if vp1[key][j][k][l] != vp2[key][j][k][l]:
                                identical = False
                                if printing:
                                    print(vp1name + ' and ' + vp2name + ' not the same. '
                                                                        'Breakpoints are different for AFSC ' + afsc +
                                          ' Objective ' + objective + '.')
                                    print(vp1name + ":", vp1[key][j][k])
                                    print(vp2name + ":", vp2[key][j][k])
                                break
                        except:  # If there was a range error, then the breakpoints are not the same
                            identical = False
                            if printing:
                                print(vp1name + ' and ' + vp2name + ' not the same. '
                                                                    'Breakpoints are different for AFSC ' + afsc +
                                      ' Objective ' + objective + '.')
                                print(vp1name + ":", vp1[key][j][k])
                                print(vp2name + ":", vp2[key][j][k])
                            break
                    if not identical:
                        break
                if not identical:
                    break
            if not identical:
                break

    if identical and printing:
        print(vp1name + ' and ' + vp2name + ' are the same.')

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
    Converts a value function string into a segment dictionary.

    Args:
        vf_string (str): Value function string.
        target (float, optional): Target objective measure.
        maximum (float, optional): Maximum objective measure.
        actual (float, optional): Proportion of eligible cadets.
        multiplier (bool, optional): Specifies whether the target values are multiplied by a scalar for quota objectives.
        minimum (float, optional): Minimum objective measure.

    Returns:
        segment_dict (dict): A dictionary representing the segments of the value function.

    Notes:
        - The function assumes that the value function string follows a specific format.
        - The segment dictionary contains keys representing the segment number and values representing the segment details.
        - Each segment is represented by a dictionary with the following keys:
            - 'x1': The starting point on the x-axis.
            - 'y1': The starting point on the y-axis.
            - 'x2': The ending point on the x-axis.
            - 'y2': The ending point on the y-axis.
            - 'rho': The value of the rho parameter for the segment.
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

        assert rho1 > 0, f"rho1 must be greater than 0 for a Quota_Direct value function, got: {rho1}"
        assert rho2 > 0, f"rho2 must be greater than 0 for a Quota_Direct value function, got: {rho2}"
        assert rho3 > 0, f"rho3 must be greater than 0 for a Quota_Direct value function, got: {rho3}"
        assert rho4 > 0, f"rho4 must be greater than 0 for a Quota_Direct value function, got: {rho4}"
        assert 0 < y1 < 1, f"y1 must be between 0 and 1 for a Quota_Direct value function, got: {y1}"
        assert 0 < y2 < 1, f"y2 must be between 0 and 1 for a Quota_Direct value function, got: {y2}"

        # Build segments
        if target == minimum:
            if target == maximum:
                segment_dict = {1: {'x1': 0, 'y1': 0, 'x2': target, 'y2': 1, 'rho': -(rho1 * target)},
                                2: {'x1': maximum, 'y1': 1, 'x2': domain_max * maximum, 'y2': 0,
                                    'rho': -(rho4 * (maximum - target))}}
            else:
                segment_dict = {1: {'x1': 0, 'y1': 0, 'x2': target, 'y2': 1, 'rho': -(rho1 * target)},
                                2: {'x1': target, 'y1': 1, 'x2': maximum, 'y2': y2,
                                    'rho': (rho3 * (maximum - target))},
                                3: {'x1': maximum, 'y1': y2, 'x2': domain_max * maximum, 'y2': 0,
                                    'rho': -(rho4 * (domain_max * maximum - maximum))}}

        else:
            if target == maximum:
                segment_dict = {1: {'x1': 0, 'y1': 0, 'x2': minimum, 'y2': y1, 'rho': -(rho1 * (minimum - 0))},
                                2: {'x1': minimum, 'y1': y1, 'x2': target, 'y2': 1,
                                    'rho': (rho2 * (target - minimum))},
                                3: {'x1': target, 'y1': 1, 'x2': domain_max * maximum, 'y2': 0,
                                    'rho': -(rho4 * (domain_max * maximum - maximum))}}
            else:
                segment_dict = {1: {'x1': 0, 'y1': 0, 'x2': minimum, 'y2': y1, 'rho': -(rho1 * (minimum - 0))},
                                2: {'x1': minimum, 'y1': y1, 'x2': target, 'y2': 1,
                                    'rho': (rho2 * (target - minimum))},
                                3: {'x1': target, 'y1': 1, 'x2': maximum, 'y2': y2,
                                    'rho': (rho3 * (maximum - target))},
                                4: {'x1': maximum, 'y1': y2, 'x2': domain_max * maximum, 'y2': 0,
                                    'rho': -(rho4 * (domain_max * maximum - maximum))}}

    else:  # Must be a "Min Increasing/Decreasing" function

        # Receive values from string
        f_param_list = split_list[1]
        split_list = f_param_list.split(',')
        rho = float(split_list[0].strip()) * target  # Multiplying by the target normalizes the domain space

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

    :param segment_dict: A dictionary representing exponential segments of the value function.
        The dictionary should have the following format:
        {
            segment_id: {
                'x1': measure of the starting point of the segment,
                'y1': value of the starting point of the segment,
                'x2': measure of the ending point of the segment,
                'y2': value of the ending point of the segment,
                'rho': rate of change for the segment,
                'r': number of breakpoints per segment (optional)
            },
            ...
        }
        If segment_dict is not provided, a default dictionary will be used.

    :param num_breakpoints: The total number of breakpoints to be generated for the value function.
        If not specified within the segment_dict for each segment, a general number of breakpoints will be used.
        If both segment_dict and num_breakpoints are provided, num_breakpoints takes precedence.

    :param derivative_locations: A boolean flag indicating whether breakpoints should be placed at locations where
        the derivative increases by some interval. If True, breakpoints will be placed based on derivative intervals;
        if False, breakpoints will be placed at fixed intervals based on x.

    :return: Two numpy arrays representing the breakpoints of the value function:
        - a: Array of breakpoint measures.
        - fhat: Array of breakpoint values.

    Note:
    - The breakpoints are determined based on the exponential segments provided.
    - The resulting breakpoints are returned as numpy arrays, with measures and values rounded to 5 decimal places.
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

    # Shorthand
    p, vp = parameters, value_parameters
    for j in p['J']:
        for k in vp['K^A'][j]:
            a = np.array(vp['a'][j][k])
            fhat = np.array(vp['f^hat'][j][k])

            # Find unnecessary zeros
            zero_indices = np.where(fhat == 0)[0]
            last_i = len(a) - 1
            removals = []
            for i in zero_indices:
                if i + 1 in zero_indices and i + 1 != last_i and i != 0:
                    removals.append(i)

            # Remove unnecessary zeros
            vp['a'][j][k] = np.delete(a, removals)
            vp['f^hat'][j][k] = np.delete(fhat, removals)

    return vp


# Rebecca's Model Parameter Translation
def translate_vft_to_gp_parameters(instance):
    """
    Translates the VFT (Value Focused Thinking) parameters to Rebecca's model parameters.

    Args:
        instance: An instance of CadetCareerProblem

    Returns:
        gp (dict): A dictionary containing the translated parameters for Rebecca's model.

    Notes:
        The function assumes that the instance object contains the following attributes:
            - parameters: A dictionary containing the VFT model parameters.
            - value_parameters: A dictionary containing the value parameters of the VFT model.
    """

    # Shorthand
    p = instance.parameters
    vp = instance.value_parameters
    objectives = vp['objectives']

    # Other parameters
    large_afscs = np.where(p['pgl'] >= 40)[0]  # set of large AFSCs
    mand_k = np.where(objectives == 'Mandatory')[0][0]  # mandatory objective index
    des_k = np.where(objectives == 'Desired')[0][0]  # desired objective index
    perm_k = np.where(objectives == 'Permitted')[0][0]  # permitted objective index
    usafa_k = np.where(objectives == 'USAFA Proportion')[0][0]  # USAFA proportion objective index

    # Initialize "gp" dictionary (Goal Programming Model Parameters)
    gp = {}

    # Main sets
    A = np.arange(p['M'])
    C = np.arange(p['N'])
    gp['A'], gp['M'] = A, len(A)  # AFSCs, number of AFSCs
    gp['C'], gp['N'] = C, len(C)  # Cadets, number of Cadets

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
    gp['param'] = {'T': p['quota_min'],  # Target quotas
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
    columns = ['Normalized Penalty', 'Normalized Reward', 'Run Penalty', 'Run Reward']
    column_dict = {column: np.array(instance.gp_df[column]) for column in columns}

    # actual reward parameters
    reward = column_dict['Normalized Reward'] * column_dict['Run Reward']

    # actual penalty parameters
    penalty = column_dict['Normalized Penalty'] * column_dict['Run Penalty']

    # mu parameters (Penalties)
    gp['mu^'] = {con: penalty[index] for index, con in enumerate(gp['con'])}

    # lambda parameters (Rewards)
    gp['lam^'] = {con: reward[index] for index, con in enumerate(gp['con'])}
    gp['lam^']['S'] = reward[len(gp['con'])]  # extra reward for preference in order of merit

    return gp

