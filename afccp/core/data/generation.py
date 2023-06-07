import random
import numpy as np
import pandas as pd
import string
from afccp.core.data.preferences import get_utility_preferences
import afccp.core.data.values as afccp_dv

def generate_random_instance(N=1600, P=6, M=32):
    """
    This procedure takes in the specified parameters (defined below) and then simulates new random "fixed" cadet/AFSC
    input parameters. These parameters are then returned and can be used to solve the VFT model.
    :param N: number of cadets
    :param P: number of preferences allowed
    :param M: number of AFSCs
    :return: model fixed parameters
    """

    # Initialize parameter dictionary
    p = {'N': N, 'P': P, 'M': M, 'num_util': P}

    # List of lowercase and uppercase letters
    alphabet = string.ascii_letters

    # Generate cadets
    p['cadets'] = np.array([''.join(random.choices(alphabet, k=8)) for _ in range(N)])

    # Generate various features of the cadets
    p['minority'] = np.random.choice([0, 1], size=N, p=[1 / 3, 2 / 3])
    p['male'] = np.random.choice([0, 1], size=N, p=[1 / 3, 2 / 3])
    p['usafa'] = np.random.choice([0, 1], size=N, p=[1 / 3, 2 / 3])
    p['merit'] = np.random.rand(N)
    p['merit_all'] = p['merit']
    p['assigned'] = np.array(['' for _ in range(N)])

    # Calculate quotas for each AFSC
    p['pgl'], p['usafa_quota'], p['rotc_quota'] = np.zeros(M), np.zeros(M), np.zeros(M)
    p['quota_min'], p['quota_max'] = np.zeros(M), np.zeros(M)
    p['quota_e'], p['quota_d'] = np.zeros(M), np.zeros(M)
    for j in range(M):

        # Get PGL target
        p['pgl'][j] = max(10, np.random.normal(1000 / M, 100))

    # Scale PGL and force integer values and minimum of 1
    p['pgl'] = np.around((p['pgl'] / np.sum(p['pgl'])) * N * 0.9)
    indices = np.where(p['pgl'] == 0)[0]
    p['pgl'][indices] = 1

    # Sort PGL by size
    p['pgl'] = np.sort(p['pgl'])[::-1]

    # USAFA/ROTC Quotas
    p['usafa_quota'] = np.around(np.random.rand(M) * 0.3 + 0.1 * p['pgl'])
    p['rotc_quota'] = p['pgl'] - p['usafa_quota']

    # Min/Max
    p['quota_min'], p['quota_max'] = p['pgl'], np.around(p['pgl'] * (1 + np.random.rand(M) * 0.5))

    # Target is a random integer between the minimum and maximum targets
    target = np.around(p['quota_min'] + np.random.rand(M) * (p['quota_max'] - p['quota_min']))
    p['quota_e'], p['quota_d'] = target, target

    # Generate AFSCs
    p['afscs'] = np.array(['R' + str(j + 1) for j in range(M)])
    p['acc_grp'] = np.array(['NRL' for _ in range(M)])

    # Add an "*" to the list of AFSCs to be considered the "Unmatched AFSC"
    p["afscs"] = np.hstack((p["afscs"], "*"))

    # Cadet preferences
    utility = np.random.rand(N, M)  # random utility matrix
    max_util = np.max(utility, axis=1)
    p['utility'] = np.around(utility / np.array([[max_util[i]] for i in range(N)]), 2)
    p['c_preferences'], p['c_utilities'] = get_utility_preferences(p)
    p['c_preferences'] = p['c_preferences'][:, :P]
    p['c_utilities'] = p['c_utilities'][:, :P]

    # Add degree tier qualifications to the set of parameters
    def generate_degree_tier_qualifications():
        """
        I made this nested function, so I could have a designated section to generate degree qualifications and such
        """

        # Determine degree tiers and qualification information
        p['qual'] = np.array([['P1' for _ in range(M)] for _ in range(N)])
        p['Deg Tiers'] = np.array([[' ' * 10 for _ in range(4)] for _ in range(M)])
        for j in range(M):

            # Determine what tiers to use on this AFSC
            if N < 100:
                random_number = np.random.rand()
                if random_number < 0.2:
                    tiers = ['M1', 'I2']
                    p['Deg Tiers'][j, :] = ['M = 1', 'I = 0', '', '']
                elif 0.2 < random_number < 0.4:
                    tiers = ['D1', 'P2']
                    target_num = round(np.random.rand(), 2)
                    p['Deg Tiers'][j, :] = ['D > ' + str(target_num), 'P < ' + str(1 - target_num), '', '']
                elif 0.4 < random_number < 0.6:
                    tiers = ['P1']
                    p['Deg Tiers'][j, :] = ['P = 1', '', '', '']
                else:
                    tiers = ['M1', 'P2']
                    target_num = round(np.random.rand(), 2)
                    p['Deg Tiers'][j, :] = ['M > ' + str(target_num), 'P < ' + str(1 - target_num), '', '']
            else:
                random_number = np.random.rand()
                if random_number < 0.1:
                    tiers = ['M1', 'I2']
                    p['Deg Tiers'][j, :] = ['M = 1', 'I = 0', '', '']
                elif 0.1 < random_number < 0.2:
                    tiers = ['D1', 'P2']
                    target_num = round(np.random.rand(), 2)
                    p['Deg Tiers'][j, :] = ['D > ' + str(target_num), 'P < ' + str(1 - target_num), '', '']
                elif 0.2 < random_number < 0.3:
                    tiers = ['P1']
                    p['Deg Tiers'][j, :] = ['P = 1', '', '', '']
                elif 0.3 < random_number < 0.4:
                    tiers = ['M1', 'P2']
                    target_num = round(np.random.rand(), 2)
                    p['Deg Tiers'][j, :] = ['M > ' + str(target_num), 'P < ' + str(1 - target_num), '', '']
                elif 0.4 < random_number < 0.5:
                    tiers = ['M1', 'D2', 'P3']
                    target_num_1 = round(np.random.rand() * 0.7, 2)
                    target_num_2 = round(np.random.rand() * (1 - target_num_1) * 0.8, 2)
                    target_num_3 = round(1 - target_num_1 - target_num_2, 2)
                    p['Deg Tiers'][j, :] = ['M > ' + str(target_num_1), 'D > ' + str(target_num_2),
                                            'P < ' + str(target_num_3), '']
                elif 0.5 < random_number < 0.6:
                    tiers = ['D1', 'D2', 'P3']
                    target_num_1 = round(np.random.rand() * 0.7, 2)
                    target_num_2 = round(np.random.rand() * (1 - target_num_1) * 0.8, 2)
                    target_num_3 = round(1 - target_num_1 - target_num_2, 2)
                    p['Deg Tiers'][j, :] = ['D > ' + str(target_num_1), 'D > ' + str(target_num_2),
                                            'P < ' + str(target_num_3), '']
                elif 0.6 < random_number < 0.7:
                    tiers = ['M1', 'D2', 'I3']
                    target_num = round(np.random.rand(), 2)
                    p['Deg Tiers'][j, :] = ['M > ' + str(target_num), 'D < ' + str(1 - target_num), 'I = 0', '']
                elif 0.7 < random_number < 0.8:
                    tiers = ['M1', 'P2', 'I3']
                    target_num = round(np.random.rand(), 2)
                    p['Deg Tiers'][j, :] = ['M > ' + str(target_num), 'P < ' + str(1 - target_num), 'I = 0', '']
                else:
                    tiers = ['M1', 'D2', 'P3', 'I4']
                    target_num_1 = round(np.random.rand() * 0.7, 2)
                    target_num_2 = round(np.random.rand() * (1 - target_num_1) * 0.8, 2)
                    target_num_3 = round(1 - target_num_1 - target_num_2, 2)
                    p['Deg Tiers'][j, :] = ['M > ' + str(target_num_1), 'D > ' + str(target_num_2),
                                            'P < ' + str(target_num_3), 'I = 0']

            # Generate the tiers for the cadets
            c_tiers = np.random.randint(0, len(tiers), N)
            p['qual'][:, j] = np.array([tiers[c_tiers[i]] for i in range(N)])

        # NxM qual matrices with various features
        p["ineligible"] = (np.core.defchararray.find(p['qual'], "I") != -1) * 1
        p["eligible"] = (p["ineligible"] == 0) * 1
        for t in [1, 2, 3, 4]:
            p["tier " + str(t)] = (np.core.defchararray.find(p['qual'], str(t)) != -1) * 1
        p["mandatory"] = (np.core.defchararray.find(p['qual'], "M") != -1) * 1
        p["desired"] = (np.core.defchararray.find(p['qual'], "D") != -1) * 1
        p["permitted"] = (np.core.defchararray.find(p['qual'], "P") != -1) * 1

        # NEW: Exception to degree qualification based on CFM ranks
        p["exception"] = (np.core.defchararray.find(p['qual'], "E") != -1) * 1

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
                if 'M' not in val and 'D' not in val and 'P' not in val and 'I' not in val:
                # if val in ["nan", "", ''] or pd.isnull(val):
                    t -= 1
                    break

                # Degree Tier Proportion
                p["t_proportion"][j, t] = val.split(" ")[2]

            # Num tiers
            p["t_count"][j] = t + 1

        return p   # Return updated parameters
    p = generate_degree_tier_qualifications()

    return p  # Return updated parameters

def generate_random_value_parameters(parameters, num_breakpoints=24):
    """
    Generates value parameters for a given problem instance from scratch
    """

    # Shorthand
    p = parameters

    # Objective to parameters lookup dictionary (if the parameter is in "p", we include the objective)
    objective_lookups = {'Norm Score': 'a_pref_matrix', 'Merit': 'merit', 'USAFA Proportion': 'usafa',
                         'Combined Quota': 'quota_d', 'USAFA Quota': 'usafa_quota', 'ROTC Quota': 'rotc_quota',
                         'Utility': 'utility', 'Male': 'male', 'Minority': 'minority'}
    for t in ["1", "2", "3", "4"]:  # Add in AFOCD Degree tiers
        objective_lookups["Tier " + t] = "tier " + t

    # Add the AFSC objectives that are included in this instance (check corresponding parameters using dict above)
    objectives = np.array([objective for objective in objective_lookups if objective_lookups[objective] in p])

    # Initialize set of value parameters
    vp = {'objectives': objectives, 'cadets_overall_weight': np.random.rand(), 'O': len(objectives),
          'K': np.arange(len(objectives)), 'num_breakpoints': num_breakpoints, 'cadets_overall_value_min': 0,
          'afscs_overall_value_min': 0}
    vp['afscs_overall_weight'] = 1- vp['cadets_overall_weight']

    # Generate AFSC and cadet weights
    weight_functions = ['Linear', 'Direct', 'Curve_1', 'Curve_2', 'Equal']
    vp['cadet_weight_function'] = np.random.choice(weight_functions)
    vp['afsc_weight_function'] = np.random.choice(weight_functions)
    vp['cadet_weight'] = afccp_dv.cadet_weight_function(p['merit_all'], func= vp['cadet_weight_function'])
    vp['afsc_weight'] = afccp_dv.afsc_weight_function(p['pgl'], func = vp['afsc_weight_function'])

    # Stuff that doesn't matter here
    vp['cadet_value_min'], vp['afsc_value_min'] = np.zeros(p['N']), np.zeros(p['N'])
    vp['USAFA-Constrained AFSCs'], vp['Cadets Top 3 Constraint'] = '', ''

    # Initialize arrays
    vp['objective_weight'], vp['objective_target'] = np.zeros([p['M'], vp['O']]), np.zeros([p['M'], vp['O']])
    vp['constraint_type'] = np.zeros([p['M'], vp['O']])
    vp['objective_value_min'] = np.array([[' ' * 20 for _ in vp['K']] for _ in p['J']])
    vp['value_functions'] = np.array([[' ' * 200 for _ in vp['K']] for _ in p['J']])

    # Initialize breakpoints
    vp['a'] = [[[] for _ in vp['K']] for _ in p["J"]]
    vp['f^hat'] = [[[] for _ in vp['K']] for _ in p["J"]]

    # Initialize objective set
    vp['K^A'] = {}

    # Get AFOCD Tier objectives
    vp = afccp_dv.generate_afocd_value_parameters(p, vp)
    vp['constraint_type'] = np.zeros([p['M'], vp['O']])  # Turn off all the constraints again

    # Loop through all AFSCs
    for j in p['J']:

        # Loop through all AFSC objectives
        for k, objective in enumerate(vp['objectives']):

            maximum, minimum, actual = None, None, None
            if objective == 'Norm Score':
                vp['objective_weight'][j, k] = np.random.rand() * 0.2 + 0.3
                vp['value_functions'][j, k] = 'Min Increasing|0.3'
                vp['objective_target'][j, k] = 1

            if objective == 'Merit':
                vp['objective_weight'][j, k] = np.random.rand() * 0.4 + 0.05
                vp['value_functions'][j, k] = 'Min Increasing|-0.3'
                vp['objective_target'][j, k] = p['sum_merit'] / p['N']
                actual = np.mean(p['merit'][p['I^E'][j]])

            elif objective == 'USAFA Proportion':
                vp['objective_weight'][j, k] = np.random.rand() * 0.3 + 0.05
                vp['value_functions'][j, k] = 'Balance|0.15, 0.15, 0.1, 0.08, 0.08, 0.1, 0.6'
                vp['objective_target'][j, k] = p['usafa_proportion']
                actual = len(p['I^D'][objective][j]) / len(p['I^E'][j])

            elif objective == 'Combined Quota':
                vp['objective_weight'][j, k] = np.random.rand() * 0.8 + 0.2
                vp['value_functions'][j, k] = 'Quota_Normal|0.2, 0.25, 0.2'
                vp['objective_target'][j, k] = p['quota_d'][j]

                # Get bounds and turn on this constraint
                minimum, maximum = p['quota_min'][j], p['quota_max'][j]
                vp['objective_value_min'][j, k] = str(int(minimum)) + ", " + str(int(maximum))
                vp['constraint_type'][j, k] = 2

            elif objective == 'USAFA Quota':
                vp['objective_weight'][j, k] = 0
                vp['value_functions'][j, k] = 'Min Increasing|0.3'
                vp['objective_target'][j, k] = p['usafa_quota'][j]

            elif objective == 'ROTC Quota':
                vp['objective_weight'][j, k] = 0
                vp['value_functions'][j, k] = 'Min Increasing|0.3'
                vp['objective_target'][j, k] = p['rotc_quota'][j]

            elif objective == 'Male':
                vp['objective_weight'][j, k] = np.random.rand() * 0.25
                vp['value_functions'][j, k] = 'Balance|0.15, 0.15, 0.1, 0.08, 0.08, 0.1, 0.6'
                vp['objective_target'][j, k] = p['male_proportion']
                actual = len(p['I^D'][objective][j]) / len(p['I^E'][j])

            elif objective == 'Minority':
                vp['objective_weight'][j, k] = np.random.rand() * 0.25
                vp['value_functions'][j, k] = 'Balance|0.15, 0.15, 0.1, 0.08, 0.08, 0.1, 0.6'
                vp['objective_target'][j, k] = p['minority_proportion']
                actual = len(p['I^D'][objective][j]) / len(p['I^E'][j])

            # If we care about this objective, we load in its value function breakpoints
            if vp['objective_weight'][j, k] != 0:

                # Create the non-linear piecewise exponential segment dictionary
                segment_dict = afccp_dv.create_segment_dict_from_string(vp['value_functions'][j, k],
                                                                        vp['objective_target'][j, k],
                                                                        minimum=minimum, maximum=maximum, actual=actual)

                # Linearize the non-linear function using the specified number of breakpoints
                vp['a'][j][k], vp['f^hat'][j][k] = afccp_dv.value_function_builder(
                    segment_dict, num_breakpoints=num_breakpoints)

        # Scale the objective weights for this AFSC, so they sum to 1
        vp['objective_weight'][j] = vp['objective_weight'][j] / sum(vp['objective_weight'][j])
        vp['K^A'][j] = np.where(vp['objective_weight'][j] != 0)[0]

    return vp
