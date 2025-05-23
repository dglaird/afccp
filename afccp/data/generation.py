import random
import numpy as np
import pandas as pd
import string
import copy
import datetime
import os

# afccp modules
import afccp.globals
import afccp.data.preferences
import afccp.data.adjustments
import afccp.data.values
import afccp.data.support
import warnings
warnings.filterwarnings('ignore')  # prevent red warnings from printing

# Import sdv if it is installed
if afccp.globals.use_sdv:
    pass


def generate_random_instance(N=1600, M=32, P=6, S=6, generate_only_nrl=False, generate_extra=False):
    """
    This procedure takes in the specified parameters (defined below) and then simulates new random "fixed" cadet/AFSC
    input parameters. These parameters are then returned and can be used to solve the VFT model.
    :param N: number of cadets
    :param M: number of AFSCs
    :param P: number of preferences allowed
    :param S: number of Bases
    :param generate_only_nrl: Only generate NRL AFSCs (default to False)
    :param generate_extra: Whether to generate extra components (bases/IST). Defaults to False.
    :return: model fixed parameters
    """

    # Initialize parameter dictionary
    # noinspection PyDictCreation
    p = {'N': N, 'P': P, 'M': M, 'num_util': P, 'cadets': np.arange(N),
         'usafa': np.random.choice([0, 1], size=N, p=[2 / 3, 1 / 3]), 'merit': np.random.rand(N)}

    # Generate various features of the cadets
    p['merit_all'] = p['merit']
    p['assigned'] = np.array(['' for _ in range(N)])
    p['soc'] = np.array(['USAFA' for _ in range(p['N'])])
    p['soc'][np.where(p['usafa'] == 0)[0]] = 'ROTC'

    # Calculate quotas for each AFSC
    p['pgl'], p['usafa_quota'], p['rotc_quota'] = np.zeros(M), np.zeros(M), np.zeros(M)
    p['quota_min'], p['quota_max'] = np.zeros(M), np.zeros(M)
    p['quota_e'], p['quota_d'] = np.zeros(M), np.zeros(M)
    for j in range(M):

        # Get PGL target
        p['pgl'][j] = max(10, np.random.normal(1000 / M, 100))

    # Scale PGL and force integer values and minimum of 1
    p['pgl'] = np.around((p['pgl'] / np.sum(p['pgl'])) * N * 0.8)
    indices = np.where(p['pgl'] == 0)[0]
    p['pgl'][indices] = 1

    # Sort PGL by size
    p['pgl'] = np.sort(p['pgl'])[::-1]

    # USAFA/ROTC Quotas
    p['usafa_quota'] = np.around(np.random.rand(M) * 0.3 + 0.1 * p['pgl'])
    p['rotc_quota'] = p['pgl'] - p['usafa_quota']

    # Min/Max
    p['quota_min'], p['quota_max'] = p['pgl'], np.around(p['pgl'] * (1 + np.random.rand(M) * 0.9))

    # Target is a random integer between the minimum and maximum targets
    target = np.around(p['quota_min'] + np.random.rand(M) * (p['quota_max'] - p['quota_min']))
    p['quota_e'], p['quota_d'] = target, target

    # Generate AFSCs
    p['afscs'] = np.array(['R' + str(j + 1) for j in range(M)])

    # Determine what "accessions group" each AFSC is in
    if generate_only_nrl:
        p['acc_grp'] = np.array(["NRL" for _ in range(M)])
    else:

        # If there are 3 or more AFSCs, we want all three accessions groups represented
        if M >= 3:
            invalid = True
            while invalid:

                # If we have 6 or fewer, limit USSF to just one AFSC
                if M <= 6:
                    p['acc_grp'] = ['USSF']
                    for _ in range(M - 1):
                        p['acc_grp'].append(np.random.choice(['NRL', 'Rated']))
                else:
                    p['acc_grp'] = [np.random.choice(['NRL', 'Rated', 'USSF']) for _ in range(M)]

                # Make sure we have at least one AFSC from each accession's group
                invalid = False  # "Innocent until proven guilty"
                for grp in ['NRL', 'Rated', 'USSF']:
                    if grp not in p['acc_grp']:
                        invalid = True
                        break

                # If we have 4 or more AFSCs, make sure we have at least two Rated
                if M >= 4:
                    if p['acc_grp'].count('Rated') < 2:
                        invalid = True
            p['acc_grp'] = np.array(p['acc_grp'])  # Convert to numpy array

        # If we only have one or two AFSCs, they'll all be NRL
        else:
            p['acc_grp'] = np.array(["NRL" for _ in range(M)])

    # Add an "*" to the list of AFSCs to be considered the "Unmatched AFSC"
    p["afscs"] = np.hstack((p["afscs"], "*"))

    # Add degree tier qualifications to the set of parameters
    def generate_degree_tier_qualifications():
        """
        I made this nested function, so I could have a designated section to generate degree qualifications and such
        """

        # Determine degree tiers and qualification information
        p['qual'] = np.array([['P1' for _ in range(M)] for _ in range(N)])
        p['Deg Tiers'] = np.array([[' ' * 10 for _ in range(4)] for _ in range(M)])
        for j in range(M):

            if p['acc_grp'][j] == 'Rated':  # All Degrees eligible for Rated
                p['qual'][:, j] = np.array(['P1' for _ in range(N)])
                p['Deg Tiers'][j, :] = ['P = 1', 'I = 0', '', '']

                # Pick 20% of the cadets at random to be ineligible for this Rated AFSC
                indices = random.sample(list(np.arange(N)), k=int(0.2 * N))
                p['qual'][indices, j] = 'I2'
            else:
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

    # Cadet preferences
    utility = np.random.rand(N, M)  # random utility matrix
    max_util = np.max(utility, axis=1)
    p['utility'] = np.around(utility / np.array([[max_util[i]] for i in range(N)]), 2)
    p['c_preferences'], p['c_utilities'] = afccp.data.preferences.get_utility_preferences(p)
    p['c_preferences'] = p['c_preferences'][:, :P]
    p['c_utilities'] = p['c_utilities'][:, :P]

    # Get cadet preferences
    p["c_pref_matrix"] = np.zeros([p["N"], p["M"]]).astype(int)
    for i in range(p['N']):

        # Sort the utilities to get the preference list
        utilities = p["utility"][i, :p["M"]]
        sorted_indices = np.argsort(utilities)[::-1]
        preferences = np.argsort(
            sorted_indices) + 1  # Add 1 to change from python index (at 0) to rank (start at 1)
        p["c_pref_matrix"][i, :] = preferences

    # If we want to generate extra components to match with, we do so here
    if generate_extra:
        p['S'] = S
        p = generate_extra_components(p)

    # Update set of parameters
    p = afccp.data.adjustments.parameter_sets_additions(p)

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
                         'Utility': 'utility', 'Mandatory': 'mandatory',
                         'Desired': 'desired', 'Permitted': 'permitted'}
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
    vp['cadet_weight'] = afccp.data.values.cadet_weight_function(p['merit_all'], func= vp['cadet_weight_function'])
    vp['afsc_weight'] = afccp.data.values.afsc_weight_function(p['pgl'], func = vp['afsc_weight_function'])

    # Stuff that doesn't matter here
    vp['cadet_value_min'], vp['afsc_value_min'] = np.zeros(p['N']), np.zeros(p['N'])
    vp['USAFA-Constrained AFSCs'], vp['Cadets Top 3 Constraint'] = '', ''
    vp['USSF OM'] = False

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
    vp = afccp.data.values.generate_afocd_value_parameters(p, vp)
    vp['constraint_type'] = np.zeros([p['M'], vp['O']])  # Turn off all the constraints again

    # Loop through all AFSCs
    for j in p['J']:

        # Loop through all AFSC objectives
        for k, objective in enumerate(vp['objectives']):

            maximum, minimum, actual = None, None, None
            if objective == 'Norm Score':
                vp['objective_weight'][j, k] = (np.random.rand() * 0.2 + 0.3) * 100  # Scale up to 100
                vp['value_functions'][j, k] = 'Min Increasing|0.3'
                vp['objective_target'][j, k] = 1

            if objective == 'Merit':
                vp['objective_weight'][j, k] = (np.random.rand() * 0.4 + 0.05) * 100
                vp['value_functions'][j, k] = 'Min Increasing|-0.3'
                vp['objective_target'][j, k] = p['sum_merit'] / p['N']
                actual = np.mean(p['merit'][p['I^E'][j]])

            elif objective == 'USAFA Proportion':
                vp['objective_weight'][j, k] = (np.random.rand() * 0.3 + 0.05) * 100
                vp['value_functions'][j, k] = 'Balance|0.15, 0.15, 0.1, 0.08, 0.08, 0.1, 0.6'
                vp['objective_target'][j, k] = p['usafa_proportion']
                actual = len(p['I^D'][objective][j]) / len(p['I^E'][j])

            elif objective == 'Combined Quota':
                vp['objective_weight'][j, k] = (np.random.rand() * 0.8 + 0.2) * 100
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

                # Bounds on this constraint (but leave it off)
                vp['objective_value_min'][j, k] = str(int(p['usafa_quota'][j])) + ", " + \
                                                  str(int(p['quota_max'][j]))

            elif objective == 'ROTC Quota':
                vp['objective_weight'][j, k] = 0
                vp['value_functions'][j, k] = 'Min Increasing|0.3'
                vp['objective_target'][j, k] = p['rotc_quota'][j]

                # Bounds on this constraint (but leave it off)
                vp['objective_value_min'][j, k] = str(int(p['rotc_quota'][j])) + ", " + \
                                                  str(int(p['quota_max'][j]))

            # If we care about this objective, we load in its value function breakpoints
            if vp['objective_weight'][j, k] != 0:

                # Create the non-linear piecewise exponential segment dictionary
                segment_dict = afccp.data.values.create_segment_dict_from_string(
                    vp['value_functions'][j, k], vp['objective_target'][j, k],
                    minimum=minimum, maximum=maximum, actual=actual)

                # Linearize the non-linear function using the specified number of breakpoints
                vp['a'][j][k], vp['f^hat'][j][k] = afccp.data.values.value_function_builder(
                    segment_dict, num_breakpoints=num_breakpoints)

        # Scale the objective weights for this AFSC, so they sum to 1
        vp['objective_weight'][j] = vp['objective_weight'][j] / sum(vp['objective_weight'][j])
        vp['K^A'][j] = np.where(vp['objective_weight'][j] != 0)[0]

    return vp


def generate_extra_components(parameters):
    """
    If we generate extra components (such as bases and IST) for the CadetCareerProblem instance
    :param parameters: instance parameters
    :return: updated parameters
    """

    # Shorthand
    p = parameters

    # Get list of ordered letters (based on Excel column names)
    alphabet = list(string.ascii_uppercase)
    excel_columns = copy.deepcopy(alphabet)
    for letter in alphabet:
        for letter_2 in alphabet:
            excel_columns.append(letter + letter_2)

    # Determine which AFSCs we assign bases for
    p['afsc_assign_base'] = np.zeros(p['M']).astype(int)
    for j in range(p['M']):
        if p['acc_grp'][j] != "Rated" and np.random.rand() > 0.3:
            p['afsc_assign_base'][j] = 1

    # Name the bases according to the Excel columns (just a method of generating unique ordered letters)
    p['bases'] = np.array(["Base " + excel_columns[b] for b in range(p['S'])])

    # Get capacities for each AFSC at each base
    p['base_min'] = np.zeros((p['S'], p['M'])).astype(int)
    p['base_max'] = np.zeros((p['S'], p['M'])).astype(int)
    afscs_with_base_assignments = np.where(p['afsc_assign_base'])[0]
    for j in afscs_with_base_assignments:
        total_max = p['pgl'][j] * 1.5
        base_max = np.array([np.random.rand() for _ in range(p['S'])])
        base_max = (base_max / np.sum(base_max)) * total_max
        p['base_max'][:, j] = base_max.astype(int)
        p['base_min'][:, j] = (base_max * 0.4).astype(int)

    # Generate random cadet preferences for bases
    bases = copy.deepcopy(p['bases'])
    p['base_preferences'] = {}
    p['b_pref_matrix'] = np.zeros((p['N'], p['S'])).astype(int)
    p['base_utility'] = np.zeros((p['N'], p['S']))
    for i in range(p['N']):
        random.shuffle(bases)
        num_base_pref = np.random.choice(np.arange(2, p['S'] + 1))
        p['base_preferences'][i] = np.array([np.where(p['bases'] == base)[0][0] for base in bases[:num_base_pref]])

        # Convert to base preference matrix
        p['b_pref_matrix'][i, p['base_preferences'][i]] = np.arange(1, len(p['base_preferences'][i]) + 1)

        utilities = np.around(np.random.rand(num_base_pref), 2)
        p['base_utility'][i, p['base_preferences'][i]] = np.sort(utilities)[::-1]
        p['base_utility'][i, p['base_preferences'][i][0]] = 1.0  # First choice is always utility of 1!

    # Get the baseline starting date (January 1st of the year we're classifying)
    next_year = datetime.datetime.now().year + 1
    p['baseline_date'] = datetime.date(next_year, 1, 1)

    # Generate training preferences for each cadet
    p['training_preferences'] = np.array(
        [random.choices(['Early', 'Late'], weights=[0.9, 0.1])[0] for _ in range(p['N'])])

    # Generate base/training "thresholds" for when these preferences kick in (based on preferences for AFSCs)
    p['training_threshold'] = np.array([np.random.choice(np.arange(p['M'] + 1)) for _ in range(p['N'])])
    p['base_threshold'] = np.array([np.random.choice(np.arange(p['M'] + 1)) for _ in range(p['N'])])

    # Generate weights for AFSCs, bases, and courses
    p['weight_afsc'], p['weight_base'], p['weight_course'] = np.zeros(p['N']), np.zeros(p['N']), np.zeros(p['N'])
    for i in range(p['N']):

        # Force some percentage of cadets to make their threshold the last possible AFSC (this means these don't matter)
        if np.random.rand() > 0.8:
            p['base_threshold'][i] = p['M']
        if np.random.rand() > 0.7:
            p['training_threshold'][i] = p['M']

        # Generate weights for bases, training (courses), and AFSCs
        if p['base_threshold'][i] == p['M']:
            w_b = 0
        else:
            w_b = np.random.triangular(0, 50, 100)
        if p['training_threshold'][i] == p['M']:
            w_c = 0
        else:
            w_c = np.random.triangular(0, 20, 100)
        w_a = np.random.triangular(0, 90, 100)

        # Scale weights so that they sum to one and load into arrays
        p['weight_afsc'][i] = w_a / (w_a + w_b + w_c)
        p['weight_base'][i] = w_b / (w_a + w_b + w_c)
        p['weight_course'][i] = w_c / (w_a + w_b + w_c)

    # Generate training start dates for each cadet
    p['training_start'] = []
    for i in range(p['N']):

        # If this cadet is a USAFA cadet
        if p['usafa'][i]:

            # Make it May 28th of this year
            p['training_start'].append(datetime.date(next_year, 5, 28))

        # If it's an ROTC cadet, we sample from two different distributions (on-time and late grads)
        else:

            # 80% should be in spring
            if np.random.rand() < 0.8:
                dt = datetime.date(next_year, 4, 15) + datetime.timedelta(int(np.random.triangular(0, 30, 60)))
                p['training_start'].append(dt)

            # 20% should be after
            else:
                dt = datetime.date(next_year, 6, 1) + datetime.timedelta(int(np.random.triangular(0, 30*5, 30*6)))
                p['training_start'].append(dt)
    p['training_start'] = np.array(p['training_start'])

    # Generate training courses for each AFSC
    p['courses'], p['course_start'], p['course_min'], p['course_max'] = {}, {}, {}, {}
    p['course_count'] = np.zeros(p['M'])
    for j in range(p['M']):

        # Determine total number of training slots to divide up
        total_max = p['pgl'][j] * 1.5

        # Determine number of courses to generate
        if total_max <= 3:
            T = 1
        elif total_max <= 10:
            T = np.random.choice([1, 2])
        elif total_max < 25:
            T = np.random.choice([2, 3])
        elif total_max < 100:
            T = np.random.choice([3, 4, 5])
        else:
            T = np.random.choice([4, 5, 6, 7, 8, 9])

        # Course minimums and maximums
        random_nums = np.random.rand(T)
        p['course_max'][j] = np.around(total_max * (random_nums / np.sum(random_nums))).astype(int)
        p['course_min'][j] = np.zeros(T).astype(int)

        # Generate course specific information
        p['courses'][j], p['course_start'][j] = [], []
        current_date = p['baseline_date'] + datetime.timedelta(int(np.random.triangular(30*5, 30*9, 30*11)))
        for _ in range(T):

            # Course names (random strings of letters)
            num_letters = random.choice(np.arange(4, 10))
            p['courses'][j].append(''.join(random.choices(alphabet, k=num_letters)))

            # Course start date
            p['course_start'][j].append(current_date)

            # Get next course start date
            current_date += datetime.timedelta(int(np.random.triangular(30, 30*4, 30*6)))

        # Convert to numpy arrays
        for param in ['courses', 'course_start', 'course_max', 'course_min']:
            p[param][j] = np.array(p[param][j])

    # Number of training courses per AFSC
    p['T'] = np.array([len(p['courses'][j]) for j in range(p['M'])])

    # Return updated parameters
    return p


def safe_round(data, decimals=0, axis=-1):
    """
    Round `data` to `decimals` decimals along `axis`, preserving the sum of each
    slice (to `decimals`), using a "difference" style strategy.
    """
    data_type = type(data)
    constructor = {}

    # 1) Scale by 10^decimals
    scale = 10.0 ** decimals
    scaled = data * scale

    # 2) Naively round each element to the nearest integer
    rounded = np.rint(scaled)

    # 3) Compute how many integer "units" the sum *should* have in each slice
    sum_rounded = np.sum(rounded, axis=axis, keepdims=True)
    sum_desired = np.rint(np.sum(scaled, axis=axis, keepdims=True))
    difference = sum_desired - sum_rounded

    n = data.shape[axis]
    leftover_div = np.floor_divide(difference, n)
    leftover_mod = difference - leftover_div * n
    rounded += leftover_div

    # 5) Select elements to tweak
    difference = scaled - rounded
    leftover_sign = np.sign(leftover_mod)
    difference_sign = np.sign(difference)
    candidate_mask = (difference_sign == leftover_sign) & (difference_sign != 0)
    sort_key = np.where(candidate_mask, -np.abs(difference), np.inf)
    sorted_idx = np.argsort(sort_key, axis=axis, kind='stable')

    ranks = np.empty_like(sorted_idx)
    shape_for_r = [1] * data.ndim
    shape_for_r[axis] = n
    r_array = np.arange(n, dtype=sorted_idx.dtype).reshape(shape_for_r)
    np.put_along_axis(ranks, sorted_idx, r_array, axis=axis)

    leftover_mod_int = np.abs(leftover_mod).astype(int)
    choose_mask = ranks < leftover_mod_int
    rounded += leftover_sign * choose_mask

    result = rounded / scale

    if data_type is np.ndarray:
        return result

    return data_type(result.squeeze(), **constructor)


# SDV functions (we may not have the SDV library!)
if afccp.globals.use_sdv:

    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    from sdv.sampling import Condition
    from scipy.stats import gaussian_kde


    def fit_kde_sampler(data):
        kde = gaussian_kde(data, bw_method='scott')

        def sampler(n):
            samples = kde.resample(n).flatten()
            # clip to ensure values stay between 0 and 1
            return np.clip(samples, 0, 1)

        return sampler


    def load_data_to_process_for_ctgan(data_to_use: list = ['2024', '2025']):
        a26 = pd.read_csv('instances/2026/4. Model Input/2026 AFSCs.csv')
        afscs = np.array(a26['AFSC'])

        # Load in the data
        dfs = {}
        arrs = {}
        for year in data_to_use:
            yr = year[2:]

            # Load in cadets/AFSC data
            dfs[f'a{yr}'] = pd.read_csv(f'instances/{year}/4. Model Input/{year} AFSCs.csv')
            dfs[f'c{yr}'] = pd.read_csv(f'instances/{year}/4. Model Input/{year} Cadets.csv')
            dfs[f'cu{yr}'] = pd.read_csv(f'instances/{year}/4. Model Input/{year} Cadets Utility.csv')
            dfs[f'cuf{yr}'] = pd.read_csv(f'instances/{year}/4. Model Input/{year} Cadets Utility (Final).csv')
            dfs[f'au{yr}'] = pd.read_csv(f'instances/{year}/4. Model Input/{year} AFSCs Utility.csv')

            # Load in cadets selected data if we have it
            if year in ['2025', '2026']:
                dfs[f'cs{yr}'] = pd.read_csv(f'instances/{year}/4. Model Input/{year} Cadets Selected.csv')

        # Modify the data for 2024
        for d_name, _ in dfs.items():
            if 'u' in d_name or 's' in d_name:
                if '24' in d_name:
                    dfs[d_name]['18X'] = dfs[d_name]['11U']
                    dfs[d_name]['USSF_R'] = 0
                    dfs[d_name]['USSF_U'] = 0
                    dfs[d_name].loc[dfs['c24']['USAFA'] == 0, 'USSF_R'] = \
                        dfs[d_name]['13S1S'].loc[dfs['c24']['USAFA'] == 0]
                    dfs[d_name].loc[dfs['c24']['USAFA'] == 1, 'USSF_U'] = \
                        dfs[d_name]['13S1S'].loc[dfs['c24']['USAFA'] == 1]
                dfs[d_name] = dfs[d_name][afscs]
                arrs[d_name] = np.array(dfs[d_name])

        return dfs, arrs, afscs


    def prepare_2024_data(dfs: dict, arrs: dict, afscs: np.ndarray):
        # Initialize CTGAN data for 2024
        df = pd.DataFrame()
        utility_matrix = np.ones(arrs['cu24'].shape) * 0.1
        df['YEAR'] = [2024 for _ in range(len(dfs['c24']))]
        df['CIP1'] = 'c' + dfs['c24']['CIP1'].fillna('').astype(str)
        df['CIP2'] = dfs['c24']['CIP2'].fillna('').astype(str)
        df.loc[df["CIP2"] != 'None', 'CIP2'] = 'c' + df.loc[df["CIP2"] != 'None', 'CIP2']
        df['Merit'] = dfs['c24']['Merit']

        # Loop through each cadet (I know, very manual process...)
        for i in range(len(dfs['c24'])):

            # Get SOC info
            soc = 'USAFA' if dfs['c24'].loc[i, 'USAFA'] else "ROTC"
            df.loc[i, 'SOC'] = soc

            columns = [col for col in dfs['c24'] if 'pref' in col.lower()]
            pref_count = (dfs['c24'].iloc[i][columns].str.strip() != '').sum()

            # Force everything to string and ensure numpy dtype is string (not object)
            prefs = np.array(dfs['c24'].iloc[i][columns][:pref_count].astype(str).values, dtype=str)

            # Replace specific AFSCs
            ussf = 'USSF_' + soc[0]
            prefs[prefs == '13S1S'] = ussf
            prefs[prefs == '11U'] = '18X'

            # Now safely use np.char.find on a proper string array
            mask_s = np.char.find(prefs, 'S') != -1
            prefs[mask_s & (prefs != ussf)] = ''
            prefs = prefs[prefs != '']
            if 'nan' in prefs:
                prefs = np.array([afsc for afsc in prefs if afsc != 'nan'])
            pref_count = len(prefs)  # New number of preferences

            # Update utilities arrays
            # print(len(prefs), prefs)
            indices = np.array([np.where(afscs == afsc)[0][0] for afsc in prefs])
            utilities = arrs['cu24'][i, indices]
            num_selected = len(np.where(utilities > 0)[0])
            indiff = min(num_selected, pref_count - 1)
            utilities[indiff:] = 0.1
            utilities[0:indiff] = utilities[0:indiff] * 0.9 + 0.1

            # Update bottom choices
            for x in np.arange(1, 4):
                afsc = prefs[pref_count - x]
                if afsc[0] == '6':
                    break

                if x == 1:
                    df.loc[i, 'Last Choice'] = afsc
                    utilities[pref_count - x] = 0
                elif x == 2:
                    df.loc[i, '2nd-Last Choice'] = afsc
                    utilities[pref_count - x] = 0.05
                else:
                    df.loc[i, '3rd-Last Choice'] = afsc
                    utilities[pref_count - x] = 0.05

            # FIll in utilities
            utility_matrix[i, indices] = utilities

        # Add in cadet data
        for j, afsc in enumerate(afscs):
            df[f'{afsc}_Cadet'] = utility_matrix[:, j]

        # Add in AFSC data
        for j, afsc in enumerate(afscs):
            df[f'{afsc}_AFSC'] = dfs['au24'][afsc]

        # Convert to integer
        df['YEAR'] = df['YEAR'].astype(int)

        # Add volunteer columns
        rated_afscs = ['11XX_R', '11XX_U', '12XX', '13B', '18X']
        df['Rated Vol'] = df[[f'{afsc}_Cadet' for afsc in rated_afscs]].sum(axis=1) != 0.5
        df['USSF Vol'] = df[[f'{afsc}_Cadet' for afsc in ['USSF_R', 'USSF_U']]].sum(axis=1) != 0.2

        # Combine AFSCs segmented by SOC (11XX/USSF)
        df = fix_soc_afscs_to_generic(df=df, afscs=afscs)
        return df


    def prepare_year_data(year: str, dfs: dict, afscs: np.ndarray):
        yr = year[2:]

        # Initialize CTGAN data for the given year
        df = pd.DataFrame()
        df['YEAR'] = [year for _ in range(len(dfs[f'c{yr}']))]
        df['CIP1'] = 'c' + dfs[f'c{yr}']['CIP1'].fillna('').astype(str)
        df['CIP2'] = dfs[f'c{yr}']['CIP2'].fillna('None').astype(str)
        df.loc[df["CIP2"] != 'None', 'CIP2'] = 'c' + df.loc[df["CIP2"] != 'None', 'CIP2']
        df['Merit'] = dfs[f'c{yr}']['Merit']
        df['SOC'] = 'USAFA'
        df.loc[dfs[f'c{yr}']['USAFA'] == 0, 'SOC'] = 'ROTC'

        # Assume your original column is named 'AFSCs' in DataFrame `df`
        # Adjust the column name as needed
        df['Last Choice'] = dfs[f'c{yr}']['Least Desired AFSC']
        df[['2nd-Last Choice', '3rd-Last Choice']] = dfs[f'c{yr}']['Second Least Desired AFSCs'].str.split(',', expand=True)

        # Optionally strip whitespace around values
        df['3rd-Last Choice'] = df['3rd-Last Choice'].str.strip()
        df['2nd-Last Choice'] = df['2nd-Last Choice'].str.strip()

        # Add in cadet data
        for afsc in afscs:
            df[f'{afsc}_Cadet'] = dfs[f'cuf{yr}'][afsc]

        # Add in AFSC data
        for afsc in afscs:
            df[f'{afsc}_AFSC'] = dfs[f'au{yr}'][afsc]

        # Add volunteer columns
        rated_afscs = ['11XX_R', '11XX_U', '12XX', '13B', '18X']
        df['Rated Vol'] = df[[f'{afsc}_Cadet' for afsc in rated_afscs]].sum(axis=1) != 0.5
        df['USSF Vol'] = df[[f'{afsc}_Cadet' for afsc in ['USSF_R', 'USSF_U']]].sum(axis=1) != 0.2

        # Combine AFSCs segmented by SOC (11XX/USSF)
        df = fix_soc_afscs_to_generic(df=df, afscs=afscs)
        return df


    def fix_soc_afscs_to_generic(df: pd.DataFrame, afscs: np.ndarray):

        # Make a generic AFSC merging in ROTC/USAFA segmented AFSCs
        for afsc in ['11XX', 'USSF']:
            for col in ["Cadet", "AFSC"]:
                df[f'{afsc}_{col}'] = df[f'{afsc}_R_{col}']
                df.loc[df['SOC'] == 'USAFA', f'{afsc}_{col}'] = df.loc[df['SOC'] == 'USAFA', f'{afsc}_U_{col}']
            for col in ['Last Choice', '2nd-Last Choice', '3rd-Last Choice']:
                df[col] = df[col].replace(f'{afsc}_U', afsc)
                df[col] = df[col].replace(f'{afsc}_R', afsc)

        # Strip out the columns we don't need anymore
        afscs_new = np.hstack((['USSF', '11XX'], afscs[4:]))
        back_cols = [f'{afsc}_Cadet' for afsc in afscs_new] + [f'{afsc}_AFSC' for afsc in afscs_new]
        front_cols = [col for col in df.columns if '_Cadet' not in col and '_AFSC' not in col]
        cols = front_cols + back_cols
        return df[cols]


    def process_instances_into_ctgan_data(data_to_use: list = ['2024', '2025']):

        print(f'Loading data to process for CTGAN: {data_to_use}.')
        dfs, arrs, afscs = load_data_to_process_for_ctgan(data_to_use=data_to_use)

        # Process data together
        df = pd.DataFrame()
        for year in data_to_use:
            print(f'Preparing data for {year}.')
            if year == '2024':
                new_df = prepare_2024_data(dfs=dfs, arrs=arrs, afscs=afscs)
            else:
                new_df = prepare_year_data(year=year, dfs=dfs, afscs=afscs)
            df = pd.concat((df, new_df))

        # Export data
        df.to_csv(afccp.globals.paths['support'] + 'data/ctgan_data.csv', index=False)


    def process_instances_into_afscs_data(data_to_use: list = ['2025', '2026']):

        # Load in dataframes
        dfs = {f'a{year[2:]}': pd.read_csv(f'instances/{year}/4. Model Input/{year} AFSCs.csv') for year in
               data_to_use}

        # Determine which AFSCs everyone qualifies for
        eligible = dfs['a25']['USAFA Eligible'] + dfs['a25']['ROTC Eligible']
        max_eligible = max(eligible)

        # Build out AFSCs proportions data for generating random PGL policies
        a_df = pd.DataFrame({'AFSC': dfs['a26']['AFSC'], 'All Eligible': eligible == max_eligible,
                             'Accessions Group': dfs['a26']['Accessions Group']})
        u_targets = dfs['a25']['USAFA Target'] + dfs['a26']['USAFA Target']
        pgl_targets = dfs['a25']['PGL Target'] + dfs['a26']['PGL Target']
        a_df['USAFA Proportion'] = u_targets / pgl_targets
        a_df['ROTC Proportion'] = 1 - a_df['USAFA Proportion']
        a_df['PGL Proportion'] = pgl_targets / sum(pgl_targets)
        for i in range(4):
            a_df[f'Deg Tier {i + 1}'] = dfs['a26'][f'Deg Tier {i + 1}']

        # Export AFSCs data
        filepath = afccp.globals.paths["support"] + 'data/afscs_data.csv'
        a_df.to_csv(filepath, index=False)


    def train_ctgan(epochs=1000, printing=True, name='CTGAN_24_25'):
        """
        Train CTGAN to produce realistic data based on the current "ctgan_data" file in the support sub-folder. This
        function then saves the ".pkl" file back to the support sub-folder
        """

        # Import data
        data = afccp.globals.import_csv_data(afccp.globals.paths['support'] + 'data/ctgan_data.csv')
        data = data[[col for col in data.columns if col not in ['YEAR']]]
        metadata = SingleTableMetadata()  # SDV requires this now
        metadata.detect_from_dataframe(data=data)  # get the metadata from dataframe

        # Create the synthesizer model
        model = CTGANSynthesizer(metadata, epochs=epochs, verbose=True)

        # List of constraints for CTGAN
        constraints = []

        # Get list of columns that must be between 0 and 1
        zero_to_one_columns = ["Merit"]
        for col in data.columns:
            if "_Cadet" in col or "_AFSC" in col:
                zero_to_one_columns.append(col)

        # Create the "zero to one" constraints and add them to our list of constraints
        for col in zero_to_one_columns:
            zero_to_one_constraint = {"constraint_class": "ScalarRange",
                                      "constraint_parameters": {
                                          'column_name': col,
                                          'low_value': 0,
                                          'high_value': 1,
                                          'strict_boundaries': False
                                      }}
            constraints.append(zero_to_one_constraint)

        # Add the constraints to the model
        model.add_constraints(constraints)

        # Train the model
        if printing:
            print("Training the model...")
        model.fit(data)

        # Save the model
        filepath = afccp.globals.paths["support"] + name + '.pkl'
        model.save(filepath)
        if printing:
            print("Model saved to", filepath)


    def generate_ctgan_instance(N=1600, name='CTGAN_24_25', pilot_condition=False, degree_qual_type='Consistent'):
        """
        This procedure takes in the specified number of cadets and then generates a representative problem
        instance using CTGAN that has been trained from a real class year of cadets
        :param pilot_condition: If we want to sample cadets according to pilot preferences
        (make this more representative)
        :param name: Name of the CTGAN model to import
        :param N: number of cadets
        :return: model fixed parameters
        """

        # Load in the model
        filepath = afccp.globals.paths["support"] + name + '.pkl'
        model = CTGANSynthesizer.load(filepath)

        # Split up the number of ROTC/USAFA cadets
        N_usafa = round(np.random.triangular(0.25, 0.33, 0.4) * N)
        N_rotc = N - N_usafa

        # Pilot is by far the #1 desired career field, let's make sure this is represented here
        N_usafa_pilots = round(np.random.triangular(0.3, 0.4, 0.43) * N_usafa)
        N_usafa_generic = N_usafa - N_usafa_pilots
        N_rotc_pilots = round(np.random.triangular(0.25, 0.3, 0.33) * N_rotc)
        N_rotc_generic = N_rotc - N_rotc_pilots

        # Condition the data generated to produce the right composition of pilot first choice preferences
        usafa_pilot_first_choice = Condition(num_rows = N_usafa_pilots, column_values={'SOC': 'USAFA', '11XX_Cadet': 1})
        usafa_generic_cadets = Condition(num_rows=N_usafa_generic, column_values={'SOC': 'USAFA'})
        rotc_pilot_first_choice = Condition(num_rows=N_rotc_pilots, column_values={'SOC': 'ROTC', '11XX_Cadet': 1})
        rotc_generic_cadets = Condition(num_rows=N_rotc_generic, column_values={'SOC': 'ROTC'})

        # Sample data  (Sampling from conditions may take too long!)
        if pilot_condition:
            data = model.sample_from_conditions(conditions=[usafa_pilot_first_choice, usafa_generic_cadets,
                                                            rotc_pilot_first_choice, rotc_generic_cadets])
        else:
            data = model.sample(N)

        # Load in AFSCs data
        filepath = afccp.globals.paths["support"] + 'data/afscs_data.csv'
        afscs_data = afccp.globals.import_csv_data(filepath)

        # Get list of AFSCs
        afscs = np.array(afscs_data['AFSC'])

        # Initialize parameter dictionary
        p = {'afscs': afscs, 'N': N, 'P': len(afscs), 'M': len(afscs), 'merit': np.array(data['Merit']),
             'cadets': np.arange(N), 'usafa': np.array(data['SOC'] == 'USAFA') * 1,
             'cip1': np.array(data['CIP1']), 'cip2': np.array(data['CIP2']), 'num_util': 10,  # 10 utilities taken
             'rotc': np.array(data['SOC'] == 'ROTC'), 'I': np.arange(N), 'J': np.arange(len(afscs))}

        # Clean up degree columns (remove the leading "c" I put there if it's there)
        for i in p['I']:
            if p['cip1'][i][0] == 'c':
                p['cip1'][i] = p['cip1'][i][1:]
            if p['cip2'][i][0] == 'c':
                p['cip2'][i] = p['cip2'][i][1:]

        # Fix percentiles for USAFA and ROTC
        re_scaled_om = p['merit']
        for soc in ['usafa', 'rotc']:
            indices = np.where(p[soc])[0]  # Indices of these SOC-specific cadets
            percentiles = p['merit'][indices]  # The percentiles of these cadets
            N = len(percentiles)  # Number of cadets from this SOC
            sorted_indices = np.argsort(percentiles)[::-1]  # Sort these percentiles (descending)
            new_percentiles = (np.arange(N)) / (N - 1)  # New percentiles we want to replace these with
            magic_indices = np.argsort(sorted_indices)  # Indices that let us put the new percentiles in right place
            new_percentiles = new_percentiles[magic_indices]  # Put the new percentiles back in the right place
            np.put(re_scaled_om, indices, new_percentiles)  # Put these new percentiles in combined SOC OM spot

        # Replace merit
        p['merit'] = re_scaled_om

        # Add AFSC features to parameters
        p['acc_grp'] = np.array(afscs_data['Accessions Group'])
        p['Deg Tiers'] = np.array(afscs_data.loc[:, 'Deg Tier 1': 'Deg Tier 4'])
        p['Deg Tiers'][pd.isnull(p["Deg Tiers"])] = ''  # TODO

        # Determine AFSCs by Accessions Group
        p['afscs_acc_grp'] = {}
        if 'acc_grp' in p:
            for acc_grp in ['Rated', 'USSF', 'NRL']:
                p['J^' + acc_grp] = np.where(p['acc_grp'] == acc_grp)[0]
                p['afscs_acc_grp'][acc_grp] = p['afscs'][p['J^' + acc_grp]]

        # Useful data elements to help us generate PGL targets
        usafa_prop, rotc_prop, pgl_prop = np.array(afscs_data['USAFA Proportion']), \
                                          np.array(afscs_data['ROTC Proportion']), \
                                          np.array(afscs_data['PGL Proportion'])

        # Total targets needed to distribute
        total_targets = int(p['N'] * min(0.95, np.random.normal(0.93, 0.08)))

        # PGL targets
        p['pgl'] = np.zeros(p['M']).astype(int)
        p['usafa_quota'] = np.zeros(p['M']).astype(int)
        p['rotc_quota'] = np.zeros(p['M']).astype(int)
        for j in p['J']:

            # Create the PGL target by sampling from the PGL proportion triangular distribution
            p_min = max(0, 0.8 * pgl_prop[j])
            p_max = 1.2 * pgl_prop[j]
            prop = np.random.triangular(p_min, pgl_prop[j], p_max)
            p['pgl'][j] = int(max(1, prop * total_targets))

            # Get the ROTC proportion of this PGL target to allocate
            if rotc_prop[j] in [1, 0]:
                prop = rotc_prop[j]
            else:
                rotc_p_min = max(0, 0.8 * rotc_prop[j])
                rotc_p_max = min(1, 1.2 * rotc_prop[j])
                prop = np.random.triangular(rotc_p_min, rotc_prop[j], rotc_p_max)

            # Create the SOC-specific targets
            p['rotc_quota'][j] = int(prop * p['pgl'][j])
            p['usafa_quota'][j] = p['pgl'][j] - p['rotc_quota'][j]

        # Initialize the other pieces of information here
        for param in ['quota_e', 'quota_d', 'quota_min', 'quota_max']:
            p[param] = p['pgl']

        # Break up USSF and 11XX AFSC by SOC
        for afsc in ['USSF', '11XX']:
            for col in ['Cadet', 'AFSC']:
                for soc in ['USAFA', 'ROTC']:
                    data[f'{afsc}_{soc[0]}_{col}'] = 0
                    data.loc[data['SOC'] == soc, f'{afsc}_{soc[0]}_{col}'] = data.loc[data['SOC'] == soc, f'{afsc}_{col}']

        c_pref_cols = [f'{afsc}_Cadet' for afsc in afscs]
        util_original = np.around(np.array(data[c_pref_cols]), 2)

        # Initialize cadet preference information
        p['c_utilities'] = np.zeros((p['N'], 10))
        p['c_preferences'] = np.array([[' ' * 6 for _ in range(p['M'])] for _ in range(p['N'])])
        p['cadet_preferences'] = {}
        p['c_pref_matrix'] = np.zeros((p['N'], p['M'])).astype(int)
        p['utility'] = np.zeros((p['N'], p['M']))

        # Loop through each cadet to tweak their preferences
        for i in p['cadets']:

            # Manually fix 62EXE preferencing from eligible cadets
            ee_j = np.where(afscs == '62EXE')[0][0]
            if '1410' in data.loc[i, 'CIP1'] or '1447' in data.loc[i, 'CIP1']:
                if np.random.rand() > 0.6:
                    util_original[i, ee_j] = np.around(max(util_original[i, ee_j], min(1, np.random.normal(0.8, 0.18))),
                                                       2)

            # Fix rated/USSF volunteer situation
            for acc_grp in ['Rated', 'USSF']:
                if data.loc[i, f'{acc_grp} Vol']:
                    if np.max(util_original[i, p[f'J^{acc_grp}']]) < 0.6:
                        util_original[i, p[f'J^{acc_grp}']] = 0
                        data.loc[i, f'{acc_grp} Vol'] = False
                else:  # Not a volunteer

                    # We have a higher preference for these kinds of AFSCs
                    if np.max(util_original[i, p[f'J^{acc_grp}']]) >= 0.6:
                        data.loc[i, f'{acc_grp} Vol'] = True  # Make them a volunteer now

            # Was this the last choice AFSC? Remove from our lists
            ordered_list = np.argsort(util_original[i])[::-1]
            last_choice = data.loc[i, 'Last Choice']
            if last_choice in afscs:
                j = np.where(afscs == last_choice)[0][0]
                ordered_list = ordered_list[ordered_list != j]

            # Add the "2nd least desired AFSC" to list
            second_last_choice = data.loc[i, '2nd-Last Choice']
            bottom = []
            if second_last_choice in afscs and afsc != last_choice:  # Check if valid and not in bottom choices
                j = np.where(afscs == second_last_choice)[0][0]  # Get index of AFSC
                ordered_list = ordered_list[ordered_list != j]  # Remove index from preferences
                bottom.append(second_last_choice)  # Add it to the list of bottom choices

            # If it's a valid AFSC that isn't already in the bottom choices
            third_last_choice = data.loc[i, '3rd-Last Choice']  # Add the "3rd least desired AFSC" to list
            if third_last_choice in afscs and afsc not in [last_choice, second_last_choice]:
                j = np.where(afscs == third_last_choice)[0][0]  # Get index of AFSC
                ordered_list = ordered_list[
                    ordered_list != j]  # Reordered_list = np.argsort(util_original[i])[::-1]move index from preferences
                bottom.append(third_last_choice)  # Add it to the list of bottom choices

            # If we have an AFSC in the bottom choices, but NOT the LAST choice, move one to the last choice
            if len(bottom) > 0 and pd.isnull(last_choice):
                afsc = bottom.pop(0)
                data.loc[i, 'Last Choice'] = afsc
            data.loc[i, 'Second Least Desired AFSCs'] = ', '.join(bottom)  # Put it in the dataframe

            # Save cadet preference information
            num_pref = 10 if np.random.rand() > 0.1 else int(np.random.triangular(11, 15, 26))
            p['c_utilities'][i] = util_original[i, ordered_list[:10]]
            p['cadet_preferences'][i] = ordered_list[:num_pref]
            p['c_preferences'][i, :num_pref] = afscs[p['cadet_preferences'][i]]
            p['c_pref_matrix'][i, p['cadet_preferences'][i]] = np.arange(1, len(p['cadet_preferences'][i]) + 1)
            p['utility'][i, p['cadet_preferences'][i][:10]] = p['c_utilities'][i]

        # Get qual matrix information
        p['Qual Type'] = degree_qual_type
        p = afccp.data.adjustments.gather_degree_tier_qual_matrix(cadets_df=None, parameters=p)

        # Get the qual matrix to know what people are eligible for
        ineligible = (np.core.defchararray.find(p['qual'], "I") != -1) * 1
        eligible = (ineligible == 0) * 1
        I_E = [np.where(eligible[:, j])[0] for j in p['J']]  # set of cadets that are eligible for AFSC j

        # Modify AFSC utilities based on eligibility
        a_pref_cols = [f'{afsc}_AFSC' for afsc in afscs]
        p['afsc_utility'] = np.around(np.array(data[a_pref_cols]), 2)
        for acc_grp in ['Rated', 'USSF']:
            for j in p['J^' + acc_grp]:
                volunteer_col = np.array(data['Rated Vol'])
                volunteers = np.where(volunteer_col)[0]
                not_volunteers = np.where(volunteer_col == False)[0]
                ranked = np.where(p['afsc_utility'][:, j] > 0)[0]
                unranked = np.where(p['afsc_utility'][:, j] == 0)[0]

                # Fill in utility values with OM for rated folks who don't have an AFSC score
                volunteer_unranked = np.intersect1d(volunteers, unranked)
                p['afsc_utility'][volunteer_unranked, j] = p['merit'][volunteer_unranked]

                # If the cadet didn't actually volunteer, they should have utility of 0
                non_volunteer_ranked = np.intersect1d(not_volunteers, ranked)
                p['afsc_utility'][non_volunteer_ranked, j] = 0

        # Remove cadets from this AFSC's preferences if the cadet is not eligible
        for j in p['J^NRL']:

            # Get appropriate sets of cadets
            eligible_cadets = I_E[j]
            ineligible_cadets = np.where(ineligible[:, j])[0]
            ranked_cadets = np.where(p['afsc_utility'][:, j] > 0)[0]
            unranked_cadets = np.where(p['afsc_utility'][:, j] == 0)[0]

            # Fill in utility values with OM for eligible folks who don't have an AFSC score
            eligible_unranked = np.intersect1d(eligible_cadets, unranked_cadets)
            p['afsc_utility'][eligible_unranked, j] = p['merit'][eligible_unranked]

            # If the cadet isn't actually eligible, they should have utility of 0
            ineligible_ranked = np.intersect1d(ineligible_cadets, ranked_cadets)
            p['afsc_utility'][ineligible_ranked, j] = 0

        # Collect AFSC preference information
        p['afsc_preferences'] = {}
        p['a_pref_matrix'] = np.zeros((p['N'], p['M'])).astype(int)
        for j in p['J']:

            # Sort the utilities to get the preference list
            utilities = p["afsc_utility"][:, j]
            ineligible_indices = np.where(utilities == 0)[0]
            sorted_indices = np.argsort(utilities)[::-1][:p['N'] - len(ineligible_indices)]
            p['afsc_preferences'][j] = sorted_indices

            # Since 'afsc_preferences' is an array of AFSC indices, we can do this
            p['a_pref_matrix'][p['afsc_preferences'][j], j] = np.arange(1, len(p['afsc_preferences'][j]) + 1)

        # Needed information for rated OM matrices
        dataset_dict = {'rotc': 'rr_om_matrix', 'usafa': 'ur_om_matrix'}
        cadets_dict = {'rotc': 'rr_om_cadets', 'usafa': 'ur_om_cadets'}
        p["Rated Cadets"] = {}

        # Create rated OM matrices for each SOC
        for soc in ['usafa', 'rotc']:

            # Rated AFSCs for this SOC
            if soc == 'rotc':
                rated_J_soc = np.array([j for j in p['J^Rated'] if '_U' not in p['afscs'][j]])
            else:  # usafa
                rated_J_soc = np.array([j for j in p['J^Rated'] if '_R' not in p['afscs'][j]])

            # Cadets from this SOC
            soc_cadets = np.where(p[soc])[0]

            # Determine which cadets are eligible for at least one rated AFSC
            p["Rated Cadets"][soc] = np.array([i for i in soc_cadets if np.sum(p['c_pref_matrix'][i, rated_J_soc]) > 0])
            p[cadets_dict[soc]] = p["Rated Cadets"][soc]

            # Initialize OM dataset
            p[dataset_dict[soc]] = np.zeros([len(p["Rated Cadets"][soc]), len(rated_J_soc)])

            # Create OM dataset
            for col, j in enumerate(rated_J_soc):

                # Get the maximum rank someone had
                max_rank = np.max(p['a_pref_matrix'][p["Rated Cadets"][soc], j])

                # Loop through each cadet to convert rank to percentile
                for row, i in enumerate(p["Rated Cadets"][soc]):
                    rank = p['a_pref_matrix'][i, j]
                    if rank == 0:
                        p[dataset_dict[soc]][row, col] = 0
                    else:
                        p[dataset_dict[soc]][row, col] = (max_rank - rank + 1) / max_rank

        # Return parameters
        return p


    def re_calculate_ots_om_and_afsc_rankings(data: pd.DataFrame):

        # Re-scale OM!
        N = len(data)
        percentiles = np.array(data['Merit'])
        sorted_indices = np.argsort(percentiles)[::-1]  # Sort these percentiles (descending)
        new_percentiles = (np.arange(N)) / (N - 1)  # New percentiles we want to replace these with
        magic_indices = np.argsort(sorted_indices)  # Indices that let us put the new percentiles in right place
        new_percentiles = new_percentiles[::-1][magic_indices]  # Put the new percentiles back in the right place
        data['Merit'] = new_percentiles  # Load back into data

        # Re-bake in OM and Cadet Utility to AFSC rankings
        for col in [col for col in data.columns if '_AFSC' in col]:
            afsc = col.split('_')[0]
            scalar = np.random.triangular(0.1, 0.5, 0.9, len(data))
            data[col] = scalar * data[col] + (1 - scalar) * data['Merit']
            scalar = np.random.triangular(0.3, 0.7, 0.9, len(data))
            data[col] = scalar * data[col] + (1 - scalar) * data[f'{afsc}_Cadet']
        return data


    def generate_data_with_degree_preference_fixes(model, full_data, afscs_df):

        # Filter dataframe to rare AFSCs (degree-wise)
        afscs_rare_eligible = ['13H', '32EXA', '32EXC', '32EXE', '32EXF',
                               '32EXJ', '61C', '61D', '62EXC', '62EXE', '62EXH', '62EXI']
        afscs_rare_df = afscs_df.set_index('AFSC')['OTS Target'].loc[afscs_rare_eligible]

        # Extract data generating parameters
        total_gen, afsc_cip_data, afsc_cip_conditions, afsc_util_samplers, cadet_util_samplers = \
            extract_afsc_cip_sampling_information(full_data, afscs_rare_eligible, afscs_rare_df)

        # Generate the data
        data = sample_cadets_for_degree_conditions(model, total_gen, afscs_rare_eligible, afsc_cip_data,
                                                   afsc_cip_conditions)

        # Modify the utilities for the cadet/AFSC pairs
        i = 0
        for afsc in afscs_rare_eligible:
            for cip, count in afsc_cip_data[afsc].items():
                count = int(count)
                data.loc[i:i + count - 1, f'{afsc}_Cadet'] = cadet_util_samplers[afsc](count)
                data.loc[i:i + count - 1, f'{afsc}_AFSC'] = afsc_util_samplers[afsc](count)
                i += count

        return data


    def sample_cadets_for_degree_conditions(model, total_gen, afscs_rare_eligible, afsc_cip_data, afsc_cip_conditions):

        # Generate dataframe
        data = pd.DataFrame()
        i = 0
        for afsc in afscs_rare_eligible:
            for cip, count in afsc_cip_data[afsc].items():
                print(f'{afsc} {cip}: {int(count)}...')
                df_gen = model.sample_from_conditions([afsc_cip_conditions[afsc][cip]])
                data = pd.concat((data, df_gen), ignore_index=True)
                i += count
                print(f'{afsc} {cip}: ({int(i)}/{int(total_gen)}) {round((i / total_gen) * 100, 2)}% complete.')

        return data


    def extract_afsc_cip_sampling_information(full_data, afscs_rare_eligible, afscs_rare_df):

        afsc_cip_data = {}
        afsc_util_samplers = {}
        cadet_util_samplers = {}
        afsc_cip_conditions = {}
        total_gen = 0
        for afsc in afscs_rare_eligible:

            # Filter the real data on people who wanted this AFSC, and the AFSC wanted them
            conditions = (full_data[f'{afsc}_AFSC'] > 0.6) & (full_data[f'{afsc}_Cadet'] > 0.6)
            columns = ['YEAR', 'CIP1', 'CIP2', 'Merit', 'SOC', f'{afsc}_Cadet', f'{afsc}_AFSC']

            # Get the degrees of these people
            d = full_data.loc[conditions][columns]['CIP1'].value_counts()
            degrees = np.array(d.index)

            # Figure out how many degrees we have to ensure are present in this newly created dataset
            val = int(afscs_rare_df.loc[afsc])
            if afsc == '62EXE':  # We struggle to fill this quota!!
                val = val / 2
            num_gen = np.ceil(max(val * 1.4, val + 3))
            proportions = np.array(d ** 3) / np.array(d ** 3).sum()  # Tip the scales in favor of the more common CIP
            counts = safe_round(proportions * num_gen)
            afsc_cip_data[afsc] = pd.Series(counts, index=degrees)  # Save the degree information for this AFSC
            afsc_cip_data[afsc] = afsc_cip_data[afsc][afsc_cip_data[afsc] > 0]

            # Save functions to sample cadet/AFSC utilities for the ones with these degrees
            afsc_util_samplers[afsc] = fit_kde_sampler(list(full_data.loc[conditions][columns][f'{afsc}_AFSC']))
            cadet_util_samplers[afsc] = fit_kde_sampler(list(full_data.loc[conditions][columns][f'{afsc}_Cadet']))

            afsc_cip_conditions[afsc] = {}
            for cip, count in afsc_cip_data[afsc].items():
                condition = Condition(num_rows=int(count), column_values={"CIP1": cip})
                afsc_cip_conditions[afsc][cip] = condition
                total_gen += count

        return total_gen, afsc_cip_data, afsc_cip_conditions, afsc_util_samplers, cadet_util_samplers


    def align_ots_preferences_and_degrees_somewhat(data: pd.DataFrame, afscs_acc_grp):

        # Clean up degree columns (remove the leading "c" I put there if it's there)
        for i in data.index:
            if data.loc[i, 'CIP1'][0] == 'c':
                data.loc[i, 'CIP1'] = str(int(data.loc[i, 'CIP1'][1:].replace('.0', '')))
            if data.loc[i, 'CIP2'][0] == 'c':
                data.loc[i, 'CIP2'] = str(int(data.loc[i, 'CIP2'][1:].replace('.0', '')))

        # # OTS candidates won't be asked these questions
        # data = data[[col for col in data.columns if col not in ['Last Choice', '2nd-Last Choice', '3rd-Last Choice']]]

        # Convert info to numpy arrays
        cadet_cols = np.array([col for col in data.columns if '_Cadet' in col])
        util_original = np.array(data[cadet_cols])

        # Loop through each cadet to tweak their preferences
        for i in data.index:

            # Fix rated/USSF volunteer situation
            for acc_grp in ['Rated', 'USSF']:
                if data.loc[i, f'{acc_grp} Vol']:
                    if np.max(util_original[i, afscs_acc_grp[acc_grp]]) < 0.6:
                        util_original[i, afscs_acc_grp[acc_grp]] = 0
                        data.loc[i, f'{acc_grp} Vol'] = False
                else:  # Not a volunteer

                    # We have a higher preference for these kinds of AFSCs
                    if np.max(util_original[i, afscs_acc_grp[acc_grp]]) >= 0.6:
                        data.loc[i, f'{acc_grp} Vol'] = True  # Make them a volunteer now

        # Save utility information back to data
        data[cadet_cols] = util_original
        return data


    def construct_parameter_dictionary_and_augment_data(data: pd.DataFrame, afscs: np.ndarray, afscs_df: pd.DataFrame,
                                                        afscs_acc_grp: dict, nrl_indices: np.ndarray):

        # Construct parameter dictionary for OTS cadets
        N = len(data)
        p = {'cip1': np.array(data['CIP1']), 'cip2': np.array(data['CIP2']), 'afscs': afscs, 'M': len(afscs),
             'Qual Type': 'Consistent', 'N': N, 'P': len(afscs), 'I': np.arange(len(data)), 'J': np.arange(len(afscs)),
             'merit': np.array(data['Merit']), 'Deg Tiers': np.array(afscs_df.loc[3:, 'Deg Tier 1': 'Deg Tier 4'])}

        p['Deg Tiers'][pd.isnull(p["Deg Tiers"])] = ''  # TODO
        p = afccp.data.adjustments.gather_degree_tier_qual_matrix(cadets_df=None, parameters=p)

        # Get the qual matrix to know what people are eligible for
        ineligible = (np.core.defchararray.find(p['qual'], "I") != -1) * 1
        eligible = (ineligible == 0) * 1
        I_E = [np.where(eligible[:, j])[0] for j in p['J']]  # set of cadets that are eligible for AFSC j

        # Modify AFSC utilities based on eligibility
        a_pref_cols = [f'{afsc}_AFSC' for afsc in afscs]
        p['afsc_utility'] = np.around(np.array(data[a_pref_cols]), 2)
        for acc_grp in ['Rated', 'USSF']:
            for j in afscs_acc_grp[acc_grp]:
                volunteer_col = np.array(data['Rated Vol'])
                volunteers = np.where(volunteer_col)[0]
                not_volunteers = np.where(volunteer_col == False)[0]
                ranked = np.where(p['afsc_utility'][:, j] > 0)[0]
                unranked = np.where(p['afsc_utility'][:, j] == 0)[0]

                # Fill in utility values with OM for rated folks who don't have an AFSC score
                volunteer_unranked = np.intersect1d(volunteers, unranked)
                p['afsc_utility'][volunteer_unranked, j] = p['merit'][volunteer_unranked]

                # If the cadet didn't actually volunteer, they should have utility of 0
                non_volunteer_ranked = np.intersect1d(not_volunteers, ranked)
                p['afsc_utility'][non_volunteer_ranked, j] = 0

        # Remove cadets from this AFSC's preferences if the cadet is not eligible
        for j in nrl_indices:
            # Get appropriate sets of cadets
            eligible_cadets = I_E[j]
            ineligible_cadets = np.where(ineligible[:, j])[0]
            ranked_cadets = np.where(p['afsc_utility'][:, j] > 0)[0]
            unranked_cadets = np.where(p['afsc_utility'][:, j] == 0)[0]

            # Fill in utility values with OM for eligible folks who don't have an AFSC score
            eligible_unranked = np.intersect1d(eligible_cadets, unranked_cadets)
            p['afsc_utility'][eligible_unranked, j] = p['merit'][eligible_unranked]

            # If the cadet isn't actually eligible, they should have utility of 0
            ineligible_ranked = np.intersect1d(ineligible_cadets, ranked_cadets)
            p['afsc_utility'][ineligible_ranked, j] = 0

        # Put new calculated utilities back into dataframe
        data[a_pref_cols] = p['afsc_utility']
        return data, p


    def construct_full_afsc_preferences_data(p, a_pref_df, afscs, nrl_indices):

        # Load in data and extra non-rated AFSC preferences/rankings
        nrl_rankings_current = np.array(a_pref_df[afscs[nrl_indices]])

        # Create full utility matrix for NRL AFSCs
        shape = (nrl_rankings_current.shape[0] + len(p['afsc_utility']), nrl_rankings_current.shape[1])
        nrl_afsc_utility_full = np.zeros(shape)

        # Loop through each AFSC to calculate utility from preferences (Current 2026 info)
        afsc_utility_curr = np.zeros(nrl_rankings_current.shape)
        for j, afsc in enumerate(afscs[nrl_indices]):

            # Indices of eligible folks for this AFSC
            indices = np.where(nrl_rankings_current[:, j])[0]
            num_nonzero = len(indices)  # How many are eligible?
            sorted_indices = np.argsort(nrl_rankings_current[:, j])  # Sort them!

            # Turn the 1, 2, 3, ... 10 list to 1, 0.9, 0.8, ..., 0.1
            utils = 1 - (np.arange(1, num_nonzero + 1) / num_nonzero) + 1 / num_nonzero

            # Place the utilities into the matrix in the correct spots
            afsc_utility_curr[sorted_indices[-num_nonzero:], j] = utils

            # Combine OTS utilities into full utility matrix for NRL AFSCs
            if afsc in afscs:
                j_o = np.where(afscs == afsc)[0][0]
                nrl_afsc_utility_full[:len(afsc_utility_curr), j] = afsc_utility_curr[:, j]
                nrl_afsc_utility_full[len(afsc_utility_curr):, j] = p['afsc_utility'][:, j_o]
            else:
                print(afsc, 'not in AFSCs')

        # Convert utilities to preferences combining USAFA, ROTC, OTS cadets for NRL
        nrl_a_pref_matrix_full = np.zeros(shape).astype(int)
        nrl_afsc_preferences = {}
        full_N = len(nrl_a_pref_matrix_full)
        for j, afsc in enumerate(afscs[nrl_indices]):
            # Sort the utilities to get the preference list
            utilities = nrl_afsc_utility_full[:, j]
            ineligible_indices = np.where(utilities == 0)[0]
            sorted_indices = np.argsort(utilities)[::-1][:full_N - len(ineligible_indices)]
            nrl_afsc_preferences[j] = sorted_indices

            # Since 'afsc_preferences' is an array of AFSC indices, we can do this
            nrl_a_pref_matrix_full[nrl_afsc_preferences[j], j] = np.arange(1, len(nrl_afsc_preferences[j]) + 1)

        # Get list of columns used in the cadet-AFSC matrices
        standard_afsc_df_columns = np.array(a_pref_df.columns)
        i = np.where(standard_afsc_df_columns == '11XX_U')[0][0]  # Add OTS pilot AFSC
        standard_afsc_df_columns = np.insert(standard_afsc_df_columns, i + 1, '11XX_O')

        full_a_pref_df = pd.DataFrame({'Cadet': np.arange(len(nrl_a_pref_matrix_full))})
        for col in standard_afsc_df_columns[1:]:
            if col in afscs[nrl_indices]:
                j = np.where(afscs[nrl_indices] == col)[0][0]
                full_a_pref_df[col] = nrl_a_pref_matrix_full[:, j]
            elif 'USSF' in col:
                full_a_pref_df.loc[:len(a_pref_df), col] = a_pref_df[col]
            else:
                full_a_pref_df[col] = 1
        full_a_pref_df = full_a_pref_df.fillna(0).astype(int)
        return full_a_pref_df


    def construct_full_cadets_data(p: dict, cadets_df: pd.DataFrame, data: pd.DataFrame, afscs: np.ndarray):

        # Get new OTS cadet indices
        N = len(data)
        ots_cadets = np.arange(len(cadets_df), len(cadets_df) + N)
        cadet_cols = np.array([col for col in data.columns if '_Cadet' in col])

        # Initialize cadet preference information
        p['c_utilities'] = np.zeros((p['N'], 10))
        p['c_preferences'] = np.array([[' ' * 6 for _ in range(p['M'])] for _ in range(p['N'])])
        p['cadet_preferences'] = {}
        p['c_pref_matrix'] = np.zeros((p['N'], p['M'])).astype(int)
        p['utility'] = np.zeros((p['N'], p['M']))
        util_original = np.around(np.array(data[cadet_cols]), 2)

        # Loop through each cadet to fix the preference information
        for i in p['I']:

            # Save cadet preference information
            ordered_list = np.argsort(util_original[i])[::-1]
            num_pref = int(np.random.triangular(3, 9, 18))
            p['c_utilities'][i, :min(10, num_pref)] = util_original[i, ordered_list[:min(10, num_pref)]]
            p['cadet_preferences'][i] = ordered_list[:num_pref]
            p['c_preferences'][i, :num_pref] = afscs[p['cadet_preferences'][i]]
            p['c_pref_matrix'][i, p['cadet_preferences'][i]] = np.arange(1, len(p['cadet_preferences'][i]) + 1)
            p['utility'][i, p['cadet_preferences'][i][:min(10, num_pref)]] = p['c_utilities'][i, :min(10, num_pref)]

            # Determine bottom choice AFSCs
            bottom_choices = []
            for col in ['Last Choice', '2nd-Last Choice', '3rd-Last Choice']:
                condition = data.loc[i, col] not in afscs[p['cadet_preferences'][i]]  # can't be a preference
                condition *= data.loc[i, col] in afscs  # has to be real AFSC (not NaN)
                condition *= (data.loc[i, col] not in bottom_choices)  # can't already be a bottom choice
                if condition:
                    bottom_choices.append(data.loc[i, col])

            # Add in bottom choices data
            if len(bottom_choices) == 1:
                data.loc[i, 'Least Desired AFSC'] = bottom_choices[0]
            elif len(bottom_choices) >= 2:
                data.loc[i, 'Second Least Desired AFSCs'] = ', '.join(bottom_choices[1:])
        p['selected'] = (p['c_pref_matrix'] > 0) * 1  # Create "selected" array

        # Create the OTS cadet dataframe
        ots_cadets_df = pd.DataFrame({'Cadet': ots_cadets, 'SOC': 'OTS', 'USAFA': 0,
                                      'CIP1': data['CIP1'], 'CIP2': data['CIP2'],
                                      'Merit': data['Merit'], 'Real Merit': data['Merit'],
                                      'Least Desired AFSC': data['Least Desired AFSC'],
                                      'Second Least Desired AFSCs': data['Second Least Desired AFSCs']})

        # Add in preferences and utilities
        for i in np.arange(10):
            ots_cadets_df[f'Util_{i + 1}'] = p['c_utilities'][:, i]
        for i in np.arange(20):
            ots_cadets_df[f'Pref_{i + 1}'] = p['c_preferences'][:, i]
        ots_cadets_df = ots_cadets_df.replace('11XX', '11XX_O')  # Add OTS pilot

        # Add in qual matrix
        for j, afsc in enumerate(afscs):
            if afsc == 'USSF':
                continue
            if afsc == '11XX':
                afsc = '11XX_O'
            ots_cadets_df[f'qual_{afsc}'] = p['qual'][:, j]

        # Add OTS cadets information to current cadets info
        new_cadets_df = pd.concat((cadets_df, ots_cadets_df))

        # Rearrange columns
        col = new_cadets_df.pop('CIP2')
        new_cadets_df.insert(5, 'CIP2', col)
        col = new_cadets_df.pop('CIP1')
        new_cadets_df.insert(5, 'CIP1', col)
        idx = np.where(new_cadets_df.columns == 'qual_11XX_U')[0][0]
        col = new_cadets_df.pop('qual_11XX_O')
        new_cadets_df.insert(int(idx + 1), 'qual_11XX_O', col)

        return new_cadets_df


    def compile_new_dataframes(new_dfs, p, cadets_df, afscs, rated, data, import_name):

        N = len(data)
        ots_cadets = np.arange(len(cadets_df), len(cadets_df) + N)

        # Create dummy buckets for OTS (I don't care about this stuff rn)
        p['afsc_buckets'] = np.ones(p['c_pref_matrix'].shape)

        # Add additional cadets data for OTS
        df_arr_dict = {"Cadets Preferences": "c_pref_matrix", "Cadets Utility": "utility",
                       "Cadets Selected": "selected", "AFSCs Buckets": 'afsc_buckets',
                       'ROTC Rated OM': '', 'USAFA Rated OM': ''}
        for df_name, arr_name in df_arr_dict.items():

            # Import current dataframe
            filepath = f'instances/{import_name}/4. Model Input/{import_name} {df_name}.csv'
            df_i = afccp.globals.import_csv_data(filepath)

            # USAFA/ROTC Rated data get pulled over directly
            if 'USAFA' in df_name or 'ROTC' in df_name:
                new_dfs[df_name] = df_i
                continue

            # Initialize OTS' dataframe
            o_df = pd.DataFrame({'Cadet': ots_cadets})

            # Add in array information
            for j, afsc in enumerate(afscs):
                if afsc == 'USSF':
                    continue
                if afsc == '11XX':
                    afsc = '11XX_O'
                o_df[afsc] = p[arr_name][:, j]

            # Add OTS cadets information to current cadets info
            new_df = pd.concat((df_i, o_df))
            col = new_df.pop('11XX_O')
            new_df.insert(5, '11XX_O', col)
            new_dfs[df_name] = new_df.fillna(0)

        # Create OTS Rated OM dataframe (Use OM as rated rankings for OTS)
        rated_cadets = np.where(p['afsc_utility'][:, rated[2]])[0]
        eligible_rated = (p['afsc_utility'][rated_cadets][:, rated] > 0) * 1
        om_arr = np.around(np.array([p['merit'][rated_cadets] for _ in rated]).T, 3) * eligible_rated
        rated_om_df = pd.DataFrame({'Cadet': rated_cadets + len(cadets_df)})
        for idx, afsc in enumerate(afscs[rated]):
            if afsc == '11XX':
                afsc = '11XX_O'
            rated_om_df[afsc] = om_arr[:, idx]
        new_dfs['OTS Rated OM'] = rated_om_df
        return new_dfs


    def augment_2026_data_with_ots(N: int=3000, import_name: str = '2026_0', export_name: str = '2026O'):

        # Load in original data
        print('Loading in data...')
        filepath = afccp.globals.paths["support"] + 'data/ctgan_data.csv'
        full_data = pd.read_csv(filepath)
        cadet_cols = np.array([col for col in full_data.columns if '_Cadet' in col])

        # Import 'AFSCs' data
        filepath = f'instances/{import_name}/4. Model Input/{import_name} AFSCs.csv'
        afscs_df = afccp.globals.import_csv_data(filepath)
        afscs = np.array([col.split('_')[0] for col in cadet_cols])

        # Load in the model
        print('Loading in model...')
        filepath = afccp.globals.paths["support"] + 'CTGAN_Full.pkl'
        model = CTGANSynthesizer.load(filepath)

        # Sample the data
        print('Sampling data...')
        data_degrees = generate_data_with_degree_preference_fixes(model, full_data, afscs_df)
        data_all_else = model.sample(N - len(data_degrees))
        data = pd.concat((data_degrees, data_all_else), ignore_index=True)

        # These are all OTS candidates now!
        data['SOC'] = 'OTS'

        # Determine AFSCs by accessions group
        rated = np.array([np.where(cadet_cols == f'{afsc}_Cadet')[0][0] for afsc in ['11XX', '12XX', '13B', '18X']])
        afscs_acc_grp = {'Rated': rated, 'USSF': np.array([0])}

        # Re-calculate OM/AFSC Rankings for OTS
        print('Modifying data...')
        data = re_calculate_ots_om_and_afsc_rankings(data)

        # OTS isn't going to USSF
        data['USSF Vol'], data['USSF_Cadet'], data['USSF_AFSC'] = False, 0, 0
        data = align_ots_preferences_and_degrees_somewhat(data, afscs_acc_grp)

        # Non-rated AFSC indices
        nrl_indices = np.array(
            [np.where(afscs == afsc)[0][0] for afsc in afscs if afsc not in ['USSF', '11XX', '12XX', '13B', '18X']])

        # Construct the parameter dictionary and adjust AFSC utilities
        data, p = construct_parameter_dictionary_and_augment_data(
            data, afscs, afscs_df, afscs_acc_grp, nrl_indices=nrl_indices)

        # Import AFSCs Preferences data
        filepath = f'instances/{import_name}/4. Model Input/{import_name} AFSCs Preferences.csv'
        a_pref_df = afccp.globals.import_csv_data(filepath)

        # Construct the full AFSC preference data
        full_a_pref_df = construct_full_afsc_preferences_data(p, a_pref_df, afscs, nrl_indices)

        # Import 'Cadets' dataframe
        filepath = f'instances/{import_name}/4. Model Input/{import_name} Cadets.csv'
        cadets_df = afccp.globals.import_csv_data(filepath)

        # Construct the cadets data
        full_cadets_df = construct_full_cadets_data(p, cadets_df, data, afscs)

        # Import CASTLE data
        filepath = f'instances/{import_name}/4. Model Input/{import_name} Castle Input.csv'
        castle_df = afccp.globals.import_csv_data(filepath)

        # Dictionary of dataframes to export with new OTS 2026 instance
        print('Compiling current 2026 data...')
        new_dfs = {'Cadets': full_cadets_df, 'AFSCs Preferences': full_a_pref_df, 'AFSCs': afscs_df, 'Raw Data': data,
                   'Castle Input': castle_df}
        new_dfs = compile_new_dataframes(new_dfs, p, cadets_df, afscs, rated, data, import_name)

        # Export new dataframes for new instance
        print('Export new data instance...')
        folder_path = f'instances/{export_name}/4. Model Input/'
        os.makedirs(folder_path, exist_ok=True)
        for df_name, df in new_dfs.items():
            print(f'Data: "{df_name}", Shape: {np.shape(df)}')
            filepath = f'{folder_path}{export_name} {df_name}.csv'
            df.to_csv(filepath, index=False)

