# import libraries
import numpy as np
import random
from math import *
from afccp.core.globals import *
from afccp.core.value_parameter_handling import *


# Value Parameter Generator
def value_parameter_realistic_generator(parameters, default_value_parameters, constraint_options=None,
                                        num_breakpoints=20, deterministic=False, constrain_merit=False,
                                        printing=False, data_name="B"):
    """
    Generates realistic value parameters based on cadet data
    :param data_name: name of data instance
    :param constrain_merit: if we want to constrain average merit
    :param deterministic: if we're generating random parameters or not
    :param num_breakpoints: number of breakpoints to use on value functions
    :param printing: if the procedure should print something
    :param parameters: fixed cadet data
    :param default_value_parameters: default value parameters
    :param constraint_options: df for constraint options
    :return: value parameters
    """

    if printing:
        print('Generating realistic value parameters...')

    # Load parameter data
    N = parameters['N']
    M = parameters['M']
    afscs = parameters['afsc_vector']
    small_afscs = np.where(parameters['quota'] < 40)[0]
    large_afscs = np.where(parameters['quota'] >= 40)[0]
    usafa_max_col = np.array(constraint_options.loc[:, 'USAFA Max']).astype(str)
    u_con_indices = np.where(usafa_max_col != '1')[0]
    usafa_con_afscs = afscs[u_con_indices]

    # Add the AFSC objectives that are included in this instance
    objective_lookups = {'Merit': 'merit', 'USAFA Proportion': 'usafa', 'Combined Quota': 'quota',
                         'USAFA Quota': 'usafa_quota', 'ROTC Quota': 'rotc_quota', 'Mandatory': 'mandatory',
                         'Desired': 'desired', 'Permitted': 'permitted', 'Utility': 'utility', 'Male': 'male',
                         'Minority': 'minority'}
    objectives = []
    for objective in list(objective_lookups.keys()):
        if objective_lookups[objective] in parameters:
            objectives.append(objective)
    objectives = np.array(objectives)
    O = len(objectives)

    # Initialize set of value parameters
    value_parameters = {'objectives': objectives, 'O': O, 'M': M, 'objective_weight': np.zeros([M, O]),
                        'objective_target': np.zeros([M, O]), 'num_breakpoints': num_breakpoints,
                        'value_functions': np.array([[" " * 100 for _ in range(O)] for _ in range(M)]),
                        'F_bp': [[[] for _ in range(O)] for _ in range(M)],
                        'F_v': [[[] for _ in range(O)] for _ in range(M)],
                        'constraint_type': np.zeros([M, O]).astype(int),
                        'objective_value_min': np.array([[" " * 20 for _ in range(O)] for _ in range(M)]),
                        'afsc_value_min': np.zeros(M), 'cadet_value_min': np.zeros(N),
                        'cadets_overall_value_min': 0, 'afscs_overall_value_min': 0}

    # Parameters that we loop through
    params = ['cadets_overall_weight', 'cadet_weight_function', 'afsc_weight_function', 'objective_weight',
              'objective_target', 'value_functions']

    # Loop through all value parameters
    for param in params:

        # Overall weights
        if param == 'cadets_overall_weight':

            # Load distribution parameters
            l, u, mu, sigma = 0.1, 0.6, 0.35, 0.2

            # Generate parameter
            if deterministic:
                v = 0.3
            else:
                v = l - 0.1
                while v < l or v > u:
                    v = np.random.normal(mu, sigma)

            # Load parameter
            value_parameters[param] = round(v, 2)
            value_parameters['afscs_overall_weight'] = round(1 - v, 2)

        # Cadet Weights
        elif param == 'cadet_weight_function':

            # Define function choices
            options = ['Equal', 'Linear', 'Exponential']
            probabilities = [0.05, 0.7, 0.25]

            # Choose Function
            if deterministic:
                func = 'Linear'
            else:
                func = random.choices(options, probabilities)[0]
            value_parameters['cadet_weight_function'] = func

            if func == 'Equal':
                value_parameters['cadet_weight'] = np.repeat(1 / N, N)
            elif func == 'Linear':
                value_parameters['cadet_weight'] = parameters['merit'] / parameters['sum_merit']
            else:

                # Generate rho parameter
                l, u, mu, sigma = 0.1, 0.5, 0.3, 0.1
                rho = l - 0.1
                while rho < l or rho > u:
                    rho = np.random.normal(mu, sigma)

                if random.uniform(0, 1) < 0.5:
                    rho = -rho

                # Generate cadet weights
                swing_weights = np.array([
                    (1 - exp(-i / rho)) / (1 - exp(-1 / rho)) for i in parameters['merit']])
                value_parameters['cadet_weight'] = swing_weights / sum(swing_weights)

        # AFSC Weights
        elif param == 'afsc_weight_function':

            # Define function choices
            options = ['Equal', 'Linear', 'Piece', 'Norm']
            probabilities = [0.1, 0.3, 0.3, 0.3]

            # Choose Function
            if deterministic:
                func = 'Piece'
            else:
                func = random.choices(options, probabilities)[0]
            value_parameters['afsc_weight_function'] = func

            if func == 'Equal':
                value_parameters['afsc_weight'] = np.repeat(1 / M, M)
            elif func == 'Linear':
                value_parameters['afsc_weight'] = parameters['quota'] / sum(parameters['quota'])
            elif func == 'Piece':

                # Generate AFSC weights
                swing_weights = np.zeros(M)
                for j, quota in enumerate(parameters['quota']):
                    if quota >= 200:
                        swing_weights[j] = 1
                    elif 150 <= quota < 200:
                        swing_weights[j] = 0.9
                    elif 100 <= quota < 150:
                        swing_weights[j] = 0.8
                    elif 50 <= quota < 100:
                        swing_weights[j] = 0.7
                    elif 25 <= quota < 50:
                        swing_weights[j] = 0.6
                    else:
                        swing_weights[j] = 0.5

                # Load weights
                value_parameters['afsc_weight'] = np.around(swing_weights / sum(swing_weights), 4)
            else:

                # Load distribution parameters
                l, u, mu, sigma = 0.5, 1.5, 1, 0.15

                # Generate AFSC weights
                swing_weights = np.zeros(M)
                for j, quota in enumerate(parameters['quota']):
                    v = l - 0.1
                    while v < l or v > u:
                        v = np.random.normal(mu, sigma)
                    swing_weights[j] = quota * v

                # Load weights
                value_parameters['afsc_weight'] = np.around(swing_weights / sum(swing_weights), 2)

        # AFSC Objective Weights
        elif param == 'objective_weight':

            swing_weights = np.zeros([M, O])

            # Choose balancing function
            options = ['Method 1', 'Method 2', 'Method 3']
            probabilities = [0.3, 0.3, 0.4]
            if deterministic:
                balancing_func = 'Method 3'
            else:
                balancing_func = random.choices(options, probabilities)[0]

            # Loop through all objectives
            loc_k = np.zeros(O).astype(int)
            for k, objective in enumerate(objectives):

                loc_k[k] = np.where(default_value_parameters['objectives'] == objective)[0][0]
                if objective == 'Merit':
                    value_parameters['merit_weight_function'] = balancing_func

                    if balancing_func == 'Method 1':
                        swing_weights[:, k] = np.array([
                            max(min(np.random.normal(30, 5), 50), 10) for _ in range(M)])

                    elif balancing_func == 'Method 2':
                        swing_weights[large_afscs, k] = np.array([
                            max(min(np.random.normal(40, 10), 60), 20) for _ in range(len(large_afscs))])

                    else:

                        if deterministic:
                            swing_weights[large_afscs, k] = np.repeat(40, len(large_afscs))
                            swing_weights[small_afscs, k] = np.repeat(10, len(small_afscs))
                        else:
                            swing_weights[large_afscs, k] = np.array([
                                max(min(np.random.normal(40, 10), 60), 20) for _ in range(len(large_afscs))])
                            swing_weights[small_afscs, k] = np.array([
                                max(min(np.random.normal(10, 2), 15), 5) for _ in range(len(small_afscs))])

                elif objective == 'USAFA Proportion':
                    value_parameters['usafa_proportion_weight_function'] = balancing_func

                    if balancing_func == 'Method 1':
                        swing_weights[:, k] = np.array([
                            max(min(np.random.normal(20, 5), 35), 5) for _ in range(M)])

                    elif balancing_func == 'Method 2':
                        swing_weights[large_afscs, k] = np.array([
                            max(min(np.random.normal(30, 10), 50), 10) for _ in range(len(large_afscs))])
                        swing_weights[u_con_indices, k] = np.array([
                            max(min(np.random.normal(30, 10), 50), 10) for _ in range(len(u_con_indices))])

                    else:

                        if deterministic:
                            swing_weights[large_afscs, k] = np.repeat(40, len(large_afscs))
                            swing_weights[small_afscs, k] = np.repeat(5, len(small_afscs))
                        else:
                            swing_weights[large_afscs, k] = np.array([
                                max(min(np.random.normal(40, 10), 60), 20) for _ in range(len(large_afscs))])
                            swing_weights[small_afscs, k] = np.array([
                                max(min(np.random.normal(5, 1), 10), 0) for _ in range(len(small_afscs))])

                elif objective == 'Combined Quota':
                    if deterministic:
                        swing_weights[:, k] = np.repeat(100, M)
                    else:
                        swing_weights[:, k] = np.array([
                            max(min(np.random.normal(100, 5), 110), 90) for _ in range(M)])

                elif objective == 'Utility':
                    if deterministic:
                        swing_weights[:, k] = np.repeat(35, M)
                    else:
                        swing_weights[:, k] = np.array([
                            max(min(np.random.normal(35, 20), 60), 10) for _ in range(M)])

            # Determine AFOCD tier makeup
            afocd_strs = np.array(['   ' for _ in range(M)])
            loc_j = np.zeros(M).astype(int)
            for j, afsc in enumerate(afscs):

                loc_j[j] = np.where(default_value_parameters['complete_afsc_vector'] == afsc)[0][0]
                a_str = ''
                for objective in ['Mandatory', 'Desired', 'Permitted']:

                    k = np.where(default_value_parameters['objectives'] == objective)[0][0]
                    if default_value_parameters['objective_target'][loc_j[j], k] != 0:
                        a_str += objective[:1]
                afocd_strs[j] = a_str

            # AFOCD Weights
            k_m = np.where(objectives == 'Mandatory')[0][0]
            k_d = np.where(objectives == 'Desired')[0][0]
            k_p = np.where(objectives == 'Permitted')[0][0]
            for j in range(M):

                if deterministic:
                    if afocd_strs[j] == 'M':
                        swing_weights[j, k_m] = 90
                    elif afocd_strs[j] == 'MD':
                        swing_weights[j, k_m] = 80
                        swing_weights[j, k_d] = 40
                    elif afocd_strs[j] == 'MP':
                        swing_weights[j, k_m] = 80
                        swing_weights[j, k_p] = 20
                    elif afocd_strs[j] == 'MDP':
                        swing_weights[j, k_m] = 80
                        swing_weights[j, k_d] = 50
                        swing_weights[j, k_p] = 30
                    elif afocd_strs[j] == 'DP':
                        swing_weights[j, k_d] = 60
                        swing_weights[j, k_p] = 30
                else:
                    if afocd_strs[j] == 'M':
                        swing_weights[j, k_m] = max(min(np.random.normal(90, 5), 100), 80)
                    elif afocd_strs[j] == 'MD':
                        swing_weights[j, k_m] = max(min(np.random.normal(80, 5), 90), 70)
                        swing_weights[j, k_d] = max(min(np.random.normal(40, 5), 50), 30)
                    elif afocd_strs[j] == 'MP':
                        swing_weights[j, k_m] = max(min(np.random.normal(80, 5), 85), 75)
                        swing_weights[j, k_p] = max(min(np.random.normal(20, 5), 30), 10)
                    elif afocd_strs[j] == 'MDP':
                        swing_weights[j, k_m] = max(min(np.random.normal(80, 10), 100), 60)
                        swing_weights[j, k_d] = max(min(np.random.normal(50, 5), 60), 40)
                        swing_weights[j, k_p] = max(min(np.random.normal(30, 5), 40), 20)
                    elif afocd_strs[j] == 'DP':
                        swing_weights[j, k_d] = max(min(np.random.normal(60, 5), 70), 50)
                        swing_weights[j, k_p] = max(min(np.random.normal(30, 5), 40), 20)

                # Load objective weights for this AFSC
                value_parameters[param][j, :] = swing_weights[j, :] / np.sum(swing_weights[j, :])

        # AFSC Objective Targets
        elif param == 'objective_target':

            for j, afsc in enumerate(afscs):

                usafa_max = constraint_options.loc[j, 'USAFA Max']
                for k, objective in enumerate(objectives):

                    if objective == 'Merit':
                        value_parameters['objective_target'][j, k] = parameters['sum_merit'] / N
                        if constrain_merit:
                            value_parameters['constraint_type'][j, k] = 4
                            value_parameters['objective_value_min'][j][k] = "0.35, 2"
                            if data_name in ['E'] and afsc in ['E31', 'E32']:
                                value_parameters['objective_value_min'][j][k] = "0.16, 2"
                            elif data_name in ['F'] and afsc in ["F31"]:
                                value_parameters['objective_value_min'][j][k] = "0.29, 2"
                            elif data_name in ['F'] and afsc in ["F13"]:
                                value_parameters['objective_value_min'][j][k] = "0.34, 2"

                    elif objective == 'USAFA Proportion':
                        if parameters['usafa_quota'][j] == 0 and data_name not in ["D", "F"]:
                            value_parameters['objective_target'][j, k] = 0

                        elif parameters['usafa_quota'][j] == parameters['quota'][j]:
                            value_parameters['objective_target'][j, k] = 1

                        else:
                            value_parameters['objective_target'][j, k] = parameters['usafa_proportion']

                        # AFSCs with maximum USAFA constraint aren't trying to balance the proportion
                        if afsc in usafa_con_afscs:
                            u_target = float(usafa_max.split(',')[0])
                            u_max = float(usafa_max.split(',')[1])
                            value_parameters['objective_target'][j, k] = u_target

                            # Constraints
                            value_parameters['constraint_type'][j, k] = 4
                            value_parameters['objective_value_min'][j][k] = str(0) + ", " + \
                                                                            str(u_max)

                    elif objective == 'Combined Quota':
                        value_parameters['objective_target'][j, k] = parameters['quota'][j]

                    elif objective == 'USAFA Quota':
                        value_parameters['objective_target'][j, k] = parameters['usafa_quota'][j]

                    elif objective == 'ROTC Quota':
                        value_parameters['objective_target'][j, k] = parameters['rotc_quota'][j]

                    elif objective == 'Male':
                        value_parameters['objective_target'][j, k] = parameters['male_proportion']

                    elif objective == 'Minority':
                        value_parameters['objective_target'][j, k] = parameters['minority_proportion']

                    elif objective == 'Utility':
                        value_parameters['objective_target'][j, k] = 1

                    else:  # AFOCD Objectives
                        value_parameters['objective_target'][j, k] = default_value_parameters[
                            'objective_target'][loc_j[j], loc_k[k]]

        # Value Functions
        elif param == 'value_functions':

            for j, afsc in enumerate(afscs):

                # AFSC variables
                cadets = parameters['I^E'][j]
                quota_str = constraint_options.loc[j, 'Combined Quota']
                mand_str = constraint_options.loc[j, 'Mandatory']

                # Loop through all objectives to create value functions
                for k, objective in enumerate(objectives):

                    # Initialize function parameters
                    target = value_parameters['objective_target'][j, k]
                    actual = None
                    maximum = None
                    q_minimum = 1

                    if value_parameters['objective_weight'][j, k] != 0:

                        if objective in ['Merit', 'USAFA Proportion']:

                            # Generate buffer y parameter
                            if deterministic:
                                buffer_y = 0.7
                            else:
                                buffer_y = 0
                                while buffer_y < 0.65 or buffer_y > 0.85:
                                    buffer_y = round(np.random.normal(0.7, 0.05), 3)

                            if objective == 'Merit':

                                # Function parameters
                                left_bm, right_bm = 0.1, 0.14
                                actual = np.mean(parameters['merit'][cadets])

                                # Rho distribution parameters
                                rho_params = {0: {'l': 0.04, 'u': 0.1, 'mu': 0.07, 'sigma': 0.01},
                                              1: {'l': 0.06, 'u': 0.1, 'mu': 0.08, 'sigma': 0.005},
                                              2: {'l': 0.06, 'u': 0.1, 'mu': 0.08, 'sigma': 0.005},
                                              3: {'l': 0.1, 'u': 0.15, 'mu': 0.125, 'sigma': 0.1}}

                                # Generate rho parameters
                                if deterministic:
                                    rhos = [0.07, 0.08, 0.08, 0.125]
                                else:
                                    rhos = []
                                    for i in range(4):
                                        r = rho_params[i]
                                        rho = r['l'] - 0.1
                                        while rho < r['l'] or rho > r['u']:
                                            rho = round(np.random.normal(r['mu'], r['sigma']), 3)
                                        rhos.append(rho)

                                # Create function string
                                vf_string = "Balance|" + str(left_bm) + ", " + str(right_bm) + ", " + \
                                            str(rhos[0]) + ", " + str(rhos[1]) + ", " + str(rhos[2]) + ", " + \
                                            str(rhos[3]) + ", " + str(buffer_y)

                            else:

                                # Function parameters
                                left_bm, right_bm = 0.12, 0.12
                                actual = np.mean(parameters['usafa'][cadets])

                                # Generate rho parameters
                                if deterministic:
                                    rhos = [0.1, 0.1, 0.1, 0.1]
                                else:
                                    rho = 0
                                    while rho < 0.08 or rho > 0.12:
                                        rho = round(np.random.normal(0.1, 0.01), 3)
                                    rhos = [rho, rho, rho, rho]

                                # Create function string
                                if target == 0 or afsc in usafa_con_afscs:
                                    vf_string = "Min Decreasing|" + str(rhos[0])
                                elif target == 1:
                                    vf_string = "Min Increasing|" + str(rhos[0])
                                else:
                                    vf_string = "Balance|" + str(left_bm) + ", " + str(right_bm) + ", " + \
                                                str(rhos[0]) + ", " + str(rhos[1]) + ", " + str(rhos[2]) + ", " + \
                                                str(rhos[3]) + ", " + str(buffer_y)

                        elif objective == 'Combined Quota':

                            # Function parameters
                            domain_max = 0.2
                            if '*' in quota_str:

                                # split string
                                split_str = quota_str.split('|')

                                # Get target upper bound
                                target_u = split_str[0].split(',')[1]
                                target_u = float(target_u[:len(target_u) - 1])  # remove *

                                # Get constraint upper bound
                                con_u = float(split_str[1].split(',')[1])

                                # Get constraint lower bound
                                q_minimum = float(split_str[1].split(',')[0])

                                # Choose Method
                                methods = ['Method 1', 'Method 2']
                                weights = [0.2, 0.8]
                                if deterministic:
                                    method = 'Method 2'
                                else:
                                    method = random.choices(methods, weights, k=1)[0]

                                if afsc in ["A5", "B3", "C3", "D5", "E6", "F5", "G5"]:
                                    method = 'Method 1'

                                actual = con_u
                                if method == 'Method 1':
                                    maximum = con_u
                                else:
                                    maximum = target_u
                            else:

                                # Pick Method 1
                                method = 'Method 1'

                                # if both ranges are valid
                                if '|' in quota_str:

                                    # split string
                                    split_str = quota_str.split('|')

                                    # Choose one of the upper bounds to use
                                    weights = [0.4, 0.6]
                                    if deterministic:
                                        index = 1
                                    else:
                                        index = random.choices([0, 1], weights, k=1)[0]

                                    maximum = float(split_str[index].split(',')[1])

                                else:
                                    maximum = float(quota_str.split(',')[1])

                                actual = maximum

                            # Rho distribution parameters
                            rho_params = {0: {'l': 0.2, 'u': 0.3, 'mu': 0.25, 'sigma': 0.03},
                                          1: {'l': 0.05, 'u': 0.1, 'mu': 0.075, 'sigma': 0.01},
                                          2: {'l': 0.03, 'u': 0.7, 'mu': 0.05, 'sigma': 0.005}}

                            if method == 'Method 1':

                                # Generate rho parameters
                                if deterministic:
                                    rhos = [0.25, 0.075]
                                else:
                                    rhos = []
                                    for i in range(2):
                                        r = rho_params[i]
                                        rho = r['l'] - 0.1
                                        while rho < r['l'] or rho > r['u']:
                                            rho = round(np.random.normal(r['mu'], r['sigma']), 3)
                                        rhos.append(rho)

                                # Create function string
                                vf_string = "Quota_Normal|" + str(domain_max) + ", " + str(rhos[0]) + ", " + \
                                            str(rhos[1])

                            else:

                                # Generate rho parameters
                                if deterministic:
                                    rhos = [0.25, 0.075, 0.05]
                                else:
                                    rhos = []
                                    for i in range(3):
                                        r = rho_params[i]
                                        rho = r['l'] - 0.1
                                        while rho < r['l'] or rho > r['u']:
                                            rho = round(np.random.normal(r['mu'], r['sigma']), 3)
                                        rhos.append(rho)

                                # Generate buffer y parameter
                                if deterministic:
                                    buffer_y = 0.6
                                else:
                                    buffer_y = 0
                                    while buffer_y < 0.5 or buffer_y > 0.7:
                                        buffer_y = round(np.random.normal(0.6, 0.05), 3)

                                # Create function string
                                vf_string = "Quota_Over|" + str(domain_max) + ", " + str(rhos[0]) + ", " + \
                                            str(rhos[1]) + ", " + str(rhos[2]) + ", " + str(buffer_y)

                            # Constraints
                            value_parameters['constraint_type'][j, k] = 4
                            value_parameters['objective_value_min'][j][k] = str(int(q_minimum * target)) + ", " + \
                                                                            str(int(actual * target))

                        elif objective in ['Mandatory', 'Desired', 'Permitted']:

                            # Generate rho parameter
                            if deterministic:
                                rho = 0.1
                            else:
                                rho = 0
                                while rho < 0.08 or rho > 0.12:
                                    rho = round(np.random.normal(0.1, 0.01), 3)

                            if objective == 'Mandatory':
                                function = 'Min Increasing'

                                if '*' in mand_str:

                                    # split string
                                    split_str = mand_str.split('|')

                                    # Get constraint bounds
                                    lower = float(split_str[1].split(',')[0])
                                    upper = float(split_str[1].split(',')[1])
                                else:
                                    lower = float(mand_str.split(',')[0])
                                    upper = float(mand_str.split(',')[1])

                                # Constraints
                                value_parameters['constraint_type'][j, k] = 3
                                value_parameters['objective_value_min'][j][k] = str(lower) + ", " + str(upper)

                                if data_name in ["D", "E", "F"] and \
                                        afsc in ["D18", "E18", "F19"]:
                                    value_parameters['constraint_type'][j, k] = 0

                            elif objective == 'Permitted':
                                function = 'Min Decreasing'
                            else:
                                if afsc in ["A26", "B33", "C26", "D27", "E26", "F26", "A13", "B15", "C14", "D16",
                                            "E12", "F15", "A1", "B2", "B11", "B34", "B36", "C1", "D2", "E3", "F2",
                                            "G1", "G13", "G27"]:
                                    function = 'Min Decreasing'
                                else:
                                    function = 'Min Increasing'

                            vf_string = function + '|' + str(rho)

                        elif objective == 'Utility':

                            # Generate rho parameter
                            if deterministic:
                                rho = 0.25
                            else:
                                rho = 0
                                while rho < 0.15 or rho > 0.35:
                                    rho = round(np.random.normal(0.25, 0.05), 3)

                            vf_string = 'Min Increasing|' + str(rho)
                        else:
                            vf_string = 'None'
                    else:
                        vf_string = 'None'

                    # Create function
                    value_parameters['value_functions'][j][k] = vf_string
                    if vf_string != 'None':
                        segment_dict = create_segment_dict_from_string(vf_string, target, actual=actual,
                                                                       maximum=maximum)
                        # print(afsc, objective, segment_dict)
                        value_parameters['F_bp'][j][k], value_parameters['F_v'][j][k] = value_function_builder(
                            segment_dict, num_breakpoints=num_breakpoints)

    return value_parameters