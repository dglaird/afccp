# Import Libraries
import random
import numpy as np
import warnings
import afccp.core.globals
from afccp.core.handling.preprocessing import cip_to_qual, cip_to_qual_direct

if afccp.core.globals.use_sdv:
    from sdv.evaluation import evaluate
    from sdv.constraints import GreaterThan
    from sdv.constraints import CustomConstraint

    # if 'workspace' in dir_path:  # Newer version!
    #     from sdv.constraints import Inequality
    #     from sdv.constraints import create_custom_constraint
    
    from sdv.tabular import CTGAN

warnings.filterwarnings('ignore')  # prevent red warnings from printing


def import_generator_parameters(filepath=afccp.core.globals.paths["support"] + "Instance_Generator_Parameters.xlsx",
                                printing=False):
    """
    Imports generator parameters used for the different generation functions
    :param filepath: excel sheet path
    :param printing: if we should print something
    :return: dataframes
    """
    if printing:
        print('Importing generator parameters...')
    targets = import_data(filepath, sheet_name="Targets")
    cip1 = import_data(filepath, sheet_name="CIP1")
    cip2 = import_data(filepath, sheet_name="CIP2")
    return targets, cip1, cip2


def simulate_model_fixed_parameters(N=1600, P=6, M=32, printing=False):
    """
    This procedure takes in the specified parameters (defined below) and then simulates new random "fixed" cadet/AFSC
    input parameters. These parameters are then returned and can be used to solve the VFT model.
    :param printing: whether or not the procedure should print out something
    :param N: number of cadets
    :param P: number of preferences allowed
    :param M: number of AFSCs
    :return: model fixed parameters
    """
    if printing:
        print('Simulating (Random)...')

    # Import CIP to Qual Matrix
    cip_qual_matrix = import_data(afccp.core.globals.paths["support"] + "Qual_CIP_Matrix_Scrubbed.xlsx",
                                  sheet_name="Qual Matrix")

    # Import Generator Parameters
    cip_df = {}
    targets, cip_df["CIP1"], cip_df["CIP2"] = import_generator_parameters()

    # Generate valid qualification matrix (every AFSC has at least one eligible cadet)
    invalid = True
    while invalid:

        # Generate CIP codes
        cips = {}
        for cip in cip_df:
            choices = np.array(cip_df[cip].loc[:, cip]).astype(str)
            probabilities = np.array(cip_df[cip].loc[:, 'Proportion'])
            cips[cip] = np.random.choice(choices, size=N, replace=True, p=probabilities)
            cips[cip] = np.where(cips[cip] == 'None', " " * 6, cips[cip])

        # Generate Qual Matrix, keep M AFSCs
        afscs = np.array(targets.loc[:, 'AFSC'])
        qual = cip_to_qual(afscs, cips['CIP1'], cips['CIP2'], cip_qual_matrix=cip_qual_matrix)[:, :M]
        afscs = afscs[:M]

        # Convert Qual Matrix into binary matrices indicating eligibility
        ineligible = (qual == 'I') * 1
        eligible = (qual != 'I') * 1

        # Ensure every AFSC has at least one eligible cadet
        invalid = False
        for j in range(M):
            if np.sum(eligible[:, j]) == 0:
                invalid = True
                break

    parameters = {'N': N, 'qual': qual, 'mandatory': (qual == 'M') * 1, 'desired': (qual == 'D') * 1,
                  'permitted': (qual == 'P') * 1, 'ineligible': ineligible, 'eligible': (ineligible == 0) * 1,
                  'utility': np.zeros([N, M]), 'afsc_vector': afscs, 'M': M}
    for cip in cips:
        parameters[cip] = cips[cip]

    # Generate cadet utilities for AFSC preferences 2 to P
    util_choices = np.array([[round(random.uniform(0, 1), 4) for _ in range(P - 1)] for _ in range(N)])

    # Create utility matrix from these utilities
    for i in range(N):
        # Put utilities generated above into random positions in utility matrix
        np.put(parameters['utility'][i, :], np.random.choice(M, P - 1, replace=False), util_choices[i, :])

        # Set one of the eligible AFSCs to have utility of 1
        indices = np.where(parameters['eligible'][i, :])[0]
        parameters['utility'][i, np.random.choice(indices)] = 1

    # Generate numbers of cadets with different demographics
    usafa_num = int(N * max(np.random.normal(0.28, 0.05), 0.20))  # number of usafa cadets
    men_num = int(N * min(np.random.normal(0.69, 0.05), 0.75))  # number of male cadets
    minority_num = int(N * max(np.random.normal(0.29, 0.05), 0.20))  # number of minority cadets

    # Generate cadet demographics
    parameters['male'] = np.append(np.ones(men_num), np.zeros(N - men_num))
    random.shuffle(parameters['male'])
    parameters['minority'] = np.append(np.ones(minority_num), np.zeros(N - minority_num))
    random.shuffle(parameters['minority'])
    parameters['usafa'] = np.append(np.ones(usafa_num), np.zeros(N - usafa_num))
    parameters['merit'] = np.append(np.round(1 - np.arange(usafa_num) / (usafa_num - 1), 4),
                                    np.round(1 - np.arange(N - usafa_num) / (N - usafa_num - 1), 4))
    parameters['sum_merit'] = parameters['merit'].sum()
    parameters['ID'] = np.arange(N)
    parameters['P'] = P

    # Specify AFSC targets
    total_quota = int(N * min(0.95, np.random.normal(0.93, 0.08)))
    total_means = np.array(targets['Total_Mean'])
    total_stds = np.array(targets['Total_Std'])
    usafa_means = np.array(targets['USAFA_Mean'])
    usafa_stds = np.array(targets['USAFA_Std'])
    over = np.array(targets['Over'])
    combined_proportions = np.array([max(np.random.normal(total_means[j], total_stds[j]), 0.001) for j in range(M)])
    combined_proportions = combined_proportions / sum(combined_proportions)
    combined_quota = np.ceil(combined_proportions * total_quota).astype(int)
    usafa_proportions = np.array([max(np.random.normal(usafa_means[j], usafa_stds[j]), 0) for j in range(M)])
    usafa_quota = np.floor(usafa_proportions * combined_quota).astype(int)
    rotc_quota = combined_quota - usafa_quota
    parameters['quota'] = combined_quota
    parameters['pgl'] = combined_quota
    parameters['usafa_quota'] = usafa_quota
    parameters['rotc_quota'] = rotc_quota
    parameters['quota_max'] = np.around(over[:M] * combined_quota)
    parameters['quota_min'] = combined_quota

    return parameters


def perfect_example_generator(N=8, P=2, M=2, printing=False):
    """
    Generates a perfect set of cadets and AFSCs. For some reason this isn't quite there yet... something is wrong but
    it is close
    :param N: number of cadets (has to be divisible by 4 and M)
    :param P: number of preferences
    :param M: number of AFSCs
    :param printing: Whether the procedure should print something
    :return: parameters, solution (vector form)
    """
    if printing:
        print('Generating perfect example...')

    # Import Generator Parameters
    cip_df = {}
    targets, cip_df["CIP1"], cip_df["CIP2"] = import_generator_parameters()

    # Get data
    complete_afsc_proportion_means = np.array(targets['Total_Mean'])
    complete_afsc_proportion_stds = np.array(targets['Total_Std'])
    complete_usafa_proportions = np.array(targets['USAFA_Mean'])
    complete_afsc_vector = np.array(targets['AFSC'])
    afsc_vector = complete_afsc_vector[:M]
    top_tier_cips = np.array(targets['Top Tier CIPs'])

    # This fancy algorithm determines how many cadets we need to generate for each AFSC
    adequate = False
    while not adequate:
        afsc_indices = np.zeros(M).astype(int)
        afsc_proportions = np.zeros(M)
        correct = False
        while not correct:
            for j in range(M):
                afsc_indices[j] = int(np.where(complete_afsc_vector == afsc_vector[j])[0])
                afsc_proportions[j] = max(np.random.normal(complete_afsc_proportion_means[afsc_indices[j]],
                                                           complete_afsc_proportion_stds[afsc_indices[j]]), 0.001)
            adjusted_proportions = afsc_proportions / sum(afsc_proportions)
            initial_amounts = np.around(adjusted_proportions * N)
            if sum(initial_amounts) == N:
                correct = True

        changes = np.zeros(M)
        min_counts = np.zeros(M)
        for j in range(M):
            min_counts[j] = 4
            if initial_amounts[j] % min_counts[j] != 0:
                if initial_amounts[j] > min_counts[j]:
                    changes[j] = -(initial_amounts[j] % min_counts[j])
                elif initial_amounts[j] > 0:
                    changes[j] = min_counts[j] - (initial_amounts[j] % min_counts[j])
                else:
                    changes[j] = min_counts[j]

        final_amounts = initial_amounts
        if sum(changes) != 0:
            for j in range(M):
                if changes[j] != 0:
                    if -changes[j] in changes:
                        j_list = np.where(changes == -changes[j])[0].astype(int)
                        if type(j_list) == int:
                            j2 = j_list
                        else:
                            j2 = j_list[0]
                        final_amounts[j] += changes[j]
                        final_amounts[j2] = final_amounts[j2] - changes[j]
                        changes[j] = 0
                        changes[j2] = 0
                    else:
                        relevant = True
                        j2 = j + 1
                        if j2 == M:
                            relevant = False
                        while relevant:
                            if changes[j2] != 0:
                                if (final_amounts[j2] - changes[j]) % min_counts[j2] == 0:
                                    final_amounts[j] += changes[j]
                                    final_amounts[j2] = final_amounts[j2] - changes[j]
                                    changes[j] = 0
                                    changes[j2] = 0
                                    relevant = False
                            j2 += 1
                            if j2 == M:
                                relevant = False
        else:
            final_amounts = initial_amounts + changes

        # Double check again for any weird scenarios!
        if sum(changes) != 0:
            absolute_changes = np.absolute(changes)
            problem_indices = np.where(changes != 0)[0].astype(int)
            if (max(absolute_changes) == 3) and (sum(absolute_changes) == 6):
                one_index = np.where(absolute_changes == 1)[0].astype(int)
                if changes[one_index] == -1:
                    negative_one = True
                else:
                    negative_one = False
                for j in problem_indices:
                    if negative_one:
                        if changes[j] == 2:
                            final_amounts[j] -= changes[j]
                        elif changes[j] == 3:
                            final_amounts[j] += changes[j]
                        elif changes[j] == -2:
                            final_amounts[j] += changes[j]
                        elif changes[j] == -3:
                            final_amounts[j] -= changes[j]
                        else:
                            final_amounts[j] += changes[j]
                    else:
                        if changes[j] == 2:
                            final_amounts[j] += changes[j]
                        elif changes[j] == 3:
                            final_amounts[j] -= changes[j]
                        elif changes[j] == -2:
                            final_amounts[j] -= changes[j]
                        elif changes[j] == -3:
                            final_amounts[j] += changes[j]
                        else:
                            final_amounts[j] += changes[j]

                    changes[j] = 0

        if (sum(changes) == 0) and (sum(final_amounts) == N):
            good = True
            for amount in final_amounts:
                if amount % 4 != 0:
                    good = False

            if good is True:
                adequate = True

    parameters = {'N': N, 'merit': np.zeros(N), 'usafa': np.zeros(N), 'utility': np.zeros([N, M]), 'quota': np.zeros(M),
                  'usafa_quota': np.zeros(M), 'male': np.zeros(N), 'minority': np.zeros(N), 'afsc_vector': afsc_vector,
                  'M': M, 'P': P, 'SS_encrypt': np.arange(N), 'sum_merit': (0.5 * N), 'quota_max': np.zeros(M),
                  'CIP1': np.array([" " * 6 for _ in range(N)]), 'CIP2': np.array([" " * 6 for _ in range(N)])}

    # Generate the cadets for each of the AFSCs
    util_choices = np.array([[round(random.uniform(0, 1), 4) for _ in range(P - 1)] for _ in range(N)])
    solution = np.zeros(N).astype(int)
    i = 0
    for j, loc in enumerate(afsc_indices):

        parameters['quota'][j] = max(round(0.8 * final_amounts[j]), 3)
        parameters['quota_max'][j] = int(parameters['quota'][j] * 2)

        if complete_usafa_proportions[loc] == 0:
            parameters['usafa_quota'][j] = 0
        elif complete_usafa_proportions[loc] == 1:
            parameters['usafa_quota'][j] = parameters['quota'][j]
        else:
            parameters['usafa_quota'][j] = max(round(0.2 * final_amounts[j]), 1)
        four_count = 1  # used to get the right demographics for every group of four cadets
        for _ in range(int(final_amounts[j])):

            # Pick a degree from the top tier CIPs for this cadet
            if ',' in top_tier_cips[j]:
                cip_options = top_tier_cips[j].split(',')
                selected = np.random.choice(cip_options, size=1)[0].strip()
                parameters['CIP1'][i] = selected
            else:
                parameters['CIP1'][i] = top_tier_cips[j]

            # Assign random utilities to the AFSCs, and then put a 1 in the correct spot
            np.put(parameters['utility'][i, :], np.random.choice(M, P - 1, replace=False),
                   util_choices[i, :])
            parameters['utility'][i, j] = 1

            if four_count == 1:
                if parameters['usafa_quota'][j] != 0:
                    parameters['usafa'][i] = 1
                else:
                    parameters['usafa'][i] = 0
                rand_merit = random.uniform(0, 1)
                parameters['merit'][i] = rand_merit
                parameters['male'][i] = 1
                parameters['minority'][i] = 1
            else:
                if parameters['usafa_quota'][j] / parameters['quota'][j] == 1:
                    parameters['usafa'][i] = 1
                else:
                    parameters['usafa'][i] = 0

                if four_count == 2:
                    parameters['merit'][i] = 1 - rand_merit
                    parameters['male'][i] = 1
                    parameters['minority'][i] = 1
                else:
                    parameters['minority'][i] = 0
                    if four_count == 3:
                        rand_merit = random.uniform(0, 1)
                        parameters['merit'][i] = rand_merit
                        parameters['male'][i] = 1
                    else:
                        parameters['merit'][i] = 1 - rand_merit
                        parameters['male'][i] = 0
                        four_count = 0

            # Next cadet
            four_count += 1
            solution[i] = int(j)
            i += 1

    parameters['qual'] = cip_to_qual(afsc_vector, parameters['CIP1'])
    parameters['rotc_quota'] = parameters['quota'] - parameters['usafa_quota']
    parameters['mandatory'] = (parameters['qual'] == 'M') * 1
    parameters['desired'] = (parameters['qual'] == 'D') * 1
    parameters['permitted'] = (parameters['qual'] == 'P') * 1
    parameters['ineligible'] = (parameters['qual'] == 'I') * 1
    parameters['eligible'] = (parameters['ineligible'] == 0) * 1
    parameters['quota_max'] = np.around(np.array(targets['Over'][:M]) * parameters['quota'])
    parameters['quota_min'] = parameters['quota']

    return parameters, solution


# "Realistic" Generation Functions
if afccp.core.globals.use_sdv:
    def simulate_realistic_fixed_data(generator=None, N=1500, printing=False):
        """
        This function calls the CTGAN to generate realistic synthetic cadets, then assigns targets to that data and returns
        the model fixed parameters in the traditional format
        :param printing: Whether the procedure should print something
        :param generator: CTGAN model
        :param N: Number of cadets to generate
        :return: synthetic data
        """
        if printing:
            print('Simulating (Realistic)...')

        if generator is None:
            generator = CTGAN.load(afccp.core.globals.paths["support"] + 'CTGAN.pkl')

        # Generate data
        data = generator.sample(N)

        # Force percentage of USAFA cadets
        usafa = np.array(data.loc[:, 'USAFA'])
        usafa_indices = np.where(usafa == 1)[0]
        current_usafa_num = len(usafa_indices)
        num_usafa_needed = round(np.random.uniform(low=0.20, high=0.31) * N)
        convert_num = num_usafa_needed - current_usafa_num
        if convert_num > 0:
            rotc_indices = np.where(usafa == 0)[0]
            convert_indices = np.random.choice(rotc_indices, size=convert_num, replace=False)
            usafa[convert_indices] = 1
            data['USAFA'] = usafa

        # Re-structure the percentiles to have a mean of 0.5
        percentiles = np.array(data.loc[:, 'percentile'])

        # Fix USAFA Percentiles
        usafa_indices = np.where(usafa == 1)[0]
        usafa_percentiles = percentiles[usafa_indices]
        N = len(usafa_percentiles)
        sorted_indices = np.argsort(usafa_percentiles)[::-1]
        usafa_percentiles = (np.arange(N) / (N - 1))
        magic_indices = np.argsort(sorted_indices)
        usafa_percentiles = usafa_percentiles[magic_indices]

        # Fix ROTC Percentiles
        rotc_indices = np.where(usafa == 0)[0]
        rotc_percentiles = percentiles[rotc_indices]
        N = len(rotc_percentiles)
        sorted_indices = np.argsort(rotc_percentiles)[::-1]
        rotc_percentiles = (np.arange(N) / (N - 1))
        magic_indices = np.argsort(sorted_indices)
        rotc_percentiles = rotc_percentiles[magic_indices]

        # Put the new percentiles back into the data
        np.put(percentiles, usafa_indices, usafa_percentiles)
        np.put(percentiles, rotc_indices, rotc_percentiles)
        data['percentile'] = percentiles
        return data


    def convert_realistic_data_parameters(data, targets=None, cip_qual_matrix=None, printing=False):
        """
        Converts the pandas dataframe of incomplete data to the two full data frames equivalent to the ones we import from
        excel
        :param printing: Whether the procedure should print something
        :param cip_qual_matrix: Matrix matching cip codes to qualifications
        :param targets: excel sheet of afsc quota historical distributions
        :param data: incomplete data generated from the CTGAN
        :return: cadet and afsc fixed data frames
        """
        if printing:
            print('Converting CTGAN generated data to data frames...')

        # Import CIP to Qual Matrix
        if cip_qual_matrix is None:
            cip_qual_matrix = import_data(afccp.core.globals.paths["support"] + "Qual_CIP_Matrix_Scrubbed.xlsx",
                                          sheet_name="Qual Matrix")

        # Import Generator Parameters
        if targets is None:
            cip_df = {}
            targets, cip_df["CIP1"], cip_df["CIP2"] = import_generator_parameters()

        # Add necessary data to cadets data frame
        afscs = np.array(targets.loc[:, 'AFSC'])
        cip1 = np.array(data.loc[:, 'CIP1']).astype(str)
        cip2 = np.array(data.loc[:, 'CIP2']).astype(str)
        qual_matrix = cip_to_qual(afscs, cip1, cip2, afscs, cip_qual_matrix)
        N = len(data)
        M = len(afscs)
        if 'NrWgt1' in data.keys():
            data['NrWgt1'] = np.ones(N)
        else:
            data.insert(5, 'NrWgt1', np.ones(N))
        for j, afsc in enumerate(afscs):
            data['qual_' + afsc] = qual_matrix[:, j]
        cadets_fixed = data

        # Specify AFSC targets
        total_quota = int(N * min(0.95, np.random.normal(0.93, 0.08)))
        total_means = np.array(targets['Total_Mean'])
        total_stds = np.array(targets['Total_Std'])
        usafa_means = np.array(targets['USAFA_Mean'])
        usafa_stds = np.array(targets['USAFA_Std'])
        combined_proportions = np.array([max(np.random.normal(total_means[j], total_stds[j]), 0.001) for j in range(M)])
        combined_proportions = combined_proportions / sum(combined_proportions)
        combined_quota = np.ceil(combined_proportions * total_quota).astype(int)
        usafa_proportions = np.array([max(np.random.normal(usafa_means[j], usafa_stds[j]), 0) for j in range(M)])
        usafa_quota = np.floor(usafa_proportions * combined_quota).astype(int)
        rotc_quota = combined_quota - usafa_quota
        max_quotas = np.around(np.array(targets['Over']) * combined_quota)
        min_quotas = combined_quota

        # Build AFSC fixed data frame
        afsc_vector = np.array(targets['AFSC'])
        afscs_fixed = pd.DataFrame({'AFSC': afsc_vector, 'USAFA Target': usafa_quota, 'ROTC Target': rotc_quota,
                                    'Combined Target': combined_quota, 'Over': np.array(targets['Over']),
                                    'Min': min_quotas,
                                    'Max': max_quotas})

        return cadets_fixed, afscs_fixed


    # CTGAN-Specific Functions
    def afscs_unique(table_data):
        """
        Check if the AFSCs are unique across the rows or they are blank
        :param table_data: pandas dataframe of cadet data
        :return: boolean
        """

        valid = ((table_data['NRat1'] != table_data['NRat2']) | (table_data['NRat1'] == "")) & \
                ((table_data['NRat1'] != table_data['NRat3']) | (table_data['NRat1'] == "")) & \
                ((table_data['NRat1'] != table_data['NRat4']) | (table_data['NRat1'] == "")) & \
                ((table_data['NRat1'] != table_data['NRat5']) | (table_data['NRat1'] == "")) & \
                ((table_data['NRat1'] != table_data['NRat6']) | (table_data['NRat1'] == "")) & \
                ((table_data['NRat2'] != table_data['NRat3']) | (table_data['NRat2'] == "")) & \
                ((table_data['NRat2'] != table_data['NRat4']) | (table_data['NRat2'] == "")) & \
                ((table_data['NRat2'] != table_data['NRat5']) | (table_data['NRat2'] == "")) & \
                ((table_data['NRat2'] != table_data['NRat6']) | (table_data['NRat2'] == "")) & \
                ((table_data['NRat3'] != table_data['NRat4']) | (table_data['NRat3'] == "")) & \
                ((table_data['NRat3'] != table_data['NRat5']) | (table_data['NRat3'] == "")) & \
                ((table_data['NRat3'] != table_data['NRat6']) | (table_data['NRat3'] == "")) & \
                ((table_data['NRat4'] != table_data['NRat5']) | (table_data['NRat4'] == "")) & \
                ((table_data['NRat4'] != table_data['NRat6']) | (table_data['NRat4'] == "")) & \
                ((table_data['NRat5'] != table_data['NRat6']) | (table_data['NRat5'] == ""))

        return valid


    def utilities_match(table_data):
        """
        Constraint ensuring blank preferences correspond to utilities of zero
        """
        valid = ((table_data['NRat2'] == "") == (table_data['NrWgt2'] == 0)) & \
                ((table_data['NRat3'] == "") == (table_data['NrWgt3'] == 0)) & \
                ((table_data['NRat4'] == "") == (table_data['NrWgt4'] == 0)) & \
                ((table_data['NRat5'] == "") == (table_data['NrWgt5'] == 0)) & \
                ((table_data['NRat6'] == "") == (table_data['NrWgt6'] == 0))

        return valid


    def ctgan_train(data=None, epochs=1000, printing=True, name='CTGAN_Scrubbed'):
        """
        Trains the CTGAN on the data given with the number of epochs specified
        :param name: name of data
        :param data: ctgan data (if none, import ctgan_data.xlsx)
        :param epochs: number of epochs for CTGAN
        :param printing: whether the procedure should print something
        :return: CTGAN model
        """

        if data is None:
            data = import_data(afccp.core.globals.paths["support"] + 'ctgan_data.xlsx', sheet_name='Data')
            data = ctgan_data_filter(data)

        print('')
        print('Data: ' + name)
        print(data)
        print('')

        # Constraints to ensure utilities descend across the columns from left to right (or are equal)  (DEPRECATED)
        three_two_constraint = GreaterThan(low='NrWgt3', high='NrWgt2', handling_strategy='reject_sampling')
        four_three_constraint = GreaterThan(low='NrWgt4', high='NrWgt3', handling_strategy='reject_sampling')
        five_four_constraint = GreaterThan(low='NrWgt5', high='NrWgt4', handling_strategy='reject_sampling')
        six_five_constraint = GreaterThan(low='NrWgt6', high='NrWgt5', handling_strategy='reject_sampling')

        # Custom constraints defined in this script
        unique_AFSC_constraint = CustomConstraint(is_valid=afscs_unique)
        utilities_pref_constraint = CustomConstraint(is_valid=utilities_match)

        # if 'workspace' in dir_path:  # Newer version!
        #
        #     # Updated version
        #     three_two_constraint = Inequality(low_column_name='NrWgt3', high_column_name='NrWgt2',
        #                                       handling_strategy='reject_sampling')
        #     four_three_constraint = Inequality(low_column_name='NrWgt4', high_column_name='NrWgt3',
        #                                        handling_strategy='reject_sampling')
        #     five_four_constraint = Inequality(low_column_name='NrWgt5', high_column_name='NrWgt4',
        #                                       handling_strategy='reject_sampling')
        #     six_five_constraint = Inequality(low_column_name='NrWgt6', high_column_name='NrWgt5',
        #                                      handling_strategy='reject_sampling')
        #
        #     # Custom Constraints
        #     unique_AFSC_constraint = create_custom_constraint(is_valid_fn=afscs_unique)
        #     utilities_pref_constraint = create_custom_constraint(is_valid_fn=utilities_match)

        constraints = [three_two_constraint, four_three_constraint, five_four_constraint, six_five_constraint,
                       unique_AFSC_constraint, utilities_pref_constraint]

        model = CTGAN(epochs=epochs, constraints=constraints, verbose=True, primary_key='Encrypt_PII')

        # Train the model
        if printing:
            print('Training CTGAN model')
        model.fit(data)

        # Save the model
        model.save(afccp.core.globals.paths["support"] + name + '.pkl')
        if printing:
            print('Model Saved.')

        return model


    def ctgan_data_filter(ctgan_data=None, export=False):
        """
        This procedure loads the ctgan data and filters it so that all the rows satisfy the CTGAN constraints
        :return: Exports to excel
        """

        # Load Data
        if ctgan_data is None:
            ctgan_data = import_data(afccp.core.globals.paths["support"] + 'ctgan_data.xlsx', sheet_name='Data')

        # Replace nans with blanks
        ctgan_data = ctgan_data.replace(np.nan, "", regex=True)

        # Replace where utilities don't match preferences
        util_match_invalid = np.array(utilities_match(ctgan_data) * 1) == 0
        index = np.where(util_match_invalid)[0]
        ctgan_data.drop(ctgan_data.index[index], inplace=True)

        # Remove rows with no first choice preference
        ctgan_data.drop(ctgan_data[ctgan_data['NRat1'] == ''].index, inplace=True)

        # Clean CIP columns
        for cip_name in ['CIP1', 'CIP2']:

            # Change instances like "10100.0" to "10100"
            cip_arr = np.array(ctgan_data.loc[:, cip_name]).astype(str)
            unique = np.unique(cip_arr)
            for i, cip in enumerate(unique):
                num = len(cip)
                if num < 6 and '.' not in cip:
                    if cip in ["220", "220.0"]:
                        unique[i] = "220000"
                    elif cip == "nan":
                        unique[i] = " " * 6
                    else:
                        unique[i] = "0" + cip
                if num == 5:
                    unique[i] = "0" + cip[:5]
                if '.' in cip and num == 7:
                    unique[i] = "0" + cip[:5]
                elif cip == '220.0':
                    unique[i] = '220000'
                if unique[i][0] == '9':
                    unique[i] = "0" + unique[i][:5]
                unique[i] = unique[i][:6]

                # Replace cip column
                cip_arr = np.where(cip_arr == cip, str(unique[i]), cip_arr.astype(str)).astype(str)

            # Reduce all blanks to ''
            indices = np.where(cip_arr == 'nan')[0]
            np.put(cip_arr, indices, '')
            indices = np.where(cip_arr == '      ')[0]
            np.put(cip_arr, indices, '')
            indices = np.where(cip_arr == '    ')[0]
            np.put(cip_arr, indices, '')
            ctgan_data[cip_name] = cip_arr

        # Remove rows with no first degree
        cip_arr = np.array(ctgan_data.loc[:, 'CIP1']).astype(str)
        index = np.where(cip_arr == '')[0]
        ctgan_data = ctgan_data.drop(ctgan_data.index[index])

        # Remove first choice utility column
        if 'NrWgt1' in ctgan_data.columns:
            ctgan_data = ctgan_data.drop(labels='NrWgt1', axis=1)

        # More blanks adjustments
        ctgan_data = ctgan_data.replace("nan", "", regex=True)
        ctgan_data = ctgan_data.replace(np.nan, "", regex=True)

        # Check where the CTGAN match constraint fails. Remove those observations
        util_match_invalid = np.array(utilities_match(ctgan_data) * 1)
        index = np.where(util_match_invalid == 0)[0]
        ctgan_data = ctgan_data.drop(ctgan_data.index[index])

        # If the utilities don't have the type "float", remove them
        utilities = np.array(ctgan_data.loc[:, 'NrWgt2':'NrWgt6']).astype(float)
        indices = []
        for i in range(len(utilities)):
            for p in range(len(utilities[i, :])):
                if type(utilities[i, p]) != np.float64:
                    indices.append(i)
        ctgan_data = ctgan_data.drop(ctgan_data.index[indices])

        # If the utilities don't descend in value across the columns, remove them
        utilities = np.array(ctgan_data.loc[:, 'NrWgt2':'NrWgt6'])
        indices = []
        for i in range(len(utilities)):
            if utilities[i, 0] >= utilities[i, 1] >= utilities[i, 2] >= utilities[i, 3] >= utilities[i, 4]:
                pass
            else:
                indices.append(i)
        ctgan_data = ctgan_data.drop(ctgan_data.index[indices])

        if export:  # Export to excel
            with pd.ExcelWriter(afccp.core.globals.paths["support"] + 'ctgan_data.xlsx') as writer:
                ctgan_data.to_excel(writer, sheet_name="Data", index=False)

        return ctgan_data
