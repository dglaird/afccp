# Import libraries
import numpy as np
from afccp.core.globals import *


# Fixed Parameter Procedures
def import_fixed_cadet_afsc_data_from_excel(filepath, printing=False):
    """
    This procedures takes in an input filepath with cadet and AFSC fixed parameter data, then returns those data frames
    :param filepath: file name of excel cadet/AFSC input data
    :param printing: whether or not the procedure should print something
    :return: cadet and AFSC fixed data frames
    """
    if printing:
        print("Importing fixed cadet/AFSC data from excel...")

    # Import datasets
    cadets_fixed = import_data(filepath, sheet_name="Cadets Fixed")
    afscs_fixed = import_data(filepath, sheet_name="AFSCs Fixed")

    return cadets_fixed, afscs_fixed


def model_fixed_parameters_from_data_frame(cadets_fixed, afscs_fixed, printing=False):
    """
    This procedure takes in fixed cadet/AFSC data frames, then converts them into the parameters structure for the
    model
    :param printing: Whether the procedure should print something
    :param cadets_fixed: data frame of cadet data
    :param afscs_fixed: data frame of AFSC data
    :return: model fixed parameters
    """
    if printing:
        print("Converting data to fixed model parameters...")

    # Get info from dataframes
    N = int(len(cadets_fixed))
    M = int(len(afscs_fixed))
    afsc_vector = np.array(afscs_fixed['AFSC'])
    qual = np.array(cadets_fixed.loc[:, 'qual_' + afsc_vector[0]:'qual_' + afsc_vector[M - 1]])
    columns = list(cadets_fixed.keys())
    afsc_columns = list(afscs_fixed.keys())

    # Number of preferences
    P = len([col for col in columns if 'NRat' in col])

    # Initialize parameters dictionary
    parameters = {'SS_encrypt': np.array(cadets_fixed.loc[:, 'Encrypt_PII']), 'afsc_vector': afsc_vector,
                  'P': P, "quota": np.array(afscs_fixed.loc[:, 'Combined Target']), 'N': N, 'M': M, 'qual': qual,
                  'quota_max': np.array(afscs_fixed.loc[:, 'Max']), 'quota_min': np.array(afscs_fixed.loc[:, 'Min']),
                  'utility': np.zeros([N, M])}

    if qual[0, 0] in [1, 0]:  # Qual Matrix is Binary
        parameters['ineligible'] = (qual == 0) * 1
        parameters['eligible'] = qual

    else:  # Qual Matrix is AFOCD Qualifications
        parameters['ineligible'] = (qual == 'I') * 1
        parameters['eligible'] = (parameters['ineligible'] == 0) * 1
        parameters['mandatory'] = (qual == 'M') * 1
        parameters['desired'] = (qual == 'D') * 1
        parameters['permitted'] = (qual == 'P') * 1

    # Load Instance Parameters (may or may not be included in this dataset)
    cadet_parameter_dictionary = {'USAFA': 'usafa', 'Male': 'male', 'Minority': 'minority', 'ASC1': 'asc1',
                                  'ASC2': 'asc2', 'CIP1': 'cip1', 'CIP2': 'cip2', 'percentile': 'merit'}
    for col_name in list(cadet_parameter_dictionary.keys()):
        if col_name in columns:
            parameters[cadet_parameter_dictionary[col_name]] = np.array(cadets_fixed.loc[:, col_name])

            # Demographic Proportions
            if col_name == 'USAFA' or col_name == 'Male' or col_name == 'Minority':
                parameters[cadet_parameter_dictionary[col_name] + '_proportion'] = np.mean(
                    parameters[cadet_parameter_dictionary[col_name]])

    if 'USAFA Target' in afsc_columns:
        parameters['usafa_quota'] = np.array(afscs_fixed.loc[:, 'USAFA Target'])
        parameters['rotc_quota'] = np.array(afscs_fixed.loc[:, 'ROTC Target'])

    # Create utility matrix from preference columns
    preferences_array = np.array(cadets_fixed.loc[:, 'NRat' + str(1):'NRat' + str(parameters['P'])])
    utilities_array = np.array(cadets_fixed.loc[:, 'NrWgt' + str(1):'NrWgt' + str(parameters['P'])])
    for i in range(N):
        for p in range(parameters['P']):
            j = np.where(preferences_array[i, p] == afsc_vector)[0]
            if len(j) != 0:
                parameters['utility'][i, j[0]] = utilities_array[i, p]

    return parameters


def model_fixed_parameters_set_additions(parameters, printing=False):
    """
    Creates subsets for AFSCs and cadets
    :param parameters: fixed parameters
    :param printing: whether the procedure should print something
    :return: updated parameters with sets
    """
    if printing:
        print('Adding subsets to parameters...')

    # Cadet Indexed Sets
    parameters['I'] = np.arange(parameters['N'])
    parameters['J'] = np.arange(parameters['M'])
    parameters['J_E'] = [np.where(
        parameters['ineligible'][i, :] == 0)[0] for i in parameters['I']]  # set of AFSCs that cadet i is eligible for

    # AFSC Indexed Sets
    parameters['I_E'] = [np.where(
        parameters['ineligible'][:, j] == 0)[0] for j in parameters['J']]  # set of cadets that are eligible for AFSC j

    # Add demographic sets if they're included
    parameters['I_D'] = {}
    if 'usafa' in parameters:
        usafa = np.where(parameters['usafa'] == 1)[0]  # set of usafa cadets
        parameters['usafa_proportion'] = np.mean(parameters['usafa'])
        parameters['I_D']['USAFA Proportion'] = [np.intersect1d(parameters['I_E'][j], usafa) for j in parameters['J']]
    if 'mandatory' in parameters:
        parameters['I_D']['Mandatory'] = [np.where(parameters['mandatory'][:, j] == 1)[0] for j in parameters['J']]
        parameters['I_D']['Desired'] = [np.where(parameters['desired'][:, j] == 1)[0] for j in parameters['J']]
        parameters['I_D']['Permitted'] = [np.where(parameters['permitted'][:, j] == 1)[0] for j in parameters['J']]

    if 'male' in parameters:
        male = np.where(parameters['male'] == 1)[0]  # set of male cadets
        parameters['I_D']['Male'] = [np.intersect1d(parameters['I_E'][j], male) for j in parameters['J']]
        parameters['male_proportion'] = np.mean(parameters['male'])

    if 'minority' in parameters:
        minority = np.where(parameters['minority'] == 1)[0]  # set of minority cadets
        parameters['I_D']['Minority'] = [np.intersect1d(parameters['I_E'][j], minority) for j in parameters['J']]
        parameters['minority_proportion'] = np.mean(parameters['minority'])

    # Merit
    if 'merit' in parameters:
        parameters['sum_merit'] = parameters['merit'].sum()  # should be close to N/2
    return parameters


def get_utility_preferences(parameters):
    """
    Converts utility matrix into two arrays of preferences and utilities (NxP for each)
    :param parameters: fixed cadet/afsc data
    :return: preference matrix and utilities matrix
    """
    preferences = np.array([[" " * 10 for _ in range(parameters['P'])] for _ in range(parameters['N'])])
    utilities_array = np.zeros([parameters['N'], parameters['P']])
    for i in range(parameters['N']):

        # Sort indices of nonzero utilities
        indices = parameters['utility'][i, :].nonzero()[0]
        sorted_init = np.argsort(parameters['utility'][i, :][indices])[::-1]
        sorted_indices = indices[sorted_init]

        # Put the utilities and preferences in the correct spots
        np.put(utilities_array[i, :], np.arange(len(sorted_indices)), parameters['utility'][i, :][sorted_indices])
        np.put(preferences[i, :], np.arange(len(sorted_indices)), parameters['afsc_vector'][sorted_indices])

    return preferences, utilities_array


# Solution Handling Procedures
def import_solution_from_excel(filepath, standard=True, printing=False):
    """
    Imports a solution from excel and converts it to a vector of AFSC indices
    :param standard: if we are importing it from standard data format
    :param filepath: file path
    :param printing: Whether the procedure should print something
    :return: solution (vector of AFSC indices)
    """

    if printing:
        print('Importing solution from excel...')

    # Load dataframes
    afscs_fixed = import_data(filepath, sheet_name="AFSCs Fixed")
    afsc_vector = np.array(afscs_fixed['AFSC'])

    if standard:
        sheet_name = "Cadet Solution Quality"
    else:
        sheet_name = "Original Solution"
    solutions_df = import_data(filepath, sheet_name=sheet_name)
    afsc_solution = np.array(solutions_df['Matched'])

    # Convert afscs to afsc indices
    solution = np.zeros(len(afsc_solution)).astype(int)
    for i in range(len(afsc_solution)):
        solution[i] = np.where(afsc_vector == afsc_solution[i])[0]

    return solution


def swap_solution_shape(solution, M):
    """
    Changes the solution from a vector of length N containing AFSC indices to an NxM binary matrix, or vice versa
    depending on the input.
    :param M: number of AFSCs
    :param solution: either vector or matrix
    :return: either matrix or vector
    """
    N = len(solution)

    # if we're dealing with a matrix, convert to vector
    if len(solution.shape) == 2:
        changed_solution = np.where(solution == 1)[1]
        return changed_solution

    # if we're dealing with a vector, convert to matrix
    elif len(solution.shape) == 1:
        changed_solution = np.zeros([N, M])
        for i in range(N):
            changed_solution[i, int(solution[i])] = 1
        return changed_solution


def compare_solutions(baseline, compared, printing=False):
    """
    Takes two solutions (in vector form) to the same problem (must be same set of cadets/AFSCs) and returns how similar
    the compared solution is to the baseline in terms of what AFSCs cadets were assigned to
    :param printing: whether or not the procedure should print something
    :param baseline: solution 1
    :param compared: solution 2, compared against the baseline
    :return: the percentage of solution 2 that is the same as solution 1
    """
    percent_similar = (sum(baseline == compared * 1) / len(baseline))
    if printing:
        print("The two solutions are " + str(percent_similar) + "% the same.")
    return percent_similar


def value_function(a, f_a, r, x):
    """
    This is the AFSC objective value function
    :param r: number of breakpoints
    :param a: measure at each breakpoint
    :param f_a: value at each breakpoint
    :param x: actual AFSC objective measure
    :return: AFSC objective value
    """
    # Find which breakpoint is immediately before this measure
    indices = np.array([a[l] <= x <= a[l + 1] for l in range(r - 1)]) * 1
    l = np.where(indices)[0]

    # Obtain value
    if len(l) == 0:
        l = r - 1
        val = f_a[l]
    else:
        l = l[0]
        val = f_a[l + 1] - ((f_a[l + 1] - f_a[l]) / (a[l + 1] - a[l])) * (a[l + 1] - x)

    # Return value
    return val


def measure_solution_quality(solution, parameters, value_parameters, printing=False, approximate=False):
    """
    This procedure takes in a solution (either vector or matrix), as well as the fixed
    cadet/AFSC model parameters and the weight/value parameters, and then evaluates the solution. The solution metrics
    are returned.
    :param approximate: whether we measure the approximate value or not (using target quota instead of count)
    :param printing: Whether or not the procedure should print the matrix
    :param solution: either vector or matrix of matched cadets to AFSCs
    :param parameters: fixed model cadet/AFSC input parameters
    :param value_parameters: weight/value parameters
    :return: solution metrics
    """

    # Load in parameters from dictionaries
    N = parameters['N']
    M = parameters['M']
    I = parameters['I']
    J = parameters['J']
    J_E = parameters['J_E']
    I_E = parameters['I_E']
    I_D = parameters['I_D']
    O = value_parameters['O']
    K_A = value_parameters['K_A']
    K_C = value_parameters['K_C']
    objectives = value_parameters['objectives']

    # Construct X matrix
    if len(np.shape(solution)) == 1:
        x_matrix = False
        X = np.zeros([N, M])
        for i in I:
            for j in J:
                if solution[i] == j:
                    X[i, j] = 1
    else:
        x_matrix = True
        X = solution

    # Create metrics dictionary
    metrics = {'objective_measure': np.zeros([M, O]), 'objective_value': np.zeros([M, O]),
               'afsc_value': np.zeros(M), 'cadet_value': np.zeros(N), 'cadet_constraint_fail': np.zeros(N),
               'afsc_constraint_fail': np.zeros(M), 'objective_score': np.zeros(O),
               'objective_constraint_fail': np.array([[" " * 30 for _ in range(O)] for _ in range(M)]),
               'total_failed_constraints': 0, 'X': X}

    # Get certain objective indices
    quota_k = np.where(objectives == 'Combined Quota')[0][0]
    if 'merit' in parameters:
        merit_k = np.where(objectives == 'Merit')[0][0]

    if 'usafa' in parameters:
        usafa_k = np.where(objectives == 'USAFA Proportion')[0][0]

    # Loop through all AFSCs to assign their individual values
    for j in J:

        count = sum(X[i, j] for i in I_E[j])  # number of assigned cadets
        if count != 0:

            # Get variables for this AFSC
            if 'usafa' in parameters:
                usafa_count = np.sum(X[i, j] for i in I_D['USAFA Proportion'][j])
            target_quota = value_parameters['objective_target'][j, quota_k]

            # Are we using approximate measures or not
            if approximate:
                num_cadets = target_quota
            else:
                num_cadets = count

            # Loop through all objectives that this AFSC cares about
            for k in K_A[j]:

                # Get the correct measure for this objective
                objective = objectives[k]
                if objective == 'Merit':
                    numerator = np.sum(parameters['merit'][i] * X[i, j] for i in I_E[j])
                    metrics['objective_measure'][j, k] = numerator / num_cadets
                elif objective == 'Utility':
                    numerator = np.sum(parameters['utility'][i, j] * X[i, j] for i in I_E[j])
                    metrics['objective_measure'][j, k] = numerator / num_cadets
                elif objective == 'Combined Quota':
                    metrics['objective_measure'][j, k] = count
                elif objective == 'USAFA Quota':
                    metrics['objective_measure'][j, k] = usafa_count
                elif objective == 'ROTC Quota':
                    metrics['objective_measure'][j, k] = count - usafa_count
                elif objective in I_D:
                    numerator = np.sum(X[i, j] for i in I_D[objective][j])
                    metrics['objective_measure'][j, k] = numerator / num_cadets

                # Get the correct value for this objective
                metrics['objective_value'][j, k] = value_function(a=value_parameters['F_bp'][j][k],
                                                                  f_a=value_parameters['F_v'][j][k],
                                                                  r=value_parameters['r'][j][k],
                                                                  x=metrics['objective_measure'][j, k])

                # AFSC Objective Constraints
                if k in K_C[j]:

                    # Constrained Value (no way for exact value constraint)
                    if value_parameters['constraint_type'][j, k] == 1 or \
                            value_parameters['constraint_type'][j, k] == 2:

                        if metrics['objective_value'][j, k] < \
                                float(value_parameters['objective_value_min'][j, k]):
                            metrics['objective_constraint_fail'][j, k] = str(
                                round(metrics['objective_value'][j, k], 3)) + ' < ' + str(
                                float(value_parameters['objective_value_min'][j, k]))
                            metrics['total_failed_constraints'] += 1

                    # Constrained Approximate Measure
                    elif value_parameters['constraint_type'][j, k] == 3:
                        value_list = value_parameters['objective_value_min'][j, k].split(",")
                        min_measure = float(value_list[0].strip())
                        max_measure = float(value_list[1].strip())

                        # Get correct measure constraint
                        if objective not in ['Combined Quota', 'USAFA Quota', 'ROTC Quota']:
                            if numerator / target_quota < min_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    round(numerator / target_quota, 3)) + ' < ' + str(
                                    min_measure) + '. ' + str(round(100 * (numerator / target_quota) /
                                                                    min_measure, 2)) + '% Met.'
                                metrics['total_failed_constraints'] += 1
                            elif numerator / target_quota > max_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    round(numerator / target_quota, 3)) + ' > ' + str(
                                    max_measure) + '. ' + str(round(100 * max_measure /
                                                                    (numerator / target_quota), 2)) + '% Met.'
                                metrics['total_failed_constraints'] += 1
                        else:
                            if metrics['objective_measure'][j, k] < min_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    metrics['objective_measure'][j, k]) + ' < ' + str(
                                    min_measure) + '. ' + str(round(100 * (metrics['objective_measure'][j, k]) /
                                                                    min_measure, 2)) + '% Met.'
                                metrics['total_failed_constraints'] += 1
                            elif metrics['objective_measure'][j, k] > max_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    metrics['objective_measure'][j, k]) + ' > ' + str(
                                    min_measure) + '. ' + str(round(100 * max_measure /
                                                                    metrics['objective_measure'][j, k], 2)) + '% Met.'
                                metrics['total_failed_constraints'] += 1

                    # Constrained Exact Measure
                    elif value_parameters['constraint_type'][j, k] == 4:

                        value_list = value_parameters['objective_value_min'][j, k].split(",")
                        min_measure = float(value_list[0].strip())
                        max_measure = float(value_list[1].strip())

                        # Get correct measure constraint
                        if objective not in ['Combined Quota', 'USAFA Quota', 'ROTC Quota']:
                            if numerator / count < min_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    round(numerator / count, 3)) + ' < ' + str(
                                    min_measure) + '. ' + str(round(100 * (numerator / count) /
                                                                    min_measure, 2)) + '% Met.'
                            elif numerator / count > max_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    round(numerator / count, 3)) + ' > ' + str(
                                    max_measure) + '. ' + str(round(100 * max_measure /
                                                                    (numerator / count), 2)) + '% Met.'
                                metrics['total_failed_constraints'] += 1
                        else:
                            if metrics['objective_measure'][j, k] < min_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    metrics['objective_measure'][j, k]) + ' < ' + str(
                                    min_measure) + '. ' + str(round(100 * (metrics['objective_measure'][j, k]) /
                                                                    min_measure, 2)) + '% Met.'
                            elif metrics['objective_measure'][j, k] > max_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    metrics['objective_measure'][j, k]) + ' > ' + str(
                                    max_measure) + '. ' + str(round(100 * max_measure /
                                                                    metrics['objective_measure'][j, k], 2)) + '% Met.'
                                metrics['total_failed_constraints'] += 1

            # AFSC individual value
            metrics['afsc_value'][j] = np.dot(value_parameters['objective_weight'][j, :],
                                              metrics['objective_value'][j, :])
            if metrics['afsc_value'][j] < value_parameters['afsc_value_min'][j]:
                metrics['afsc_constraint_fail'][j] = 1
                metrics['total_failed_constraints'] += 1

            # Also calculate Average Merit and USAFA proportion even if not in the AFSC's objectives
            if 'merit' in parameters:
                if merit_k not in K_A[j]:
                    numerator = np.sum(parameters['merit'][i] * X[i, j] for i in I_E[j])
                    metrics['objective_measure'][j, merit_k] = numerator / num_cadets
            if 'usafa' in parameters:
                if usafa_k not in K_A[j]:
                    numerator = np.sum(X[i, j] for i in I_D['USAFA Proportion'][j])
                    metrics['objective_measure'][j, usafa_k] = numerator / num_cadets

    # Loop through all cadets to assign their values
    for i in I:
        metrics['cadet_value'][i] = np.sum(X[i, j] * parameters['utility'][i, j] for j in J_E[i])
        if metrics['cadet_value'][i] < value_parameters['cadet_value_min'][i]:
            metrics['cadet_constraint_fail'][i] = 1
            metrics['total_failed_constraints'] += 1

    # Generate objective scores for each objective
    for k in range(O):
        new_weights = value_parameters['afsc_weight'] * value_parameters['objective_weight'][:, k]
        new_weights = new_weights / sum(new_weights)
        metrics['objective_score'][k] = np.dot(new_weights, metrics['objective_value'][:, k])

    # Solution Matched Vector
    metrics['afsc_solution'] = np.array([" " * 10 for _ in range(parameters['N'])])

    # turns the X matrix back into a vector of AFSC indices
    solution = np.where(X)[1]

    # Translate AFSC indices into the AFSCs themselves
    for i in I:
        metrics['afsc_solution'][i] = parameters['afsc_vector'][int(solution[i])]

    # Define overall metrics
    metrics['cadets_overall_value'] = np.dot(value_parameters['cadet_weight'], metrics['cadet_value'])
    metrics['afscs_overall_value'] = np.dot(value_parameters['afsc_weight'], metrics['afsc_value'])
    metrics['z'] = value_parameters['cadets_overall_weight'] * metrics['cadets_overall_value'] + \
                   value_parameters['afscs_overall_weight'] * metrics['afscs_overall_value']
    metrics['num_ineligible'] = np.sum(X[i, j] * parameters['ineligible'][i, j] for j in
                                    J for i in I)
    if printing:

        if approximate:
            model_type = 'approximate'
        else:
            model_type = 'exact'
        if x_matrix:
            solution_type = 'matrix'
        else:
            solution_type = 'vector'

        print("Measured " + model_type + " solution " + solution_type +
              " objective value: " + str(round(metrics['z'], 4)))

    return metrics


def ga_fitness_function(chromosome, parameters, value_parameters, constraints='Fail', penalty_scale=1.3,
                        con_fail_dict=None, printing=False, first=True):
    """
    This function takes in a chromosome (solution vector) and evaluates it.
    :param first: if this is a solution in the initial population
    :param con_fail_dict: dictionary used for constraints
    :param penalty_scale: how much to penalize failed constraints
    :param constraints: how we handle failed constraints
    :param printing: whether the procedure should print something
    :param value_parameters: weight and value parameters
    :param parameters: cadet/AFSC parameters
    :param chromosome: solution vector
    :return: fitness score
    """
    # Parameter Sets
    N = parameters['N']  # Number of Cadets
    M = parameters['M']  # Number of AFSCs
    I = parameters['I']  # Set of Cadets
    J = parameters['J']  # Set of AFSCs
    I_D = parameters['I_D']  # Set of cadets with some demographic for AFSC j

    # Value Parameter Sets
    O = value_parameters['O']  # Number of objectives
    K_A = value_parameters['K_A']  # Set of objectives for AFSC j
    K_C = value_parameters['K_C']  # Set of objectives with constraints for AFSC j
    I_C = value_parameters['I_C']  # Set of cadets with constrained minimum values
    J_C = value_parameters['J_C']  # Set of AFSCs with constrained minimum values
    r = value_parameters['r']  # Number of breakpoints for value function on objective k for AFSC j
    a = value_parameters['F_bp']  # Set of breakpoint measures for value function on objective k for AFSC j
    f_a = value_parameters['F_v']  # Set of breakpoint values for value function on objective k for AFSC j

    # More Parameters
    merit = parameters['merit']
    utility = parameters['utility']
    quota = parameters['quota']
    afsc_vector = parameters['afsc_vector']
    objectives = value_parameters['objectives']
    objective_weight = value_parameters['objective_weight']
    objective_constraint_type = value_parameters['constraint_type']
    afsc_weight = value_parameters['afsc_weight']
    afscs_overall_weight = value_parameters['afscs_overall_weight']
    cadet_weight = value_parameters['cadet_weight']
    cadets_overall_weight = value_parameters['cadets_overall_weight']
    afsc_value_min = value_parameters['afsc_value_min']
    cadet_value_min = value_parameters['cadet_value_min']
    objective_min = np.zeros([M, O])
    objective_max = np.zeros([M, O])
    soc_counts = np.zeros(M)

    metrics = {'objective_measure': np.zeros([M, O]), 'objective_value': np.zeros([M, O]),
               'afsc_value': np.zeros(M), 'cadet_value': np.zeros(N), 'cadet_constraint_fail': np.zeros(N),
               'afsc_constraint_fail': np.zeros(M), 'objective_constraint_fail': np.zeros([M, O]),
               'total_failed_constraints': 0}

    for j in J:

        # If we care about USAFA Count and ROTC Count
        if 'USAFA Count' in objectives[K_A[j]]:
            soc_counts[j] = 1

        for k in K_C[j]:
            if objective_constraint_type[j, k] == 1 or objective_constraint_type[j, k] == 2:
                objective_min[j, k] = float(value_parameters['objective_value_min'][j, k])
            elif objective_constraint_type[j, k] == 3 or objective_constraint_type[j, k] == 4:
                value_list = value_parameters['objective_value_min'][j, k].split(",")
                objective_min[j, k] = float(value_list[0].strip())
                objective_max[j, k] = float(value_list[1].strip())

    # If USAFA counts are used
    if sum(soc_counts) == 0:
        ignore_uc = True
    else:
        ignore_uc = False

    failed = False
    penalty = 0
    for j in J:

        # Initialize objective measures and values
        metrics['objective_measure'][j, :] = np.zeros(O)
        metrics['objective_value'][j, :] = np.ones(O)

        # list of indices of assigned cadets
        cadets = np.where(chromosome == j)[0]

        # Only calculate measures for AFSCs with at least one cadet
        count = len(cadets)
        usafa_count = count

        if count > 0:
            if not ignore_uc:
                usafa_cadets = np.intersect1d(I_D['USAFA Proportion'][j], cadets)
                usafa_count = len(usafa_cadets)

            # Loop through all AFSC objectives
            for k in K_A[j]:
                objective = objectives[k]
                if objective == 'Merit':
                    metrics['objective_measure'][j, k] = np.mean(merit[cadets])
                elif objective == 'Utility':
                    metrics['objective_measure'][j, k] = np.mean(utility[cadets, j])
                elif objective == 'Combined Quota':
                    metrics['objective_measure'][j, k] = count
                elif objective == 'USAFA Quota':
                    metrics['objective_measure'][j, k] = usafa_count
                elif objective == 'ROTC Quota':
                    metrics['objective_measure'][j, k] = count - usafa_count
                elif objective in I_D:
                    metrics['objective_measure'][j, k] = len(np.intersect1d(I_D[objective][j], cadets)) / count

                # Assign AFSC objective value
                metrics['objective_value'][j, k] = value_function(a[j][k], f_a[j][k], r[j][k],
                                                                  metrics['objective_measure'][j, k])

                # AFSC Objective Constraints
                if k in K_C[j]:

                    # We're really only ever going to constrain the approximate measure for Mandatory
                    if objective == 'Mandatory':
                        constrained_measure = (metrics['objective_measure'][j, k] * count) / quota[j]
                    else:
                        constrained_measure = metrics['objective_measure'][j, k]

                    # Use the real constraint (potentially different as a result of approximate model)
                    constrained_min, constrained_max = objective_min[j, k], objective_max[j, k]
                    if con_fail_dict is not None:
                        if (j, k) in con_fail_dict:
                            split_list = con_fail_dict[(j, k)].split(' ')
                            if split_list[0] == '>':
                                constrained_min = float(split_list[1])
                                constrained_max = objective_max[j, k]
                            else:
                                constrained_min = objective_min[j, k]
                                constrained_max = float(split_list[1])

                    if (constrained_measure < constrained_min or constrained_measure > constrained_max) \
                            and not first:

                        if con_fail_dict is not None and constraints == 'Fail':
                            failed = True
                            break
                        else:
                            if constrained_measure < constrained_min:
                                p_con_met = constrained_measure / constrained_min
                            else:
                                p_con_met = constrained_max / constrained_measure

                            if objective == 'USAFA Proportion':
                                adj_con_tolerance = 0.9
                            elif objective == 'Combined Quota':
                                adj_con_tolerance = min((constrained_min - 1) / constrained_min,
                                                        constrained_max / (constrained_max + 1))
                            elif objective == 'Mandatory':
                                adj_con_tolerance = min((constrained_min - (1 / quota[j])) / constrained_min,
                                                        constrained_max / (constrained_max + (1 / quota[j])))
                            else:
                                adj_con_tolerance = 0.95

                            # Either we reduce z by some penalty or z is set to 0
                            if constraints == 'Penalty' and p_con_met < adj_con_tolerance:
                                penalty += afscs_overall_weight * afsc_weight[j]
                            elif constraints == 'Fail' and p_con_met < adj_con_tolerance:
                                failed = True
                                break
            if failed:
                break
            else:

                # Calculate AFSC value
                metrics['afsc_value'][j] = np.dot(objective_weight[j, :], metrics['objective_value'][j, :])
                if j in J_C:
                    if metrics['afsc_value'][j] < afsc_value_min[j]:

                        # Either we reduce z by some penalty or z is set to 0
                        metrics['afsc_constraint_fail'][j] = 1
                        metrics['total_failed_constraints'] += 1
                        if constraints == 'Penalty':
                            penalty += afscs_overall_weight * afsc_weight[j]
                        elif constraints == 'Fail':
                            failed = True
                            break

        else:

            # Either we reduce z by some penalty or z is set to 0
            metrics['afsc_constraint_fail'][j] = 1
            metrics['total_failed_constraints'] += 1
            if constraints == 'Penalty':
                penalty += afscs_overall_weight * afsc_weight[j]
            elif constraints == 'Fail':
                failed = True
                break

    if not failed:
        metrics['cadet_value'] = np.array([utility[i, int(chromosome[i])] for i in I])
        for i in I_C:
            if metrics['cadet_value'][i] < cadet_value_min[i]:

                # Either we reduce z by some penalty or z is set to 0
                metrics['cadet_constraint_fail'][i] = 1
                metrics['total_failed_constraints'] += 1
                if constraints == 'Penalty':
                    penalty += cadets_overall_weight * cadet_weight[i]

    # Solution Matched Vector
    metrics['afsc_solution'] = np.array([afsc_vector[int(chromosome[i])] for i in I])

    # Calculate Overall Values
    metrics['cadets_overall_value'] = np.dot(cadet_weight, metrics['cadet_value'])
    metrics['afscs_overall_value'] = np.dot(afsc_weight, metrics['afsc_value'])
    metrics['z'] = cadets_overall_weight * metrics['cadets_overall_value'] + \
                   afscs_overall_weight * metrics['afscs_overall_value']
    metrics['num_ineligible'] = 0

    if failed:
        metrics['z'] = 0

    if constraints == 'Penalty':
        penalized_z = metrics['z'] - penalty ** (1 / penalty_scale)
        if printing:
            print("Measured solution vector fitness: " + str(round(metrics['z'], 4)))
            print("Measured solution vector penalized fitness: " + str(round(penalized_z, 4)))
        return metrics, penalized_z
    else:
        if printing:
            print("Measured solution vector fitness: " + str(round(metrics['z'], 4)))
        return metrics


def find_original_solution_ineligibility(parameters, solution=None, printing=True):
    """
    This procedure takes a solution, presumably an original AFPC solution, and finds the cadets that were matched
    to an ineligible AFSC
    :param solution: afsc solution vector
    :param parameters: cadet afsc parameters
    :param printing: whether the procedure should print something
    """

    if printing:
        print('Finding ineligibility of matched cadets...')

    for i in parameters['I']:
        afsc_index = int(solution[i])
        if afsc_index not in parameters['J_E'][i]:
            if printing:
                print('cadet', i, '->', parameters['afsc_vector'][afsc_index])


def solution_similarity_coordinates(similarity_matrix):
    """
    This procedure takes in a similarity matrix then performs MDS and returns the coordinates
    to plot the solutions in terms of how similar they are to each other
    :param similarity_matrix: similarity matrix
    :return: coordinates
    """
    # Change similarity matrix into distance matrix
    distances = 1 - similarity_matrix

    # Get coordinates
    if use_manifold:
        mds = manifold.MDS(n_components=2, dissimilarity='precomputed', random_state=10)
        results = mds.fit(distances)
        coords = results.embedding_
    else:
        coords = np.zeros([len(distances), 2])
        print('Sklearn manifold not available')

    return coords


# Export Procedures
def pyomo_measures_to_excel(X, measures, values, parameters, value_parameters, filepath=None, printing=False):
    """
    Exports x matrix to excel along with objective values and measures
    :param values: objective values
    :param measures: objective measures
    :param parameters: cadet parameters
    :param value_parameters: value parameters
    :param X: X matrix
    :param filepath: filepath
    :param printing: if we print something out
    :return: None
    """
    if printing:
        print('Exporting Pyomo measures to excel...')

    X_df = pd.DataFrame({'Encrypt_PII': parameters['SS_encrypt']})
    for j, afsc in enumerate(parameters['afsc_vector']):
        X_df[afsc] = X[:, j]

    measures_df = pd.DataFrame({'AFSC': parameters['afsc_vector']})
    values_df = pd.DataFrame({'AFSC': parameters['afsc_vector']})
    for k, objective in enumerate(value_parameters['objectives']):
        measures_df[objective] = measures[:, k]
        values_df[objective] = values[:, k]

    if filepath is None:
        filepath = paths['Data Processing Support'] + 'X_Matrix.xlsx'

    with pd.ExcelWriter(filepath) as writer:  # Export to excel
        X_df.to_excel(writer, sheet_name="X", index=False)
        measures_df.to_excel(writer, sheet_name="Measures", index=False)
        values_df.to_excel(writer, sheet_name="Values", index=False)


def data_to_excel(filepath, parameters, value_parameters=None, metrics=None, printing=False):
    """
    This procedures takes in an output filepath, as well as an array of parameters, then exports the parameters as a
    dataframe to that excel filepath
    :param printing: whether or not the procedure should print out something
    :param metrics: optional solution metrics
    :param value_parameters: optional user defined parameters, for if we want to print out those too
    :param filepath: The filepath we wish to write the dataframe to
    :param parameters: The array of fixed cadet/AFSC parameters we wish to write to excel (see above for structure of
    parameters)
    :return: None.
    """

    if printing:
        print("Exporting to excel...")

    # Convert utility matrix to utility columns
    preferences, utilities_array = get_utility_preferences(parameters)

    # Build Cadets Fixed data frame
    cadets_fixed = pd.DataFrame(
        {'Encrypt_PII': parameters['SS_encrypt']})

    # Load Instance Parameters (may or may not be included)
    cadet_parameter_dictionary = {'Male': 'male', 'Minority': 'minority', 'USAFA': 'usafa', 'ASC1': 'asc1',
                                  'ASC2': 'asc2', 'CIP1': 'cip1', 'CIP2': 'cip2', 'percentile': 'merit'}
    for col_name in list(cadet_parameter_dictionary.keys()):
        if cadet_parameter_dictionary[col_name] in parameters:
            cadets_fixed[col_name] = parameters[cadet_parameter_dictionary[col_name]]

    # Loop through all the choices
    for i in range(parameters['P']):
        cadets_fixed['NrWgt' + str(i + 1)] = utilities_array[:, i]
    for i in range(parameters['P']):
        cadets_fixed['NRat' + str(i + 1)] = preferences[:, i]

    M = parameters['M']

    # Loop through all the AFSCs
    for j, afsc in enumerate(parameters['afsc_vector']):
        cadets_fixed['qual_' + afsc] = parameters['qual'][:, j]

    # Build AFSCs Fixed data frame
    afscs_fixed = pd.DataFrame({'AFSC': parameters['afsc_vector']})

    if 'usafa' in parameters:
        afscs_fixed['USAFA Target'] = parameters['usafa_quota']
        afscs_fixed['ROTC Target'] = parameters['rotc_quota']

    afscs_fixed['Combined Target'] = parameters['quota']
    afscs_fixed['Min'] = parameters['quota_min']
    afscs_fixed['Max'] = parameters['quota_max']
    afscs_fixed['Eligible Cadets'] = [len(parameters['I_E'][j]) for j in range(M)]

    if 'usafa' in parameters:
        afscs_fixed['USAFA Cadets'] = [len(parameters['I_D']['USAFA Proportion'][j]) for j in range(M)]

    if 'mandatory' in parameters:
        afscs_fixed['Mandatory Cadets'] = [len(parameters['I_D']['Mandatory'][j]) for j in range(M)]
        afscs_fixed['Desired Cadets'] = [len(parameters['I_D']['Desired'][j]) for j in range(M)]
        afscs_fixed['Permitted Cadets'] = [len(parameters['I_D']['Permitted'][j]) for j in range(M)]

    # Build value parameters dataframes if need be
    if value_parameters is not None:
        O = len(value_parameters['objectives'])
        F_bp_strings = np.array([[" " * 400 for _ in range(O)] for _ in range(M)])
        F_v_strings = np.array([[" " * 400 for _ in range(O)] for _ in range(M)])
        for j, afsc in enumerate(parameters['afsc_vector']):
            for k, objective in enumerate(value_parameters['objectives']):
                string_list = [str(x) for x in value_parameters['F_bp'][j][k]]
                F_bp_strings[j, k] = ",".join(string_list)
                string_list = [str(x) for x in value_parameters['F_v'][j][k]]
                F_v_strings[j, k] = ",".join(string_list)

        F_bps = np.ndarray.flatten(F_bp_strings)
        F_vs = np.ndarray.flatten(F_v_strings)
        afsc_objective_min_values = np.ndarray.flatten(value_parameters['objective_value_min'])
        afsc_objective_convex_constraints = np.ndarray.flatten(value_parameters['constraint_type'])
        afsc_objective_targets = np.ndarray.flatten(value_parameters['objective_target'])
        afsc_objectives = np.tile(value_parameters['objectives'], parameters['M'])
        afsc_objective_weights = np.ndarray.flatten(value_parameters['objective_weight'])
        afsc_value_functions = np.ndarray.flatten(value_parameters['value_functions'])
        afscs = np.ndarray.flatten(np.array(list(np.repeat(parameters['afsc_vector'][j],
                                                           value_parameters['O']) for j in range(parameters['M']))))
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
                                        'Function Breakpoints': F_bps, 'Function Breakpoint Values': F_vs,
                                        'Value Functions': afsc_value_functions})

        cadet_weights_df = pd.DataFrame({'Cadet': parameters['SS_encrypt'],
                                         'Weight': value_parameters['cadet_weight'],
                                         'Min Value': value_parameters['cadet_value_min']})

        overall_weights_df = pd.DataFrame({'Cadets Weight': [value_parameters['cadets_overall_weight']],
                                           'AFSCs Weight': [value_parameters['afscs_overall_weight']],
                                           'Cadets Min Value': [value_parameters['cadets_overall_value_min']],
                                           'AFSCs Min Value': [value_parameters['afscs_overall_value_min']],
                                           'AFSC Weight Function': [value_parameters['afsc_weight_function']],
                                           'Cadet Weight Function': [value_parameters['cadet_weight_function']]})

    # Build the solution metrics dataframes if need be
    if metrics is not None:

        cadet_solution_df = pd.DataFrame({'Cadet': parameters['SS_encrypt'], 'Matched': metrics['afsc_solution'],
                                          'Value': metrics['cadet_value'],
                                          'Weight': value_parameters['cadet_weight'],
                                          'Value Fail': metrics['cadet_constraint_fail']})

        objective_measures = pd.DataFrame({'AFSC': parameters['afsc_vector']})
        objective_values = pd.DataFrame({'AFSC': parameters['afsc_vector']})
        afsc_constraints_df = pd.DataFrame({'AFSC': parameters['afsc_vector']})
        for k in range(value_parameters['O']):
            objective_measures[value_parameters['objectives'][k]] = metrics['objective_measure'][:, k]
            objective_values[value_parameters['objectives'][k]] = metrics['objective_value'][:, k]
            afsc_constraints_df[value_parameters['objectives'][k]] = metrics['objective_constraint_fail'][:, k]

        objective_values['AFSC Value'] = metrics['afsc_value']
        afsc_constraints_df['AFSC Value Fail'] = metrics['afsc_constraint_fail']

        metric_names = ['Z', 'Cadet Value', 'AFSC Value', 'Num Ineligible', 'Failed Constraints']
        metric_results = [metrics['z'], metrics['cadets_overall_value'], metrics['afscs_overall_value'],
                          metrics['num_ineligible'], metrics['total_failed_constraints']]
        for k, objective in enumerate(value_parameters['objectives']):
            metric_names.append(objective + ' Score')
            metric_results.append(metrics['objective_score'][k])

        overall_solution = pd.DataFrame({'Solution Metric': metric_names, 'Result': metric_results})

    with pd.ExcelWriter(filepath) as writer:  # Export to excel
        cadets_fixed.to_excel(writer, sheet_name="Cadets Fixed", index=False)
        afscs_fixed.to_excel(writer, sheet_name="AFSCs Fixed", index=False)
        if value_parameters is not None:
            overall_weights_df.to_excel(writer, sheet_name="Overall Weights", index=False)
            cadet_weights_df.to_excel(writer, sheet_name="Cadet Weights", index=False)
            afsc_weights_df.to_excel(writer, sheet_name="AFSC Weights", index=False)
        if metrics is not None:
            cadet_solution_df.to_excel(writer, sheet_name="Cadet Solution Quality", index=False)
            objective_measures.to_excel(writer, sheet_name="AFSC Objective Measures", index=False)
            objective_values.to_excel(writer, sheet_name="AFSC Solution Quality", index=False)
            afsc_constraints_df.to_excel(writer, sheet_name="AFSC Constraint Fails", index=False)
            overall_solution.to_excel(writer, sheet_name="Overall Solution Quality", index=False)

