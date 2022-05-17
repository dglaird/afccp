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
    try:
        info_df = import_data(filepath, sheet_name="All Cadet Info")
    except:
        info_df = None
    cadets_fixed = import_data(filepath, sheet_name="Cadets Fixed")
    afscs_fixed = import_data(filepath, sheet_name="AFSCs Fixed")

    return info_df, cadets_fixed, afscs_fixed


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
    if "NR_Pref_1" in columns:
        P = len([col for col in columns if 'NR_Pref_' in col])
    else:
        P = len([col for col in columns if 'NRat' in col])

    # Initialize parameters dictionary
    parameters = {'afsc_vector': afsc_vector,
                  'P': P, "quota": np.array(afscs_fixed.loc[:, 'Combined Target']), 'N': N, 'M': M, 'qual': qual,
                  'quota_max': np.array(afscs_fixed.loc[:, 'Max']), 'quota_min': np.array(afscs_fixed.loc[:, 'Min']),
                  'utility': np.zeros([N, M])}

    # Phasing out the old ID
    if "Cadet" in columns:
        parameters["ID"] = np.array(cadets_fixed.loc[:, 'Cadet'])
    else:
        parameters["ID"] = np.array(cadets_fixed.loc[:, 'Encrypt_PII'])

    if "Assigned" in columns:
        parameters["assigned"] = np.array(cadets_fixed.loc[:, 'Assigned'])
    else:
        parameters["assigned"] = np.array(["" for _ in range(N)])

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
                                  'ASC2': 'asc2', 'CIP1': 'cip1', 'CIP2': 'cip2', 'percentile': 'merit',
                                  'percentile_all': 'merit_all'}
    for col_name in cadet_parameter_dictionary:
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
    try:  # Phasing out "NRat"
        preferences_array = np.array(cadets_fixed.loc[:, 'NR_Pref_' + str(1):'NR_Pref_' + str(parameters['P'])])
    except:
        preferences_array = np.array(cadets_fixed.loc[:, 'NRat' + str(1):'NRat' + str(parameters['P'])])
    try:  # Phasing out "NrWgt"
        utilities_array = np.array(cadets_fixed.loc[:, 'NR_Util_' + str(1):'NR_Util_' + str(parameters['P'])])
    except:
        utilities_array = np.array(cadets_fixed.loc[:, 'NrWgt' + str(1):'NrWgt' + str(parameters['P'])])
    for i in range(N):
        for p in range(parameters['P']):
            j = np.where(preferences_array[i, p] == afsc_vector)[0]
            if len(j) != 0:
                parameters['utility'][i, j[0]] = utilities_array[i, p]

    return parameters


def model_data_frame_from_fixed_parameters(parameters):
    """
    This procedure takes a set of parameters and constructs the two fixed dataframes: cadets and AFSCs
    :param parameters: fixed model parameters
    :return: dataframes
    """

    # Convert utility matrix to utility columns
    preferences, utilities_array = get_utility_preferences(parameters)

    # Build Cadets Fixed data frame
    cadets_fixed = pd.DataFrame(
        {'Cadet': parameters['ID'], 'Assigned': parameters['assigned']})

    # Load Instance Parameters (may or may not be included)
    cadet_parameter_dictionary = {'Male': 'male', 'Minority': 'minority', 'USAFA': 'usafa', 'ASC1': 'asc1',
                                  'ASC2': 'asc2', 'CIP1': 'cip1', 'CIP2': 'cip2', 'percentile': 'merit',
                                  'percentile_all': 'merit_all'}
    for col_name in cadet_parameter_dictionary:
        if cadet_parameter_dictionary[col_name] in parameters:
            cadets_fixed[col_name] = parameters[cadet_parameter_dictionary[col_name]]

    # Loop through all the choices
    for i in range(parameters['P']):
        cadets_fixed['NR_Util_' + str(i + 1)] = utilities_array[:, i]
    for i in range(parameters['P']):
        cadets_fixed['NR_Pref_' + str(i + 1)] = preferences[:, i]

    # Number of AFSCs
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
    afscs_fixed['Eligible Cadets'] = [len(parameters['I^E'][j]) for j in range(M)]

    if 'usafa' in parameters:
        afscs_fixed['USAFA Cadets'] = [len(parameters['I^D']['USAFA Proportion'][j]) for j in range(M)]

    if 'mandatory' in parameters:
        afscs_fixed['Mandatory Cadets'] = [len(parameters['I^D']['Mandatory'][j]) for j in range(M)]
        afscs_fixed['Desired Cadets'] = [len(parameters['I^D']['Desired'][j]) for j in range(M)]
        afscs_fixed['Permitted Cadets'] = [len(parameters['I^D']['Permitted'][j]) for j in range(M)]

    return cadets_fixed, afscs_fixed


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
    parameters['J^E'] = [np.where(
        parameters['ineligible'][i, :] == 0)[0] for i in parameters['I']]  # set of AFSCs that cadet i is eligible for

    # AFSC Indexed Sets
    parameters['I^E'] = [np.where(
        parameters['ineligible'][:, j] == 0)[0] for j in parameters['J']]  # set of cadets that are eligible for AFSC j

    # Add demographic sets if they're included
    parameters['I^D'] = {}
    if 'usafa' in parameters:
        usafa = np.where(parameters['usafa'] == 1)[0]  # set of usafa cadets
        parameters['usafa_proportion'] = np.mean(parameters['usafa'])
        parameters['I^D']['USAFA Proportion'] = [np.intersect1d(parameters['I^E'][j], usafa) for j in parameters['J']]
    if 'mandatory' in parameters:
        parameters['I^D']['Mandatory'] = [np.where(parameters['mandatory'][:, j] == 1)[0] for j in parameters['J']]
        parameters['I^D']['Desired'] = [np.where(parameters['desired'][:, j] == 1)[0] for j in parameters['J']]
        parameters['I^D']['Permitted'] = [np.where(parameters['permitted'][:, j] == 1)[0] for j in parameters['J']]

    if 'male' in parameters:
        male = np.where(parameters['male'] == 1)[0]  # set of male cadets
        parameters['I^D']['Male'] = [np.intersect1d(parameters['I^E'][j], male) for j in parameters['J']]
        parameters['male_proportion'] = np.mean(parameters['male'])

    if 'minority' in parameters:
        minority = np.where(parameters['minority'] == 1)[0]  # set of minority cadets
        parameters['I^D']['Minority'] = [np.intersect1d(parameters['I^E'][j], minority) for j in parameters['J']]
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
def import_solution_from_excel(filepath, solution_name=None, afsc_vector=None, excel_format='Specific',
                               printing=False):
    """
    Imports a solution from excel and converts it to a vector of AFSC indices
    :param solution_name: name of solution
    :param afsc_vector: list of AFSCs
    :param excel_format: the kind of excel file we're importing from
    :param filepath: file path
    :param printing: Whether the procedure should print something
    :return: solution (vector of AFSC indices)
    """

    if printing:
        print('Importing solution from excel...')

    if afsc_vector is None:
        afscs_fixed = import_data(filepath, sheet_name="AFSCs Fixed")
        afsc_vector = np.array(afscs_fixed['AFSC'])

    if excel_format == 'Specific':
        sheet_name = "Cadet Solution Quality"
    elif excel_format == 'Original':
        sheet_name = "Original Solution"
    else:  # From Aggregate File
        sheet_name = "Solutions"
    solutions_df = import_data(filepath, sheet_name=sheet_name)

    if solution_name is None and excel_format not in ['Specific', 'Original']:
        raise ValueError('No solution name provided')
    elif excel_format in ['Specific', 'Original']:
        afsc_solution = np.array(solutions_df['Matched'])
    else:
        afsc_solution = np.array(solutions_df[solution_name])

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

    # Shorthand notation
    p = parameters
    vp = value_parameters

    # Construct X matrix from vector
    if len(np.shape(solution)) == 1:
        x_matrix = False
        x = np.array([[1 if solution[i] == j else 0 for j in p['J']] for i in p['I']])
    else:
        x_matrix = True
        x = solution

    # Create metrics dictionary
    metrics = {'objective_measure': np.zeros([p['M'], vp['O']]), 'objective_value': np.zeros([p['M'], vp['O']]),
               'afsc_value': np.zeros(p['M']), 'cadet_value': np.zeros(p['N']),
               'cadet_constraint_fail': np.zeros(p['N']), 'afsc_constraint_fail': np.zeros(p['M']),
               'objective_score': np.zeros(vp['O']), 'total_failed_constraints': 0, 'x': x,
               'objective_constraint_fail': np.array([[" " * 30 for _ in range(vp['O'])] for _ in range(p['M'])])}

    # Get certain objective indices
    if 'merit' in p:
        merit_k = np.where(vp['objectives'] == 'Merit')[0][0]

    if 'usafa' in p:
        usafa_k = np.where(vp['objectives'] == 'USAFA Proportion')[0][0]

    # Loop through all AFSCs to assign their individual values
    for j in p['J']:

        # Number of assigned cadets
        count = np.sum(x[i, j] for i in p['I^E'][j])

        # Only calculate value if we assigned at least one cadet (otherwise AFSC has a value of 0)
        if count != 0:

            # Get more variables for this AFSC
            if 'usafa' in p:
                usafa_count = np.sum(x[i, j] for i in p['I^D']['USAFA Proportion'][j])

            # Are we using approximate measures or not
            if approximate:
                num_cadets = p['quota'][j]
            else:
                num_cadets = count

            # Loop through all objectives that this AFSC cares about
            for k in vp['K^A'][j]:

                # Get the correct measure for this objective
                objective = vp['objectives'][k]
                if objective == 'Merit':
                    numerator = np.sum(p['merit'][i] * x[i, j] for i in p['I^E'][j])
                    metrics['objective_measure'][j, k] = numerator / num_cadets
                elif objective == 'Utility':
                    numerator = np.sum(p['utility'][i, j] * x[i, j] for i in p['I^E'][j])
                    metrics['objective_measure'][j, k] = numerator / num_cadets
                elif objective == 'Combined Quota':
                    metrics['objective_measure'][j, k] = count
                elif objective == 'USAFA Quota':
                    metrics['objective_measure'][j, k] = usafa_count
                elif objective == 'ROTC Quota':
                    metrics['objective_measure'][j, k] = count - usafa_count
                elif objective in vp['K^D']:
                    numerator = np.sum(x[i, j] for i in p['I^D'][objective][j])
                    metrics['objective_measure'][j, k] = numerator / num_cadets

                # Get the correct value for this objective
                metrics['objective_value'][j, k] = value_function(a=vp['a'][j][k],
                                                                  f_a=vp['f^hat'][j][k],
                                                                  r=vp['r'][j][k],
                                                                  x=metrics['objective_measure'][j, k])

                # AFSC Objective Constraints
                if k in vp['K^C'][j]:

                    # Constrained Value (this isn't used though)
                    if vp['constraint_type'][j, k] == 1 or vp['constraint_type'][j, k] == 2:

                        # If the objective value is less than the lower bound on value, constraint is failed
                        if metrics['objective_value'][j, k] < float(vp['objective_value_min'][j, k]):
                            metrics['objective_constraint_fail'][j, k] = str(
                                round(metrics['objective_value'][j, k], 3)) + ' < ' + str(
                                float(vp['objective_value_min'][j, k]))
                            metrics['total_failed_constraints'] += 1

                    # Constrained Approximate Measure
                    elif vp['constraint_type'][j, k] == 3:
                        value_list = vp['objective_value_min'][j, k].split(",")
                        min_measure = float(value_list[0].strip())
                        max_measure = float(value_list[1].strip())

                        # Get correct measure constraint
                        if objective not in ['Combined Quota', 'USAFA Quota', 'ROTC Quota']:
                            if numerator / p['quota'][j] < min_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    round(numerator / p['quota'][j], 3)) + ' < ' + str(
                                    min_measure) + '. ' + str(round(100 * (numerator / p['quota'][j]) /
                                                                    min_measure, 2)) + '% Met.'
                                metrics['total_failed_constraints'] += 1
                            elif numerator / p['quota'][j] > max_measure:
                                metrics['objective_constraint_fail'][j, k] = str(
                                    round(numerator / p['quota'][j], 3)) + ' > ' + str(
                                    max_measure) + '. ' + str(round(100 * max_measure /
                                                                    (numerator / p['quota'][j]), 2)) + '% Met.'
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
                    elif vp['constraint_type'][j, k] == 4:

                        value_list = vp['objective_value_min'][j, k].split(",")
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
            metrics['afsc_value'][j] = np.dot(vp['objective_weight'][j, :], metrics['objective_value'][j, :])
            if metrics['afsc_value'][j] < vp['afsc_value_min'][j]:
                metrics['afsc_constraint_fail'][j] = 1
                metrics['total_failed_constraints'] += 1

            # Also calculate Average Merit and USAFA proportion even if not in the AFSC's objectives
            if 'merit' in p:
                if merit_k not in vp['K^A'][j]:
                    numerator = np.sum(p['merit'][i] * x[i, j] for i in p['I^E'][j])
                    metrics['objective_measure'][j, merit_k] = numerator / num_cadets
            if 'usafa' in p:
                if usafa_k not in vp['K^A'][j]:
                    numerator = np.sum(x[i, j] for i in p['I^D']['USAFA Proportion'][j])
                    metrics['objective_measure'][j, usafa_k] = numerator / num_cadets

    # Loop through all cadets to assign their values
    for i in p['I']:
        metrics['cadet_value'][i] = np.sum(x[i, j] * p['utility'][i, j] for j in p['J^E'][i])
        if metrics['cadet_value'][i] < vp['cadet_value_min'][i]:
            metrics['cadet_constraint_fail'][i] = 1
            metrics['total_failed_constraints'] += 1

    # Generate objective scores for each objective
    for k in vp['K']:
        new_weights = vp['afsc_weight'] * vp['objective_weight'][:, k]
        new_weights = new_weights / sum(new_weights)
        metrics['objective_score'][k] = np.dot(new_weights, metrics['objective_value'][:, k])

    # Solution Matched Vector
    metrics['afsc_solution'] = np.array([" " * 10 for _ in p['I']])

    # turns the x matrix back into a vector of AFSC indices
    solution = np.where(x)[1]

    # Translate AFSC indices into the AFSCs themselves
    for i in p['I']:
        metrics['afsc_solution'][i] = p['afsc_vector'][int(solution[i])]

    # Define overall metrics
    metrics['cadets_overall_value'] = np.dot(vp['cadet_weight'], metrics['cadet_value'])
    metrics['afscs_overall_value'] = np.dot(vp['afsc_weight'], metrics['afsc_value'])
    metrics['z'] = vp['cadets_overall_weight'] * metrics['cadets_overall_value'] + \
                   vp['afscs_overall_weight'] * metrics['afscs_overall_value']
    metrics['num_ineligible'] = np.sum(x[i, j] * p['ineligible'][i, j] for j in
                                    p['J'] for i in p['I'])
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
    This function takes in a chromosome (solution vector) and evaluates it. This function is only here because I wanted
    to test the function to make sure it was returning the same objective value as the "measure_solution_quality"
    function since the two calculate the objective value a little differently. This one uses a vector of AFSCs and
    can therefore use sets more efficiently while the other one uses a binary solution matrix which is more like
    how the model actually calculates the objective value. Assuming integer solutions (and no constraints are violated),
    the two functions will return the same value.
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

    # Shorthand
    p = parameters
    vp = value_parameters
    objective_min = np.zeros((p['M'], vp['O']))
    objective_max = np.zeros((p['M'], vp['O']))

    # Initialize metrics
    metrics = {'objective_measure': np.zeros([p['M'], vp['O']]), 'objective_value': np.zeros([p['M'], vp['O']]),
               'afsc_value': np.zeros(p['M']), 'cadet_value': np.zeros(p['N']),
               'afsc_constraint_fail': np.zeros(p['M']), 'objective_constraint_fail': np.zeros([p['M'], vp['O']]),
               'total_failed_constraints': 0, 'cadet_constraint_fail': np.zeros(p['N'])}

    # Determine if we should calculate usafa count
    soc_counts = np.zeros(p['M'])
    for j in p['J']:

        # If we care about USAFA Count and ROTC Count
        if 'USAFA Count' in vp['objectives'][vp['K^A'][j]]:
            soc_counts[j] = 1

        # Get minimum/maximum values for constraints
        for k in vp['K^C'][j]:

            # Approximate or Exact Value Constraint
            if vp['objective_constraint_type'][j, k] == 1 or vp['objective_constraint_type'][j, k] == 2:
                objective_min[j, k] = float(vp['objective_value_min'][j, k])

            # Approximate or Exact Measure Constraint
            elif vp['objective_constraint_type'][j, k] == 3 or vp['objective_constraint_type'][j, k] == 4:
                value_list = vp['objective_value_min'][j, k].split(",")
                objective_min[j, k] = float(value_list[0].strip())
                objective_max[j, k] = float(value_list[1].strip())

    # If USAFA counts are used
    if sum(soc_counts) == 0:
        ignore_uc = True
    else:
        ignore_uc = False

    # Constraint penalty variables
    failed = False
    penalty = 0

    # Loop through all AFSCs to assign their values
    for j in p['J']:

        # list of indices of assigned cadets
        cadets = np.where(chromosome == j)[0]

        # Only calculate measures for AFSCs with at least one cadet
        count = len(cadets)
        usafa_count = count

        # If there's at least one cadet assigned to the AFSC, we can calculate a non-zero value
        if count > 0:
            if not ignore_uc:
                usafa_cadets = np.intersect1d(p['I^D']['USAFA Proportion'][j], cadets)
                usafa_count = len(usafa_cadets)

            # Loop through all AFSC objectives that apply to this AFSC
            for k in vp['K^A'][j]:
                objective = vp['objectives'][k]
                if objective == 'Merit':
                    metrics['objective_measure'][j, k] = np.mean(p['merit'][cadets])
                elif objective == 'Utility':
                    metrics['objective_measure'][j, k] = np.mean(p['utility'][cadets, j])
                elif objective == 'Combined Quota':
                    metrics['objective_measure'][j, k] = count
                elif objective == 'USAFA Quota':
                    metrics['objective_measure'][j, k] = usafa_count
                elif objective == 'ROTC Quota':
                    metrics['objective_measure'][j, k] = count - usafa_count
                elif objective in vp['K^D']:
                    metrics['objective_measure'][j, k] = len(np.intersect1d(p['I^D'][objective][j], cadets)) / count

                # Calculate AFSC objective value
                metrics['objective_value'][j, k] = value_function(vp['a'][j][k], vp['f^hat'][j][k], vp['r'][j][k],
                                                                  metrics['objective_measure'][j, k])

                # AFSC Objective Constraints
                if k in vp['K^C'][j]:

                    # We're really only ever going to constrain the approximate measure for Mandatory Tier
                    if objective == 'Mandatory':
                        constrained_measure = (metrics['objective_measure'][j, k] * count) / p['quota'][j]
                    else:
                        constrained_measure = metrics['objective_measure'][j, k]

                    # Use the real constraint (potentially different as a result of approximate model)
                    constrained_min, constrained_max = objective_min[j, k], objective_max[j, k]

                    # This constraint fail dictionary keeps track of the constraints that are failed by the
                    # optimization model by maybe 1 constraint
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
                                adj_con_tolerance = min((constrained_min - (1 / p['quota'][j])) / constrained_min,
                                                        constrained_max / (constrained_max + (1 / p['quota'][j])))
                            else:
                                adj_con_tolerance = 0.95

                            # Either we reduce z by some penalty or z is set to 0
                            if constraints == 'Penalty' and p_con_met < adj_con_tolerance:
                                penalty += vp['afscs_overall_weight'] * vp['afsc_weight'][j]
                            elif constraints == 'Fail' and p_con_met < adj_con_tolerance:
                                failed = True
                                break
            if failed:
                break
            else:

                # Calculate AFSC value
                metrics['afsc_value'][j] = np.dot(vp['objective_weight'][j, :], metrics['objective_value'][j, :])
                if j in vp['J^C']:
                    if metrics['afsc_value'][j] < vp['afsc_value_min'][j]:

                        # Either we reduce z by some penalty or z is set to 0
                        metrics['afsc_constraint_fail'][j] = 1
                        metrics['total_failed_constraints'] += 1
                        if constraints == 'Penalty':
                            penalty += vp['afscs_overall_weight'] * vp['afsc_weight'][j]
                        elif constraints == 'Fail':
                            failed = True
                            break

        else:

            # Either we reduce z by some penalty or z is set to 0
            metrics['afsc_constraint_fail'][j] = 1
            metrics['total_failed_constraints'] += 1
            if constraints == 'Penalty':
                penalty += vp['afscs_overall_weight'] * vp['afsc_weight'][j]
            elif constraints == 'Fail':
                failed = True
                break

    if not failed:
        metrics['cadet_value'] = np.array([p['utility'][i, int(chromosome[i])] for i in p['I']])
        for i in vp['I^C']:
            if metrics['cadet_value'][i] < vp['cadet_value_min'][i]:

                # Either we reduce z by some penalty or z is set to 0
                metrics['cadet_constraint_fail'][i] = 1
                metrics['total_failed_constraints'] += 1
                if constraints == 'Penalty':
                    penalty += vp['cadets_overall_weight'] * vp['cadet_weight'][i]

    # Solution Matched Vector
    metrics['afsc_solution'] = np.array([p['afsc_vector'][int(chromosome[i])] for i in p['I']])

    # Calculate Overall Values
    metrics['cadets_overall_value'] = np.dot(vp['cadet_weight'], metrics['cadet_value'])
    metrics['afscs_overall_value'] = np.dot(vp['afsc_weight'], metrics['afsc_value'])
    metrics['z'] = vp['cadets_overall_weight'] * metrics['cadets_overall_value'] + \
                   vp['afscs_overall_weight'] * metrics['afscs_overall_value']
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
        if afsc_index not in parameters['J^E'][i]:
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


# Export solution metrics
def pyomo_measures_to_excel(x, measures, values, parameters, value_parameters, filepath=None, printing=False):
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

    x_df = pd.DataFrame({'Encrypt_PII': parameters['SS_encrypt']})
    for j, afsc in enumerate(parameters['afsc_vector']):
        x_df[afsc] = x[:, j]

    measures_df = pd.DataFrame({'AFSC': parameters['afsc_vector']})
    values_df = pd.DataFrame({'AFSC': parameters['afsc_vector']})
    for k, objective in enumerate(value_parameters['objectives']):
        measures_df[objective] = measures[:, k]
        values_df[objective] = values[:, k]

    if filepath is None:
        filepath = paths_out['results'] + 'X_Matrix.xlsx'

    with pd.ExcelWriter(filepath) as writer:  # Export to excel
        x_df.to_excel(writer, sheet_name="X", index=False)
        measures_df.to_excel(writer, sheet_name="Measures", index=False)
        values_df.to_excel(writer, sheet_name="Values", index=False)




