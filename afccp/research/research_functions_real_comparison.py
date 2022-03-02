import pandas as pd
import datetime
from problem_class import *
from research_graphs import *


# Thesis Comparison Against Real Solutions
def solve_class_year_real_instances(years=None, instances=None, solve_ga_from_excel=False, solve_ga=True,
                                    ga_max_time=60*10, pyomo_max_time=10, printing=True):
    """
    This procedure loads in the variously generated instances for each class year, solves them, then exports the
    results back to excel
    :param pyomo_max_time: max solve time for pyomo
    :param ga_max_time: max solve time for GA
    :param solve_ga: if we want to solve the genetic algorithm
    :param solve_ga_from_excel: if we want to take a solved solution from excel and evolve it using GA
    :param instances: list of instances to solve
    :param years: years to solve/compare
    :param printing: whether the procedure should print something
    :return: Results table
    """

    if years is None:
        years = [2021]

    if instances is None:
        instances = np.arange(1, 11).astype(int)

    if printing:
        print('Conducting Class Year Instance Analysis...')

    # Loop through all years
    for year in years:

        # Print Year
        if printing:
            print('')
            print('')
            print_str = 22 * ' '
            print_str += '<' + str(year) + '>'
            print_str += (60 - len(print_str)) * ' '
            print(print_str)
            print('')

        for i in instances:

            if printing:
                print('')
                print_str = 20 * '='
                print_str += '<Instance ' + str(i) + '>'
                print_str += (56 - len(print_str)) * '='
                print(print_str)

            # Load instance
            data_name = str(year) + '_VP_' + str(i)
            instance = CadetCareerProblem(data_name, filepath=paths['Problem Instances'] +
                                                                   'Class Year VP Instances\\' + data_name + '.xlsx')
            instance.import_value_parameters()

            # Load AFPC Solution
            instance.import_solution(filepath=paths['Problem Instances'] + str(year) + '_Original.xlsx')
            z_original = instance.measure_solution()
            original_fail = str(instance.metrics['total_failed_constraints'])

            if printing:
                print('AFPC Model Solution Value: ' + str(z_original) + '.')

            # Export Instance
            instance.export_to_excel()

            if solve_ga_from_excel:
                instance.import_solution()
                VFT_fail = str(instance.metrics['total_failed_constraints'])
            else:

                if printing:
                    now = datetime.datetime.now()
                    print('Solving VFT Model for ' + str(
                        pyomo_max_time) + ' seconds at ' + now.strftime('%H:%M:%S') + '...')

                # Solve Pyomo Model
                instance.solve_vft_pyomo_model(max_time=pyomo_max_time, printing=False)
                VFT_fail = str(instance.metrics['total_failed_constraints'])

            if printing:
                print('Solution value of ' + str(round(instance.metrics['z'], 4)) + ' obtained. ' +
                      VFT_fail + ' failed constraints.')

            if solve_ga:
                if printing:
                    now = datetime.datetime.now()
                    print('Solving Genetic Algorithm for ' + str(ga_max_time) + ' seconds at ' +
                          now.strftime('%H:%M:%S') + '...')

                # Solve GA
                instance.genetic_algorithm(initialize=True, stopping_time=ga_max_time, printing=False)

            # Get metrics
            z_VFT = round(instance.metrics['z'], 4)
            VFT_fail = str(instance.metrics['total_failed_constraints'])

            # Export Instance
            instance.export_to_excel()

            # Compare to AFPC
            p_i = round(((z_VFT - z_original)/z_original) * 100, 3)

            # Last print statement
            if printing:
                print('Model Solved. VFT Z: ' + str(z_VFT) +
                      ', % Improvement: ' + str(p_i) + '.\nAFPC Failed Constraints: ' + original_fail +
                      ', VFT Failed Constraints: ' + VFT_fail + '.')
            if printing:
                print(60 * '=')


def generate_solve_class_year_instances(years=None, instances=None, solve_ga=True,
                                        ga_max_time=60*10, pyomo_max_time=10, printing=True):
    """
    This procedure generates an instance for each class year and then solves it and exports it to excel
    :param pyomo_max_time: max solve time for pyomo
    :param ga_max_time: max solve time for GA
    :param solve_ga: if we want to solve the genetic algorithm
    :param instances: list of instances to solve
    :param years: years to solve/compare
    :param printing: whether the procedure should print something
    :return: Results table
    """

    if years is None:
        years = [2016, 2017, 2018, 2019, 2020, 2021]

    if instances is None:
        instances = np.arange(31, 51).astype(int)

    if printing:
        print('Conducting Class Year Instance Analysis...')

    # Import all data
    options_filename = paths['Data Processing Support'] + 'Value_Parameter_Sets_Options_Real.xlsx'
    defaults_filepath = paths['Data Processing Support'] + 'Value_Parameters_Defaults_Real.xlsx'
    default_value_parameters = default_value_parameters_from_excel(filepath=defaults_filepath)
    year_instances = {}
    year_options = {}
    year_solutions = {}
    for year in years:
        year_options[year] = import_data(options_filename, sheet_name=str(year) + ' Constraint Options')
        filepath = paths['Problem Instances'] + str(year) + '_Original.xlsx'
        year_instances[year] = CadetCareerProblem(str(year), filepath=filepath)
        year_instances[year].import_value_parameters()
        year_instances[year].import_solution()
        year_solutions[year] = copy.deepcopy(year_instances[year].solution)

    # Loop through all instances
    for i in instances:

        # Print Instance
        if printing:
            print('')
            print('')
            print_str = 20 * ' '
            print_str += '<Instance ' + str(i) + '>'
            print_str += (56 - len(print_str)) * ' '
            print(print_str)
            print('')

        # Loop through each year
        for year in years:

            # Print Year
            if printing:
                print('')
                print_str = 22 * '='
                print_str += '<' + str(year) + '>'
                print_str += (60 - len(print_str)) * '='
                print(print_str)

            # Load instance
            data_name = str(year) + '_VP_' + str(i)
            instance = copy.deepcopy(year_instances[year])
            instance.generate_realistic_value_parameters(default_value_parameters, year_options[year],
                                                         deterministic=False, constrain_merit=True)

            # Load AFPC Solution
            z_original = instance.measure_solution()
            original_fail = str(instance.metrics['total_failed_constraints'])

            if printing:
                print('AFPC Model Solution Value: ' + str(z_original) + '.')

            if printing:
                now = datetime.datetime.now()
                print('Solving VFT Model for ' + str(
                    pyomo_max_time) + ' seconds at ' + now.strftime('%H:%M:%S') + '...')

            # Solve Pyomo Model
            instance.solve_vft_pyomo_model(max_time=pyomo_max_time, printing=False)
            VFT_fail = str(instance.metrics['total_failed_constraints'])

            if printing:
                print('Solution value of ' + str(round(instance.metrics['z'], 4)) + ' obtained. ' +
                      VFT_fail + ' failed constraints.')

            if solve_ga:
                if printing:
                    now = datetime.datetime.now()
                    print('Solving Genetic Algorithm for ' + str(ga_max_time) + ' seconds at ' +
                          now.strftime('%H:%M:%S') + '...')

                # Solve GA
                instance.genetic_algorithm(initialize=True, stopping_time=ga_max_time, printing=False)

            # Get metrics
            z_VFT = round(instance.metrics['z'], 4)
            VFT_fail = str(instance.metrics['total_failed_constraints'])

            # Export to Excel
            year_filepath = paths['Problem Instances'] + 'Class Year VP Instances\\' + data_name + '.xlsx'
            instance.export_to_excel(year_filepath)

            # Compare to AFPC
            p_i = round(((z_VFT - z_original)/z_original) * 100, 3)

            # Last print statement
            if printing:
                print('Model Solved. VFT Z: ' + str(z_VFT) +
                      ', % Improvement: ' + str(p_i) + '.\nAFPC Failed Constraints: ' + original_fail +
                      ', VFT Failed Constraints: ' + VFT_fail + '.')
            if printing:
                print(60 * '=')


def compile_real_results(years=None, instances=None, printing=True):
    """
    This function loads in all of the instance data and compiles the results
    :param printing:
    :param years:
    :param instances:
    :return: None
    """
    if printing:
        print('Compiling Real results...')

    if years is None:
        years = [2016, 2017, 2018, 2019, 2020, 2021]

    if instances is None:
        instances = np.arange(1, 10).astype(int)

    num_years = len(years)
    num_instances = len(instances)
    num_rows = num_years * num_instances
    data_dict = {'Instance': np.array([' ' * 20 for _ in range(num_rows)]), 'AFPC Z': np.zeros(num_rows),
                 'VFT Z': np.zeros(num_rows), 'Percent Improvement': np.zeros(num_rows),
                 'AFPC Cadets Value': np.zeros(num_rows), 'VFT Cadets Value': np.zeros(num_rows),
                 'AFPC AFSCs Value': np.zeros(num_rows), 'VFT AFSCs Value': np.zeros(num_rows)}
    row = 0
    similarities = {}
    similarities_df = {}
    for year in years:

        # Print Year
        if printing:
            print('')
            print('')
            print_str = 22 * ' '
            print_str += '<' + str(year) + '>'
            print_str += (60 - len(print_str)) * ' '
            print(print_str)
            print('')

        original_filepath = paths['Problem Instances'] + str(year) + '_Original.xlsx'
        original_solution = import_solution_from_excel(original_filepath)
        solutions = {}
        data_names = []
        for i in instances:

            if printing:
                print('')
                print_str = 20 * '='
                print_str += '<Instance ' + str(i) + '>'
                print_str += (56 - len(print_str)) * '='
                print(print_str)

            # Load data
            data_name = str(year) + '_VP_' + str(i)
            data_names.append(data_name)
            data_dict['Instance'][row] = data_name
            filepath = paths['Problem Instances'] + 'Class Year VP Instances\\' + data_name + '.xlsx'
            instance = CadetCareerProblem(data_name, filepath)
            instance.import_value_parameters()
            instance.import_solution()
            vft_metrics = instance.metrics
            solutions[data_name] = instance.solution
            instance.measure_solution(original_solution)
            original_metrics = instance.metrics

            # Gather Information
            data_dict['VFT Z'][row] = round(vft_metrics['z'], 4)
            data_dict['AFPC Z'][row] = round(original_metrics['z'], 4)
            data_dict['VFT Cadets Value'][row] = round(vft_metrics['cadets_overall_value'], 4)
            data_dict['AFPC Cadets Value'][row] = round(original_metrics['cadets_overall_value'], 4)
            data_dict['VFT AFSCs Value'][row] = round(vft_metrics['afscs_overall_value'], 4)
            data_dict['AFPC AFSCs Value'][row] = round(original_metrics['afscs_overall_value'], 4)
            data_dict['Percent Improvement'][row] = round((data_dict['VFT Z'][row] - data_dict['AFPC Z'][row]) /
                                                          data_dict['AFPC Z'][row], 4)

            if printing:
                print('VFT Z: ' + str(data_dict['VFT Z'][row]) + ', AFPC Z: ' + str(data_dict['AFPC Z'][row]) +
                      ', VFT Cadets: ' + str(data_dict['VFT Cadets Value'][row]))
                print('AFPC Cadets: ' + str(data_dict['AFPC Cadets Value'][row]) + ', VFT AFSCs: ' +
                      str(data_dict['VFT AFSCs Value'][row]) +
                      ', AFPC AFSCs: ' + str(data_dict['AFPC AFSCs Value'][row]))
                print('VFT solution is ' + str(round(data_dict['Percent Improvement'][row] * 100, 4)) +
                      '% better than AFPC solution.')
            row += 1

        # Add AFPC Solution
        data_name = 'AFPC'
        data_names.append(data_name)
        solutions[data_name] = original_solution
        similarities[year] = np.ones([num_instances + 1, num_instances + 1])
        for r, data_name1 in enumerate(data_names):
            for c, data_name2 in enumerate(data_names):
                similarity = compare_solutions(solutions[data_name1], solutions[data_name2])
                similarities[year][r, c] = round(similarity, 2)

        # Create Similarity dataframe
        similarities_df[year] = pd.DataFrame({'Solution Similarity': data_names})
        for c, data_name in enumerate(data_names):
            similarities_df[year][data_name] = similarities[year][:, c]

    compiled_results_df = pd.DataFrame({})
    for column in data_dict:
        compiled_results_df[column] = data_dict[column]

    export_file = paths['Analysis & Results'] + 'Real_Solution_Comparison_Results.xlsx'
    with pd.ExcelWriter(export_file) as writer:
        compiled_results_df.to_excel(writer, sheet_name='Results', index=False)

        for year in years:
            similarities_df[year].to_excel(writer, sheet_name=str(year), index=False)


def plot_real_results(years=None, printing=True):
    """
    Takes the results table and plots it, along with the solution similarities
    :param years:
    :param printing:
    :return:
    """
    if printing:
        print('Compiling Real results...')

    if years is None:
        years = [2016, 2017, 2018, 2019, 2020, 2021]

    # Load df
    results_df = import_data(paths['Analysis & Results'] + 'Real_Solution_Comparison_Results.xlsx',
                             sheet_name='Results')

    # Similarity Plots
    for year in years:

        # Load Similarity Matrix
        filepath = paths['Analysis & Results'] + 'Real_Solution_Comparison_Results.xlsx'

        similarity_df = import_data(filepath, sheet_name=str(year))
        data_names = np.array(similarity_df.loc[:, 'Solution Similarity'])
        similarity_matrix = np.array(similarity_df.loc[:, data_names[0]:data_names[len(data_names) - 1]])

        # Get coordinates
        coords = solution_similarity_coordinates(similarity_matrix)

        # Plot similarity
        chart = solution_similarity_graph(coords, data_names, thesis_chart=True, year=year)
        chart.show()

    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white', tight_layout=True)

    p_i = np.array(results_df.loc[:, 'Percent Improvement'])

    # Histogram
    bins = np.arange(0.025, 0.155, 0.01)
    ax.hist(p_i, bins=bins, color='black', edgecolor='white', alpha=0.5)

    # Ticks
    ax.tick_params(axis='y', labelsize=30)
    ax.tick_params(axis='x', labelsize=30)
    x_ticks = np.array([bins[i] for i in range(len(bins)) if i % 2 == 0])
    x_ticklabels = np.array([str(round(bins[i] * 100, 2)) + "%" for i in range(len(bins)) if i % 2 == 0])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)

    # Labels
    ax.set_ylabel('Number of Instances')
    ax.yaxis.label.set_size(35)
    ax.set_xlabel('Percent Improvement')
    ax.xaxis.label.set_size(35)

    # Save and show
    # fig.savefig(paths['Charts & Figures'] + 'Real_Results_Histogram.png', bbox_inches='tight')
    # fig.show()







