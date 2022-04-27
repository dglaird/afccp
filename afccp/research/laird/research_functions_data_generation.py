from afccp.core.problem_class import *
from afccp.research.research_graphs import *
import copy


# Generated Data Comparison Helper Functions
def evaluate_data(data_dict, baseline_data, printing=False):
    """
    Conducts an aggregate score of the Kolmogorov-Smirnov statistic and Kullback-Leibler Divergence for the
    synthetic data compared to the real data and returns a dictionary of these statistics
    :param data_dict: dictionary of dataframes
    :param baseline_data: some dataframe to compare against
    :param printing: Whether the procedure should print something
    :return: dictionary of evaluation scores
    """
    if printing:
        print('Evaluating Data...')
    evaluation_score = {}
    for name in data_dict.keys():
        evaluation_score[name] = round(evaluate(data_dict[name], baseline_data, aggregate=True,
                                                metrics=['KSTest', 'ContinuousKLDivergence',
                                                         'DiscreteKLDivergence']), 3)
    return evaluation_score


def compare_data(data_dict=None, baseline_data=None, colors=None, averaged=False, both=False, metrics=None,
                 printing=False):
    """
    This is the main procedure to compare a dictionary of dataframes (most likely synthetic)
    with some baseline dataframe (most likely real data). Although it is more often
    than not comparing real with synthetic data, this function could also
    be used for comparing different class years with each other.
    :param metrics: optional metrics df if we already have that
    :param both: if we want to get both averaged data and full data
    :param averaged: if we want to average CTGAN data
    :param data_dict: dictionary of dataframes
    :param baseline_data: some dataframe to compare against
    :param colors: what colors to use for the charts
    :param printing: Whether the procedure should print something
    :return: comparison metrics and chart
    """
    if printing:
        print('Comparing dataframes...')

    # Arbitrarily compare 2019 against the whole ctgan data
    if data_dict is None:
        data_dict = {'2019': import_data('ctgan_data.xlsx', sheet_name='2019')}
    if baseline_data is None:
        baseline_data = ctgan_data_filter()
    if colors is None:
        num_compare = len(data_dict.keys())
        colors = ['green', 'yellow', 'orange', 'purple', 'red', 'cyan', 'magenta', 'lime', 'pink', 'chocolate',
                  'deepskyblue', 'gold', 'deeppink', 'sandybrown', 'olive', 'maroon', 'navy', 'coral', 'teal',
                  'darkorange']
        colors = colors[:num_compare]
        colors.append('blue')

    if 'NrWgt1' in baseline_data.keys():
        baseline_data.drop(labels='NrWgt1', axis=1, inplace=True)

    # Import Generator Parameters
    cip_df = {}
    targets, cip_df["CIP1"], cip_df["CIP2"] = import_generator_parameters()
    afscs = np.array(targets['AFSC'])

    # Evaluate data
    if metrics is None:
        evaluation_score = evaluate_data(data_dict, baseline_data, printing)
        evaluation_score['Base'] = 'N/A'

    # Add baseline data to dictionary
    data_dict['Base'] = baseline_data

    # Names for Metrics Dataframe
    names = list(data_dict.keys())
    metric_names = ['KS/KL Aggregate Score', 'Number of Cadets', 'Usafa Proportion', 'Average Percentile', 'NrWgt2_Avg',
                    'NrWgt3_Avg', 'NrWgt4_Avg', 'NrWgt5_Avg', 'NrWgt6_Avg', 'Num Wrong Utilities',
                    'Num Wrong Preferences', 'Num Wrong Zero Utilities']

    # Initialize dictionaries
    N = {}
    usafa_proportion = {}
    average_percentile = {}
    choice_utility_averages = {}
    num_wrong_utils = {}
    num_wrong_prefs = {}
    num_wrong_util_blanks = {}

    if metrics is None:
        metrics = pd.DataFrame({}, index=metric_names)

        # Loop through each dataset
        for num, name in enumerate(names):
            N[name] = len(data_dict[name])
            usafa_proportion[name] = round(data_dict[name]['USAFA'].mean(), 3)
            average_percentile[name] = round(data_dict[name]['percentile'].mean(), 3)

            # Get average choice utilities
            choice_utility_averages[name] = [1, 0, 0, 0, 0, 0]  # going to alter choices 2 through 6
            for p in range(5):
                choice_utility_averages[name][p + 1] = round(data_dict[name]['NrWgt' + str(p + 2)].mean(), 3)

            # Number of wrong utility rows
            num_wrong_utils[name] = sum(np.array(utilities_match(data_dict[name])) == 0)

            # Number of wrong preference rows
            num_wrong_prefs[name] = sum(np.array(afscs_unique(data_dict[name])) == 0)

            # Number of wrong utility blank rows (blank preferences must translate to 0 utilities)
            num_wrong_util_blanks[name] = 0
            utilities = np.array(data_dict[name].loc[:, 'NrWgt2':'NrWgt6'])
            for i in range(len(utilities)):
                for p in range(4):
                    if utilities[i, p] < utilities[i, p + 1]:
                        num_wrong_util_blanks[name] += 1
                        break

            # Assemble dataframe
            metrics[name] = [evaluation_score[name], int(N[name]), usafa_proportion[name], average_percentile[name],
                             choice_utility_averages[name][1], choice_utility_averages[name][2],
                             choice_utility_averages[name][3], choice_utility_averages[name][4],
                             choice_utility_averages[name][5], num_wrong_utils[name],
                             num_wrong_prefs[name], num_wrong_util_blanks[name]]

    # Get CIP table
    cip_table, cip_table_df = baseline_cip_hist_data(data=baseline_data, num_bins=49, printing=printing)
    with pd.ExcelWriter('CIP_Table.xlsx') as writer:
        cip_table_df.to_excel(writer, sheet_name="Results", index=False)

    # Return requested results
    if both:
        cip_proportions_averaged = cip_hist_data(cip_table=cip_table, averaged=True, data_dict=data_dict,
                                                 printing=printing)
        cip_proportions_full = cip_hist_data(cip_table=cip_table, averaged=False, data_dict=data_dict,
                                             printing=printing)
        cip_hist_chart_averaged = CIP_Hist_Chart(cip_proportions_averaged)
        cip_hist_chart_full = CIP_Hist_Chart(cip_proportions_full, colors=colors)
        average_utility_chart_averaged = Average_Utility_Data_Graph(data_dict=data_dict, colors=colors, averaged=True,
                                                                    afscs=afscs)
        average_utility_chart_full = Average_Utility_Data_Graph(data_dict=data_dict, colors=colors, averaged=False,
                                                                afscs=afscs)
        return cip_hist_chart_averaged, cip_hist_chart_full, average_utility_chart_averaged, \
               average_utility_chart_full, metrics

    else:
        if averaged:
            cip_proportions_averaged = cip_hist_data(cip_table=cip_table, averaged=True, data_dict=data_dict,
                                                     printing=printing)
            cip_hist_chart_averaged = CIP_Hist_Chart(cip_proportions_averaged)
            instance = CadetCareerProblem(filepath=paths['Problem Instances'] + '2019_AFPC.xlsx')
            instance.Scrub_AFSC_Indices(year=2019)
            average_utility_chart_averaged = Average_Utility_Data_Graph(data_dict=data_dict, colors=colors, afscs=afscs,
                                                                        averaged=True, parameters=instance.parameters)
            return cip_hist_chart_averaged, average_utility_chart_averaged, metrics
        else:
            cip_proportions_full = cip_hist_data(cip_table=cip_table, averaged=False, data_dict=data_dict,
                                                 printing=printing)
            cip_hist_chart_full = CIP_Hist_Chart(cip_proportions_full, colors)
            average_utility_chart_full = Average_Utility_Data_Graph(data_dict=data_dict, colors=colors,
                                                                    averaged=False, afscs=afscs)
            return cip_hist_chart_full, average_utility_chart_full, metrics


def baseline_cip_hist_data(data=None, num_bins=49, printing=False):
    """
    This procedure takes the CTGAN historic data and generates the CIP -> group table.
    Essentially, we first get a list of all the unique CIPs in the data as well as the number
    of occurrences of those CIPs. We then put those CIPs into N bins (based on alpha-numeric
    position), and then sort the bins based on number of occurrences of CIPs within the bins.
    We can now plot a histogram of the proportions of cadets with different CIPs.
    :param num_bins: number of bins
    :param data: ctgan data
    :param printing: Whether the procedure should print something
    :return: cip to group table
    """
    if printing:
        print('Generating Baseline CIP Histogram Data Tables...')

    if data is None:
        data = ctgan_data_filter()

    cip_arr = np.array(data.loc[:, 'CIP1']).astype(str)
    unique_cip = np.unique(cip_arr)
    num_unique = len(unique_cip)
    num_cips_per_bin = int(num_unique / num_bins)
    cip_table = np.array([["      " for _ in range(num_cips_per_bin)] for _ in range(num_bins)])
    counts = np.zeros(num_bins)
    for i in range(num_bins):
        for j in range(num_cips_per_bin):
            cip_table[i, j] = unique_cip[i * num_cips_per_bin + j]
            counts[i] += len(np.where(cip_arr == cip_table[i, j])[0])

    sorted_indices = np.argsort(counts)[::-1]
    sorted_table = cip_table[sorted_indices, :]

    table_df = pd.DataFrame({'Bin': np.arange(1, len(sorted_table) + 1)})
    table_df['Count'] = counts[sorted_indices]
    for j in range(num_cips_per_bin):
        table_df['CIP ' + str(j + 1)] = sorted_table[:, j]
    return sorted_table, table_df


def cip_hist_data(cip_table=None, averaged=False, data_dict=None, printing=False):
    """
    This procedure gets the data on the number of occurrences of CIPs
    in each dataset
    :param averaged: If we should average the comparison data or not
    :param cip_table: cip sorted bin table
    :param data_dict: dictionary of datasets
    :param printing: whether we should print something
    :return: dictionary of cip proportions
    """
    if printing:
        print('Collecting CIP occurrence data...')

    if data_dict is None:
        data_dict = {'2019': import_data('ctgan_data.xlsx', sheet_name='2019'),
                     'Base': ctgan_data_filter()}
        if cip_table is None:
            cip_table = baseline_cip_hist_data(data=data_dict['Base'], printing=printing)

    if cip_table is None:
        cip_table = baseline_cip_hist_data(printing=printing)

    # Flatten CIP table and row numbers
    num_bins = len(cip_table)
    num_cips_per_bin = len(cip_table[0, :])
    flattened_table = np.ndarray.flatten(cip_table)
    flattened_group_nums = np.repeat(np.arange(num_bins), num_cips_per_bin)

    # Retrieve number of occurrences for each CIP group
    cip_counts = {}
    cip_proportions = {}
    names = list(data_dict.keys())
    for name in names:

        cip_counts[name] = np.zeros(num_bins + 1).astype(int)  # add one group for if the CIP code is not in table
        cip_arr = np.array(data_dict[name].loc[:, 'CIP1']).astype(str)
        unique_cip = np.unique(cip_arr)

        # Loop through all unique CIPs for this data set
        for cip in unique_cip:

            # Number of cadets with this CIP code
            cadets = np.where(cip_arr == cip)[0]
            count = len(cadets)

            index = np.where(flattened_table == cip)[0]  # location in flattened array

            # If this cip is in the table, we add the count to the bin count
            if len(index) != 0:
                bin_num = flattened_group_nums[index[0]]
                cip_counts[name][bin_num] += count

            else:  # If it is not, we add it to the "other" bin
                cip_counts[name][num_bins] += count

        sum_cips = sum(cip_counts[name])
        cip_proportions[name] = cip_counts[name] / sum_cips
    if averaged:
        cip_proportions2 = []
        for d_num in range(len(names) - 1):
            cip_proportions2.append(cip_proportions[names[d_num]])
        cip_proportions2 = np.array(cip_proportions2)
        cip_proportions2 = np.mean(cip_proportions2, axis=0)
        cip_proportions = {'CTGAN': cip_proportions2,
                           names[len(names) - 1]: cip_proportions[names[len(names) - 1]]}
    return cip_proportions


# Thesis: Appendix B (Data Generation) and Section 4.1.2
def generate_thesis_problem_instances(N_mean=1500, N_std=100, num_instances=10, printing=True):
    """
    This procedure generates the CTGAN data that will be used to conduct the thesis analysis for Appendix B
    and section 4.1.2
    :param N_mean: Mean number of cadets to generate
    :param N_std: standard deviation of that number
    :param num_instances: number of data sets to generate
    :param printing: whether the procedure should print something
    :return: dictionaries of data sets
    """
    if printing:
        print('Generating CTGAN data instances for thesis research...')

    # Import CIP to Qual Matrix
    cip_qual_matrix = import_data(paths['Data Processing Support'] + "Qual_CIP_Matrix_Generated.xlsx",
                                  sheet_name="Qual Matrix")

    # Import Generator Parameters
    cip_df = {}
    targets, cip_df["CIP1"], cip_df["CIP2"] = import_generator_parameters()

    # Load generator
    generator = CTGAN.load(paths['Data Processing Support'] + 'CTGAN.pkl')

    # Initialize data dictionaries
    data_dict = {}
    cadets_fixed_dict = {}
    afscs_fixed_dict = {}
    names = []

    # Generate data instances
    for i in range(num_instances):
        N = int(np.random.normal(N_mean, N_std))  # number of cadets to generate

        # Name of Dataset
        data_name = 'Generated_Instance_' + str(i + 1)
        names.append(data_name)

        if printing:
            print('')
            print('Generating ' + str(N) + ' cadets for ' + data_name)

        # Create data
        data_dict[data_name] = simulate_realistic_fixed_data(generator, N=N)
        data = copy.deepcopy(data_dict[data_name])
        data = ctgan_data_filter(data)
        cadets_fixed_dict[data_name], afscs_fixed_dict[data_name] = convert_realistic_data_parameters(
            data, targets, cip_qual_matrix, printing)

        # Export to excel
        with pd.ExcelWriter(paths['Problem Instances'] + data_name + '.xlsx') as writer:
            cadets_fixed_dict[data_name].to_excel(writer, sheet_name="Cadets Fixed", index=False)
            afscs_fixed_dict[data_name].to_excel(writer, sheet_name="AFSCs Fixed", index=False)

    return data_dict, cadets_fixed_dict, afscs_fixed_dict


def import_thesis_problem_instances(num_instances=10, printing=True):
    """
    This procedure imports the CTGAN data that will be used to conduct the thesis analysis for Appendix B and
    Section 4.1.2
    :param num_instances: number of CTGAN data instances to import
    :param printing: whether the procedure should print something
    :return: dictionaries of data sets
    """
    if printing:
        print('Importing thesis problem instances into dictionaries of data sets...')

    data_dict = {}
    cadets_fixed_dict = {}
    afscs_fixed_dict = {}

    for i in range(num_instances):
        data_name = 'Generated_Instance_' + str(i + 1)  # name of data set

        # Import Datasets
        cadets_fixed = import_data(paths['Problem Instances'] + data_name + '.xlsx', sheet_name='Cadets Fixed')
        afscs_fixed = import_data(paths['Problem Instances'] + data_name + '.xlsx', sheet_name='AFSCs Fixed')
        cadets_fixed = cadets_fixed.replace(np.nan, '', regex=True)

        # Convert cadets dataframe to same structure as CTGAN data
        data = copy.deepcopy(cadets_fixed)
        afscs = np.array(afscs_fixed.loc[:, 'AFSC'])
        drop_columns = ['qual_' + afsc for afsc in afscs]
        drop_columns.append('NrWgt1')
        data.drop(drop_columns, axis=1, inplace=True)

        # Load Data Dictionaries
        data_dict[data_name] = data
        cadets_fixed_dict[data_name] = cadets_fixed
        afscs_fixed_dict[data_name] = afscs_fixed

    return data_dict, cadets_fixed_dict, afscs_fixed_dict


def generated_data_comparison_analysis(N_mean=1500, N_std=100, num_instances=10, generate=True,
                                       ctgan_name='Full', data_dict=None, import_metrics=False,
                                       printing=True):
    """
    This procedure compares the generated CTGAN data with the real data. This procedure
    supports Appendix B: Data Generation of the thesis. Table is exported to excel and charts are saved
    :param import_metrics: optional metrics df if we already have the metrics
    :param data_dict: data dictionary
    :param ctgan_name: Which CTGAN to use
    :param N_mean: Mean number of cadets to generate
    :param N_std: standard deviation of that number
    :param num_instances: number of data sets to generate
    :param generate: Whether we are generating the data for the first time here, or importing from excel
    :param printing: whether the procedure should print something
    :return: None
    """
    if printing:
        print('Conducting Generated Data Comparison Analysis (Appendix B)...')
        print('')

    if data_dict is None:
        if generate:
            data_dict, cadets_fixed_dict, afscs_fixed_dict = generate_thesis_problem_instances(
                N_mean, N_std, num_instances, printing=printing)
        else:
            data_dict, cadets_fixed_dict, afscs_fixed_dict = import_thesis_problem_instances(
                num_instances, printing=printing)

    # Collect Data
    if ctgan_name == 'Full':
        baseline_data = ctgan_data_filter()
    else:
        data = import_data(paths['Data Processing Support'] + 'ctgan_data.xlsx', sheet_name=ctgan_name)
        baseline_data = ctgan_data_filter(data)

    if import_metrics:
        cip_hist_chart_averaged, average_utility_chart_averaged, metrics = compare_data(
            data_dict, both=False, baseline_data=baseline_data, metrics=4, averaged=True, printing=printing)
        metrics = import_data(paths['Analysis & Results'] + 'CTGAN_Data_Metrics.xlsx')
    else:
        cip_hist_chart_averaged, cip_hist_chart_full, average_utility_chart_averaged, \
        average_utility_chart_full, metrics = compare_data(data_dict, both=True, baseline_data=baseline_data,
                                                           printing=printing)

    # Save and Show Metrics
    if not import_metrics:
        metrics['CTGAN Averaged'] = metrics.loc[
                                    :, 'Generated_Instance_1':'Generated_Instance_' + str(num_instances)].mean(axis=1)
        metrics['CTGAN Averaged'] = metrics['CTGAN Averaged'].round(3)
        with pd.ExcelWriter(paths['Analysis & Results'] + 'Generated_Instance_Metrics.xlsx') as writer:
            metrics.to_excel(writer, sheet_name="Results", index=True)

    if printing:
        print(metrics)

    # Save and Show Charts
    cip_hist_chart_averaged.savefig(paths['Charts & Figures'] + 'CTGAN CIP Averaged Histogram.png', bbox_inches='tight')
    cip_hist_chart_averaged.show()
    average_utility_chart_averaged.savefig(paths['Charts & Figures'] + 'CTGAN Averaged Utility Chart.png',
                                           bbox_inches='tight')
    average_utility_chart_averaged.show()


def class_year_comparison_analysis(class_years=None, printing=True):
    """
    This procedure compares the class years against each other. This procedure
    supports Appendix B: Data Generation of the thesis. Table is exported to excel and charts are saved
    :param class_years: list of class years to use
    :param printing: whether the procedure should print something
    :return: None
    """
    if printing:
        print('Conducting Class Year Similarity Analysis (Appendix B)...')
        print('')

    if class_years is None:
        class_years = [2015, 2016, 2017, 2018, 2019]

    # Gather Data
    data_dict = {}
    for year in class_years:
        data_dict[year] = import_data('ctgan_data.xlsx', sheet_name=str(year))
    num_years = len(class_years)

    # Compare all data against all data
    evaluation_matrix = np.ones([num_years, num_years])
    for i in range(num_years - 1):

        year1 = class_years[i]
        for j in range(i + 1, num_years):

            year2 = class_years[j]
            eval_score = round(evaluate(data_dict[year1], data_dict[year2], aggregate=True,
                                        metrics=['KSTest', 'ContinuousKLDivergence', 'DiscreteKLDivergence']), 3)
            evaluation_matrix[i, j] = eval_score
            evaluation_matrix[j, i] = eval_score

    # Evaluation Matrix Df
    evaluation_df = pd.DataFrame({"Year": class_years})
    for j, year in enumerate(class_years):
        evaluation_df[year] = evaluation_matrix[:, j]

    # Export to excel
    with pd.ExcelWriter(paths['Analysis & Results'] + 'Class Year Evaluation Matrix.xlsx') as writer:
        evaluation_df.to_excel(writer, sheet_name="Results", index=False)

    # CIP proportions and chart
    cip_table, cip_table_df = baseline_cip_hist_data()
    cip_proportions_full = cip_hist_data(cip_table=cip_table, averaged=False, data_dict=data_dict,
                                         printing=printing)
    cip_hist_chart_full = CIP_Hist_Chart(cip_proportions_full)

    # Average Utility Chart
    average_utility_chart_full = Average_Utility_Data_Graph(data_dict=data_dict, years=True,
                                                            averaged=False)

    # Save and Show Charts
    average_utility_chart_full.savefig(paths['Charts & Figures'] + 'Class Year Full Utility Chart.png',
                                       bbox_inches='tight')
    average_utility_chart_full.show()
    cip_hist_chart_full.savefig(paths['Charts & Figures'] + 'Class Year CIP Full Histogram.png', bbox_inches='tight')
    cip_hist_chart_full.show()