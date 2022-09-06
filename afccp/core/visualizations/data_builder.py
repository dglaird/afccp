import pandas as pd
import numpy as np
from collections import OrderedDict


# test variables

# cadets_fixed_df = pd.read_excel(r'C:\Users\lucas\downloads\Real 2023.xlsx', sheet_name='Cadets Fixed')
# solutions_df = pd.read_excel(r'C:\Users\lucas\downloads\Real 2023.xlsx', sheet_name='Solutions')
# afsc_fixed_df = pd.read_excel(r'C:\Users\lucas\downloads\Real 2023.xlsx', sheet_name='AFSCs Fixed')
# combined_df = pd.merge(cadets_fixed_df, solutions_df, on='Cadet')
# solution_name = 'Solution B'


def main(cadets_fixed_df, solutions_df, afsc_fixed_df, combined_df, solution_name):
    """Main function to create the final dataset to be used with our plotly dashboard.

    Parameters:
        cadets_fixed_df : pandas dataframe
            Cadet information file.
        solutions_df : pandas dataframe
            Solutions information with what cadets were assigned to specific AFSCs
        afsc_fixed_df : pandas dataframe
            Information regarding an AFSCs requirements.
        combined_df : pandas dataframe
            merged dataframe of cadets_fixed_df and solutions_df. merged on cadet ids
        solution_name : str
            name of solution to be used to generate data from. Exact format from solutions_df

    Returns:
        pandas dataframe:
            Final information to be used to gerenate graphics from.

    """
    global overall_df

    # Defines the AFSC Matches dictionary. This creates the dictionary using all AFSCs by
    # assigning the keys to the dictionary first and then appending cadets to it.
    afsc_matches = {}
    afsc_list = []

    for a in afsc_fixed_df['AFSC']:  # Creating AFSC list to make as keys to the dictionary
        afsc_list.append(a)

    for a in afsc_list:
        afsc_matches[a] = []  # Creating the dictionary keys as the lists of AFSCs.

    # Building the dictionary for AFSC_matches
    for c in range(len(solutions_df['Cadet'])):
        if solutions_df.loc[:, solution_name][c] in afsc_matches.keys():
            afsc_matches[solutions_df.loc[:, solution_name][c]].append(solutions_df.loc[:, 'Cadet'][c])

    cadet_index_map = {i: x for x, i in enumerate(cadets_fixed_df.loc[:, 'Cadet'])}

    overall_df = pd.DataFrame()

    # USAFA number per AFSC

    def usafa_cadets_per_afsc_dictionary():
        """Creates a dictionary of AFSCs with the number of USAFA cadets assigned
        to that AFSC as the value

        Returns:
            None. Directly appends to the overall_df

        """
        usafa_cadets_per_afsc = {}

        for a in afsc_matches:
            usafa_count = 0

            for c in afsc_matches[a]:
                c = cadet_index_map[c]
                if cadets_fixed_df.loc[:, 'USAFA'][c] == 1:
                    usafa_count += 1
            usafa_cadets_per_afsc[a] = usafa_count

        overall_df["Number of USAFA Cadets"] = usafa_cadets_per_afsc.values()

    # Function for Overall ratio of ROTC Cadets
    def percent_rotc_cadets_per_afsc():
        """Calculates the percent of ROTC cadets assigned to an AFSC and appends
        the results to the overall_df.

        Returns:
            None. Directly appends results to the overall_df

        """
        rotc_cadets_per_afsc = {}

        # for every afsc get the percent of rotc cadets
        for a in afsc_matches:
            rotc_count = 0

            for c in afsc_matches[a]:
                c = cadet_index_map[c]
                if cadets_fixed_df.loc[:, 'USAFA'][c] == 0:
                    rotc_count += 1

            # handle the assignment of zero cadets
            try:
                rotc_cadets_per_afsc[a] = 100 * rotc_count / len(afsc_matches[a])
            except:
                rotc_cadets_per_afsc[a] = 0

        overall_df["Percent of ROTC Cadets in AFSC"] = rotc_cadets_per_afsc.values()

    # USAFA percentage per AFSC
    def percent_usafa_cadets_per_afsc():
        """Calculates the percent of USAFA cadets assigned to an AFSC and appends
        the results to the overall_df.

        Returns:
            None. Directly appends results to the overall_df.

        """
        usafa_percent_per_afsc = {}

        for a in afsc_matches:
            usafa_count = 0

            for c in afsc_matches[a]:
                c = cadet_index_map[c]
                if cadets_fixed_df.loc[:, 'USAFA'][c] == 1:
                    usafa_count += 1

            # handle the assignment of zero cadets
            try:
                usafa_percent_per_afsc[a] = 100 * usafa_count / len(afsc_matches[a])
            except:
                usafa_percent_per_afsc[a] = 0

        overall_df["Percent of USAFA Cadets in AFSC"] = usafa_percent_per_afsc.values()

    def cadet_merit_per_afsc():
        """Calculates the average merit across cadets assigned to an AFSC, for
        each AFSC. Then, directly appends results to the overall_df

        Returns:
            None. Directly appends results to overall_df.

        """
        afsc_avg_merit = {}
        list_of_cadet_merit_in_matched_afsc = []
        total_num_cadets = 0

        for a in afsc_matches:
            for c in range(len(afsc_matches[a])):
                total_num_cadets += 1
                list_of_cadet_merit_in_matched_afsc.append(cadets_fixed_df.loc[:, 'percentile'][c])
            afsc_summed_merit = sum(list_of_cadet_merit_in_matched_afsc)
            try:
                afsc_avg_cadet_merit = afsc_summed_merit / total_num_cadets
            except:
                afsc_avg_cadet_merit = 0
            afsc_avg_merit[a] = afsc_avg_cadet_merit

        overall_df["Cadet Merit Per AFSC"] = afsc_avg_merit.values()

    # This is omitting AFSC 61C, why?
    def count_of_nonvol_people_afsc(df1, df2, pref=6, df1_solution_set=solution_name):
        """Returns the number of people who have not matched with an AFSC they
        had selectioned, per AFSC.

        Args:
          df1 (pandas data frame):
              the solutions data frame
          df2 (pandas data frame):
              the Cadets Fixed data frame or the combined dataframe
          pref (int):
              an integer of either 1, 3, or 6. This is the number of preferences
              that are being considered
          df1_solution (str):
              this is the solution set to compare to the preferences of the cadets.
              Can be any column header in the solution dataframe but not Cadets

        Returns:
          None.

        """
        # dict1 is structured like this... 'AFSC': ['Number of Non-Vol Cadets', 'Total Number Cadets Assigned']
        dict1 = {}

        # Check to make sure the string entered is a solution set
        if df1_solution_set in list(df1.columns)[1:]:

            # for the next 4 if statements, this picks the number of preferences to consider
            if pref == 6:
                x = df2.loc[:, ['NR_Pref_1', 'NR_Pref_2', 'NR_Pref_3', 'NR_Pref_4', 'NR_Pref_5', 'NR_Pref_6']].eq(
                    df1.loc[:, df1_solution_set], axis=0).any(axis='columns')

            if pref == 3:
                x = df2.loc[:, ['NR_Pref_1', 'NR_Pref_2', 'NR_Pref_3']].eq(
                    df1.loc[:, df1_solution_set], axis=0).any(axis='columns')

            if pref == 1:
                x = df2.loc[:, ['NR_Pref_1']].eq(
                    df1.loc[:, df1_solution_set], axis=0).any(axis='columns')

            if pref != 6:
                if pref != 3:
                    if pref != 1:
                        x = df2.loc[:,
                            ['NR_Pref_1', 'NR_Pref_2', 'NR_Pref_3', 'NR_Pref_4', 'NR_Pref_5', 'NR_Pref_6']].eq(
                            df1.loc[:, df1_solution_set], axis=0).any(axis='columns')

            # makes the dictionary of AFSCs and the relevant infromation
            for i in range(len(df1.loc[:, df1_solution_set])):

                if df1.loc[:, df1_solution_set][i] in dict1.keys():
                    if x[i] == True:
                        dict1[df1.loc[:, df1_solution_set][i]] = [dict1[df1.loc[:, df1_solution_set][i]][0],
                                                                  dict1[df1.loc[:, df1_solution_set][i]][1] + 1]
                    else:
                        dict1[df1.loc[:, df1_solution_set][i]] = [dict1[df1.loc[:, df1_solution_set][i]][0] + 1,
                                                                  dict1[df1.loc[:, df1_solution_set][i]][1] + 1]
                else:
                    if x[i] == True:
                        dict1[df1.loc[:, df1_solution_set][i]] = [0, 1]
                    else:
                        dict1[df1.loc[:, df1_solution_set][i]] = [1, 1]

        dict1 = OrderedDict((sorted(dict1.items())))
        return dict1

    def percent_max_afsc(df1=solutions_df, df2=combined_df, df3=afsc_fixed_df, df1_solution_set=solution_name):
        """Returns the percent of the max allowed people per AFSC.

        Args:
          df1 (data frame):
              the solutions data frame
          df2 (data frame):
              the Cadets Fixed data frame or the combined dataframe
          df3 (data frame):
              the AFSC Fixed data frame
          df1_solution (str):
              This is the solution set to compare to the prefrences of the cadets.
              Can be any column header inthe solution dataframe but not Cadets

        Returns:
          None. total number of happy people

        """
        # dict2 is organized like this... 'AFSC': ['Total', 'Max', 'Min', 'Percent to Max',
        dict1 = count_of_nonvol_people_afsc(df1, df2, df1_solution_set)
        dict2 = {}

        for i in range(len(df3.loc[:, 'AFSC'])):
            if df3.loc[:, 'AFSC'][i] in dict1.keys():
                dict2[df3.loc[:, 'AFSC'][i]] = [dict1[df3.loc[:, 'AFSC'][i]][1],
                                                df3.loc[:, 'Max'][i],
                                                df3.loc[:, 'Min'][i],
                                                (dict1[df3.loc[:, 'AFSC'][i]][1]) / (df3.loc[:, 'Max'][i])]
            else:
                dict2[df3.loc[:, 'AFSC'][i]] = [0, df3.loc[:, 'Max'][i], df3.loc[:, 'Min'][i], 0]

        dict2
        dict2 = OrderedDict(sorted(dict2.items()))
        total_cadets_list = [i[0] for i in dict2.values()]
        afsc_max_capacities = [i[1] for i in dict2.values()]
        afsc_min_quotas = [i[2] for i in dict2.values()]
        afsc_percent_to_max_capacity = [i[3] for i in dict2.values()]
        overall_df["AFSCs"] = dict2.keys()
        overall_df["Total Cadets Assigned"] = total_cadets_list
        overall_df["AFSC Max Capacities"] = afsc_max_capacities
        overall_df["AFSC Min Quotas"] = afsc_min_quotas
        overall_df["Percent of AFSC to Max Capacity"] = afsc_percent_to_max_capacity

    numbers = list(range(0, len(combined_df.columns)))  # df_cadets is the combined solutions and the cadets_fixed sheet
    convert_header_to_indices = dict(zip(combined_df.columns, numbers))

    test_df = np.array(combined_df)

    def nump_nonvol_degree_counts(solution, overall_df):
        """Calculates the number of cadets that were nonvold per AFSC per degree requirement

        Args:
        solution (str):
            solution name in the combined solution/cadets_fixed dataframe
        overall_df (pandas dataframe):
            the overall_df dataframe created within this file

        Returns:
        2-d numpy array:
            row entries are [AFSC,M,D,P,I]

        """

        # initiate numpy array
        nonvol_bydegree = np.zeros([1, 5])

        # calculate the nonvolunteered status
        nonvol = combined_df[['NR_Pref_1', 'NR_Pref_2', 'NR_Pref_3', 'NR_Pref_4', 'NR_Pref_5', 'NR_Pref_6']].eq(
            combined_df[solution], axis=0).any(axis=1)

        # get numpy index of header
        solution = convert_header_to_indices[solution]

        # calc the number of people nonvoluntired by degree requirement by afsc
        for afsc in afsc_fixed_df['AFSC']:
            nonvol_degree_array = np.array([[afsc,

                                             sum([1 if i[convert_header_to_indices['qual_' + afsc]] == 'M'
                                                 and i[solution] == afsc
                                                 and j == False
                                                  else 0
                                                  for i, j in zip(test_df, nonvol)]
                                                 ),

                                             sum([1 if i[convert_header_to_indices['qual_' + afsc]] == 'D'
                                                 and i[solution] == afsc
                                                 and j == False
                                                  else 0
                                                  for i, j in zip(test_df, nonvol)]
                                                 ),

                                             sum([1 if i[convert_header_to_indices['qual_' + afsc]] == 'P'
                                                  and i[solution] == afsc
                                                  and j == False
                                                  else 0
                                                  for i, j in zip(test_df, nonvol)]
                                                 ),

                                             sum([1 if i[convert_header_to_indices['qual_' + afsc]] == 'I'
                                                  and i[solution] == afsc
                                                  and j == False
                                                  else 0
                                                  for i, j in zip(test_df, nonvol)]
                                                 )
                                             ]])

            # add row to array
            nonvol_bydegree = np.concatenate((nonvol_bydegree, nonvol_degree_array), axis=0)

        (nonvol_bydegree[1:])

        degree_df = pd.DataFrame(nonvol_bydegree[1:],
                                 columns=['AFSCs',
                                          'Mandatory Non-Vol',
                                          'Desired Non-Vol',
                                          'Permitted Non-Vol',
                                          'Ineligible']
                                 )
        degree_df[['Mandatory Non-Vol', 'Desired Non-Vol', 'Permitted Non-Vol', 'Ineligible']
        ] = degree_df[['Mandatory Non-Vol', 'Desired Non-Vol', 'Permitted Non-Vol',
                       'Ineligible']].astype('int')

        overall_df = pd.merge(overall_df, degree_df, on='AFSCs')

        return overall_df

    def overall_df_function_build():  # func1, func2, func3, func4, func5, func6
        """Builds the final dataframe from all earlier fucntions.

        Args:
            None.

        Returns:
        (pandas dataframe):
            Contains all the infomration to create the graphics for the plotly dashboard.
        """
        global overall_df
        overall_df = pd.DataFrame()

        percent_max_afsc()
        count_of_nonvol_people_afsc(solutions_df, combined_df)
        cadet_merit_per_afsc()
        usafa_cadets_per_afsc_dictionary()
        percent_rotc_cadets_per_afsc()
        percent_usafa_cadets_per_afsc()
        overall_df = nump_nonvol_degree_counts(solution_name, overall_df)
        overall_df['Voluntary'] = (overall_df['Total Cadets Assigned'] - overall_df['Mandatory Non-Vol'] - overall_df[
            'Desired Non-Vol']
                                   - overall_df['Permitted Non-Vol'] - overall_df[
                                       'Ineligible'])  # Built in the number of volunatary cadets within this function .

        return overall_df

    return overall_df_function_build()


if __name__ == '__main__':
    main(cadets_fixed_df, solutions_df, afsc_fixed_df, combined_df, solution_name)