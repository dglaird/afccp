# Import modules
import pandas as pd
import numpy as np
import copy
import datetime
import openpyxl
import os
from afccp.core.globals import *
from afccp.processing.support import *


class DataAggregator:
    def __init__(self, cy='2023', processing_path=None, printing=True, log_file=True):

        if printing:
            print('Loading in required dataframes for the class of ' + cy + '...')

        # Initialize attributes
        self.printing, self.cy, self.ic, self.current_time = printing, cy, None, None
        self.afscs, self.log_file = None, log_file

        # Specify path to cloud folder if not already specified (for databricks)
        if processing_path is None:
            if databricks:
                self.processing_path = '/dbfs/FileStore/shared_uploads/1523477583.LAIRD.DAN/NRL_Data/raw/'
            else:
                self.processing_path = dir_path + 'afccp/processing/'
        else:
            self.processing_path = processing_path

        # Initialize dictionary of dataframes
        df_list = ['ROTC', 'USAFA', 'AFSCs', 'Compiled', 'Cadets']
        self.dfs = {df_name: None for df_name in df_list}

        # Load in dataframes
        for df_name in df_list:
            try:
                filepath = self.processing_path + df_name + '_' + cy[2:] + '_Data.xlsx'
                self.dfs[df_name] = pd.read_excel(filepath, engine='openpyxl')

                if printing:
                    print('Successfully imported ' + df_name + ' dataframe for the class of ' + self.cy + '.')
            except:
                self.dfs[df_name] = None
                if printing:
                    print('Failed to import ' + df_name + ' dataframe for the class of ' + self.cy + '.')

        # Get array of AFSCs
        if self.dfs["AFSCs"] is None:
            raise ValueError('Error. We need data on the AFSCs in order to create a '
                             'problem instance file for the class of ' + self.cy)
        else:
            self.afscs = np.array(self.dfs["AFSCs"]['AFSC'])

        # Figure out if we have USAFA data or not
        if self.dfs["USAFA"] is not None:
            self.usafa_data = True
        else:
            self.usafa_data = False

    # These methods support the creation of the "All Cadet Info" dataframe
    def view_rotc_columns(self):
        """
        See unique values in the ROTC dataframe
        """

        r = self.dfs["ROTC"]  # Shorthand -> ROTC excel sheet
        for col in r:
            try:
                unique_vals = r[col].unique()
                if len(unique_vals) > 12:
                    print(col, "too many unique")
                else:
                    print(col, unique_vals)
            except:
                print(col, "Something went wrong with this column")

    def init_info_columns(self, printing=None):
        """
        Initialize the info columns (creates "self.ic" as a dictionary of lists that will be the cadet columns)
        """

        if printing is None:
            printing = self.printing

        # All cadet info columns (these are the columns that I want the data to look like with all cadets)
        info_columns = ['Cadet', 'ID', 'SSN', 'Last Name', 'First Name', 'Middle Initial', 'AFSC_Assigned', 'Gender',
                        'Race', 'Ethnicity', 'USAFA', 'University', 'Det Number',
                        'Det Ranking', 'GOM_All', "GOM_NR", 'ASC1', 'ASC2', 'CIP1', 'CIP2', 'Major1', 'Major2', 'GPA']

        # AFOQT Columns
        afoqt_types = ['Pilot', 'CSO', 'ABM', 'Navigator', 'Academic', 'Verbal', 'Quantitative']
        for afoqt_type in afoqt_types:
            info_columns.append("AFOQT Score [" + afoqt_type + "]")
        more_columns = ['Commission Date', 'Prior Service Months']

        # Preference Columns
        for i in range(1, 7):
            more_columns.append('NR_Pref_' + str(i))

        # Utility Columns
        for i in range(1, 7):
            more_columns.append('NR_Util_' + str(i))

        # Add columns to the info columns
        for col in more_columns:
            info_columns.append(col)

        # Create a dictionary of lists that will be the columns for the info dataframe
        self.ic = {col: [] for col in info_columns}

        # Final Columns of "Info" Sheet
        if printing:
            print('\nInfo Columns:', info_columns)

    def rotc_detachment_logs(self, lines, r_cols, printing=None):
        """
        This method determines what is wrong with the rotc detachments
        """
        if printing is None:
            printing = self.printing

        # ROTC Detachment rankings log
        dt_lines = ['_' * 20 + 'Class of ' + self.cy + ' Log file for ROTC detachment rankings' + '_' * 20 + '\n',
                    'Current time: ' + str(self.current_time) + '\n']

        # List of detachments
        dets = np.unique(r_cols['Det Number'])
        num_dets = len(dets)

        # Dictionary of cadet indices for each detachment
        det_cadets = {det: np.where(r_cols['Det Number'] == det)[0] for det in dets}

        # If there is no rank, change the rank to zero (temporarily)
        r_cols['Cadet Ranking'] = np.nan_to_num(r_cols['Cadet Ranking'], nan=0)
        no_det_ranking_indices = np.where(r_cols['Cadet Ranking'] == 0)[0]
        num_no_rank = len(no_det_ranking_indices)
        if num_no_rank > 0:
            if num_no_rank == 1:
                indices_str = "\nROTC Cadet Rankings. " \
                              "There is one cadet without a detachment ranking. " \
                              "The cadet is at index " + str(no_det_ranking_indices[0]) + "."
            else:
                indices_str = "\nROTC Cadet Rankings. There are " + str(
                    num_no_rank) + " cadets with no det ranking. They are at indices "
                for i, index in enumerate(no_det_ranking_indices):
                    if i == num_no_rank - 1 and i != 0:
                        indices_str += "and " + str(index) + "."
                    else:
                        indices_str += str(index) + ", "
        else:
            indices_str = "\nROTC Cadet Rankings. All cadets have detachment rankings."
        lines.append(indices_str + "\n")

        # Dictionary of detachment rankings of the cadets for each detachment
        det_rankings = {det: r_cols['Cadet Ranking'][det_cadets[det]] for det in dets}
        # det_cadet_counts = {det: 0 for det in dets}

        # Issues with the detachment rankings
        det_rank_cats = {'Normal': [], 'Ties': [], 'Only 1s': []}
        det_rank_cat_counts = {'Normal': 0, 'Ties': 0, 'Only 1s': 0}

        # See if we have a ranking issue with the detachments
        for det in dets:

            # Sort the rankings and get the required lists
            sort_indices = np.argsort(det_rankings[det])
            rankings = det_rankings[det][sort_indices]
            # cadets = det_cadets[det][sort_indices]

            # Add to the "detachment" log file to see all the rankings that the commanders are giving
            rank_str = str(rankings[0])
            for i, rank in enumerate(rankings):
                if i > 0:
                    rank_str += ", " + str(rank)
            dt_lines.append('Detachment: ' + str(det) + ', Rankings: ' + rank_str)

            # Figure out the patterns that are happening
            unique_ranks = np.unique(rankings)
            num_cadets = len(rankings)
            num_unique_ranks = len(unique_ranks)
            if num_unique_ranks == num_cadets:
                det_rank_cats['Normal'].append(det)
                det_rank_cat_counts['Normal'] += 1
            elif num_unique_ranks == 1:
                det_rank_cats['Only 1s'].append(det)
                det_rank_cat_counts['Only 1s'] += 1
            else:  # other kinds of ties
                det_rank_cats['Ties'].append(det)
                det_rank_cat_counts['Ties'] += 1
        new_str = "ROTC Detachments. There are " + str(num_dets) + " ROTC detachments (Det " + str(
            min(dets)) + " -> " + str(max(dets)) + ")."
        lines.append(new_str)
        new_str = "Of the " + str(num_dets) + " detachments, " + str(
            det_rank_cat_counts['Normal']) + " dets have normal rankings (1 to n lists with no ties), " + \
                  str(det_rank_cat_counts['Ties']) + " dets have given ties of some sort, and " + str(
            det_rank_cat_counts['Only 1s']) + " have ranked all of their cadets as #1."
        lines.append(new_str + '\n')

        return lines, dt_lines

    def add_rotc_cadets(self, r_cols, lines, printing=None):
        """
        Add ROTC cadets to the info dataframes
        """
        if printing is None:
            printing = self.printing

        # Fix SSN columns so we can sanity check data  (replace dashes)
        ssns = np.array([ssn.strip().replace('-', '') for ssn in r_cols['SSAN']])

        # Number of cadets
        n = len(r_cols['Last Name'])

        # Assigned AFSC stuff
        afsc_assigned_vals = np.unique(r_cols['Assigned AFSC'].astype(str))
        new_str = "\nAssigned Career Fields. The values in the 'Assigned AFSC' column are: "
        for val in afsc_assigned_vals:
            new_str += val + ", "

        if 'Rated Select' in r_cols:

            # Fix the column if necessary
            r_cols['Rated Select'] = r_cols['Rated Select'].astype(str)

            # Unique values
            rated_selects = np.unique(r_cols['Rated Select'])
            new_str += "the values in the 'Rated Select' column are: "
            for val in rated_selects:
                new_str += val + ", "

            # Indices of cadets with characteristics based on their rated column (main sheet)
            abm_indices = np.where(r_cols['Rated Select'] == 'ABM')[0]
            zero_indices = np.where(r_cols['Rated Select'] == '0')[0]
            rated_indices = np.hstack((abm_indices, zero_indices))

            # Little sloppy, but 13S is always matched after Rated so we don't need to check this down below
            if '13S Select' in r_cols:

                # Fix the column if necessary
                r_cols['13S Select'] = r_cols['13S Select'].astype(str)

                # Find subset of cadets that were not assigned 13S or Rated AFSCs
                nonrated_indices = np.where((r_cols['Rated Select'] == 'nan') & (r_cols['13S Select'] == 'nan'))[0]
            else:
                nonrated_indices = np.where((r_cols['Rated Select'] == 'nan'))[0]
        else:
            new_str += "there is no 'Rated Select' column, "
            rated_indices = np.array([])
            nonrated_indices = np.arange(n)

        if '13S Select' in r_cols:

            # Fix the column if necessary
            r_cols['13S Select'] = r_cols['13S Select'].astype(str)

            # Unique values
            sf_selects = np.unique(r_cols['13S Select'])
            new_str += "and the values in the '13S Select' column are: "
            for i, val in enumerate(sf_selects):
                if i < len(sf_selects) - 1:
                    new_str += val + ", "
            new_str += sf_selects[len(sf_selects) - 1] + "."

            # Indices of cadets with characteristics based on their 13S column (main sheet)
            sf_indices = np.where(r_cols["13S Select"] == '13S')[0]

        else:
            new_str += "and there is no '13S Select' column."
            sf_indices = np.array([])
        lines.append(new_str)

        # AFSC preferences (Initial clean (trimming off white spaces from AFSC names))
        pref_col_names = ['AFSC Preference ' + str(i + 1) + ' [Cadet]' for i in range(6)]
        pref_df = self.dfs["ROTC"][pref_col_names]
        for col in pref_df:
            pref_df[col].str.strip()
        original_preferences = np.array(pref_df)

        # AFSC Utilities (Need to clean them up before we convert to numpy arrays
        # because numpy is confused about strings and integers for some reason)
        util_col_names = ['CADET WEIGHT ' + str(i + 1) for i in range(6)]
        util_df = self.dfs["ROTC"][util_col_names]

        # Replace blanks with zeros
        for col in util_col_names:
            util_df.loc[util_df[col] == ' ', [col]] = 0
        original_utilities = np.array(util_df)

        # Clean up preferences and utilities further using the list of AFSCs
        cleaned_preferences, cleaned_utilities = clean_problem_instance_preferences_utilities(
            self.afscs, original_preferences, original_utilities, mapping=False)

        # Cadet counting logs
        num_rated, num_nonrated, num_sf = len(rated_indices), len(nonrated_indices), len(sf_indices)
        check_sum = num_rated + num_nonrated + num_sf
        lines.append("\nROTC Cadets. There are " + str(n) + " ROTC cadets in the main sheet.")
        lines.append("Of those ROTC cadets, " + str(num_rated) + ' are rated selects, ' + str(
            num_nonrated) + ' are non-rated, and ' + str(num_sf) + ' are space-force selects.')
        lines.append(str(num_rated) + " + " + str(num_nonrated) + " + " + str(num_sf) + " = " + str(check_sum))

        # Loop through all the ROTC cadets in the "main" sheet
        # (Taking it one column at a time so I can really make sure everything is clean)
        no_degree_count = 0
        valid_second_degree_count = 0
        invalid_second_degree_count = 0
        invalid_second_degree_indices = []
        for i in range(n):  # Loop through all ROTC cadets from "main sheet"

            # Cadet Identifiers
            ssn = ssns[i]
            self.ic['SSN'].append(ssn)
            self.ic['Cadet'].append(i)
            id = r_cols['EmployeeID'][i]
            self.ic['ID'].append(id)

            # Same Column Names
            for col in ['Last Name', 'First Name', 'Middle Initial', 'Gender', 'Race', 'Ethnicity']:
                str_val = str(r_cols[col][i])

                # I want to map race and ethnicity to something more understandable than just the letters soon
                if col in ['Gender', 'Race', 'Ethnicity']:
                    self.ic[col].append(str_val.strip())
                else:
                    self.ic[col].append(str_val)

            # ROTC cadets, not USAFA
            self.ic["USAFA"].append(0)

            # Standardize university stuff  (future work could actually clean up university names too)
            university = r_cols['School Name'][i]
            if university in ['', ' ', '   ', np.nan]:
                university = ''
            self.ic["University"].append(university)

            # ROTC detachments
            self.ic["Det Number"].append(r_cols['Det Number'][i])
            self.ic["Det Ranking"].append(r_cols['Cadet Ranking'][i])

            # Order of Merit
            self.ic["GOM_All"].append(r_cols["GOM"][i])
            self.ic["GOM_NR"].append('')

            # Degrees
            if self.cy == '2022':

                # First Degree
                asc1 = str(r_cols['Degree Code'][i]).strip()
                self.ic['ASC1'].append(asc1)
                if len(asc1) != 4 or asc1[0] in range(10):  # ensure there is a valid ASC code
                    no_degree_count += 1

                # Second Degree
                asc2 = str(r_cols['Second Degree Code'][i]).strip()
                if asc2 in ['None', '', ' ', 'N/A', 'NA', 'NONE', '0']:
                    pass  # No second degree
                    self.ic['ASC2'].append('')
                else:
                    if len(asc2) == 4 and asc2[0] in range(
                            10):  # check for valid entries (ASC codes start with a number)
                        invalid_second_degree_count += 1
                        invalid_second_degree_indices.append(i)
                        self.ic['ASC2'].append('Inv')
                    else:
                        valid_second_degree_count += 1
                        self.ic['ASC2'].append(asc2)

                # Don't know CIPs yet
                self.ic['CIP1'].append(''), self.ic['CIP2'].append('')

                # Don't think 22 had a degree title column...
                self.ic['Major1'].append(''), self.ic['Major2'].append('')

            elif self.cy == '2023':

                # Don't know ASCs (and we don't care)
                self.ic['ASC1'].append(''), self.ic['ASC2'].append('')

                # Nice and easy for degree titles
                self.ic['Major1'].append(r_cols['Degree Title'][i])
                self.ic['Major2'].append(r_cols['Second Degree Title'][i])

                # Get CIPs (This step is very important! These columns determine qualifications. Must be done with care)
                cip1 = str(r_cols['Degree Code'][i])
                if '.' in cip1:
                    split_list = cip1.split('.')
                    w1, w2 = split_list[0], split_list[1]

                    # Fancy formula to get the CIP in a format I can use
                    # (add * to keep all the numbers aligned as a string format rather than numeric)
                    cip1 = "*" + "0" * (2 - len(w1)) + w1 + w2 + "0" * (4 - len(w2))
                else:
                    cip1 = "Unk"  # We know they have a degree, but we don't know what it is
                    no_degree_count += 1

                # Second CIP
                cip2 = str(r_cols['Second Degree Code'][i])
                if '.' in cip2:
                    split_list = cip2.split('.')
                    w1, w2 = split_list[0], split_list[1]

                    # Fancy formula to get the CIP in a format I can use
                    # (add * to keep all the numbers aligned as a string format rather than numeric)
                    cip2 = "*" + "0" * (2 - len(w1)) + w1 + w2 + "0" * (4 - len(w2))
                    valid_second_degree_count += 1
                else:
                    cip2 = "None"  # We assume they don't have a second degree

                # Add CIPs
                self.ic['CIP1'].append(cip1), self.ic['CIP2'].append(cip2)

            # AFOQT
            afoqt_types = ['Pilot', 'CSO', 'ABM', 'Navigator', 'Academic', 'Verbal', 'Quantitative']
            for afoqt_type in afoqt_types:
                col = "AFOQT Score [" + afoqt_type + "]"
                if col in r_cols:
                    self.ic[col].append(r_cols[col][i])
                else:
                    self.ic[col].append("")

            # GPA
            gpa = r_cols['GPA'][i]
            try:
                gpa = float(gpa)
            except:
                gpa = 1.9

            self.ic['GPA'].append(gpa)

            # Commission Date
            yr, mo, day = r_cols['Year of Commission'][i], r_cols['Month of Commission'][i], \
                          r_cols['Day of Commission'][i]
            dt = str(mo) + "/" + str(day) + "/" + str(yr)
            self.ic['Commission Date'].append(dt)

            # Months of prior service
            try:
                pr_months = r_cols['Number of Prior Service Months'][i] + r_cols['Number of Prior Service Years'][
                    i] * 12
            except:
                pr_months = r_cols['Number of Prior Service Months'][i] + int(
                    r_cols['Number of Prior Service Years'][i].replace('+', '')) * 12  # some have "10+" years
            self.ic['Prior Service Months'].append(pr_months)

            # Assigned AFSC
            if i in rated_indices:
                self.ic['AFSC_Assigned'].append('Rated')  # Don't know more specifics on
            elif i in sf_indices:
                self.ic['AFSC_Assigned'].append('13S')
            else:
                self.ic['AFSC_Assigned'].append('')  # Non-Rated! (We're matching these ones)

            # Preferences and Utilities
            for p in range(6):
                self.ic['NR_Pref_' + str(p + 1)].append(cleaned_preferences[i, p])
                self.ic['NR_Util_' + str(p + 1)].append(cleaned_utilities[i, p])

        lines.append(str(no_degree_count) + " ROTC cadets have no degree identified. " + str(
            valid_second_degree_count) + " ROTC cadets have a second degree identified.")
        r_n, r_num_rated, r_num_nonrated, r_num_sf = n, num_rated, num_nonrated, num_sf  # Save ROTC variables
        return lines, r_n  # We need these variables

    def add_usafa_cadets(self, r_n, lines, printing=None):
        """
        Add USAFA cadets to the info dataframes
        """
        if printing is None:
            printing = self.printing

        # Dictionary of numpy arrays of the data in the USAFA excel sheet
        u_cols = {col: np.array(self.u_df[col]) for col in self.u_df.columns}

        # USAFA SSNs
        ssns = u_cols['SSN'].astype(str)

        # Assigned AFSCs
        assigned_afscs = u_cols['Awarded_AFSC'].astype(str)
        afsc_assigned_vals = np.unique(assigned_afscs)

        # Map the AFSCs in the raw data to the cleaned standardized versions
        mapping_dict = {'13S1': '13S', '14N1': '14N', '17D1': '17X', '62E1A': '62EXA', '62E1B': '62EXB',
                        '62E1C': '62EXC', '62E1E': '62EXE', '62E1G': '62EXG', '62E1H': '62EXH', '62E1I': '62EXI',
                        '63A1': '63A', '92T0': '92T0', '92T1': '92T1', '92T2': '92T2', '92T3': '92T3', 'nan': ''}
        for afsc in afsc_assigned_vals:
            if afsc in mapping_dict:
                indices = np.where(assigned_afscs == afsc)[0]
                np.put(assigned_afscs, indices, mapping_dict[afsc])
            else:
                if printing:
                    print('AFSC', afsc, 'not in mapping_dict')

        # Assigned AFSC indices
        sf_indices = np.where(assigned_afscs == '13S')[0]
        rated_indices = assigned_afscs[
            (assigned_afscs == '92T0') + (assigned_afscs == '92T1') + (assigned_afscs == '92T2') + (
                    assigned_afscs == '92T3')]

        # AFSC preferences (Initial clean (trimming off white spaces from AFSC names))
        pref_col_names = ['NR' + str(p + 1) + '_CODE' for p in range(6)]
        pref_df = self.u_df[pref_col_names]
        for col in pref_df:
            pref_df[col].str.strip()
        original_preferences = np.array(pref_df)

        # AFSC Utilities (Need to clean them up before we convert to numpy arrays
        # because numpy is confused about strings and integers for some reason)
        util_col_names = ['NR' + str(p + 1) + '_WEIGHT' for p in range(6)]
        util_df = self.u_df[util_col_names]

        # Replace blanks with zeros
        for col in util_col_names:
            util_df.loc[util_df[col] == ' ', [col]] = 0
        original_utilities = np.array(util_df)

        # Clean up preferences and utilities further using the list of AFSCs
        cleaned_preferences, cleaned_utilities = clean_problem_instance_preferences_utilities(self.afscs,
                                                                                              original_preferences,
                                                                                              original_utilities)

        # USAFA majors
        majors_1, majors_2 = u_cols['Major'], u_cols['Sec_Major']

        # This is where I'm going to map majors to CIP codes (if it's necessary)
        mapping_dict = {'AeroEngr': '', 'AstroEngr': '', 'BehSci': '', 'BioChem': '', 'Biology': '',
                        'Chemistry': '', 'CivEngr': '',
                        'CompSci': '', 'CyberSci': '', 'Economics': '', 'ElCompEngr': '', 'English': '',
                        'EngrChem': '', 'FAS-Hist': '',
                        'FAS-MSS': '', 'FAS-PolSci': '', 'GenEngr': '', 'GeoSci': '', 'History': '',
                        'HistoryAme': '', 'HistoryInt': '',
                        'HistoryMil': '', 'Humanities': '', 'LegalStu': '', 'MSS': '', 'Management': '', 'Math': '',
                        'MathApp': '',
                        'MechEngr': '', 'Meteor': '', 'OpsRsch': '', 'Philosophy': '', 'Physics': '', 'PolSci': '',
                        'SocSci': '',
                        'SpaceOps': '', 'SysEngr': '', }
        #     print('\nMajor 1', np.unique(majors_1))
        #     print('\nMajor 2', np.unique(majors_2))

        # Cadet counting logs
        n = len(self.dfs["USAFA"])

        # USAFA numbers will always add up to N (I can trust this one)
        num_rated, num_nonrated, num_sf = len(rated_indices), n - len(sf_indices) - len(rated_indices), len(
            sf_indices)
        lines.append("\nUSAFA Cadets. There are " + str(n) + " USAFA cadets.")
        lines.append("Of those USAFA cadets, " + str(num_rated) + ' are rated selects, ' + str(
            num_nonrated) + ' are non-rated, and ' + str(num_sf) + ' are space-force selects.')
        check_sum = num_rated + num_nonrated + num_sf
        lines.append(str(num_rated) + " + " + str(num_nonrated) + " + " + str(num_sf) + " = " + str(check_sum))

        # Figure out which AFSCs are already assigned and how many of them have been given
        m_intersect = np.intersect1d(self.afscs, np.unique(assigned_afscs))
        totals = 0
        if len(m_intersect) == 0:
            new_str = "There are no USAFA cadets assigned to any non-rated AFSC yet."
        elif len(m_intersect) == 1:
            afsc = m_intersect[0]
            cadets = np.where(assigned_afscs == afsc)[0]
            num_cadets = len(cadets)
            new_str = "There are " + str(num_cadets) + " cadets already assigned to " + afsc + "."
            totals = num_cadets
        else:
            new_str = "There are "
            for j, afsc in enumerate(m_intersect):
                cadets = np.where(assigned_afscs == afsc)[0]
                num_cadets = len(cadets)
                totals += num_cadets
                if j != len(m_intersect) - 1:
                    new_str += str(num_cadets) + " cadets already assigned to " + afsc + ", "
                else:
                    new_str += "and " + str(num_cadets) + " cadets already assigned to " + afsc + "."
        lines.append(new_str)
        lines.append('In total, ' + str(
            totals) + ' USAFA cadets have been assigned to Non-Rated AFSCs outside of our model. There are then ' + str(
            num_nonrated - totals) + ' USAFA cadets that will be matched in the NRL optimization model.')

        # Loop through all USAFA cadets
        for i in range(n):

            # Cadet Identifiers
            ssn = ssns[i]
            id = u_cols['PID'][i]
            self.ic['SSN'].append(ssn), self.ic['Cadet'].append(r_n + i), self.ic['ID'].append(id)

            # Same Column Names
            for col in ['First Name', 'Gender', 'Race']:
                self.ic[col].append(u_cols[col][i])

            # Others
            self.ic['Middle Initial'].append(''), self.ic['Ethnicity'].append(''), self.ic['Last Name'].append(
                u_cols['Last_Name'][i])

            # Graduating order of merit
            self.ic['GOM_All'].append(u_cols['GOM'][i]), self.ic['GOM_NR'].append("")

            # USAFA, not ROTC
            self.ic['USAFA'].append(1), self.ic['University'].append('USAFA'), self.ic['Det Number'].append(0)
            self.ic['Det Ranking'].append(u_cols['GOM'][i])

            # Don't know what the data is going to look like for 2023 yet!
            # (As far as I can tell, 22 didn't have ASC or CIP codes)
            self.ic['Major1'].append(majors_1[i]), self.ic['Major2'].append(majors_2[i])

            # Don't know CIPs yet
            self.ic['CIP1'].append(''), self.ic['CIP2'].append('')

            # Don't know ASCs (and we don't care)
            self.ic['ASC1'].append(''), self.ic['ASC2'].append('')

            # AFOQT
            afoqt_types = ['Pilot', 'CSO', 'ABM', 'Navigator', 'Academic', 'Verbal', 'Quantitative']
            for afoqt_type in afoqt_types:
                col = "AFOQT Score [" + afoqt_type + "]"
                if "AFOQT_" + afoqt_type in u_cols:
                    self.ic[col].append(u_cols["AFOQT_" + afoqt_type][i])
                else:
                    self.ic[col].append("")

            # GPA
            gpa = u_cols['GPA'][i]
            try:
                gpa = float(gpa)
            except:
                gpa = 1.9
            self.ic['GPA'].append(gpa)

            # Commission Date
            self.ic['Commission Date'].append("5/18/2022")  # 2022 Graduation Date

            # Don't have prior enlisted months yet
            self.ic['Prior Service Months'].append('')

            # Assigned AFSC
            self.ic['AFSC_Assigned'].append(assigned_afscs[i])

            # Preferences and Utilities
            for p in range(6):
                self.ic['NR_Pref_' + str(p + 1)].append(cleaned_preferences[i, p])
                self.ic['NR_Util_' + str(p + 1)].append(cleaned_utilities[i, p])

        return lines

    # This is the main method to create the "All Cadet Info" dataframe (pulls from the methods above)
    def create_info_df(self, view_columns=False, log_file=None, view_det_log=False,
                       export_df=False, printing=None):
        """
        This is the main class method that aggregates the ROTC and USAFA data into one dataframe for all cadets.
        This generates the "All Cadet Info" dataframe.
        """

        if printing is None:
            printing = self.printing

        if log_file is None:
            log_file = self.log_file

        # Main Log file
        self.current_time = datetime.datetime.now()
        lines = ['_' * 20 + 'Class of ' + self.cy +
                 ' Log file for pre-processing cadet data from ROTC (Form 53) and from USAFA' + '_' * 20 + '\n',
                 'Current time: ' + str(self.current_time)]

        # Dictionary of numpy arrays of the data in the "main" ROTC excel sheet
        r_cols = {col: np.array(self.dfs["ROTC"][col]) for col in self.dfs["ROTC"].columns}

        # Structure numpy arrays if necessary
        r_cols['Ethnicity'] = r_cols['Ethnicity'].astype(str)

        if view_columns:
            self.view_rotc_columns()

        # Initialize Info Dataframe columns
        self.init_info_columns(printing=printing)

        # Determine which ROTC cadets have incorrect detachments
        lines, dt_lines = self.rotc_detachment_logs(lines, r_cols, printing=printing)

        # Add the cadets from ROTC and USAFA to the info columns
        lines, r_n = self.add_rotc_cadets(r_cols, lines, printing=printing)
        lines = self.add_usafa_cadets(r_n)

        self.dfs["Compiled"] = pd.DataFrame(self.ic)
        if printing:
            print('\nFinal Compiled Dataframe:')
            print(self.dfs["Compiled"])

        # Determine folder to export to
        if databricks:
            export_path = ''
        else:
            export_path = self.processing_path

        # Log File
        if log_file:

            # Write the "lines" to a text file
            with open(export_path + self.cy + '_aggregate_data_log.txt', 'w') as f:
                for line in lines:
                    f.write(line)
                    f.write('\n')

        # Export compiled dataframe
        if export_df:

            # Write "All Cadet Info" dataframe to excel
            with pd.ExcelWriter(export_path + 'Compiled_' + self.cy[2:] + '_Data.xlsx') as writer:
                self.dfs["Compiled"].to_excel(writer, sheet_name="All Cadet Info", index=False)

        # Print the log to console
        if printing:

            # Main Log
            print('\n\n')
            for line in lines:
                print(line)

        # ROTC detachment log
        if view_det_log:

            # Detachment log
            print('\n\n')
            for line in dt_lines:
                print(line)

        if printing:
            print('\nLoaded.')

    # This method creates the "Cadets Fixed" dataframe which contains the actual parameters to the model
    def create_cadets_fixed_df(self, use_all=False):
        """
        This method will use the "All Cadet Info" dataframe in conjunction with some supporting material and
        create the cadets fixed dataframe used in the instance file. The "use_all" parameter controls which
        cadets go into the NRL model. DSYA only matches non-rated non-Space Force cadets at the moment, but
        it may be a fun experiment to see what kind of solutions we could get to if we were able to match
        everybody.
        """

        # Shorthand
        i_df = self.dfs['Compiled']
        a_df = self.dfs['AFSCs']

    # This does everything! Theoretically I could just execute this method and have an instance file ready to go
    def compile_problem_instance_file(self, printing=None):
        """
        This is the main method that does everything. Aggregate data, clean it, and then turn it
        into a problem instance file that can be loaded into the main "CadetCareerProblem" Class
        :return:
        """

        if printing is None:
            printing = self.printing

        if printing:
            print('Compiling problem instance file for the class of ' + self.cy)

        if self.dfs["Compiled"] is None:
            self.create_info_df()

        self.create_cadets_fixed_df()
