import numpy as np
import copy
from afccp.core.globals import *


def generate_cip_to_qual_matrix(printing=True, year=2016):
    """
    This procedure takes all the CIP codes from the ASC_CIP excel sheet and then creates a matrix of AFSCs that the
    degrees qualify for. This matrix is for AFSC qualifications as designated by the AFOCD in 2021
    :return: None
    """
    if printing:
        print('Importing dataframe...')
    cip_df = import_data(paths['support'] + 'ASC_2_CIP.xlsx', sheet_name="ASC_CIP")

    if printing:
        print('Generating Matrix...')

    cip_codes = np.unique(np.array(cip_df.loc[:, 'CIP']).astype(str)).astype(str)
    for i, cip in enumerate(cip_codes):
        num = len(cip)
        if num < 6:
            new_cip = "0" + cip
            cip_codes[i] = new_cip
    afscs = ['13H', '13M', '13N', '13S', '14F', '14N', '15A', '15W', '17S', '17D', '21A', '21M', '21R', '31P', '32EXA',
             '32EXC', '32EXE', '32EXF', '32EXG', '32EXJ', '35P', '38F', '38P', '61A', '61B', '61C', '61D', '62EXA',
             '62EXB', '62EXC', '62EXE', '62EXG', '62EXH', '62EXI', '62EXS', '63A', '64P', '65F']

    # AFOCD CLASSIFICATION
    N = len(cip_codes)
    M = len(afscs)
    qual_matrix = np.array([[" " for _ in range(M)] for _ in range(N)])
    for i, cip in enumerate(cip_codes):
        for j, afsc in enumerate(afscs):
            if afsc == '13H':
                if cip[:4] in ['2607', '3027', '3105', '2609']:
                    qual_matrix[i, j] = 'M'
                elif cip[:4] in ['2904', '4228']:
                    qual_matrix[i, j] = 'D'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '13M':
                if cip == '290402' or cip[:4] in ['4901', '5201', '5202', '5206', '5207', '5211', '5212', '5213',
                                                  '5214', '5218', '5219', '5220']:
                    qual_matrix[i, j] = 'D'
                else:
                    qual_matrix[i, j] = 'P'
            elif afsc == '13N':
                m_list4 = ['0402', '0403', '0404', '0405', '1427', '1428', '1437', '2903', '5202', '5213']
                for x in range(3, 10):
                    m_list4.append('140' + str(x))
                for x in range(10, 15):
                    m_list4.append('14' + str(x))
                m_list6 = ['290203', '290205', '290207']
                for x in ['0601', '0801', '1601', '3001', '3101']:
                    m_list6.append('30' + x)
                if cip[:4] in m_list4 or cip in m_list6 or cip[:2] in ['11', '27', '40']:
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'P'
            elif afsc == '13S':
                m_list4 = ['1402', '1410', '1419', '1427', '4002']
                d_list4 = ['1101', '1102', '1104', '1105', '1107', '1108', '1109', '1110', '1404', '1406', '1407',
                           '1408', '1409', '1411', '1412', '1413', '1414', '1418', '1420', '1423', '1432', '1435',
                           '1436', '1437', '1438', '1439', '1441', '1442', '1444', '3006', '3008', '3030', '4008']
                d_list6 = ['140101', '290203', '290204', '290205', '290207', '290301', '290302', '290304']
                if cip[:4] in m_list4 or cip[:2] == '27' or cip == '290305':
                    qual_matrix[i, j] = 'M'
                elif cip[:4] in d_list4 or cip in d_list6:
                    qual_matrix[i, j] = 'D'
                else:
                    qual_matrix[i, j] = 'P'
            elif afsc == '14F':
                m_list4 = ['3017', '4201', '4227', '4501', '4511', '5214', '3023', '3026']
                d_list4 = ['0909', '4509', '4502', '3025', '0901']
                if cip[:4] in m_list4:
                    qual_matrix[i, j] = 'M'
                elif cip[:4] in d_list4:
                    qual_matrix[i, j] = 'D'
                elif cip[:4] == '0501':
                    qual_matrix[i, j] = 'P'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '14N':
                m_list2 = ['11', '14', '27', '40', '05', '16', '22', '24', '28', '29', '45', '54']
                d_list2 = ['13', '09', '23', '28', '29', '30', '35', '38', '42', '43', '52']
                if cip[:2] in m_list2 or cip in ['307001', '301701']:
                    qual_matrix[i, j] = 'M'
                elif cip[:2] in d_list2 or cip == '490101':
                    qual_matrix[i, j] = 'D'
                else:
                    qual_matrix[i, j] = 'P'
            elif afsc == '15A' or afsc == '61A':
                if cip[:4] in ['1437', '1435'] or cip[:2] == '27' or cip in ['303001', '307001']:
                    qual_matrix[i, j] = 'M'
                elif cip in ['110701', '450603'] or cip[:4] == '1427':
                    qual_matrix[i, j] = 'D'
                elif (cip[:2] == '14' and cip != '140102') or cip[:4] in ['4008', '4506']:
                    qual_matrix[i, j] = 'P'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '15W':
                if cip[:4] == '4004':
                    qual_matrix[i, j] = 'M'
                elif cip[:4] in ['3008', '3030'] or cip[:2] in ['27', '41'] or (cip[:2] == '40' and cip[:4] != '4004'):
                    qual_matrix[i, j] = 'P'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '17D' or afsc == '17S' or afsc == '17X':
                m_list6 = ['150303', '151202', '290207', '303001', '307001', '270103', '270303', '270304']
                d_list4 = ['1503', '1504', '1508', '1512', '1514', '4008', '4005']
                if cip[:4] in ['3008', '3016', '5212'] or cip in m_list6 or \
                        (cip[:2] == '11' and cip[:4] not in ['1103', '1106']) or (cip[:2] == '14' and cip != '140102'):
                    qual_matrix[i, j] = 'M'
                elif cip[:4] in d_list4 or cip[:2] == '27':
                    qual_matrix[i, j] = 'D'
                else:
                    qual_matrix[i, j] = 'P'
            elif afsc == '21A':
                d_list4 = ['5202', '5206', '1101', '1102', '1103', '1104', '1107', '1110', '5212']
                if cip[:4] in d_list4 or cip[:2] in ['14', '40'] or cip in ['151501', '520409', '490104', '490101']:
                    qual_matrix[i, j] = 'D'
                else:
                    qual_matrix[i, j] = 'P'
            elif afsc == '21M':
                d_list4 = ['1107', '1101', '1110', '5202', '5206', '5213']
                d_list2 = ['14', '27', '40']
                d_list6 = ['290407', '290408', '151501', '520409']
                if cip[:2] in d_list2 or cip[:4] in d_list4 or cip in d_list6:
                    qual_matrix[i, j] = 'D'
                else:
                    qual_matrix[i, j] = 'P'
            elif afsc == '21R':
                d_list4 = ['1425', '1407', '1101', '1102', '1103', '1104', '1107', '1110', '4506', '5202', '5203',
                           '5206', '5208', '5212']
                d_list6 = ['151501', '490101', '520409']
                if cip[:4] in d_list4 or cip in d_list6:
                    qual_matrix[i, j] = 'D'
                else:
                    qual_matrix[i, j] = 'P'
            elif afsc == '31P':
                d_list6 = ['450902', '430301', '430302', '430303', '430304', '439999', '450401', '451101', '301701',
                           '220000', '220001', '229999']
                for x in ['03', '04', '07', '11', '12', '14', '18', '19', '20', '02', '13', '99']:
                    d_list6.append('4301' + x)
                if cip in d_list6:
                    qual_matrix[i, j] = 'D'
                else:
                    qual_matrix[i, j] = 'P'
            elif afsc == '32EXA':
                if cip[:4] == '0402' or cip in ['140401', '402010']:
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '32EXC':
                if cip[:4] == '1408':
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '32EXE':
                if cip[:4] == '1410':
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '32EXF':
                if cip == '141901':
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '32EXG':
                if cip[:4] in ['1408', '1410'] or cip in ['140401', '141401', '141901', '143301', '143501']:
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '32EXJ':
                if cip == '141401':
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '35P':
                if cip[:2] == '09':
                    qual_matrix[i, j] = 'M'
                elif cip[:4] in ['2313', '4509', '4510', '5214'] or cip[:2] == '42':
                    qual_matrix[i, j] = 'D'
                else:
                    qual_matrix[i, j] = 'P'
            elif afsc == '38F' or afsc == '38P':
                d_list4 = ['4404', '4405', '4506', '5202', '5203', '5206', '5208', '5210']
                if cip[:2] == '27' or cip[:4] == '5213' or cip in ['143701', '143501']:
                    qual_matrix[i, j] = 'M'
                elif cip[:4] in d_list4 or cip in ['301601', '301701', '422804']:
                    qual_matrix[i, j] = 'D'
                else:
                    qual_matrix[i, j] = 'P'
            elif afsc == '61B':
                m_list4 = ['3017', '4502', '4511', '4513', '4514', '4501']
                if cip[:2] == '42' or cip[:4] in m_list4 or cip in ['450501', '451201']:
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '61C':
                d_list6 = ['140601', '141801', '143201', '144301', '144401', '260299']
                if cip[:4] in ['1407', '4005'] or cip in ['260202', '260205']:
                    qual_matrix[i, j] = 'M'
                elif cip in d_list6 or cip[:5] == '26021' or cip[:4] == '4010':
                    qual_matrix[i, j] = 'D'
                elif cip in ['140501', '142001', '142501', '144501']:
                    qual_matrix[i, j] = 'P'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '61D':
                if cip[:4] in ['1412', '1423', '4002', '4008']:
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '62EXA':
                if cip[:4] == '1402' or (cip == '142701' and year == 2016):
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '62EXB':
                if cip[:4] == '1402':
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '62EXC':
                if cip[:4] in ['1409', '1447']:
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '62EXE':
                if cip[:4] in ['1410', '1447']:
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '62EXF':
                if cip[:2] in ['27', '40'] or (cip[:2] == '14' and cip != '140102'):
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '62EXG':
                if cip[:2] == '14' and cip != '140102' or (cip == '401002' and year in [2016]):
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '62EXH':
                if cip[:4] == '1419':
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '62EXI' or afsc == '62EXS':
                if cip[:4] in ['1427', '1435']:
                    qual_matrix[i, j] = 'M'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '63A':
                if cip[:2] in ['14', '40']:
                    qual_matrix[i, j] = 'M'
                elif cip[:2] in ['11', '27'] or cip[:4] == '4506' or (cip[:2] == '52' and cip[:4] != '5204'):
                    qual_matrix[i, j] = 'D'
                else:
                    qual_matrix[i, j] = 'P'
            elif afsc == '64P':
                d_list2 = ['52', '14', '15', '26', '27', '29', '40', '41', '28', '44', '54', '16', '23', '05', '42']
                if cip[:2] in d_list2 or (cip[:2] == '45' and cip[:4] != '4506') or \
                        cip[:4] in ['2200', '2202'] or cip == '220101':
                    qual_matrix[i, j] = 'D'
                else:
                    qual_matrix[i, j] = 'P'
            elif afsc == '65F':
                d_list4 = ['4506', '5203', '5206', '5213', '5208']
                if cip[:4] in d_list4 or cip[:2] in ['27', '52', '14']:
                    qual_matrix[i, j] = 'D'
                else:
                    qual_matrix[i, j] = 'P'
            else:
                qual_matrix[i, j] = 'I'

    qual_matrix_df = pd.DataFrame({'CIP': cip_codes})
    for j, afsc in enumerate(afscs):
        qual_matrix_df['qual_' + afsc] = qual_matrix[:, j]
    full_afscs_df = pd.DataFrame({"AFSC": afscs})

    if printing:
        print('Exporting Matrix...')
    with pd.ExcelWriter(paths['s_support'] + 'Qual_CIP_Matrix.xlsx') as writer:  # Export to excel
        qual_matrix_df.to_excel(writer, sheet_name="Qual Matrix", index=False)
        full_afscs_df.to_excel(writer, sheet_name="Full AFSCS", index=False)


def cip_to_qual(afscs, cip1, cip2=None, full_afscs=None, cip_qual_matrix=None):
    """
    This procedure takes two cip arrays: cadet first and second degrees and then
    returns a qual matrix for those cadets
    :param full_afscs: list of the full afscs
    :param afscs: list of afscs used for this problem
    :param cip_qual_matrix: Matrix matching cip codes to qualifications
    :param cip1: cip codes for first degree
    :param cip2: cip codes for second degree
    :return: qual matrix
    """
    # Get correct data type
    if len(afscs[0]) == 2:  # "G1"
        data_type = 'Generated'
        full_afscs = afscs
    else:  # We know it's a real AFSC
        data_type = 'Real'

    # Load CIP to Qual matrix
    if cip_qual_matrix is None:
        cip_qual_matrix = import_data(paths['s_support'] + "Qual_CIP_Matrix_" + data_type + ".xlsx",
                                      sheet_name="Qual Matrix")

    # Load full afscs
    if full_afscs is None:
        full_afscs = np.array(import_data(
            paths['s_support'] + "Qual_CIP_Matrix_" + data_type + ".xlsx", sheet_name="Full AFSCS"))

    afsc_indices = np.where(full_afscs == afscs)[0]  # afscs used in this instance
    full_cip_codes = np.array(cip_qual_matrix.loc[:, "CIP"]).astype(str)  # full list of CIP codes
    full_qual_matrix = np.array(cip_qual_matrix.iloc[:, 1:])  # full qual matrix
    N = len(cip1)
    M = len(afscs)

    # Initialize Qual Matrix
    qual_matrix = np.array([[" " for _ in range(M)] for _ in range(N)])
    tier_dict = {'M': 3, 'D': 2, 'P': 1, 'I': 0}

    # Loop through all cadets
    for i in range(N):

        # Load qualifications for this cadet
        cip_index = np.where(full_cip_codes == cip1[i])[0]
        if len(cip_index) == 0:
            cip_index = 0
        else:
            cip_index = cip_index[0]
        qual_matrix[i, :] = full_qual_matrix[cip_index][afsc_indices]

        # Check second degree
        if cip2 is not None:
            cip_index2 = np.where(full_cip_codes == cip2[i])[0]
            if len(cip_index2) != 0:
                cip_index2 = cip_index2[0]
                check_row = full_qual_matrix[cip_index2][afsc_indices]
                for j in range(M):
                    qual1 = qual_matrix[i, j]
                    qual2 = check_row[j]

                    # If second degree tier trumps first, we use it instead
                    if tier_dict[qual2] > tier_dict[qual1]:
                        qual_matrix[i, j] = qual2

    return qual_matrix


def asc_to_cip(asc_dict, asc_cip=None):
    """
    This procedure takes a dictionary of arrays of ASC codes and converts them to CIP codes
    :param asc_cip: optional sheet with conversions (we'll just import it manually otherwise)
    :param asc_dict: ASC codes dict
    :return: CIP codes
    """

    if asc_cip is None:
        asc_cip = import_data(paths['support'] + "ASC_2_CIP.xlsx", sheet_name="ASC_CIP")

    asc_list = np.array(asc_cip.loc[:, 'ASC'])
    cip_list = np.array(asc_cip.loc[:, 'CIP'])
    cip_dict = {}
    for x, name in enumerate(list(asc_dict.keys())):

        cip_dict[name] = np.array(["      " for _ in range(len(asc_dict[name]))])
        for i, asc in enumerate(asc_dict[name]):

            index = np.where(asc_list == asc)[0]
            if len(index) != 0:
                cip_dict[name][i] = cip_list[index[0]]

    return cip_dict


def clean_problem_instance_preferences_utilities(afscs, original_preferences, original_utilities=None,
                                                 year_2020=False):
    """
    This procedure takes the afscs that are matched this year, as well as the original preferences
    and optionally the original utilities and returns the cleaned preferences and utilities for
    the problem instance.
    :param year_2020: if this year is 2020 or not
    :param afscs: original year afscs
    :param original_preferences: raw cadet preferences
    :param original_utilities: raw cadet utilities
    :return: cleaned preferences, cleaned utilities
    """
    N = len(original_preferences)
    if original_utilities is not None:
        original_utilities = original_utilities / 100

    # Reduce preferences to only include the AFSCs defined above
    unique_preferences = np.unique(original_preferences)
    flattened_preferences = np.ndarray.flatten(original_preferences)
    for afsc in unique_preferences:
        indices = np.where(flattened_preferences == afsc)[0]
        if year_2020:
            if afsc not in afscs:
                np.put(flattened_preferences, indices, '')
        else:
            if afsc[:-1] in afscs:
                np.put(flattened_preferences, indices, afsc[:-1])
            elif afsc not in afscs:
                np.put(flattened_preferences, indices, '')

    # Clean up preferences and add utilities
    preferences = flattened_preferences.reshape(N, 6)
    new_preferences = np.array([[" " * 8 for _ in range(6)] for _ in range(N)])
    utility_vector = np.array([1, 0.50, 0.33, 0.25, 0.2, 0.17])
    utilities = np.zeros([N, 6])
    for i in range(N):

        # Get indices of year-specific AFSCs
        correct_indices = np.where(preferences[i, :] != '')[0]

        # Place preferences and utilities in the correct spots
        first_indices = np.arange(len(correct_indices))
        np.put(new_preferences[i, :], first_indices, preferences[i, correct_indices])
        if original_utilities is not None:
            np.put(utilities[i, :], first_indices, original_utilities[i, correct_indices])
        else:
            np.put(utilities[i, :], first_indices, utility_vector[correct_indices])

        # Reduce the "larger" blanks to the simple ''
        big_blank_indices = np.where(new_preferences[i, :] == " " * 8)[0]
        np.put(new_preferences[i, :], big_blank_indices, '')

    return new_preferences, utilities


def clean_ctgan_data_preferences_utilities(original_preferences, original_utilities=None, printing=False):
    """
    This procedure takes the afscs that are matched this year, as well as the original preferences
    and optionally the original utilities and returns the cleaned preferences and utilities for
    the CTGAN data.
    :param printing: Whether to print something
    :param original_preferences: raw cadet preferences
    :param original_utilities: raw cadet utilities
    :return: cleaned preferences, cleaned utilities
    """
    if original_utilities is not None:
        original_utilities = original_utilities / 100

    # Standardized AFSCs
    full_afscs = ['13M', '13N', '13S', '14F', '14N', '15A', '15W', '17D', '21A', '21M', '21R', '31P', '32EXA',
                  '32EXC', '32EXE', '32EXF', '32EXG', '32EXJ', '35P', '38F', '61C', '61D', '62EXA', '62EXB', '62EXC',
                  '62EXE', '62EXG', '62EXH', '62EXI', '63A', '64P', '65F']

    N = len(original_preferences)
    flattened_preferences = np.ndarray.flatten(original_preferences)
    for afsc in np.unique(original_preferences):
        check_afsc = afsc[:3] + 'X' + afsc[len(afsc) - 1]
        if afsc == '38P':
            indices = np.where(flattened_preferences == '38P')[0]
            np.put(flattened_preferences, indices, '38F')
            if printing:
                print('38P -> 38F')
        elif afsc in ['61A', '61A1', '61AX']:
            indices = np.where(flattened_preferences == afsc)[0]
            np.put(flattened_preferences, indices, '15A')
            if printing:
                print(afsc + ' -> 15A')
        elif afsc == '62EXS':
            indices = np.where(flattened_preferences == afsc)[0]
            np.put(flattened_preferences, indices, '14N')
            if printing:
                print(afsc + ' -> 62EXI')
        elif afsc == '63AXS':
            indices = np.where(flattened_preferences == afsc)[0]
            np.put(flattened_preferences, indices, '63A')
            if printing:
                print(afsc + ' -> 63A')
        elif afsc in ['14NJX', '14NXS']:
            indices = np.where(flattened_preferences == afsc)[0]
            np.put(flattened_preferences, indices, '14N')
            if printing:
                print(afsc + ' -> 14N')
        elif afsc in ['17NX', '17SX', '17DXS', '17DXY', '17S1']:
            indices = np.where(flattened_preferences == afsc)[0]
            np.put(flattened_preferences, indices, '17D')
            if printing:
                print(afsc + ' -> 17D')
        elif afsc[:-1] in full_afscs:
            indices = np.where(flattened_preferences == afsc)[0]
            np.put(flattened_preferences, indices, afsc[:-1])
            if printing:
                print(afsc + ' -> ' + afsc[:-1])
        elif check_afsc in full_afscs:
            indices = np.where(flattened_preferences == afsc)[0]
            np.put(flattened_preferences, indices, check_afsc)
            if printing:
                print(afsc + ' -> ' + check_afsc)
        elif afsc not in full_afscs:
            indices = np.where(flattened_preferences == afsc)[0]
            np.put(flattened_preferences, indices, '')
            if printing:
                print(afsc + ' -> Blank')

    # Clean up preferences and add utilities
    preferences = flattened_preferences.reshape(N, 6)
    new_preferences = np.array([["      " for _ in range(6)] for _ in range(N)])
    utility_vector = np.array([1, 0.50, 0.33, 0.25, 0.2, 0.17])
    utilities = np.zeros([N, 6])
    for i in range(N):

        # Get list of unique preferences in order
        s, indices = np.unique(preferences[i, :], return_index=True)
        unique_pref = preferences[i, :][np.sort(indices)]

        # Remove empty preferences
        blank_index = np.where(unique_pref == '')[0]
        if len(blank_index) != 0:
            unique_pref = np.delete(unique_pref, blank_index)

        # Put the condensed preferences into the new array
        np.put(new_preferences[i, :], range(len(unique_pref)), unique_pref)

        # Reduce the "larger" blanks to the simple ''
        big_blank_indices = np.where(new_preferences[i, :] == "      ")[0]
        np.put(new_preferences[i, :], big_blank_indices, '')

        # Utilities
        util_indices = np.where(new_preferences[i, :] != '')[0]
        if original_utilities is not None:
            np.put(utilities[i, :], util_indices, original_utilities[i, :][:len(util_indices)])
        else:
            np.put(utilities[i, :], util_indices, utility_vector[:len(util_indices)])

    preferences = new_preferences

    return preferences, utilities


def generate_afsc_quota(afscs, assigned, historic_target_means, historic_target_max, historic_target_afscs,
                        historic_usafa_means=None):
    """
    This procedure generates AFSC quotas based on historical means for the given class year
    :param afscs: class year AFSCs
    :param assigned: assigned AFSCs
    :param historic_target_means: historical average quotas
    :param historic_target_max: historical maximum quota
    :param historic_target_afscs: historical afscs
    :param historic_usafa_means: historical average usafa quotas
    :return: AFSC quotas
    """
    year_target_indices = np.array([np.where(historic_target_afscs == afsc)[0][0] for afsc in afscs])
    year_target_means = historic_target_means[year_target_indices]
    year_target_max = historic_target_max[year_target_indices]
    afsc_quota = year_target_means / sum(year_target_means)
    minimums = np.zeros(len(afscs))
    maximums = np.zeros(len(afscs))
    usafa_quota = np.zeros(len(afscs))
    for j, afsc in enumerate(afscs):
        indices = np.where(assigned == afsc)[0]
        num = len(indices)
        if year_target_max[j] == 1:
            minimums[j] = num
            maximums[j] = num
        elif num < afsc_quota[j]:
            minimums[j] = num
            maximums[j] = int(num * year_target_max[j])
        else:
            minimums[j] = int(max(num - num * 0.1, 1))
            maximums[j] = int(max(minimums[j] * year_target_max[j], 2))
        if afsc == '61B':
            minimums[j] = int(max(num - num * 0.1, 1))
            maximums[j] = int(max(minimums[j] * 2, 2))
        if historic_usafa_means is not None:
            usafa_quota[j] = int(minimums[j] * historic_usafa_means[j])

    if historic_usafa_means is None:
        return minimums, maximums
    else:
        return minimums, maximums, usafa_quota


def clean_2020_2021_data(years=None, ctgan_df=None, ctgan_combined=None, asc_cip_df=None, get_ctgan=False):
    """
    This procedure cleans the 2020 and 2021 NO_PII data and exports it to excel
    :return: None
    """

    # Import dataframes
    if asc_cip_df is None:
        asc_cip_df = import_data(paths['support'] + "ASC_2_CIP.xlsx", sheet_name="ASC_CIP")

    if years is None:
        years = [2020]
    for year in years:

        if year == 2020:
            print('Acquiring Problem Instance and CTGAN data for year ' + str(year))
        else:
            print('Acquiring Problem Instance for year ' + str(year))

        # Import Dataframes
        results_df = import_data(paths['s_raw'] + 'result_Detailed_NO_PII_' + str(year) + '.xlsx',
                                 sheet_name='Result')
        targets_df = import_data(paths['s_raw'] + 'No_PII_input_data_' + str(year) + '.xlsx',
                                 sheet_name='Targets')
        usafa_input_df = import_data(paths['s_raw'] + 'No_PII_input_data_' + str(year) + '.xlsx',
                                     sheet_name='USAFA_Input')
        rotc_input_df = import_data(paths['s_raw'] + 'No_PII_input_data_' + str(year) + '.xlsx',
                                    sheet_name='ROTC_Input')

        # Useful numpy arrays
        encrypt_SS = np.array(results_df.loc[:, 'Encrypt_PII_S'])
        assigned = np.array(results_df.loc[:, 'afsc']).astype(str)
        original_preferences = np.array(results_df.loc[:, 'NRAT1':'NRAT6']).astype(str)
        percentiles = np.array(results_df.loc[:, 'percentile'])
        usafa = np.array(results_df.loc[:, 'USAFA'])
        afscs = np.unique(assigned)
        if 'nan' in afscs:
            afscs = afscs[:-1]

        if year == 2020:
            original_utilities = np.array(results_df.loc[:, 'NrWgt1':'NrWgt6'])
            year_qual_df = results_df.loc[:, 'qual_1':'qual_44']

            # Clean preferences and utilities for the Problem Instance
            preferences, utilities = clean_problem_instance_preferences_utilities(afscs, original_preferences,
                                                                                  original_utilities, year_2020=True)
        else:
            original_utilities = np.array(results_df.loc[:, 'NRWGT1':'NRWGT6'])
            year_qual_df = results_df.loc[:, 'qual_1':'qual_32']

            # Clean preferences and utilities for the Problem Instance
            preferences, utilities = clean_problem_instance_preferences_utilities(afscs, original_preferences,
                                                                                  original_utilities, year_2020=True)

        # Load AFSC Quotas
        quota = np.array(targets_df.loc[:, 'Target'])
        usafa_quota = np.array(targets_df.loc[:, 'USAFA Target'])
        rotc_quota = np.array(targets_df.loc[:, 'ROTC Target'])
        minimums = (np.array(targets_df.loc[:, 'Min']) * quota).astype(int)
        maximums = (np.array(targets_df.loc[:, 'Max']) * quota).astype(int)

        # Build dataframes
        afscs_fixed = pd.DataFrame({'AFSC': afscs, 'USAFA Target': usafa_quota,
                                    'ROTC Target': rotc_quota, 'Combined Target': quota,
                                    'Min': minimums, 'Max': maximums})
        cadets_fixed = pd.DataFrame({'Encrypt_PII': encrypt_SS, 'USAFA': usafa, 'percentile': percentiles})
        solution_df = pd.DataFrame({"Encrypt_PII": encrypt_SS, "afsc": assigned})

        # Utilities
        for p in range(6):
            cadets_fixed['NrWgt' + str(p + 1)] = utilities[:, p]

        # Preferences
        for p in range(6):
            cadets_fixed['NRat' + str(p + 1)] = preferences[:, p]

        # Qualifications
        year_qual_df = year_qual_df.replace(np.nan, 'I', regex=True)
        qual_matrix = np.array(year_qual_df).astype(str)
        for j, afsc in enumerate(afscs):
            cadets_fixed['qual_' + afsc] = qual_matrix[:, j]

        print('Exporting Problem Instance for year ' + str(year) + ' to excel')

        # Export to Excel
        with pd.ExcelWriter(paths['s_instances'] + str(year) + ".xlsx") as writer:  # Export to excel
            cadets_fixed.to_excel(writer, sheet_name='Cadets Fixed', index=False)
            afscs_fixed.to_excel(writer, sheet_name='AFSCs Fixed', index=False)
            solution_df.to_excel(writer, sheet_name="Original Solution", index=False)

        if year == 2020 and ctgan_df is not None and get_ctgan:

            # Clean CTGAN data (disregard results data, and only use input data)
            usafa_encrypt_SS = np.array(usafa_input_df.loc[:, 'Encrypt_PII_S'])
            rotc_encrypt_SS = np.array(rotc_input_df.loc[:, 'Encrypt_PII_L'])
            encrypt_SS = np.hstack((usafa_encrypt_SS, rotc_encrypt_SS))

            # Preferences and Utilities
            usafa_original_preferences = np.array(usafa_input_df.loc[:, 'NRAT1':'NRAT6']).astype(str)
            usafa_original_utilities = np.array(usafa_input_df.loc[:, 'NrWgt1':'NrWgt6']).astype(float)
            rotc_original_utilities = np.zeros([len(rotc_input_df), 6])
            rotc_original_preferences = np.array([["      " for _ in range(6)] for _ in range(len(rotc_input_df))])
            for p in range(6):
                rotc_original_utilities[:, p] = np.array(rotc_input_df.loc[:, 'CADET WEIGHT ' + str(p + 1)])
                rotc_original_preferences[:, p] = np.array(rotc_input_df.loc[
                                                           :, 'AFSC Preference ' + str(p + 1) + ' (Cadet)'])

            original_preferences = np.vstack((usafa_original_preferences, rotc_original_preferences))
            original_utilities = np.vstack((usafa_original_utilities, rotc_original_utilities))
            preferences, utilities = clean_ctgan_data_preferences_utilities(original_preferences, original_utilities)

            # CIP
            usafa_cip1 = np.array(usafa_input_df.loc[:, 'CIP_DEGREE_1'])
            usafa_cip2 = np.array(usafa_input_df.loc[:, 'CIP_DEGREE_2'])
            rotc_asc1 = np.array(rotc_input_df.loc[:, 'Degree Code'])
            rotc_asc2 = np.array(rotc_input_df.loc[:, 'Second Degree Code'])
            rotc_asc = {1: rotc_asc1, 2: rotc_asc2}
            rotc_cip = asc_to_cip(rotc_asc, asc_cip_df)
            rotc_cip1 = rotc_cip[1]
            rotc_cip2 = rotc_cip[2]
            cip1 = np.hstack((usafa_cip1, rotc_cip1))
            cip2 = np.hstack((usafa_cip2, rotc_cip2))

            # Percentile
            rotc_gpa = np.array(rotc_input_df.loc[:, 'GPA'])
            sorted_indices = np.argsort(rotc_gpa)
            rotc_percentiles = (np.arange(len(rotc_encrypt_SS)) /
                                (len(rotc_encrypt_SS) - 1))
            magic_indices = np.argsort(sorted_indices)
            rotc_percentiles = rotc_percentiles[magic_indices]
            usafa_gom = np.array(usafa_input_df.loc[:, 'GOM'])
            sorted_indices = np.argsort(usafa_gom)[::-1]
            usafa_percentiles = (np.arange(len(usafa_encrypt_SS)) /
                                 (len(usafa_encrypt_SS) - 1))
            magic_indices = np.argsort(sorted_indices)
            usafa_percentiles = usafa_percentiles[magic_indices]
            percentiles = np.hstack((usafa_percentiles, rotc_percentiles))

            # USAFA column
            usafa = np.hstack((np.repeat(1, len(usafa_encrypt_SS)),
                               np.repeat(0, len(rotc_encrypt_SS))))

            # Build CTGAN dataframe
            ctgan_df[year] = pd.DataFrame({'Encrypt_PII': encrypt_SS, 'CIP1': cip1, 'CIP2': cip2, 'USAFA': usafa,
                                           'percentile': percentiles})

            # Utilities
            for p in range(6):
                ctgan_df[year]['NrWgt' + str(p + 1)] = utilities[:, p]

            # Preferences
            for p in range(6):
                ctgan_df[year]['NRat' + str(p + 1)] = preferences[:, p]

            ctgan_df[year] = ctgan_data_filter(ctgan_df[year])  # More Cleaning

            # Add this year data to ctgan combined dataframe
            ctgan_combined = ctgan_combined.append(ctgan_df[year])

    return ctgan_df, ctgan_combined


def ctgan_data_filter(ctgan_data=None):
    """
    This procedure loads the ctgan data and filters it so that all the rows satisfy the CTGAN constraints
    :return: Exports to excel
    """

    # Load Data
    if ctgan_data is None:
        ctgan_data = import_data(paths['s_support'] + 'ctgan_data_scrubbed.xlsx', sheet_name='Data')

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
        # for cip in unique:
        #     real_cip = cip.split('.')[0]
        #     indices = np.where(cip_arr == cip)[0]
        #     np.put(cip_arr, indices, real_cip)
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
    # ctgan_data = ctgan_data.drop(labels='NrWgt1', axis=1)

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

    # with pd.ExcelWriter(paths['s_support'] + 'ctgan_data_scrubbed.xlsx') as writer:  # Export to excel
    #     ctgan_data.to_excel(writer, sheet_name="Data", index=False)

    return ctgan_data


def afscs_unique(table_data):
    """
    Constraint enforcing the cadet preferences to be unique across the rows (no repeats)
    """

    # Check the row to see if there are no repeated AFSCs, unless the cell is empty, in which case
    # blanks can be repeated.
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


def data_structure_constraint_fails(table_data):
    """
    This procedure takes in cadet parameter data and reports the number of data constraint fails
    for that dataset. This includes non-unique preferences, non-descending utilities, preferences
    and utilities not matching up where there are zero utilities etc.
    :param table_data: CTGAN-like data
    :return: Table of constraint fails
    """

    afscs_unique_valid = np.array(afscs_unique(table_data)) * 1
    utilities_match_valid = np.array(utilities_match(table_data)) * 1
    num_wrong_au = sum(afscs_unique_valid == 0)
    num_wrong_um = sum(utilities_match_valid == 0)
    num_wrong_ud = 0
    utilities = np.array(table_data.loc[:, 'NrWgt2':'NrWgt6'])
    for i in range(len(utilities)):
        for p in range(4):
            if utilities[i, p] < utilities[i, p + 1]:
                num_wrong_ud += 1
                break

    constraint_table = pd.DataFrame({'Non-Unique AFSC Rows': [num_wrong_au],
                                     'Zero-Utilities & Non-Blank': [num_wrong_um],
                                     'Utilities Non-Descending': [num_wrong_ud]})

    return constraint_table


def create_scrubbed_data(years=None, create_defaults=True, create_instances=True):
    """
    creates value parameter defaults for the class years
    :param create_instances:
    :param create_defaults:
    :param years: years to generate
    :return: None
    """

    # Get years
    full_years_dict = {'2016': 'F', '2017': 'D', '2018': 'E', '2019': 'A', '2020': 'B', '2021': 'C', 'CTGAN': 'CTGAN'}
    if years is not None:
        years_dict = {}
        for year in years:
            years_dict[year] = full_years_dict[year]
    else:
        years_dict = full_years_dict

    # Load in defaults dataframes
    df_names = ['Overall Weights', 'AFSC Weights', 'AFSC Objective Weights', 'AFSC Objective Targets',
                'AFSC Objective Min Value', 'Constraint Type', 'Value Functions']
    real_df_dict = {}
    for df_name in df_names:
        real_df_dict[df_name] = import_data(paths['s_support'] + 'Value_Parameters_Defaults_Real.xlsx',
                                            sheet_name=df_name)
    default_afscs = np.array(real_df_dict['AFSC Weights'].loc[:, 'AFSC'])

    # Want to know which scrubbed names correspond to each of the real AFSCs for CIP table
    afsc_locator_dict = {}
    for afsc in default_afscs:
        afsc_locator_dict[afsc] = []

    # Loop through all years
    for year in years_dict:
        print('Loading year ' + year)

        # Load in year AFSCs
        year_afscs_df = import_data(paths['s_support'] + 'Year_AFSCs_Table.xlsx', sheet_name=year)
        real_afscs = np.array(year_afscs_df.loc[:, 'AFSC'])
        new_names = np.array(year_afscs_df.loc[:, 'Name'])
        default_indices = [np.where(default_afscs == afsc)[0][0] for afsc in real_afscs]

        # Add each scrubbed name to the real afscs
        for default_afsc in default_afscs:
            if default_afsc in real_afscs:
                index = np.where(real_afscs == default_afsc)[0][0]
                afsc_locator_dict[default_afsc].append(new_names[index])

        # Create new default value parameters
        if create_defaults:
            new_df_dict = {}
            for df_name in df_names:

                if df_name == 'Overall Weights':
                    new_df_dict[df_name] = copy.deepcopy(real_df_dict[df_name])
                else:
                    columns = list(real_df_dict[df_name].keys())
                    new_df_dict[df_name] = pd.DataFrame({})
                    for column in columns:

                        if column == 'AFSC':
                            new_df_dict[df_name][column] = new_names
                        else:
                            real_column = np.array(real_df_dict[df_name].loc[:, column])
                            new_df_dict[df_name][column] = real_column[default_indices]

            # Export default value parameters to Excel
            filepath = paths['s_support'] + 'Value_Parameters_Defaults_' + years_dict[year] + '.xlsx'
            with pd.ExcelWriter(filepath) as writer:  # Export to excel
                for df_name in df_names:
                    new_df_dict[df_name].to_excel(writer, sheet_name=df_name, index=False)

        # Create new problem instances
        if create_instances and year != "CTGAN":

            # Create new Cadets Fixed dataframe
            cadets_fixed = import_data(paths['s_instances'] + year + '_AFPC.xlsx', sheet_name='Cadets Fixed')
            N = len(cadets_fixed)
            cadets_columns = list(cadets_fixed.keys())
            unchanged_columns = cadets_columns[1:9]
            preferences = cadets_columns[9:15]

            # Some columns don't change
            encrypt_column = np.array([(i + 1) for i in np.arange(N)])
            new_cadets_fixed = pd.DataFrame({'Encrypt_PII': encrypt_column})
            for column in unchanged_columns:
                new_cadets_fixed[column] = np.array(cadets_fixed[column])

            # Preference columns replace old afscs with new ones
            for column in preferences:
                column_array = np.array(cadets_fixed[column]).astype(str)
                cadet_afsc_indices = [np.where(column_array == afsc)[0] for afsc in real_afscs]
                for j, new_name in enumerate(new_names):
                    column_array[cadet_afsc_indices[j]] = new_name
                new_cadets_fixed[column] = column_array

            # Qualification columns change
            real_qual_columns = np.array(["qual_" + afsc for afsc in real_afscs])
            for j, column in enumerate(real_qual_columns):
                new_column = "qual_" + new_names[j]
                new_cadets_fixed[new_column] = np.array(cadets_fixed[column])

            # Create new AFSCs Fixed dataframe
            afscs_fixed = import_data(paths['s_instances'] + year + '_AFPC.xlsx', sheet_name='AFSCs Fixed')
            afscs_columns = list(afscs_fixed.keys())
            solution_quality = import_data(paths['s_instances'] + year + '_AFPC.xlsx',
                                           sheet_name='Cadet Solution Quality')

            # Get correct indices
            original_afscs = np.array(afscs_fixed.loc[:, 'AFSC'])
            new_indices = [np.where(original_afscs == afsc)[0][0] for afsc in real_afscs]

            # Replace columns
            new_afscs_fixed = pd.DataFrame({'AFSC': new_names})
            for column in afscs_columns[1:]:
                new_afscs_fixed[column] = np.array(afscs_fixed.loc[:, column])[new_indices]

            # Change AFSCs in solution
            afsc_solution = np.array(solution_quality.loc[:, 'Matched'])
            cadet_afsc_indices = np.array([np.where(afsc_solution == afsc)[0] for afsc in real_afscs])
            for j, new_name in enumerate(new_names):
                afsc_solution[cadet_afsc_indices[j]] = new_name
            original_solution_df = pd.DataFrame({"Encrypt_PII": encrypt_column, "Matched": afsc_solution})

            # Export problem instance to Excel
            filepath = paths['s_instances'] + 'Instance_' + years_dict[year] + '_Original.xlsx'
            with pd.ExcelWriter(filepath) as writer:  # Export to excel
                new_cadets_fixed.to_excel(writer, sheet_name='Cadets Fixed', index=False)
                new_afscs_fixed.to_excel(writer, sheet_name='AFSCs Fixed', index=False)
                original_solution_df.to_excel(writer, sheet_name='Original Solution', index=False)

    # Get Scrubbed CIP locator table
    afsc_locator_column = np.array([" " * 100 for _ in range(len(default_afscs))])
    for j, afsc in enumerate(default_afscs):
        afsc_string = ""
        for new_name in afsc_locator_dict[afsc]:
            afsc_string += new_name + ", "
        if len(afsc_string) > 0:
            afsc_string = afsc_string[:-2]
        afsc_locator_column[j] = afsc_string

    # Export to Excel
    locator_df = pd.DataFrame({'AFSC': default_afscs, 'Names': afsc_locator_column})
    filepath = paths['s_support'] + 'AFSC_Translation.xlsx'
    with pd.ExcelWriter(filepath) as writer:  # Export to excel
        locator_df.to_excel(writer, sheet_name='AFSCs', index=False)


def scrub_ctgan_afscs():
    """
    Scrub AFSCs in CTGAN data
    :return:
    """
    ctgan_data = import_data(paths['s_support'] + 'ctgan_data_real.xlsx', sheet_name='Data')
    year_afscs_df = import_data(paths['s_support'] + 'Year_AFSCs_Table.xlsx', sheet_name="CTGAN")
    real_afscs = np.array(year_afscs_df.loc[:, 'AFSC'])
    new_names = np.array(year_afscs_df.loc[:, 'Name'])

    # Create new CTGAN dataframe
    N = len(ctgan_data)
    cadets_columns = list(ctgan_data.keys())
    unchanged_columns = cadets_columns[1:10]
    preferences = cadets_columns[10:]

    # Some columns don't change
    encrypt_column = np.array([(i + 1) for i in np.arange(N)])
    new_ctgan_data = pd.DataFrame({'Encrypt_PII': encrypt_column})
    for column in unchanged_columns:
        new_ctgan_data[column] = np.array(ctgan_data[column])

    # Preference columns replace old afscs with new ones
    for column in preferences:
        column_array = np.array(ctgan_data[column]).astype(str)
        cadet_afsc_indices = [np.where(column_array == afsc)[0] for afsc in real_afscs]
        for j, new_name in enumerate(new_names):
            column_array[cadet_afsc_indices[j]] = new_name
        new_ctgan_data[column] = column_array

    # Export new CTGAN data to Excel
    filepath = paths['s_support'] + 'ctgan_data_scrubbed.xlsx'
    with pd.ExcelWriter(filepath) as writer:  # Export to excel
        new_ctgan_data.to_excel(writer, sheet_name='Data', index=False)


def scrub_qual_cip_matrix():
    """
    Creates a new CIP to Qual matrix using scrubbed qualifications
    :return: None
    """

    # Load data
    qual_matrix = import_data(paths['s_support'] + 'Qual_CIP_Matrix.xlsx', sheet_name='Qual Matrix')
    cip_column = np.array(qual_matrix.loc[:, 'CIP']).astype(str)
    for i, cip in enumerate(cip_column):
        num = len(cip)
        if num < 6:
            new_cip = "0" + cip
            cip_column[i] = new_cip
    year_afscs_df = import_data(paths['s_support'] + 'Year_AFSCs_Table.xlsx', sheet_name="CTGAN")
    real_afscs = np.array(year_afscs_df.loc[:, 'AFSC'])
    new_names = np.array(year_afscs_df.loc[:, 'Name'])

    # Create qualification matrix
    new_qual_matrix = pd.DataFrame({'CIP': cip_column})
    real_qual_columns = np.array(["qual_" + afsc for afsc in real_afscs])
    for j, column in enumerate(real_qual_columns):
        new_column = "qual_" + new_names[j]
        new_qual_matrix[new_column] = np.array(qual_matrix[column])

    # AFSCs df
    full_afscs_df = pd.DataFrame({'AFSC': new_names})

    # Export data to Excel
    filepath = paths['s_support'] + 'Qual_CIP_Matrix_Generated.xlsx'
    with pd.ExcelWriter(filepath) as writer:
        new_qual_matrix.to_excel(writer, sheet_name='Qual Matrix', index=False)
        full_afscs_df.to_excel(writer, sheet_name='Full AFSCs', index=False)


def clean_cip_stuff():
    """
    There's been some issues with CIPs, so we're cleaning them here
    :return:
    """

    # Load data
    print('Loading data...')
    ctgan_data = import_data(paths['s_support'] + 'ctgan_data_scrubbed.xlsx', sheet_name='Data')
    qual_matrix = import_data(paths['s_support'] + 'Qual_CIP_Matrix_Generated.xlsx',
                              sheet_name='Qual Matrix')
    targets = import_data(paths['s_support'] + 'Instance_Generator_Parameters.xlsx',
                          sheet_name='Targets')

    # Fix CIPs for CTGAN data
    cadet_cip_dict = {}
    cip_df = {}
    for cip_name in ['CIP1', 'CIP2']:

        # Fix CTGAN
        cadet_cip_dict[cip_name] = np.array(ctgan_data[cip_name]).astype(str)
        unique_cips = np.unique(cadet_cip_dict[cip_name]).astype(str)
        for i, cip in enumerate(unique_cips):
            num = len(cip)
            if num < 6 and '.' not in cip:
                if cip in ["220", "220.0"]:
                    unique_cips[i] = "220000"
                elif cip == "nan":
                    unique_cips[i] = " " * 6
                else:
                    unique_cips[i] = "0" + cip
            if '.' in cip and num == 6:
                unique_cips[i] = "0" + cip[:5]

            unique_cips[i] = unique_cips[i][:6]

            # Replace cip column
            cadet_cip_dict[cip_name] = np.where(cadet_cip_dict[cip_name] == cip,
                                                str(unique_cips[i]), cadet_cip_dict[cip_name].astype(str)).astype(str)
        ctgan_data[cip_name] = cadet_cip_dict[cip_name]

        # Fix Generator Parameters
        cip_df[cip_name] = import_data(paths['s_support'] + 'Instance_Generator_Parameters.xlsx',
                                       sheet_name=cip_name)
        cip_column = np.array(cip_df[cip_name][cip_name]).astype(str)
        for i, cip in enumerate(cip_column):
            num = len(cip)
            if num < 6:
                if cip in ["220", "220.0"]:
                    cip_column[i] = "220000"
                elif cip == "nan":
                    cip_column[i] = " " * 6
                else:
                    cip_column[i] = "0" + cip
        cip_df[cip_name][cip_name] = cip_column

    # Export CTGAN data to Excel
    filepath = paths['s_support'] + 'ctgan_data_scrubbed.xlsx'
    with pd.ExcelWriter(filepath) as writer:  # Export to excel
        ctgan_data.to_excel(writer, sheet_name='Data', index=False)

    # Load Data
    afscs = np.array(targets['AFSC'])
    high_tiers = np.array(targets['High_Tier'])
    M = len(afscs)
    cips = np.array(qual_matrix['CIP']).astype(str)
    for i, cip in enumerate(cips):
        num = len(cip)
        if num < 6:
            new_cip = "0" + cip
            cips[i] = new_cip

    # Get CIPs that are preferred for each AFSC
    print('Getting CIP column...')
    target_cip_dict = {}
    for j, afsc in enumerate(afscs):
        qual_col = np.array(qual_matrix['qual_' + afsc])
        tier = high_tiers[j]
        indices = np.where(qual_col == tier)[0]
        target_cip_dict[afsc] = cips[indices]

    # Put them in the column
    target_cip_col = np.array([" " * 10000 for _ in range(M)])
    for j, afsc in enumerate(afscs):
        cip_str = ""
        for cip in target_cip_dict[afsc]:
            cip_str += cip + ", "
        if len(cip_str) > 0:
            cip_str = cip_str[:-2]
        target_cip_col[j] = cip_str

    # Add column to dataframe and export to excel
    targets['Top Tier CIPs'] = target_cip_col
    filepath = paths['s_support'] + 'Instance_Generator_Parameters.xlsx'
    with pd.ExcelWriter(filepath) as writer:
        targets.to_excel(writer, sheet_name='Targets', index=False)
        for cip_name in cip_df:
            cip_df[cip_name].to_excel(writer, sheet_name=cip_name, index=False)