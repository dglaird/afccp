import copy

import numpy as np
import pandas as pd
import afccp.core.globals


def generate_cip_to_qual_matrix(printing=True, year=2016):
    """
    This procedure takes all the CIP codes from the ASC_CIP excel sheet and then creates a matrix of AFSCs that the
    degrees qualify for. This matrix is for AFSC qualifications as designated by the AFOCD (CAO 20 Apr 2022)
    :return: None
    """
    if printing:
        print('Importing dataframe...')
    cip_df = import_data(paths["support"] + 'ASC_2_CIP.xlsx', sheet_name="CIP")

    if printing:
        print('Generating Matrix...')

    cip_codes = np.unique(np.array(cip_df.loc[:, 'CIP']).astype(str)).astype(str)
    for i, cip in enumerate(cip_codes):
        cip_codes[i] = (6 - len(cip)) * "0" + cip  # Add leading 0s if necessary
    afscs = ['13H', '13M', '13N', '13S', '14F', '14N', '15A', '15W', '17X', '17S', '17D', '21A', '21M', '21R', '31P',
             '32EXA',
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
                m_list4 = ['3017', '4201', '4227', '4501', '4511']
                d_list4 = ['0909', '5214', '3023', '3026']
                p_list4 = ['0501', '4509', '4502', '3025', '0901']
                if cip[:4] in m_list4:
                    qual_matrix[i, j] = 'M'
                elif cip[:4] in d_list4:
                    qual_matrix[i, j] = 'D'
                elif cip[:4] in p_list4:
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
                m_list4 = ['1437', '1435', '3070', '3030', '3008']
                if cip[:4] in m_list4 or cip[:2] in ['27'] or cip in ['110102']:
                    qual_matrix[i, j] = 'M'
                elif cip in ['110804', '450603'] or cip[:4] in ['1427', '1107', '3039', '3049']:
                    qual_matrix[i, j] = 'D'
                elif (cip[:2] == '14' and cip != '140102') or cip[:4] in ['4008', '4506', '2611', '3071', '5213',
                                                                          '4506']:
                    qual_matrix[i, j] = 'P'
                else:
                    qual_matrix[i, j] = 'I'
            elif afsc == '15W':
                if cip[:4] == '4004':
                    qual_matrix[i, j] = 'M'
                elif cip[:2] in ['27', '41'] or (cip[:2] == '40' and cip[:4] != '4004') or cip[:4] in ['3008', '3030']:
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

                # Added Ops Research and Data Processing (no CIPs in AFOCD)
                d_list6_add = ['143701', '110301']
                if cip[:2] in d_list2 or cip[:4] in d_list4 or cip in d_list6 or cip in d_list6_add:
                    qual_matrix[i, j] = 'D'
                else:
                    qual_matrix[i, j] = 'P'
            elif afsc == '21R':
                d_list4 = ['1425', '1407', '1101', '1102', '1103', '1104', '1107', '1110', '4506', '5202', '5203',
                           '5206', '5208']
                d_list6 = ['151501', '490101', '520409']

                # Added Ops Research and Data Processing (no CIPs in AFOCD)
                d_list6_add = ['143701', '110301']
                if cip[:4] in d_list4 or cip in d_list6 or cip in d_list6_add or cip[:3] == "521":
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
                if cip[:4] == '0402' or cip in ['140401']:  # Sometimes 402010 is included
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
                elif cip in ["140701"] or cip[:4] in ["1405", "1425", "1402", "5220"]:
                    qual_matrix[i, j] = 'D'  # CY23 added a desired tier to 32EXG!
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
                if cip[:4] == '1402':  # or (cip == '142701' and year == 2016):
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
                if cip[:2] == '14' and cip != '140102' and cip[:4] != "1437":  # (cip == '401002' and year in [2016]):
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
    with pd.ExcelWriter(paths["support"] + 'Qual_CIP_Matrix.xlsx') as writer:  # Export to excel
        qual_matrix_df.to_excel(writer, sheet_name="Qual Matrix", index=False)
        full_afscs_df.to_excel(writer, sheet_name="Full AFSCS", index=False)


def cip_to_qual(afscs, cip1, cip2=None, full_afscs=None, cip_qual_matrix=None, report_cips_not_found=False):
    """
    This procedure takes two cip arrays: cadet first and second degrees and then
    returns a qual matrix for those cadets
    :param report_cips_not_found: return a list of cips of cadets that aren't in the list
    :param full_afscs: list of the full afscs
    :param afscs: list of afscs used for this problem
    :param cip_qual_matrix: Matrix matching cip codes to qualifications
    :param cip1: cip codes for first degree
    :param cip2: cip codes for second degree
    :return: qual matrix
    """
    # Get correct data type
    if len(afscs[0]) == 2:  # "G1"
        data_type = 'scrubbed'
        full_afscs = afscs
    else:  # We know it's a real AFSC
        data_type = 'real'

    # Load CIP to Qual matrix
    if cip_qual_matrix is None:
        cip_qual_matrix = import_data(paths["support"] + "Qual_CIP_Matrix.xlsx",
                                      sheet_name="Qual Matrix")

    # Load full afscs
    if full_afscs is None:
        full_afscs = np.array(import_data(
            paths["support"] + "Qual_CIP_Matrix.xlsx", sheet_name="Full AFSCS"))

    afsc_indices = np.where(full_afscs == afscs)[0]  # afscs used in this instance
    full_cip_codes = np.array(cip_qual_matrix.loc[:, "CIP"]).astype(str)  # full list of CIP codes
    full_qual_matrix = np.array(cip_qual_matrix.iloc[:, 1:])  # full qual matrix
    N = len(cip1)
    M = len(afscs)

    # Adjust CIPs in case python cut the leading 0s off
    for i, cip in enumerate(full_cip_codes):
        full_cip_codes[i] = "0" * (6 - len(cip)) + cip

    # Initialize Qual Matrix
    qual_matrix = np.array([[" " for _ in range(M)] for _ in range(N)])
    tier_dict = {'M': 3, 'D': 2, 'P': 1, 'I': 0}
    unknown_cips = {}

    # Loop through all cadets
    for i in range(N):

        # Clean up CIPs (Add leading 0s)
        if cip1[i] != "Unk":
            cip1[i] = "0" * (6 - len(cip1[i])) + cip1[i]

        # Load qualifications for this cadet
        cip_index = np.where(full_cip_codes == cip1[i])[0]
        if len(cip_index) == 0:
            cip_index = 0  # Set degree qualifications to arbitrary degree

            if report_cips_not_found:
                if cip1[i] in unknown_cips:
                    unknown_cips[cip1[i]] += 1
                else:
                    unknown_cips[cip1[i]] = 1
        else:
            cip_index = cip_index[0]
        qual_matrix[i, :] = full_qual_matrix[cip_index][afsc_indices]

        # Check second degree
        if cip2 is not None:

            if len(cip2[i]) >= 4 and cip2[i] != "None":
                cip2[i] = "0" * (6 - len(cip2[i])) + cip2[i]
            else:
                cip2[i] = "None"
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

    if report_cips_not_found:
        return qual_matrix, unknown_cips
    else:
        return qual_matrix


def cip_to_qual_direct(afscs, cip1, cip2=None, business_hours=None, printing=True):
    """
    This procedure takes in a list of AFSCs, CIP codes, and optionally a second set of CIP codes
    (the cadets' second degrees) and generates a qual matrix for this years' specific cadets. AFOCD cao April 2022
    :return: None
    """

    # AFOCD CLASSIFICATION
    N = len(cip1)
    M = len(afscs)
    cips = {1: cip1, 2: cip2}
    qual = {}
    if cip2 is None:
        degrees = [1]
    else:
        degrees = [1, 2]

    # Loop through both sets of degrees (if applicable)
    for d in degrees:

        # Initialize qual matrix (for this set of degrees)
        qual[d] = np.array([["I" for _ in range(M)] for _ in range(N)])
        for i in range(N):
            cip = "0" * (6 - len(cips[d][i])) + cips[d][i]
            for j, afsc in enumerate(afscs):
                if afsc == '13H':
                    if cip[:4] in ['2607', '3027', '3105', '2609']:
                        qual[d][i, j] = 'M'
                    elif cip[:4] in ['2904', '4228']:
                        qual[d][i, j] = 'D'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '13M':
                    if cip == '290402' or cip[:4] in ['4901', '5201', '5202', '5206', '5207', '5211', '5212', '5213',
                                                      '5214', '5218', '5219', '5220']:
                        qual[d][i, j] = 'D'
                    else:
                        qual[d][i, j] = 'P'
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
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'P'
                elif afsc == '13S':
                    m_list4 = ['1402', '1410', '1419', '1427', '4002']
                    d_list4 = ['1101', '1102', '1104', '1105', '1107', '1108', '1109', '1110', '1404', '1406', '1407',
                               '1408', '1409', '1411', '1412', '1413', '1414', '1418', '1420', '1423', '1432', '1435',
                               '1436', '1437', '1438', '1439', '1441', '1442', '1444', '3006', '3008', '3030', '4008']
                    d_list6 = ['140101', '290203', '290204', '290205', '290207', '290301', '290302', '290304']
                    if cip[:4] in m_list4 or cip[:2] == '27' or cip == '290305':
                        qual[d][i, j] = 'M'
                    elif cip[:4] in d_list4 or cip in d_list6:
                        qual[d][i, j] = 'D'
                    else:
                        qual[d][i, j] = 'P'
                elif afsc == '14F':
                    m_list4 = ['3017', '4201', '4227', '4501', '4511']
                    d_list4 = ['0909', '5214', '3023', '3026']
                    p_list4 = ['0501', '4509', '4502', '3025', '0901']
                    if cip[:4] in m_list4:
                        qual[d][i, j] = 'M'
                    elif cip[:4] in d_list4:
                        qual[d][i, j] = 'D'
                    elif cip[:4] in p_list4:
                        qual[d][i, j] = 'P'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '14N':
                    m_list2 = ['11', '14', '27', '40', '05', '16', '22', '24', '28', '29', '45', '54']
                    d_list2 = ['13', '09', '23', '28', '29', '30', '35', '38', '42', '43', '52']
                    if cip[:2] in m_list2 or cip in ['307001', '301701']:
                        qual[d][i, j] = 'M'
                    elif cip[:2] in d_list2 or cip == '490101':
                        qual[d][i, j] = 'D'
                    else:
                        qual[d][i, j] = 'P'
                elif afsc == '15A' or afsc == '61A':
                    m_list4 = ['1437', '1435', '3070', '3030', '3008']
                    if cip[:4] in m_list4 or cip[:2] in ['27'] or cip in ['110102']:
                        qual[d][i, j] = 'M'
                    elif cip in ['110804', '450603'] or cip[:4] in ['1427', '1107', '3039', '3049']:
                        qual[d][i, j] = 'D'
                    elif (cip[:2] == '14' and cip != '140102') or cip[:4] in ['4008', '4506', '2611', '3071', '5213',
                                                                              '4506']:
                        qual[d][i, j] = 'P'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '15W':
                    if cip[:4] == '4004':
                        qual[d][i, j] = 'M'
                    elif cip[:2] in ['27', '41'] or (cip[:2] == '40' and cip[:4] != '4004') or cip[:4] in ['3008',
                                                                                                           '3030']:
                        qual[d][i, j] = 'P'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '17D' or afsc == '17S' or afsc == '17X':
                    m_list6 = ['150303', '151202', '290207', '303001', '307001', '270103', '270303', '270304']
                    d_list4 = ['1503', '1504', '1508', '1512', '1514', '4008', '4005']
                    if cip[:4] in ['3008', '3016', '5212'] or cip in m_list6 or \
                            (cip[:2] == '11' and cip[:4] not in ['1103', '1106']) or (
                            cip[:2] == '14' and cip != '140102'):
                        qual[d][i, j] = 'M'
                    elif cip[:4] in d_list4 or cip[:2] == '27':
                        qual[d][i, j] = 'D'
                    else:
                        qual[d][i, j] = 'P'
                elif afsc == '21A':
                    d_list4 = ['5202', '5206', '1101', '1102', '1103', '1104', '1107', '1110', '5212']
                    if cip[:4] in d_list4 or cip[:2] in ['14', '40'] or cip in ['151501', '520409', '490104', '490101']:
                        qual[d][i, j] = 'D'
                    else:
                        qual[d][i, j] = 'P'
                elif afsc == '21M':
                    d_list4 = ['1107', '1101', '1110', '5202', '5206', '5213']
                    d_list2 = ['14', '27', '40']
                    d_list6 = ['290407', '290408', '151501', '520409']

                    # Added Ops Research and Data Processing (no CIPs in AFOCD)
                    d_list6_add = ['143701', '110301']
                    if cip[:2] in d_list2 or cip[:4] in d_list4 or cip in d_list6 or cip in d_list6_add:
                        qual[d][i, j] = 'D'
                    else:
                        qual[d][i, j] = 'P'
                elif afsc == '21R':
                    d_list4 = ['1425', '1407', '1101', '1102', '1103', '1104', '1107', '1110', '4506', '5202', '5203',
                               '5206', '5208']
                    d_list6 = ['151501', '490101', '520409']

                    # Added Ops Research and Data Processing (no CIPs in AFOCD)
                    d_list6_add = ['143701', '110301']
                    if cip[:4] in d_list4 or cip in d_list6 or cip in d_list6_add or cip[:3] == "521":
                        qual[d][i, j] = 'D'
                    else:
                        qual[d][i, j] = 'P'
                elif afsc == '31P':
                    d_list6 = ['450902', '430301', '430302', '430303', '430304', '439999', '450401', '451101', '301701',
                               '220000', '220001', '229999']
                    for x in ['03', '04', '07', '11', '12', '14', '18', '19', '20', '02', '13', '99']:
                        d_list6.append('4301' + x)
                    if cip in d_list6:
                        qual[d][i, j] = 'D'
                    else:
                        qual[d][i, j] = 'P'
                elif afsc == '32EXA':
                    if cip[:4] == '0402' or cip in ['140401']:  # Sometimes 402010 is included
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '32EXC':
                    if cip[:4] == '1408':
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '32EXE':
                    if cip[:4] == '1410':
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '32EXF':
                    if cip == '141901':
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '32EXG':
                    if cip[:4] in ['1408', '1410'] or cip in ['140401', '141401', '141901', '143301', '143501']:
                        qual[d][i, j] = 'M'
                    elif cip in ["140701"] or cip[:4] in ["1405", "1425", "1402", "5220"]:
                        qual[d][i, j] = 'D'  # CY23 added a desired tier to 32EXG!
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '32EXJ':
                    if cip == '141401':
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '35P':
                    if cip[:2] == '09':
                        qual[d][i, j] = 'M'
                    elif cip[:4] in ['2313', '4509', '4510', '5214'] or cip[:2] == '42':
                        qual[d][i, j] = 'D'
                    else:
                        qual[d][i, j] = 'P'
                elif afsc == '38F' or afsc == '38P':
                    d_list4 = ['4404', '4405', '4506', '5202', '5203', '5206', '5208', '5210']
                    if cip[:2] == '27' or cip[:4] == '5213' or cip in ['143701', '143501']:
                        qual[d][i, j] = 'M'
                    elif cip[:4] in d_list4 or cip in ['301601', '301701', '422804']:
                        qual[d][i, j] = 'D'
                    else:
                        qual[d][i, j] = 'P'
                elif afsc == '61B':
                    m_list4 = ['3017', '4502', '4511', '4513', '4514', '4501']
                    if cip[:2] == '42' or cip[:4] in m_list4 or cip in ['450501', '451201']:
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '61C':
                    d_list6 = ['140601', '141801', '143201', '144301', '144401', '260299']
                    if cip[:4] in ['1407', '4005'] or cip in ['260202', '260205']:
                        qual[d][i, j] = 'M'
                    elif cip in d_list6 or cip[:5] == '26021' or cip[:4] == '4010':
                        qual[d][i, j] = 'D'
                    elif cip in ['140501', '142001', '142501', '144501']:
                        qual[d][i, j] = 'P'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '61D':
                    if cip[:4] in ['1412', '1423', '4002', '4008']:
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '62EXA':
                    if cip[:4] == '1402':  # or (cip == '142701' and year == 2016):
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '62EXB':
                    if cip[:4] == '1402':
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '62EXC':
                    if cip[:4] in ['1409', '1447']:
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '62EXE':
                    if cip[:4] in ['1410', '1447']:
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '62EXF':
                    if cip[:2] in ['27', '40'] or (cip[:2] == '14' and cip != '140102'):
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '62EXG':
                    if cip[:2] == '14' and cip != '140102' and cip[
                                                               :4] != "1437":  # (cip == '401002' and year in [2016]):
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '62EXH':
                    if cip[:4] == '1419':
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '62EXI' or afsc == '62EXS':
                    if cip[:4] in ['1427', '1435']:
                        qual[d][i, j] = 'M'
                    else:
                        qual[d][i, j] = 'I'
                elif afsc == '63A':
                    if cip[:2] in ['14', '40']:
                        qual[d][i, j] = 'M'
                    elif cip[:2] in ['11', '27'] or cip[:4] == '4506' or (cip[:2] == '52' and cip[:4] != '5204'):
                        qual[d][i, j] = 'D'
                    else:

                        if business_hours is not None:
                            if business_hours[i] >= 24:
                                qual[d][i, j] = 'P'
                            else:
                                qual[d][i, j] = 'I'
                        else:
                            qual[d][i, j] = 'P'
                elif afsc == '64P':
                    d_list2 = ['52', '14', '15', '26', '27', '29', '40', '41', '28', '44', '54', '16', '23', '05', '42']
                    if cip[:2] in d_list2 or (cip[:2] == '45' and cip[:4] != '4506') or \
                            cip[:4] in ['2200', '2202'] or cip == '220101':
                        qual[d][i, j] = 'D'
                    else:
                        qual[d][i, j] = 'P'
                elif afsc == '65F':
                    d_list4 = ['4506', '5203', '5206', '5213', '5208']
                    if cip[:4] in d_list4 or cip[:2] in ['27', '52', '14']:
                        qual[d][i, j] = 'D'
                    else:
                        qual[d][i, j] = 'P'
                else:
                    qual[d][i, j] = 'I'

    if cip2 is None:
        qual_matrix = copy.deepcopy(qual[1])
    else:
        qual_matrix = np.array([["I" for _ in range(M)] for _ in range(N)])
        tier = {'M': 3, 'D': 2, 'P': 1, 'I': 0}
        for i in range(N):
            for j in range(M):
                qual_1 = qual[1][i, j]
                qual_2 = qual[2][i, j]
                if tier[qual_1] > tier[qual_2]:
                    qual_matrix[i, j] = qual_1
                else:
                    qual_matrix[i, j] = qual_2

    return qual_matrix


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


def cip_to_qual_tiers(afscs, cip1, cip2=None, business_hours=None, true_tiers=True):
    """
    This procedure takes in a list of AFSCs, CIP codes, and optionally a second set of CIP codes
    (the cadets' second degrees) and generates a qual matrix for this year's specific cadets. AFOCD c/ao October 2022,
    however discussions with CFMs have altered the AFOCD "unofficially" but we honor these modifications as they're
    directly from the CFM. They should go into effect in future AFOCD iterations. (The Latest Modification: Apr 2023)
    This qualification matrix incorporates both tier and requirement (M1, D2, etc.)
    :return: None
    """

    # AFOCD CLASSIFICATION
    N = len(cip1)
    M = len(afscs)
    cips = {1: cip1, 2: cip2}
    qual = {}
    if cip2 is None:
        degrees = [1]
    else:
        degrees = [1, 2]

    # Loop through both sets of degrees (if applicable)
    for d in degrees:

        # Initialize qual matrix (for this set of degrees)
        qual[d] = np.array([["I5" for _ in range(M)] for _ in range(N)])

        # Loop through each cadet and AFSC pair
        for i in range(N):
            cip = str(cips[d][i])
            cip = "0" * (6 - len(cip)) + cip
            for j, afsc in enumerate(afscs):

                # Rated
                if afsc in ["11U", "11XX", "12XX", "13B", "18X", "92T0", "92T1", "92T2", "92T3"]:
                    qual[d][i, j] = "P1"

                # Aerospace Physiologist
                elif afsc == '13H':
                    if cip[:4] in ['2607', '3027']:
                        qual[d][i, j] = 'M1'
                    elif cip[:4] in ['3105', '2609']:
                        qual[d][i, j] = 'M2'
                    elif cip[:4] in ['2904', '4228']:
                        qual[d][i, j] = 'D3'
                    else:
                        qual[d][i, j] = 'I4'

                # Airfield Ops
                elif afsc == '13M':
                    if cip == '290402' or cip[:4] == '4901':
                        qual[d][i, j] = 'D1'
                    elif cip[:4] in ['5201', '5202', '5206', '5207', '5211', '5212', '5213',
                                     '5214', '5218', '5219', '5220']:
                        qual[d][i, j] = 'D2'
                    else:
                        qual[d][i, j] = 'P3'

                # Nuclear and Missile Operations
                elif afsc == '13N':
                    if true_tiers:
                        qual[d][i, j] = 'P1'
                    else:
                        m_list4 = ['0402', '0403', '0404', '0405', '1427', '1428', '1437', '2903', '5202', '5213']
                        for x in range(3, 10):
                            m_list4.append('140' + str(x))
                        for x in range(10, 15):
                            m_list4.append('14' + str(x))
                        m_list6 = ['290203', '290205', '290207']
                        for x in ['0601', '0801', '1601', '3001', '3101']:
                            m_list6.append('30' + x)
                        if cip[:4] in m_list4 or cip in m_list6 or cip[:2] in ['11', '27', '40']:
                            qual[d][i, j] = 'M1'
                        else:
                            qual[d][i, j] = 'P2'

                # Space Operations
                elif afsc in ['13S', '13S1S']:
                    m_list4 = ['1402', '1410', '1419', '1427', '4002']
                    d_list4 = ['1101', '1102', '1104', '1105', '1107', '1108', '1109', '1110', '1404', '1406', '1407',
                               '1408', '1409', '1411', '1412', '1413', '1414', '1418', '1420', '1423', '1432', '1435',
                               '1436', '1437', '1438', '1439', '1441', '1442', '1444', '3006', '3008', '3030', '4008']
                    d_list6 = ['140101', '290203', '290204', '290205', '290207', '290301', '290302', '290304']
                    if cip[:4] in m_list4 or cip[:2] == '27' or cip == '290305':
                        qual[d][i, j] = 'M1'
                    elif cip[:4] in d_list4 or cip in d_list6:
                        qual[d][i, j] = 'D2'
                    else:
                        qual[d][i, j] = 'P3'

                # Information Operations
                elif afsc == '14F':
                    m_list4 = ['3017', '4201', '4227', '4511']
                    d_list4 = ['5214', '3023', '3026']
                    p_list4 = ['4509', '4502', '3025', '0901']
                    if cip[:4] in m_list4:
                        qual[d][i, j] = 'M1'
                    elif cip[:4] in d_list4 or cip in ["090902", "090903", "090907"]:
                        qual[d][i, j] = 'D2'
                    elif cip[:4] in p_list4:
                        qual[d][i, j] = 'P3'
                    else:
                        qual[d][i, j] = 'I4'

                # Intelligence
                elif afsc in ['14N', '14N1S']:
                    m_list2 = ['05', '16', '22', '24', '28', '29', '45', '54']
                    d_list2 = ['13', '09', '23', '28', '29', '30', '35', '38', '42', '43', '52']
                    if cip[:2] in ['11', '14', '27', '40'] or cip == '307001':
                        qual[d][i, j] = 'M1'
                    elif cip[:2] in m_list2 or cip == '301701':
                        qual[d][i, j] = 'M2'
                    elif cip[:2] in d_list2 or cip == '490101':
                        qual[d][i, j] = 'D3'
                    else:
                        qual[d][i, j] = 'P4'

                # Operations Research Analyst
                elif afsc in ['15A', '61A']:
                    m_list4 = ['1437', '1435', '3070', '3030', '3008']
                    if cip[:4] in m_list4 or cip[:2] == '27' or cip == '110102':
                        qual[d][i, j] = 'M1'
                    elif cip in ['110804', '450603'] or cip[:4] in ['1427', '1107', '3039', '3049']:
                        qual[d][i, j] = 'D2'
                    elif (cip[:2] == '14' and cip != '140102') or cip[:4] in ['4008', '4506', '2611', '3071', '5213']:
                        qual[d][i, j] = 'P3'
                    else:
                        qual[d][i, j] = 'I4'

                # Weather and Environmental Sciences (DOESN'T MAKE SENSE)
                elif afsc == '15W':
                    if cip[:4] == '4004':
                        qual[d][i, j] = 'M1'
                    elif cip[:2] in ['27', '41'] or (cip[:2] == '40' and cip[:4] != '4004') or \
                            cip[:4] in ['3008', '3030']:
                        qual[d][i, j] = 'P2'
                    else:
                        qual[d][i, j] = 'I3'

                # Cyberspace Operations
                elif afsc in ['17D', '17S', '17X', '17S1S']:
                    m_list6 = ['150303', '151202', '290207', '303001', '307001', '270103', '270303', '270304']
                    d_list4 = ['1503', '1504', '1508', '1512', '1514', '4008', '4005']
                    if cip[:4] in ['3008', '3016', '5212'] or cip in m_list6 or \
                            (cip[:2] == '11' and cip[:4] not in ['1103', '1106']) or (
                            cip[:2] == '14' and cip != '140102'):
                        qual[d][i, j] = 'M1'
                    elif cip[:4] in d_list4 or cip[:2] == '27':
                        qual[d][i, j] = 'D2'
                    else:
                        qual[d][i, j] = 'P3'

                # Aircraft Maintenance
                elif afsc == '21A':
                    d_list4 = ['5202', '5206', '1101', '1102', '1103', '1104', '1107', '1110', '5212']
                    if cip[:2] == '14':
                        qual[d][i, j] = 'D1'
                    elif cip[:4] in d_list4 or cip[:2] == '40' or cip in ['151501', '520409', '490104', '490101']:
                        qual[d][i, j] = 'D2'
                    else:
                        qual[d][i, j] = 'P3'

                # Munitions and Missile Maintenance
                elif afsc == '21M':
                    d_list4 = ['1107', '1101', '1110', '5202', '5206', '5213']
                    d_list2 = ['27', '40']

                    # Added "Data Processing" (no CIPs in AFOCD, and others are already captured in other tiers)
                    d_list6 = ['290407', '290408', '151501', '520409', "110301"]
                    if cip[:2] == "14":
                        qual[d][i, j] = 'D1'
                    elif cip[:2] in d_list2 or cip[:4] in d_list4 or cip in d_list6:
                        qual[d][i, j] = 'D2'
                    else:
                        qual[d][i, j] = 'P3'

                # Logistics Readiness  (Current a/o 4/28/2023)
                elif afsc == '21R':
                    if true_tiers:
                        cip_list = ['520203', '520409', '142501', '490199', '520209', '499999', '521001', '520201',
                                    '140101', '143501', '280799', '450601', '520601', '520304', '520899', '520213',
                                    '520211', '143701', '110802']
                        if cip in cip_list:
                            qual[d][i, j] = 'D1'
                        else:
                            qual[d][i, j] = 'P2'
                    else:
                        d_list4 = ['1101', '1102', '1103', '1104', '1107', '1110', '4506', '5202', '5203', '5206',
                                   '5208']
                        d_list6 = ['151501', '490101', '520409']

                        # Added Ops Research and Data Processing (no CIPs in AFOCD)
                        d_list6_add = ['143701', '110301']
                        if cip[:4] in ['1425', '1407']:
                            qual[d][i, j] = 'D1'
                        elif cip[:4] in d_list4 or cip in d_list6 or cip in d_list6_add or cip[:3] == "521":
                            qual[d][i, j] = 'D2'
                        else:
                            qual[d][i, j] = 'P3'

                # Security Forces
                elif afsc == '31P':
                    if true_tiers:
                        qual[d][i, j] = 'P1'
                    else:
                        d_list6 = ['450902', '430301', '430302', '430303', '430304', '439999', '450401', '451101',
                                   '301701', '220000', '220001', '229999']
                        for x in ['03', '04', '07', '11', '12', '14', '18', '19', '20', '02', '13', '99']:
                            d_list6.append('4301' + x)
                        if cip in d_list6:
                            qual[d][i, j] = 'D1'
                        else:
                            qual[d][i, j] = 'P2'

                # Civil Engineering: Architect/Architectural Engineer
                elif afsc == '32EXA':
                    if cip[:4] == '0402' or cip in ['140401']:  # Sometimes 402010 is included
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Civil Engineering: Civil Engineer
                elif afsc == '32EXC':
                    if cip[:4] == '1408':
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Civil Engineering: Electrical Engineer
                elif afsc == '32EXE':
                    if cip[:4] == '1410':
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Civil Engineering: Mechanical Engineer
                elif afsc == '32EXF':
                    if cip == '141901':
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Civil Engineering: General Engineer
                elif afsc == '32EXG':
                    if cip[:4] in ['1408', '1410'] or cip in ['140401', '141401', '141901', '143301', '143501']:
                        qual[d][i, j] = 'M1'
                    elif cip in ["140701"] or cip[:4] in ["1405", "1425", "1402", "5220"]:
                        qual[d][i, j] = 'D2'  # FY23 added a desired tier to 32EXG!
                    else:
                        qual[d][i, j] = 'I3'

                # Civil Engineering: Environmental Engineer
                elif afsc == '32EXJ':
                    if cip == '141401':
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Public Affairs
                elif afsc == '35P':
                    if cip[:2] == '09':
                        qual[d][i, j] = 'M1'
                    elif cip[:4] in ['2313', '4509', '4510', '5214'] or cip[:2] == '42':
                        qual[d][i, j] = 'D2'
                    else:
                        qual[d][i, j] = 'P3'

                # Force Support
                elif afsc in ['38F', '38P']:
                    d_list4 = ['4404', '4405', '4506', '5202', '5203', '5206', '5208', '5210']
                    if cip[:2] == '27' or cip[:4] == '5213' or cip in ['143701', '143501']:
                        qual[d][i, j] = 'M1'
                    elif cip[:4] in d_list4 or cip in ['301601', '301701', '422804']:
                        qual[d][i, j] = 'D2'
                    else:
                        qual[d][i, j] = 'P3'

                # Old 14F (Information Operations)
                elif afsc == '61B':
                    m_list4 = ['3017', '4502', '4511', '4513', '4514', '4501']
                    if cip[:2] == '42' or cip[:4] in m_list4 or cip in ['450501', '451201']:
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Chemist/Nuclear Chemist
                elif afsc == '61C':
                    d_list6 = ['140601', '141801', '143201', '144301', '144401', '260299']
                    if cip[:4] in ['1407', '4005'] or cip in ['260202', '260205']:
                        qual[d][i, j] = 'M1'
                    elif cip in d_list6 or cip[:5] == '26021' or cip[:4] == '4010':
                        qual[d][i, j] = 'D2'
                    elif cip in ['140501', '142001', '142501', '144501']:
                        qual[d][i, j] = 'P3'
                    else:
                        qual[d][i, j] = 'I4'

                # Physicist/Nuclear Engineer
                elif afsc == '61D':
                    if cip[:4] in ['1412', '1423', '4002', '4008']:
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Developmental Engineering: Aeronautical Engineer
                elif afsc in ['62EXA', '62E1A1S']:
                    if cip[:4] == '1402':  # or (cip == '142701' and year == 2016):
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Developmental Engineering: Astronautical Engineer
                elif afsc in ['62EXB', '62E1B1S']:
                    if cip[:4] == '1402':
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Developmental Engineering: Computer Systems Engineer
                elif afsc in ['62EXC', '62E1C1S']:
                    if cip[:4] in ['1409', '1447']:
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Developmental Engineering: Electrical/Electronic Engineer
                elif afsc in ['62EXE', '62E1E1S']:
                    if cip[:4] in ['1410', '1447']:
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Developmental Engineering: Flight Test Engineer
                elif afsc == '62EXF':
                    if cip[:2] in ['27', '40'] or (cip[:2] == '14' and cip != '140102'):
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Developmental Engineering: Project/General Engineer
                elif afsc in ['62EXG', '62E1G1S']:
                    if cip[:2] == '14' and cip != '140102' and cip[
                                                               :4] != "1437":  # (cip == '401002' and year in [2016]):
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Developmental Engineering: Mechanical Engineer
                elif afsc in ['62EXH', '62E1H1S']:
                    if cip[:4] == '1419':
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Developmental Engineering: Systems/Human Factors Engineer
                elif afsc in ['62EXI', '62EXS', '62E1I1S']:
                    if cip[:4] in ['1427', '1435']:
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Acquisition Manager
                elif afsc in ['63A', '63A1S']:
                    if cip[:2] in ['14', '40']:
                        qual[d][i, j] = 'M1'
                    elif cip[:2] in ['11', '27'] or cip[:4] == '4506' or (cip[:2] == '52' and cip[:4] != '5204'):
                        qual[d][i, j] = 'D2'
                    else:
                        if business_hours is not None:
                            if business_hours[i] >= 24:
                                qual[d][i, j] = 'P3'
                            else:
                                qual[d][i, j] = 'I4'
                        else:
                            qual[d][i, j] = 'P3'

                # Contracting
                elif afsc == '64P':
                    d_list2 = ['28', '44', '54', '16', '23', '05', '42']
                    if cip[:2] == "52":
                        qual[d][i, j] = 'D1'
                    elif cip[:2] in ['14', '15', '26', '27', '29', '40', '41']:
                        qual[d][i, j] = 'D2'
                    elif cip[:2] in d_list2 or (cip[:2] == '45' and cip[:4] != '4506') or \
                            cip[:4] in ['2200', '2202'] or cip == '220101':
                        qual[d][i, j] = 'D3'
                    else:
                        qual[d][i, j] = 'P4'

                # Financial Management
                elif afsc == '65F':
                    if cip[:4] in ['4506', '5203', '5206', '5213', '5208']:
                        qual[d][i, j] = 'D1'
                    elif cip[:2] in ['27', '52', '14']:
                        qual[d][i, j] = 'D2'
                    else:
                        qual[d][i, j] = 'P3'

                # This shouldn't happen... but all others!
                else:
                    qual[d][i, j] = 'I5'

    # If CIP2 is not specified, we just take the qual matrix from the first degrees
    if cip2 is None:
        qual_matrix = copy.deepcopy(qual[1])

    # If CIP2 is specified, we take the highest tier that the cadet qualifies for
    else:
        qual_matrix = np.array([["I5" for _ in range(M)] for _ in range(N)])

        # Loop though each cadet and AFSC pair
        for i in range(N):
            for j in range(M):

                # Get degree tier qualifications from both degrees
                qual_1 = qual[1][i, j]
                qual_2 = qual[2][i, j]

                # If the first degree qualification is higher (lower number ex. 1 < 2), we take that. Otherwise, #2
                if int(qual_1[1]) < int(qual_2[1]):
                    qual_matrix[i, j] = qual_1
                else:
                    qual_matrix[i, j] = qual_2

    return qual_matrix
