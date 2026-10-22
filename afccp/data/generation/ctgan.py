import random
import numpy as np
import pandas as pd
import pickle
import datetime
from typing import Optional, Dict, Any
from scipy.stats import beta

# afccp modules
import afccp.globals
import afccp.data.preferences
import afccp.data.adjustments
import afccp.data.values
import afccp.data.support

# SDV modules
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
from scipy.stats import gaussian_kde


# _______________________________________MAIN CTGAN DATA GENERATION_____________________________________________________
def generate_ctgan_instance(
        N=1600, name='CTGAN_Full', pilot_condition=False, rare_degrees_adjust=True, degree_qual_type='Consistent',
        ussf_sampling=True, include_ots=False, printing=True
):
    """
    This procedure takes in the specified number of cadets and then generates a representative problem
    instance using CTGAN that has been trained from a real class year of cadets
    :param pilot_condition: If we want to sample cadets according to pilot preferences
    (make this more representative)
    :param name: Name of the CTGAN model to import
    :param N: number of cadets
    :return: model fixed parameters
    """

    # Load in the model
    filepath = afccp.globals.paths["support"] + name + '.pkl'
    model = CTGANSynthesizer.load(filepath)

    # Load in AFSCs data
    if include_ots:
        filepath = afccp.globals.paths["support"] + 'data/afscs_data_ots.csv'
        socs = ['usafa', 'rotc', 'ots']
    else:
        filepath = afccp.globals.paths["support"] + 'data/afscs_data.csv'
        socs = ['usafa', 'rotc']
    afscs_data = afccp.globals.import_csv_data(filepath)

    # Initialize parameter dictionary (initially only AFSCs data)
    afscs = np.array(afscs_data['AFSC'])  # List of AFSCs
    p = {'afscs': afscs, 'M': len(afscs), 'acc_grp': np.array(afscs_data['Accessions Group']),
         'Deg Tiers': np.array(afscs_data.loc[:, 'Deg Tier 1': 'Deg Tier 4']), 'P': len(afscs),
         'J': np.arange(len(afscs)), 'num_util': 10, 'Qual Type': degree_qual_type, 'SOCs': socs,

         # Some cadet data needed too
         'N': N, 'cadets': np.arange(N), 'I': np.arange(N)}

    # Add AFSC features to parameters
    p = generate_afscs_data(p, afscs_data)

    # Generate the cadet data and add it to the parameters
    data = generate_cadet_data(p, model, pilot_condition, rare_degrees_adjust, ussf_sampling, printing=printing)
    p = initialize_cadet_parameters_in_dictionary(p, data)
    p = determine_cadet_preference_information(p, data)
    p = generate_afsc_eligibility_and_preferences_data(p, data)
    p = generate_rated_om_matrices(p)

    # Add an "*" to the list of AFSCs to be considered the "Unmatched AFSC"
    p["afscs"] = np.hstack((p["afscs"], "*"))

    # Return parameters
    return p


def generate_afscs_data(p, afscs_data):

    # Edit Deg Tiers null values to make string null
    p['Deg Tiers'][pd.isnull(p["Deg Tiers"])] = ''

    # Determine AFSCs by Accessions Group
    p['afscs_acc_grp'] = {}
    if 'acc_grp' in p:
        for acc_grp in ['Rated', 'USSF', 'NRL']:
            p['J^' + acc_grp] = np.where(p['acc_grp'] == acc_grp)[0]
            p['afscs_acc_grp'][acc_grp] = p['afscs'][p['J^' + acc_grp]]

    # Useful data elements to help us generate PGL targets
    socs_props = {soc: np.array(afscs_data[f'{soc.upper()} Proportion']) for soc in p['SOCs']}
    pgl_prop = np.array(afscs_data['PGL Proportion'])

    # Total targets needed to distribute
    if 'ots' in p['SOCs']:  # Many OTS candidates will not make it!
        total_targets = int(p['N'] * min(0.75, np.random.normal(0.72, 0.08)))
    else:
        total_targets = int(p['N'] * min(0.94, np.random.normal(0.92, 0.08)))

    # PGL targets
    p['pgl'] = np.zeros(p['M']).astype(int)
    for soc in p['SOCs']:
        p[f'{soc}_quota'] = np.zeros(p['M']).astype(int)
    for j in p['J']:

        # Create the PGL target by sampling from the PGL proportion triangular distribution
        p_min = max(0, 0.8 * pgl_prop[j])
        p_max = 1.2 * pgl_prop[j]
        prop = np.random.triangular(p_min, pgl_prop[j], p_max)
        p['pgl'][j] = int(max(1, prop * total_targets))

        # If this AFSC is a SOC-specific proportion
        if '_U' in p['afscs'][j] or '_R' in p['afscs'][j] or '_O' in p['afscs'][j]:
            calc_soc_props = {soc: socs_props[soc][j] for soc in p['SOCs']}
        else:
            # Draw a random sample from the proportions for each SOC for this AFSC
            calc_soc_props = {soc: max(0, np.random.normal(socs_props[soc][j], 0.04)) for soc in p['SOCs']}
            total = sum(calc_soc_props[soc] for soc in p['SOCs'])

            # Re-scale proportions so they sum to 1
            calc_soc_props = {soc: calc_soc_props[soc] / total for soc in p['SOCs']}

        # Create the SOC-specific targets
        if 'ots' in p['SOCs']:
            p['rotc_quota'][j] = int(calc_soc_props['rotc'] * p['pgl'][j])
            p['usafa_quota'][j] = int(calc_soc_props['usafa'] * p['pgl'][j])
            p['ots_quota'][j] = p['pgl'][j] - p['rotc_quota'][j] - p['usafa_quota'][j]
        else:

            p['rotc_quota'][j] = int(calc_soc_props['rotc'] * p['pgl'][j])
            p['usafa_quota'][j] = p['pgl'][j] - p['rotc_quota'][j]

    # Initialize the other pieces of information here
    for param in ['quota_e', 'quota_d', 'quota_min']:
        p[param] = p['pgl']
    p['quota_max'] = np.around(p['pgl'] * (1 + 0.4 + np.random.rand(p['M']) * 0.6))

    # Force Rated/USSF AFSCs to have maximum/minimum equal to PGL
    for j in p['J']:
        if j not in p['J^NRL']:
            p['quota_max'][j] = p['pgl'][j]


    return p


def generate_cadet_data(p, model, pilot_condition, rare_degrees_adjust=True, ussf_sampling=True, printing=True):

    # Sample the cadets to fill the AFSCs with rare degrees
    if rare_degrees_adjust:
        if printing:
            print('Sampling cadets with rare degrees...')
        data_rare_degrees = sample_rare_afscs_degrees(model=model, p=p)
    else:
        data_rare_degrees = pd.DataFrame()  # Empty dataframe (don't care about it)

    # Sample USSF interested cadets
    if ussf_sampling:
        if printing:
            print('\nSampling cadets with USSF preferences...')
        data_ussf_sampling = sample_ussf_cadets_condition(p, model)
    else:
        data_ussf_sampling = pd.DataFrame()  # Empty dataframe (don't care about it)

    # Sample the majority of cadets
    n_generated = len(data_rare_degrees) + len(data_ussf_sampling)
    if pilot_condition:
        if printing:
            print('\nSampling remaining cadets (using pilot sampling method)...')
        data_all_else = sample_cadet_data_with_pilot_condition(N=p['N'] - n_generated, model=model, p=p)
    else:
        if printing:
            print('Sampling remaining cadets...')
        data_all_else = model.sample(p['N'] - n_generated)

    # Combine majority cadets and rare degree cadets
    data = pd.concat((data_rare_degrees, data_ussf_sampling, data_all_else), ignore_index=True)

    # Pick N*40% random cadets to swap over to OTS
    if 'ots' in p['SOCs']:

        random_idx = np.random.choice(data.index, size=int(p['N'] * 0.40), replace=False)
        data.loc[random_idx, 'SOC'] = 'OTS'

    # Break up USSF and 11XX AFSC by SOC
    for afsc in ['USSF', '11XX']:
        for col in ['Cadet', 'AFSC']:
            for soc in p['SOCs']:
                soc = soc.upper()
                data[f'{afsc}_{soc[0]}_{col}'] = 0
                data.loc[data['SOC'] == soc, f'{afsc}_{soc[0]}_{col}'] = data.loc[data['SOC'] == soc, f'{afsc}_{col}']

    return data


class KDESampler:
    def __init__(self, data=None, kde=None):
        if kde is not None:
            self.kde = kde
        else:
            self.kde = gaussian_kde(data, bw_method='scott')

    def sample(self, n):
        return np.clip(self.kde.resample(n).flatten(), 0, 1)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.kde, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            kde = pickle.load(f)
        return KDESampler(kde=kde)


def safe_round(data, decimals: int = 0, axis: int = -1):
    """
    Round values while preserving the sum along a given axis.

    This function rounds `data` to `decimals` decimal places but adjusts a minimal
    subset of elements so that the rounded values sum to the same (rounded) total
    as the original, slice‑by‑slice along `axis`. It does this by distributing the
    leftover rounding “units” to the entries whose fractional parts are most
    favorable (largest magnitude residuals with the correct sign), using a stable
    tie‑break so results are deterministic.

    Parameters
    ----------
    data : numpy.ndarray
        Input array to round. Must be numeric. (Other array‑likes are coerced;
        behavior is only guaranteed for NumPy arrays.)
    decimals : int, optional
        Number of decimal places to keep (default 0).
    axis : int, optional
        Axis along which to preserve the slice sums (default -1). Each 1D slice
        along this axis will have its rounded sum equal to the original sum
        rounded to `decimals`.

    Returns
    -------
    numpy.ndarray or same type as `data` when feasible
        Rounded array with the same shape as `data`. If `data` is a NumPy array,
        a NumPy array is returned. For some other types, the function attempts to
        reconstruct the input type after rounding.

    Notes
    -----
    - Let `S = sum(data, axis)` and `S_r = round(S, decimals)`. The output `y`
      satisfies `sum(y, axis) == S_r` exactly (up to floating‑point representation).
    - Within each slice, the adjustment is minimal in the sense that only the
      elements with the largest compatible residuals are modified by ± one unit
      in the scaled space (10**decimals).
    - Time complexity is `O(n log n)` per slice due to sorting; memory usage is
      linear in the slice size.
    - This procedure does not enforce monotonicity or ordering of values.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.24, 0.24, 0.24, 0.24, 0.04])
    >>> x.sum(), round(x.sum(), 2)
    (1.0, 1.0)
    >>> y = safe_round(x, decimals=1, axis=0)
    >>> y
    array([0.2, 0.2, 0.2, 0.2, 0.2])
    >>> y.sum()
    1.0

    >>> X = np.array([[0.333, 0.333, 0.334],
    ...               [0.125, 0.125, 0.750]])
    >>> Y = safe_round(X, decimals=2, axis=1)
    >>> Y
    array([[0.33, 0.33, 0.34],
           [0.12, 0.13, 0.75]])
    >>> Y.sum(axis=1)
    array([1.  , 1.  ])
    """
    data_type = type(data)
    constructor = {}

    # 1) Scale by 10^decimals
    scale = 10.0 ** decimals
    scaled = data * scale

    # 2) Naively round each element to the nearest integer
    rounded = np.rint(scaled)

    # 3) Compute how many integer "units" the sum *should* have in each slice
    sum_rounded = np.sum(rounded, axis=axis, keepdims=True)
    sum_desired = np.rint(np.sum(scaled, axis=axis, keepdims=True))
    difference = sum_desired - sum_rounded

    n = data.shape[axis]
    leftover_div = np.floor_divide(difference, n)
    leftover_mod = difference - leftover_div * n
    rounded += leftover_div

    # 5) Select elements to tweak
    difference = scaled - rounded
    leftover_sign = np.sign(leftover_mod)
    difference_sign = np.sign(difference)
    candidate_mask = (difference_sign == leftover_sign) & (difference_sign != 0)
    sort_key = np.where(candidate_mask, -np.abs(difference), np.inf)
    sorted_idx = np.argsort(sort_key, axis=axis, kind='stable')

    ranks = np.empty_like(sorted_idx)
    shape_for_r = [1] * data.ndim
    shape_for_r[axis] = n
    r_array = np.arange(n, dtype=sorted_idx.dtype).reshape(shape_for_r)
    np.put_along_axis(ranks, sorted_idx, r_array, axis=axis)

    leftover_mod_int = np.abs(leftover_mod).astype(int)
    choose_mask = ranks < leftover_mod_int
    rounded += leftover_sign * choose_mask

    result = rounded / scale

    if data_type is np.ndarray:
        return result

    return data_type(result.squeeze(), **constructor)


def sample_rare_afscs_degrees(model, p):

    # Load in rare AFSCs degree data
    deg_df = pd.read_csv('support/data/afscs_rare_eligibility.csv').set_index(['AFSC', 'Degree'])
    afscs_rare_eligible = np.array(deg_df.index.get_level_values('AFSC').unique())
    afsc_util_samplers = {}
    cadet_util_samplers = {}
    for afsc in afscs_rare_eligible:
        afsc_util_samplers[afsc] = KDESampler.load(f"support/kde_samplers/{afsc}_AFSC_KDESampler.pkl")
        cadet_util_samplers[afsc] = KDESampler.load(f"support/kde_samplers/{afsc}_Cadet_KDESampler.pkl")

    # Determine how many cadets to generate with these degrees (after AFSC generated data)
    total_gen = 0
    afsc_cip_conditions, afsc_cip_data = {}, {}
    for afsc in afscs_rare_eligible:
        afsc_cip_conditions[afsc], afsc_cip_data[afsc] = {}, {}
        j = np.where(p['afscs'] == afsc)[0][0]
        val = p['pgl'][j]
        if afsc == '62EXE':  # We struggle to fill this quota!!
            val = val / 2
        num_gen = np.ceil(max(val * 1.4, val + 3))
        proportions = deg_df.loc[afsc]['Proportion']
        cip_data = pd.Series(safe_round(np.array(proportions) * num_gen), index=proportions.index)
        cip_data = cip_data[cip_data > 0]
        for cip, count in cip_data.items():
            afsc_cip_data[afsc][cip] = int(count)
            afsc_cip_conditions[afsc][cip] = Condition(num_rows=int(count), column_values={"CIP1": cip})
            total_gen += count

    # Generate cadets dataframe
    data = pd.DataFrame()
    i = 0
    for afsc in afscs_rare_eligible:
        for cip, count in afsc_cip_data[afsc].items():
            print(f'{afsc} {cip}: {count}...')
            try:
                df_gen = model.sample_from_conditions([afsc_cip_conditions[afsc][cip]])
                data = pd.concat((data, df_gen), ignore_index=True)
                i += count
                print(f'\n{afsc} {cip}: ({int(i)}/{int(total_gen)}) {round((i / total_gen) * 100, 2)}% complete.')
            except:
                df_gen = model.sample(count)
                data = pd.concat((data, df_gen), ignore_index=True)
                i += count
                print(f'\n{afsc} {cip}: ({int(i)}/{int(total_gen)}) {round((i / total_gen) * 100, 2)}% complete. '
                      f'However, we failed to generate this AFSC-cip combination after many attempts.'
                      f' Filled with {count} random cadet(s).')

    # Modify the utilities for the cadet/AFSC pairs
    i = 0
    for afsc in afscs_rare_eligible:
        for cip, count in afsc_cip_data[afsc].items():
            data.loc[i:i + count - 1, f'{afsc}_Cadet'] = cadet_util_samplers[afsc].sample(count)
            data.loc[i:i + count - 1, f'{afsc}_AFSC'] = afsc_util_samplers[afsc].sample(count)
            i += count
    return data


def sample_cadet_data_with_pilot_condition(N, model, p):

    # Split up the number of ROTC/USAFA cadets
    N_usafa = round(np.random.triangular(0.25, 0.33, 0.4) * N)
    N_rotc = N - N_usafa

    # Pilot is by far the #1 desired career field, let's make sure this is represented here

    N_usafa_pilots = round(np.random.triangular(0.35, 0.4, 0.43) * N_usafa)
    N_usafa_generic = N_usafa - N_usafa_pilots
    N_rotc_pilots = round(np.random.triangular(0.27, 0.3, 0.33) * N_rotc)
    N_rotc_generic = N_rotc - N_rotc_pilots

    # Condition the data generated to produce the right composition of pilot first choice preferences
    usafa_pilot_first_choice = Condition(num_rows=N_usafa_pilots, column_values={'SOC': 'USAFA', '11XX_Cadet': 1})
    usafa_generic_cadets = Condition(num_rows=N_usafa_generic, column_values={'SOC': 'USAFA'})
    rotc_pilot_first_choice = Condition(num_rows=N_rotc_pilots, column_values={'SOC': 'ROTC', '11XX_Cadet': 1})
    rotc_generic_cadets = Condition(num_rows=N_rotc_generic, column_values={'SOC': 'ROTC'})

    # Sample data  (Sampling from conditions may take too long!)
    data = model.sample_from_conditions(
        conditions=[usafa_pilot_first_choice, usafa_generic_cadets, rotc_pilot_first_choice, rotc_generic_cadets])

    return data


def sample_ussf_cadets_condition(p, model):

    conditions = []
    for soc in p['SOCs']:  # This only works if "USSF" is the only "USSF" AFSC....confusing ;)
        afsc = f'USSF_{soc[0].upper()}'
        j = np.where(p['afscs'] == afsc)[0][0]
        soc_targets = p[f'{soc.lower()}_quota'][j]
        N = int(0.9 * soc_targets)  # We're only going to do this for 90% of the targets. The other 10% will be
        # made up from other cadet sampling methods
        conditions.append(Condition(num_rows=N,
                                    column_values={'SOC': soc.upper(), 'USSF_Cadet': 1, 'USSF Vol': 1}))

    # Sample data
    data = model.sample_from_conditions(conditions=conditions)

    return data


def initialize_cadet_parameters_in_dictionary(p, data):

    # Add cadet data features to parameter dictionary
    p['merit'] = np.array(data['Merit'])
    for soc in p['SOCs']:
        p[soc] = np.array(data['SOC'] == soc.upper()) * 1
    p['cip1'] = np.array(data['CIP1']).astype(str)
    p['cip2'] = np.array(data['CIP2']).astype(str)

    # Clean up degree columns (remove the leading "c" I put there if it's there)
    for i in p['I']:
        if p['cip1'][i][0] == 'c':
            p['cip1'][i] = p['cip1'][i][1:]
        if p['cip2'][i][0] == 'c':
            p['cip2'][i] = p['cip2'][i][1:]

    # Create "SOC" variable
    p['soc'] = np.array(data['SOC'])

    # Fix percentiles for USAFA and ROTC
    re_scaled_om = p['merit']
    for soc in p['SOCs']:
        indices = np.where(p[soc])[0]  # Indices of these SOC-specific cadets
        percentiles = p['merit'][indices]  # The percentiles of these cadets
        N = len(percentiles)  # Number of cadets from this SOC
        sorted_indices = np.argsort(percentiles)[::-1]  # Sort these percentiles (descending)
        new_percentiles = (np.arange(N)) / (N - 1)  # New percentiles we want to replace these with
        magic_indices = np.argsort(sorted_indices)  # Indices that let us put the new percentiles in right place
        new_percentiles = new_percentiles[magic_indices]  # Put the new percentiles back in the right place
        np.put(re_scaled_om, indices, new_percentiles)  # Put these new percentiles in combined SOC OM spot

    # Replace merit
    p['merit'] = re_scaled_om

    return p


def determine_cadet_preference_information(p, data):

    c_pref_cols = [f'{afsc}_Cadet' for afsc in p['afscs']]
    util_original = np.around(np.array(data[c_pref_cols]), 2)

    # Initialize cadet preference information
    p['c_utilities'] = np.zeros((p['N'], 10))
    p['c_preferences'] = np.array([[' ' * 6 for _ in range(p['M'])] for _ in range(p['N'])])
    p['cadet_preferences'] = {}
    p['c_pref_matrix'] = np.zeros((p['N'], p['M'])).astype(int)
    p['utility'] = np.zeros((p['N'], p['M']))

    # Loop through each cadet to tweak their preferences
    for i in p['cadets']:

        # Fix rated/USSF volunteer situation
        for acc_grp in ['Rated', 'USSF']:
            if data.loc[i, f'{acc_grp} Vol']:
                if np.max(util_original[i, p[f'J^{acc_grp}']]) < 0.6:
                    util_original[i, p[f'J^{acc_grp}']] = 0
                    data.loc[i, f'{acc_grp} Vol'] = False
            else:  # Not a volunteer

                # We have a higher preference for these kinds of AFSCs
                if np.max(util_original[i, p[f'J^{acc_grp}']]) >= 0.6:
                    data.loc[i, f'{acc_grp} Vol'] = True  # Make them a volunteer now

        # Was this the last choice AFSC? Remove from our lists
        ordered_list = np.argsort(util_original[i])[::-1]
        last_choice = data.loc[i, 'Last Choice']
        if last_choice in p['afscs']:
            j = np.where(p['afscs'] == last_choice)[0][0]
            ordered_list = ordered_list[ordered_list != j]

        # Add the "2nd least desired AFSC" to list
        second_last_choice = data.loc[i, '2nd-Last Choice']
        bottom = []
        if second_last_choice in p['afscs'] and second_last_choice != last_choice:  # Check if valid and not in bottom choices
            j = np.where(p['afscs'] == second_last_choice)[0][0]  # Get index of AFSC
            ordered_list = ordered_list[ordered_list != j]  # Remove index from preferences
            bottom.append(second_last_choice)  # Add it to the list of bottom choices

        # If it's a valid AFSC that isn't already in the bottom choices
        third_last_choice = data.loc[i, '3rd-Last Choice']  # Add the "3rd least desired AFSC" to list
        if third_last_choice in p['afscs'] and third_last_choice not in [last_choice, second_last_choice]:
            j = np.where(p['afscs'] == third_last_choice)[0][0]  # Get index of AFSC
            ordered_list = ordered_list[
                ordered_list != j]  # Reordered_list = np.argsort(util_original[i])[::-1]move index from preferences
            bottom.append(third_last_choice)  # Add it to the list of bottom choices

        # If we have an AFSC in the bottom choices, but NOT the LAST choice, move one to the last choice
        if len(bottom) > 0 and pd.isnull(last_choice):
            afsc = bottom.pop(0)
            data.loc[i, 'Last Choice'] = afsc
        data.loc[i, 'Second Least Desired AFSCs'] = ', '.join(bottom)  # Put it in the dataframe

        # Save cadet preference information
        num_pref = 10 if np.random.rand() > 0.1 else int(np.random.triangular(11, 15, 26))
        p['c_utilities'][i] = util_original[i, ordered_list[:10]]
        p['cadet_preferences'][i] = ordered_list[:num_pref]
        p['c_preferences'][i, :num_pref] = p['afscs'][p['cadet_preferences'][i]]
        p['c_pref_matrix'][i, p['cadet_preferences'][i]] = np.arange(1, len(p['cadet_preferences'][i]) + 1)
        p['utility'][i, p['cadet_preferences'][i][:10]] = p['c_utilities'][i]

    # Save least desired AFSC info
    p['second_to_last_afscs'] = np.array(data['Second Least Desired AFSCs'])
    p['last_afsc'] = np.array(data['Last Choice'])

    # Save selected pref information
    p['c_selected_matrix'] = (p['c_pref_matrix'] > 0) * 1
    return p


def generate_afsc_eligibility_and_preferences_data(p, data):

    # Get qual matrix information
    p = afccp.data.adjustments.gather_degree_tier_qual_matrix(cadets_df=None, parameters=p)

    # Get the qual matrix to know what people are eligible for
    ineligible = (np.core.defchararray.find(p['qual'], "I") != -1) * 1
    eligible = (ineligible == 0) * 1
    I_E = [np.where(eligible[:, j])[0] for j in p['J']]  # set of cadets that are eligible for AFSC j

    # Modify AFSC utilities based on eligibility
    a_pref_cols = [f'{afsc}_AFSC' for afsc in p['afscs']]
    p['afsc_utility'] = np.around(np.array(data[a_pref_cols]), 2)
    for acc_grp in ['Rated', 'USSF']:
        for j in p['J^' + acc_grp]:
            volunteer_col = np.array(data[f'{acc_grp} Vol'])
            volunteers = np.where(volunteer_col)[0]
            not_volunteers = np.where(volunteer_col == False)[0]
            ranked = np.where(p['afsc_utility'][:, j] > 0)[0]
            unranked = np.where(p['afsc_utility'][:, j] == 0)[0]

            # Fill in utility values with OM for folks who don't have an AFSC score
            volunteer_unranked = np.intersect1d(volunteers, unranked)
            p['afsc_utility'][volunteer_unranked, j] = p['merit'][volunteer_unranked]

            # If the cadet didn't actually volunteer, they should have utility of 0
            non_volunteer_ranked = np.intersect1d(not_volunteers, ranked)
            p['afsc_utility'][non_volunteer_ranked, j] = 0

    # Remove cadets from this AFSC's preferences if the cadet is not eligible
    for j in p['J^NRL']:
        # Get appropriate sets of cadets
        eligible_cadets = I_E[j]
        ineligible_cadets = np.where(ineligible[:, j])[0]
        ranked_cadets = np.where(p['afsc_utility'][:, j] > 0)[0]
        unranked_cadets = np.where(p['afsc_utility'][:, j] == 0)[0]

        # Fill in utility values with OM for eligible folks who don't have an AFSC score
        eligible_unranked = np.intersect1d(eligible_cadets, unranked_cadets)
        p['afsc_utility'][eligible_unranked, j] = p['merit'][eligible_unranked]

        # If the cadet isn't actually eligible, they should have utility of 0
        ineligible_ranked = np.intersect1d(ineligible_cadets, ranked_cadets)
        p['afsc_utility'][ineligible_ranked, j] = 0

    # Collect AFSC preference information
    p['afsc_preferences'] = {}
    p['a_pref_matrix'] = np.zeros((p['N'], p['M'])).astype(int)
    for j in p['J']:
        # Sort the utilities to get the preference list
        utilities = p["afsc_utility"][:, j]
        ineligible_indices = np.where(utilities == 0)[0]
        sorted_indices = np.argsort(utilities)[::-1][:p['N'] - len(ineligible_indices)]
        p['afsc_preferences'][j] = sorted_indices

        # Since 'afsc_preferences' is an array of AFSC indices, we can do this
        p['a_pref_matrix'][p['afsc_preferences'][j], j] = np.arange(1, len(p['afsc_preferences'][j]) + 1)

    return p


def generate_rated_om_matrices(p):

    # Needed information for rated OM matrices
    dataset_dict = {soc: f'{soc[0]}r_om_matrix' for soc in p['SOCs']}
    cadets_dict = {soc: f'{soc[0]}r_om_cadets' for soc in p['SOCs']}
    p["Rated Cadets"] = {}

    # Create rated OM matrices for each SOC
    for soc in p['SOCs']:

        # Rated AFSCs for this SOC
        if soc == 'rotc':
            rated_J_soc = np.array([j for j in p['J^Rated'] if '_U' not in p['afscs'][j] and '_O' not in p['afscs'][j]])
        elif soc == 'usafa':
            rated_J_soc = np.array([j for j in p['J^Rated'] if '_R' not in p['afscs'][j] and '_O' not in p['afscs'][j]])
        else:  # ots
            rated_J_soc = np.array([j for j in p['J^Rated'] if '_U' not in p['afscs'][j] and '_R' not in p['afscs'][j]])

        # Cadets from this SOC
        soc_cadets = np.where(p[soc])[0]

        # Determine which cadets are eligible for at least one rated AFSC
        p["Rated Cadets"][soc] = np.array([i for i in soc_cadets if np.sum(p['c_pref_matrix'][i, rated_J_soc]) > 0])
        p[cadets_dict[soc]] = p["Rated Cadets"][soc]

        # Initialize OM dataset
        p[dataset_dict[soc]] = np.zeros([len(p["Rated Cadets"][soc]), len(rated_J_soc)])

        # Create OM dataset
        for col, j in enumerate(rated_J_soc):

            # Get the maximum rank someone had
            max_rank = np.max(p['a_pref_matrix'][p["Rated Cadets"][soc], j])

            # Loop through each cadet to convert rank to percentile
            for row, i in enumerate(p["Rated Cadets"][soc]):
                rank = p['a_pref_matrix'][i, j]
                if rank == 0:
                    p[dataset_dict[soc]][row, col] = 0
                else:
                    p[dataset_dict[soc]][row, col] = (max_rank - rank + 1) / max_rank
    return p


# ________________________________AUGMENTED SUPPORT FOR BASE/TRAINING ASSIGNMENTS_______________________________________
def augment_instance_with_base_training_components(parameters, printing=True):

    # Shorthand
    p = parameters

    if printing:
        print('Augmenting CTGAN data with extra base/training components...')

    # Get the baseline starting date (January 1st of the year we're classifying)
    next_year = datetime.datetime.now().year + 1
    p['baseline_date'] = datetime.date(next_year, 1, 1)

    # Generate start preferences, weights, thresholds
    with open(afccp.globals.paths["support"] + 'prob_samplers.pkl', "rb") as f:
        prob_samplers = pickle.load(f)  # Load in probability samplers
    p = generate_cadet_extra_component_preferences(p, prob_samplers)

    # Generate start dates
    p = generate_start_dates(p, next_year)

    # Delineate AFSCs based on IST/base status and generate courses data
    p, base_afscs = generate_courses_afsc_status_data(p, next_year)

    # Load in data for base-billet generation
    probability_df = pd.read_csv(afccp.globals.paths["support"] + 'probability_df.csv', index_col='ilc_title')
    valid_billets_df = pd.read_csv(afccp.globals.paths["support"] + 'valid_billets_df.csv', index_col='ilc_title')

    # Load CTGAN model
    filepath = afccp.globals.paths["support"] + 'Base_Preferences_CTGAN' + '.pkl'
    ctgan_model = CTGANSynthesizer.load(filepath)

    # Generate bases and preference data
    p = generate_bases_data(p, base_afscs, probability_df, valid_billets_df, prob_samplers, ctgan_model)

    # Generate UPT preferences
    p = generate_upt_preferences_df(p)
    return p


def generate_cadet_extra_component_preferences(p, prob_samplers):
    N = p['N']

    # Initialize arrays
    p['base_threshold'] = np.full(N, 100, dtype=int)
    p['training_threshold'] = np.full(N, 100, dtype=int)
    p['weight_afsc'] = np.full(N, 100.0)
    p['weight_base'] = np.zeros(N)
    p['weight_course'] = np.zeros(N)

    # Sample start preferences
    p['training_preferences'] = np.random.choice(
        prob_samplers['Start Pref'].index,
        size=p['N'],
        p=prob_samplers['Start Pref'].values
    )

    # Should've been consistent between IST/Course terms
    threshold_map = {'base_threshold': 'Base', 'training_threshold': 'IST'}
    for threshold_key, sampler_name in threshold_map.items():
        affect_sample = np.random.choice(
            prob_samplers[f'{sampler_name} Affect AFSC'].index,
            size=N,
            p=prob_samplers[f'{sampler_name} Affect AFSC'].values
        )
        yes_mask = affect_sample == 'Yes'
        num_yes = yes_mask.sum()
        if num_yes > 0:
            sampled_thresholds = np.random.choice(
                prob_samplers[f'{sampler_name} Affect AFSC Num'].index,
                size=num_yes,
                p=prob_samplers[f'{sampler_name} Affect AFSC Num'].values
            )
            p[threshold_key][yes_mask] = sampled_thresholds

    # Fix course thresholds
    none_mask = p['training_preferences'] == 'None'
    p['training_threshold'][none_mask] = 100

    # Generate AFSC weight
    mask = (p['base_threshold'] != 100) | (p['training_threshold'] != 100)
    num = mask.sum()
    if num > 0:
        p['weight_afsc'][mask] = sample_one_inflated_beta(prob_samplers['AFSC Weight'], num)

    # Generate Base weight
    mask = p['base_threshold'] != 100
    num = mask.sum()
    if num > 0:
        p['weight_base'][mask] = sample_one_inflated_beta(prob_samplers['Base Weight'], num)

    # Generate Course weight
    mask = p['training_threshold'] != 100
    num = mask.sum()
    if num > 0:
        p['weight_course'][mask] = sample_one_inflated_beta(prob_samplers['Course Weight'], num)

    # Normalize weights so they sum to 1
    total = (p['weight_afsc'] + p['weight_base'] + p['weight_course'])
    nonzero_mask = total > 0  # Avoid divide-by-zero
    p['weight_afsc'][nonzero_mask] /= total[nonzero_mask]
    p['weight_base'][nonzero_mask] /= total[nonzero_mask]
    p['weight_course'][nonzero_mask] /= total[nonzero_mask]

    # If total == 0 (no thresholds active), assign all weight to AFSC
    zero_mask = ~nonzero_mask
    p['weight_afsc'][zero_mask] = 1.0
    p['weight_base'][zero_mask] = 0.0
    p['weight_course'][zero_mask] = 0.0
    return p


def generate_start_dates(p, next_year):

    # Generate training start dates for each cadet
    p['training_start'] = []

    for i in range(p['N']):

        # If this cadet is a USAFA cadet
        if p['usafa'][i]:

            # Make it May 28th of this year
            p['training_start'].append(datetime.date(next_year, 5, 28))

        else:

            # 92% Spring Graduates
            if np.random.rand() < 0.92:

                # Spring window: April 15 – June 15
                base_date = datetime.date(next_year, 4, 15)

                # Triangular gives clustering toward May
                offset_days = int(np.random.triangular(0, 30, 60))
                dt = base_date + datetime.timedelta(offset_days)
                p['training_start'].append(dt)

            # 8% Fall Graduates (Oct–Dec of previous calendar year)
            else:
                fall_year = next_year - 1

                # Fall window: October 15 – December 15
                base_date = datetime.date(fall_year, 10, 15)

                # Cluster toward November
                offset_days = int(np.random.triangular(0, 30, 60))
                dt = base_date + datetime.timedelta(offset_days)
                p['training_start'].append(dt)

    p['training_start'] = np.array(p['training_start'])
    return p


def generate_bases_data(p, base_afscs, probability_df, valid_billets_df, prob_samplers, ctgan_model):

    # Translator to standardize base names
    base_translation_df = pd.read_csv(afccp.globals.paths["support"] + 'base_translation.csv')
    d = base_translation_df.loc[base_translation_df['Match'] == 1]
    translator = {d.loc[idx, 'Faces Data']: d.loc[idx, 'Spaces Data'] for idx in d.index}

    # Generate base preferences data
    synthetic_rankings = sample_and_convert_to_rankings(
        ctgan_model,
        prob_samplers,
        n_samples=p['N']
    )
    base_pref_columns = [col for col in synthetic_rankings if col != 'num_base_preferences']
    base_pref_df = synthetic_rankings[base_pref_columns]
    base_pref_df.index.name = 'Cadet'

    # Standardize base preference names
    base_pref_df = base_pref_df.rename(columns=translator)

    # Fill out base preferences as 0 for bases that were in spaces but not faces
    spaces_only = list(base_translation_df.loc[(base_translation_df['Unique'] == 1) &
                                               ~pd.isnull(base_translation_df['Spaces Data'])]['Spaces Data'])
    for col in spaces_only:
        base_pref_df[col] = 0
    base_pref_df.sort_index(axis=1, inplace=True)

    # Generate billets for each of these AFSCs!
    billets_to_generate = {}
    for afsc in base_afscs:
        j = np.where(p['afscs'] == afsc)[0][0]
        if p['pgl'][j] <= 10:
            scalar = 4
        elif 10 < p['pgl'][j] <= 50:
            scalar = 3
        elif 50 < p['pgl'][j] <= 100:
            scalar = 2
        elif 100 < p['pgl'][j] <= 200:
            scalar = 1.5
        else:
            scalar = 1.2
        billets_to_generate[afsc] = int(np.ceil(p['pgl'][j] * scalar))
    assignment_dict = generate_billets_by_afsc(
        billets_by_afsc=billets_to_generate,
        probability_df=probability_df,
        valid_billets_df=valid_billets_df,
    )

    # Add in other AFSC fixed bases billets
    others = {'12XX': {'PENSACOLA NAS': 1000}, '13B': {'PENSACOLA NAS': 1000}, '13N': {'VANDENBERG': 1000},
              '14N': {'GOODFELLOW AFB': 1000}, '17X': {'KEESLER': 1000}, '18X': {'JBSA RANDOLPH': 1000}, }
    for afsc, items in others.items():
        assignment_dict[afsc] = items

    # Collect all possible bases from every source
    pref_bases = set(base_pref_df.columns)
    prob_bases = set(probability_df.index)
    valid_bases = set(valid_billets_df.index)
    others_bases = set()
    for afsc_dict in assignment_dict.values():
        others_bases.update(afsc_dict.keys())
    all_bases = sorted(pref_bases | prob_bases | valid_bases | others_bases)
    p['bases'] = np.array(all_bases)
    p['S'] = len(p['bases'])

    base_pref_df = base_pref_df.reindex(columns=p['bases'], fill_value=0)
    p['b_pref_matrix'] = base_pref_df.to_numpy()

    # Create the "Bases" dataframe file and then load it into the arrays
    base_afscs_all = np.sort(list(assignment_dict.keys()))
    columns = []
    for kind in ['Min', 'Max']:
        for afsc in base_afscs_all:
            columns.append(f'{afsc} {kind}')
    base_billets_df = pd.DataFrame(index=p['bases'], columns=columns, data=0)
    for afsc, base_billets_dict in assignment_dict.items():
        for base, count in base_billets_dict.items():
            base_billets_df.loc[base, f'{afsc} Max'] = count

    # Loop through each base AFSC to load the arrays
    p['base_min'] = np.zeros((p['S'], p['M']))
    p['base_max'] = np.zeros((p['S'], p['M']))
    for j, afsc in enumerate(p['afscs']):
        if afsc in base_afscs_all:
            p['base_min'][:, j] = base_billets_df[f'{afsc} Min']
            p['base_max'][:, j] = base_billets_df[f'{afsc} Max']
    return p


def generate_courses_afsc_status_data(p, next_year):

    # Load AFSC dictionary of data
    if 'ots' in p['SOCs']:
        filename = 'afscs_status_dict_ots.pkl'
    else:
        filename = 'afscs_status_dict.pkl'
    with open(afccp.globals.paths["support"] + filename, "rb") as f:
        afscs_status_dict = pickle.load(f)  # Load in probability samplers

    # Load courses data
    with open(afccp.globals.paths["support"] + 'learned_course_data.pkl', "rb") as f:
        learned_course_data = pickle.load(f)  # Load in course generation data

    # Load extra by-AFSC information
    p['tau'] = np.array([t for _, t in afscs_status_dict['training_afscs_dict'].items()])
    p['base_ist_status'] = np.array([t for _, t in afscs_status_dict['base_ist_status'].items()])

    # List of AFSCs to assign initial duty stations for new accessions
    base_afscs = afscs_status_dict['base_afscs']

    # Training AFSCs
    p['T'] = afscs_status_dict['training_afscs']

    # Generate courses data
    df = pd.DataFrame()
    for t in p['T']:
        if t == '11XX':  # There's a glitch where sometimes we don't generate enough courses for specific UPT bases

            # This block ensures we have diversity of the courses represented across each UPT base
            iterating = True
            while iterating:
                new_df = generate_courses_for_fy(
                    learned_data=learned_course_data[t], fy=next_year, afsc=t, seat_scalar=1, n_fys=3)
                if len(new_df) > 70:
                    iterating = False
        else:
            new_df = generate_courses_for_fy(
                learned_data=learned_course_data[t], fy=next_year, afsc=t, seat_scalar=1, n_fys=3)
        df = pd.concat((df, new_df))
    df['Course'] = df['Training Class']
    df['Min'] = 0
    df['Max'] = df['Allocated']
    df = df[['AFSC', 'Course', 'Start Date', 'Min', 'Max']]

    # Dictionary to translate parameter names to column names
    column_translation = {"Course": 'courses', 'Start Date': 'course_start', 'Min': 'course_min',
                          'Max': 'course_max'}

    # Get each parameter from the columns of this dataset
    for col, param in column_translation.items():
        p[param] = {}
        for t in p['T']:  # Loop through each training AFSC to extract its info
            p[param][t] = np.array(df.loc[df['AFSC'] == t][col])

    # Number of courses per training AFSC
    p['Q'] = {t: len(p['courses'][t]) for t in p['T']}
    p['num_courses_full'] = np.zeros(p['M'])
    for j in range(p['M']):
        t = p['tau'][j]  # tau: J -> T mapping!! tau(j) = t returns training AFSC 't' from regular AFSC 'j'
        if t in p['T']:
            p['num_courses_full'][j] = p['Q'][t]

    return p, base_afscs


def generate_upt_preferences_df(p):

    # Bias: Vance and Columbus more desirable
    bases = ["VANCE", "LAUGHLIN", "COLUMBUS"]
    first_choice_probs = {
        "VANCE": 0.40,
        "COLUMBUS": 0.40,
        "LAUGHLIN": 0.20
    }
    rows = []

    # UPT qualified cadets  (Had to modify this to make UPT prefs for all rated cadets- issue with pilot specific prefs)
    pilot_j = [j for j in p['J'] if p['base_ist_status'][j] == 'UPT Base & IST']
    cadets = [i for i in range(p['N']) if np.sum(p['c_pref_matrix'][i, p['J^Rated']]) > 0]

    # Generate preferences for cadets
    rng = np.random.default_rng()
    for i in cadets:
        # Select first choice with bias
        first = rng.choice(bases, p=[first_choice_probs[b] for b in bases])

        remaining = [b for b in bases if b != first]
        rng.shuffle(remaining)
        rankings = {
            first: 1,
            remaining[0]: 2,
            remaining[1]: 3
        }
        rows.append({
            "Cadet": i,
            "VANCE": rankings["VANCE"],
            "LAUGHLIN": rankings["LAUGHLIN"],
            "COLUMBUS": rankings["COLUMBUS"]
        })
    p['upt_preferences_df'] = pd.DataFrame(rows).sort_values("Cadet").reset_index(drop=True)
    return p


def sample_one_inflated_beta(params, n_samples):
    p_one = params["p_one"]
    a = params["a"]
    b = params["b"]

    # Decide which samples are exactly 1
    is_one = np.random.rand(n_samples) < p_one

    samples = np.empty(n_samples)

    # Assign 1s
    samples[is_one] = 1

    # Sample Beta for the rest
    n_beta = (~is_one).sum()
    samples[~is_one] = beta.rvs(a, b, size=n_beta)

    return samples * 100


def sample_and_convert_to_rankings(ctgan, prob_sampler, n_samples):
    """
    Sample base utilities from CTGAN, sample num_base_preferences from
    empirical distribution, and convert utilities into ranked preferences.
    """

    # -------------------------------------------------------
    # 1️⃣ Sample utilities from CTGAN
    # -------------------------------------------------------
    synthetic = ctgan.sample(num_rows=n_samples)

    # Identify base columns
    base_cols = [c for c in synthetic.columns if c != "num_base_preferences"]

    # Keep only utility columns
    utilities_df = synthetic[base_cols].copy()

    # Clip utilities to valid range
    utilities_df = utilities_df.clip(lower=0, upper=1)

    # -------------------------------------------------------
    # 2️⃣ Sample num_base_preferences from empirical distribution
    # -------------------------------------------------------
    dist = prob_sampler['num_base_preferences']

    sampled_num_prefs = np.random.choice(
        dist.index,
        size=n_samples,
        p=dist.values
    )

    num_prefs = pd.Series(sampled_num_prefs, index=utilities_df.index)

    # -------------------------------------------------------
    # 3️⃣ Convert utilities → rankings
    # -------------------------------------------------------
    ranking_df = pd.DataFrame(
        0,
        index=utilities_df.index,
        columns=base_cols
    )
    ranking_df['num_base_preferences'] = num_prefs

    for idx in utilities_df.index:

        N = int(num_prefs.loc[idx])
        N = min(N, len(base_cols))  # safety guard

        if N <= 0:
            continue

        row_utilities = utilities_df.loc[idx]

        # Add tiny noise to avoid ties
        row_utilities = row_utilities + np.random.normal(
            0, 1e-6, size=len(row_utilities)
        )

        # Sort descending
        top_bases = row_utilities.sort_values(ascending=False).index[:N]

        # Assign ranks 1..N
        for rank, base in enumerate(top_bases, start=1):
            ranking_df.loc[idx, base] = rank

    return ranking_df


def number_to_alpha(n):
    """
    Convert 1 -> A, 2 -> B, ..., 26 -> Z, 27 -> AA, etc.
    """
    result = ""
    while n > 0:
        n -= 1
        result = chr(65 + (n % 26)) + result
        n //= 26
    return result


def generate_courses_for_fy(
    learned_data: Dict[str, Any],
    fy: int,
    afsc: str,
    *,
    seat_scalar: float = 1.0,
    n_fys: int = 1,
    random_state: Optional[int] = None,
    afscs_alpha: list = ['6XX', '64P']
) -> pd.DataFrame:
    """
    Generate synthetic course classes for one or more fiscal years.

    Parameters
    ----------
    learned_data : dict
        Output from learn_course_generation_data().
    fy : int
        First fiscal year to generate. Example: FY25 means 2024-10-01 through 2025-09-30.
    afsc : str
        AFSC to generate for.
    seat_scalar : float
        Scalar applied to generated seat counts. Default is 1.0 (100% of normal).
    n_fys : int
        Number of consecutive fiscal years to generate, starting with `fy`.
        Example:
        - n_fys=1 -> only FY25
        - n_fys=2 -> FY25 and FY26
    random_state : int | None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Generated classes with course, class id, start date, seats, and FY.
    """

    rng = np.random.default_rng(random_state)

    if learned_data.get("afsc") != afsc:
        raise ValueError("The supplied learned_data does not match the requested AFSC.")

    generated_rows = []

    for fy_i in range(fy, fy + n_fys):
        fy_start = pd.Timestamp(datetime.datetime(fy_i - 1, 10, 1))
        fy_end = pd.Timestamp(datetime.datetime(fy_i, 9, 30))

        for course, profile in learned_data["courses"].items():

            # -----------------------------
            # Number of classes to generate
            # -----------------------------
            hist_counts = profile["classes_per_fy_values"]
            if len(hist_counts) == 0:
                continue

            n_classes = int(rng.choice(hist_counts))
            if n_classes <= 0:
                continue

            # -----------------------------
            # Generate class start days
            # -----------------------------
            first_class_days = profile["first_class_days"]
            interarrival_days = profile["interarrival_days"]
            all_fy_days = profile["all_fy_days"]

            generated_days = []

            if len(first_class_days) > 0:
                first_day = int(rng.choice(first_class_days))
            elif len(all_fy_days) > 0:
                first_day = int(rng.choice(all_fy_days))
            else:
                first_day = int(rng.integers(0, 365))

            first_day = max(0, min(first_day, 364))
            generated_days.append(first_day)

            while len(generated_days) < n_classes:
                if len(interarrival_days) > 0:
                    gap = int(rng.choice(interarrival_days))
                else:
                    gap = 30

                next_day = generated_days[-1] + max(1, gap)

                if next_day > 364:
                    break

                generated_days.append(next_day)

            # fallback if too few classes got generated within FY
            if len(generated_days) < n_classes and len(all_fy_days) > 0:
                remaining = n_classes - len(generated_days)
                extra_days = rng.choice(all_fy_days, size=remaining, replace=True).tolist()
                generated_days.extend(extra_days)
                generated_days = sorted([int(x) for x in generated_days if x <= 364])[:n_classes]

            # Deduplicate / nudge collisions
            generated_days = sorted(generated_days)
            for i in range(1, len(generated_days)):
                if generated_days[i] <= generated_days[i - 1]:
                    generated_days[i] = generated_days[i - 1] + 1
            generated_days = [d for d in generated_days if d <= 364]

            # -----------------------------
            # Generate seats
            # -----------------------------
            seat_values = profile["seat_values"]
            if len(seat_values) == 0:
                seat_values = [1]

            generated_seats = []
            for _ in range(len(generated_days)):
                base_seats = float(rng.choice(seat_values))
                seats = int(round(base_seats * seat_scalar))
                seats = max(1, seats)
                generated_seats.append(seats)

            # -----------------------------
            # Build rows
            # -----------------------------
            for day_offset, seats in zip(generated_days, generated_seats):
                start_date = fy_start + pd.Timedelta(days=int(day_offset))

                if start_date < fy_start or start_date > fy_end:
                    continue

                generated_rows.append({
                    "AFSC": afsc,
                    "Training Course": course,
                    "Start Date": start_date,
                    "FY": fy_i,
                    "Allocated": seats,
                })

    out = pd.DataFrame(generated_rows)

    if not out.empty:
        out = out.sort_values(["FY", "Training Course", "Start Date"]).reset_index(drop=True)

        # Number classes sequentially within each FY + course
        out["class_num_within_fy"] = out.groupby(["FY", "Training Course"]).cumcount() + 1

        if afsc in afscs_alpha:
            out["Training Class"] = out.apply(
                lambda r: "{0}{1} {2}".format(
                    int(r["FY"]),
                    number_to_alpha(int(r["class_num_within_fy"])),
                    r["Training Course"]
                ),
                axis=1
            )
        else:
            out["Training Class"] = out.apply(
                lambda r: "{0}{1:03d} {2}".format(
                    int(r["FY"]),
                    int(r["class_num_within_fy"]),
                    r["Training Course"]
                ),
                axis=1
            )

        out = out[["AFSC", "Training Course", "Training Class", "Start Date", "FY", "Allocated"]]

    return out


def generate_billets_by_afsc(
    billets_by_afsc: Dict[str, int],
    probability_df: pd.DataFrame,
    valid_billets_df: pd.DataFrame,
    *,
    total_col: str = "TOTAL_BILLETS",
    random_state: Optional[int] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Allocate (sample) open billets across bases for each AFSC, respecting per-base capacity.

    Parameters
    ----------
    billets_by_afsc : dict[str, int]
        {afsc: n_open_billets_to_generate}
    probability_df : pd.DataFrame
        index = bases, columns include AFSCs (probabilities) and optionally TOTAL_BILLETS.
        Each AFSC column should sum to 1 (after your filtering + renormalization).
    valid_billets_df : pd.DataFrame
        index = bases, columns include AFSCs (capacities) and optionally TOTAL_BILLETS.
        Values are max billets that can be assigned at that base for that AFSC.
    total_col : str
        Name of the total billets column to ignore, if present.
    random_state : int | None
        Random seed for reproducibility.

    Returns
    -------
    dict[str, dict[str, int]]
        {afsc: {base: count_assigned}}
    """
    rng = np.random.default_rng(random_state)

    # Align indices (bases) and drop TOTAL column if present in the AFSC column sets
    common_bases = probability_df.index.intersection(valid_billets_df.index)
    p_df = probability_df.loc[common_bases].copy()
    v_df = valid_billets_df.loc[common_bases].copy()

    # Determine AFSC columns present in each df
    p_cols = [c for c in p_df.columns if c != total_col]
    v_cols = [c for c in v_df.columns if c != total_col]

    out: dict[str, dict[str, int]] = {}

    for afsc, n_open in billets_by_afsc.items():
        if n_open <= 0:
            out[afsc] = {}
            continue

        if afsc not in p_cols or afsc not in v_cols:
            raise KeyError(f"AFSC '{afsc}' not found as a column in both probability_df and valid_billets_df.")

        probs = p_df[afsc].to_numpy(dtype=float)
        caps = v_df[afsc].to_numpy(dtype=int)

        # Only consider bases with capacity > 0
        mask = caps > 0
        if not np.any(mask):
            raise ValueError(f"AFSC '{afsc}' has zero capacity across all bases in valid_billets_df.")

        bases = p_df.index.to_numpy()[mask]
        probs = probs[mask]
        caps = caps[mask]

        total_cap = int(caps.sum())
        if n_open > total_cap:
            print(f"Requested {n_open} billets for AFSC '{afsc}', "
                f"but total valid capacity is {total_cap}. Will reduce to {total_cap}.")
        n_open = min(total_cap, n_open)

        # Normalize probs within the feasible set (and handle all-zero probs)
        if probs.sum() <= 0:
            probs = np.ones_like(probs, dtype=float)
        probs = probs / probs.sum()

        remaining = caps.copy()
        counts = np.zeros(len(bases), dtype=int)

        # Sequential sampling with capacity constraints
        for _ in range(n_open):
            avail = remaining > 0
            p_avail = probs.copy()
            p_avail[~avail] = 0.0

            if p_avail.sum() <= 0:
                # fallback uniform over available
                idxs = np.flatnonzero(avail)
                choice = rng.choice(idxs)
            else:
                p_avail = p_avail / p_avail.sum()
                choice = rng.choice(len(bases), p=p_avail)

            counts[choice] += 1
            remaining[choice] -= 1

        out[afsc] = {str(bases[i]): int(counts[i]) for i in range(len(bases)) if counts[i] > 0}

    return out