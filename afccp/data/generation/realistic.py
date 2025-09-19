"""
`afccp.data.generation.realistic`
=================================

Generates **realistic** AFCCP problem instances by learning from historical data and sampling
with a conditional tabular GAN (CTGAN). This module prepares training datasets from past
instances, trains/loads CTGAN models, samples synthetic cadets and AFSC utilities, and can
augment an existing instance (e.g., 2026) with **OTS** candidates. It also rebuilds the
derived AFCCP parameter structures needed by optimization models (preferences, utilities,
eligibility/qual matrices, quotas, and rated OM datasets).

What this module does
---------------------
- **AFSC facts & proportions (for policy generation)**
  - `process_instances_into_afscs_data`: Builds an `afscs_data.csv` with AFSC list, accession groups,
    “all-eligible” flags, USAFA/ROTC proportions, overall PGL proportions, and degree-tier strings.

- **CTGAN training data assembly**

  - `process_instances_into_ctgan_data`: Merges selected years (e.g., 2024/2025) into a single table of
    features (SOC, CIP1/2, Merit, least‑desired AFSCs) plus per‑AFSC cadet/AFSC utilities.

  - Handles 2024 column harmonization (e.g., `13S1S → USSF_{R/U}`, `11U → 18X`) and merges SOC‑segmented
    AFSCs into generic columns via `fix_soc_afscs_to_generic`.

- **Model training**

  - `train_ctgan`: Detects metadata with SDV, enforces [0,1] constraints on Merit and all utility columns,
    trains the CTGAN, and saves to `<support>/CTGAN_*.pkl`.

- **Sampling realistic instances**

  - `generate_ctgan_instance`: Loads a trained CTGAN and samples **N** cadets; optionally conditions on
    pilot first‑choice composition (USAFA/ROTC) and sets degree qualification style (`degree_qual_type`).

  - Re‑scales OM percentiles within SOC, builds cadet preference lists/matrices, AFSC utilities, and
    rated OM datasets (USAFA/ROTC) consistent with AFCCP expectations.

- **OTS augmentation pipeline**

  - `augment_2026_data_with_ots`: Adds a large OTS cohort to an existing instance (e.g., `2026_0`) by
    sampling with CTGAN and stitching the new cadets into all required CSVs (Cadets, Preferences, Utility,
    Selected, AFSCs Preferences, Rated OM, etc.).

  - Degree‑scarce AFSCs are boosted with targeted sampling:
    `generate_data_with_degree_preference_fixes` → `extract_afsc_cip_sampling_information` →
    `sample_cadets_for_degree_conditions` (+ KDE‑based utility samplers).

  - Recomputes OM and AFSC rankings for OTS (`re_calculate_ots_om_and_afsc_rankings`), aligns volunteer flags
    and degrees (`align_ots_preferences_and_degrees_somewhat`), rebuilds qual matrices and utilities
    with eligibility rules (`construct_parameter_dictionary_and_augment_data`), and emits fully formed
    dataframes via `construct_full_afsc_preferences_data`, `construct_full_cadets_data`, and `compile_new_dataframes`.

Key outputs & file layout
-------------------------
- Writes training/derived data under `<support>/data/`:

  - `afscs_data.csv` (AFSC facts/proportions)
  - `ctgan_data.csv` (CTGAN training table)

- Writes a trained model under `<support>/CTGAN_*.pkl`.

- For instance augmentation, writes CSVs under `instances/<export_name>/4. Model Input/`.

Important details & conventions
-------------------------------
- **SOC merging**: `fix_soc_afscs_to_generic` consolidates `11XX_{R/U}` → `11XX` and `USSF_{R/U}` → `USSF`
  for training and downstream sampling while preserving least‑desired columns.
- **Bounds**: Merit and all utility columns are constrained to `[0,1]` during CTGAN training.
- **OM re-scaling**: Within‑SOC percentile normalization ensures comparable distributions for USAFA/ROTC.
- **Eligibility coupling**: AFSC utilities are zeroed for ineligible/ non‑volunteer cases; missing but
  eligible entries may be backfilled with OM (rated/USSF and NRL logic differs accordingly).
- **Quotas**: PGL and SOC quotas are sampled from empirical proportions stored in `afscs_data.csv`,
  then propagated to `quota_*` parameters.

Minimal examples
----------------
- Train a model:
    >>> process_instances_into_ctgan_data(['2024','2025'])
    >>> train_ctgan(epochs=1000, name='CTGAN_Full')

- Sample a synthetic instance:
    >>> p = generate_ctgan_instance(N=1600, name='CTGAN_Full', pilot_condition=True, degree_qual_type='Consistent')

- Augment 2026 with OTS:
    >>> augment_2026_data_with_ots(N=3000, import_name='2026_0', export_name='2026O')

Dependencies
------------
- **SDV** (`sdv`): `CTGANSynthesizer`, `SingleTableMetadata`, `Condition`
- **NumPy**, **Pandas**, **SciPy** (`gaussian_kde`) for sampling and table ops
- AFCCP submodules: `globals`, `data.adjustments`, `data.preferences`, `data.values`, `data.support`

See also
--------
- [`data.preferences`](../../../../../afccp/reference/data/preferences/#data.preferences)
- [`data.adjustments`](../../../../../afccp/reference/data/adjustments/#data.adjustments)
- [`data.values`](../../../../../afccp/reference/data/values/#data.values)
- [`data.support`](../../../../../afccp/reference/data/support/#data.support)
"""
import random
import numpy as np
import pandas as pd
import os
import warnings
import pickle
warnings.filterwarnings('ignore')  # prevent red warnings from printing

# afccp modules
import afccp.globals
import afccp.data.preferences
import afccp.data.adjustments
import afccp.data.values
import afccp.data.support

# SDV modules
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from scipy.stats import gaussian_kde


# ___________________________________________CTGAN DATA PREPARATION_____________________________________________________
def process_instances_into_afscs_data(data_to_use=None):

    # Load in dataframes
    if data_to_use is None:
        data_to_use = ['2025', '2026']
    dfs = {f'a{year[2:]}': pd.read_csv(f'instances/{year}/4. Model Input/{year} AFSCs.csv') for year in
           data_to_use}

    # Determine which AFSCs everyone qualifies for
    eligible = dfs['a25']['USAFA Eligible'] + dfs['a25']['ROTC Eligible']
    max_eligible = max(eligible)

    # Build out AFSCs proportions data for generating random PGL policies
    a_df = pd.DataFrame({'AFSC': dfs['a26']['AFSC'], 'All Eligible': eligible == max_eligible,
                         'Accessions Group': dfs['a26']['Accessions Group']})
    u_targets = dfs['a25']['USAFA Target'] + dfs['a26']['USAFA Target']
    pgl_targets = dfs['a25']['PGL Target'] + dfs['a26']['PGL Target']
    a_df['USAFA Proportion'] = u_targets / pgl_targets
    a_df['ROTC Proportion'] = 1 - a_df['USAFA Proportion']
    a_df['PGL Proportion'] = pgl_targets / sum(pgl_targets)
    for i in range(4):
        a_df[f'Deg Tier {i + 1}'] = dfs['a26'][f'Deg Tier {i + 1}']

    # Export AFSCs data
    filepath = afccp.globals.paths["support"] + 'data/afscs_data.csv'
    a_df.to_csv(filepath, index=False)


def process_instances_into_ctgan_data(data_to_use=None):

    if data_to_use is None:
        data_to_use = ['2024', '2025']
    print(f'Loading data to process for CTGAN: {data_to_use}.')
    dfs, arrs, afscs = load_data_to_process_for_ctgan(data_to_use=data_to_use)

    # Process data together
    df = pd.DataFrame()
    for year in data_to_use:
        print(f'Preparing data for {year}.')
        if year == '2024':
            new_df = prepare_2024_data(dfs=dfs, arrs=arrs, afscs=afscs)
        else:
            new_df = prepare_year_data(year=year, dfs=dfs, afscs=afscs)
        df = pd.concat((df, new_df))

    # Export data
    df.to_csv(afccp.globals.paths['support'] + 'data/ctgan_data.csv', index=False)


def load_data_to_process_for_ctgan(data_to_use=None):
    if data_to_use is None:
        data_to_use = ['2024', '2025']
    a26 = pd.read_csv('instances/2026/4. Model Input/2026 AFSCs.csv')
    afscs = np.array(a26['AFSC'])

    # Load in the data
    dfs = {}
    arrs = {}
    for year in data_to_use:
        yr = year[2:]

        # Load in cadets/AFSC data
        dfs[f'a{yr}'] = pd.read_csv(f'instances/{year}/4. Model Input/{year} AFSCs.csv')
        dfs[f'c{yr}'] = pd.read_csv(f'instances/{year}/4. Model Input/{year} Cadets.csv')
        dfs[f'cu{yr}'] = pd.read_csv(f'instances/{year}/4. Model Input/{year} Cadets Utility.csv')
        dfs[f'cuf{yr}'] = pd.read_csv(f'instances/{year}/4. Model Input/{year} Cadets Utility (Final).csv')
        dfs[f'au{yr}'] = pd.read_csv(f'instances/{year}/4. Model Input/{year} AFSCs Utility.csv')

        # Load in cadets selected data if we have it
        if year in ['2025', '2026']:
            dfs[f'cs{yr}'] = pd.read_csv(f'instances/{year}/4. Model Input/{year} Cadets Selected.csv')

    # Modify the data for 2024
    for d_name, _ in dfs.items():
        if 'u' in d_name or 's' in d_name:
            if '24' in d_name:
                dfs[d_name]['18X'] = dfs[d_name]['11U']
                dfs[d_name]['USSF_R'] = 0
                dfs[d_name]['USSF_U'] = 0
                dfs[d_name].loc[dfs['c24']['USAFA'] == 0, 'USSF_R'] = \
                    dfs[d_name]['13S1S'].loc[dfs['c24']['USAFA'] == 0]
                dfs[d_name].loc[dfs['c24']['USAFA'] == 1, 'USSF_U'] = \
                    dfs[d_name]['13S1S'].loc[dfs['c24']['USAFA'] == 1]
            dfs[d_name] = dfs[d_name][afscs]
            arrs[d_name] = np.array(dfs[d_name])

    return dfs, arrs, afscs


def prepare_2024_data(dfs: dict, arrs: dict, afscs: np.ndarray):
    # Initialize CTGAN data for 2024
    df = pd.DataFrame()
    utility_matrix = np.ones(arrs['cu24'].shape) * 0.1
    df['YEAR'] = [2024 for _ in range(len(dfs['c24']))]
    df['CIP1'] = 'c' + dfs['c24']['CIP1'].fillna('').astype(str)
    df['CIP2'] = dfs['c24']['CIP2'].fillna('').astype(str)
    df.loc[df["CIP2"] != 'None', 'CIP2'] = 'c' + df.loc[df["CIP2"] != 'None', 'CIP2']
    df['Merit'] = dfs['c24']['Merit']

    # Loop through each cadet (I know, very manual process...)
    for i in range(len(dfs['c24'])):

        # Get SOC info
        soc = 'USAFA' if dfs['c24'].loc[i, 'USAFA'] else "ROTC"
        df.loc[i, 'SOC'] = soc

        columns = [col for col in dfs['c24'] if 'pref' in col.lower()]
        pref_count = (dfs['c24'].iloc[i][columns].str.strip() != '').sum()

        # Force everything to string and ensure numpy dtype is string (not object)
        prefs = np.array(dfs['c24'].iloc[i][columns][:pref_count].astype(str).values, dtype=str)

        # Replace specific AFSCs
        ussf = 'USSF_' + soc[0]
        prefs[prefs == '13S1S'] = ussf
        prefs[prefs == '11U'] = '18X'

        # Now safely use np.char.find on a proper string array
        mask_s = np.char.find(prefs, 'S') != -1
        prefs[mask_s & (prefs != ussf)] = ''
        prefs = prefs[prefs != '']
        if 'nan' in prefs:
            prefs = np.array([afsc for afsc in prefs if afsc != 'nan'])
        pref_count = len(prefs)  # New number of preferences

        # Update utilities arrays
        # print(len(prefs), prefs)
        indices = np.array([np.where(afscs == afsc)[0][0] for afsc in prefs])
        utilities = arrs['cu24'][i, indices]
        num_selected = len(np.where(utilities > 0)[0])
        indiff = min(num_selected, pref_count - 1)
        utilities[indiff:] = 0.1
        utilities[0:indiff] = utilities[0:indiff] * 0.9 + 0.1

        # Update bottom choices
        for x in np.arange(1, 4):
            afsc = prefs[pref_count - x]
            if afsc[0] == '6':
                break

            if x == 1:
                df.loc[i, 'Last Choice'] = afsc
                utilities[pref_count - x] = 0
            elif x == 2:
                df.loc[i, '2nd-Last Choice'] = afsc
                utilities[pref_count - x] = 0.05
            else:
                df.loc[i, '3rd-Last Choice'] = afsc
                utilities[pref_count - x] = 0.05

        # FIll in utilities
        utility_matrix[i, indices] = utilities

    # Add in cadet data
    for j, afsc in enumerate(afscs):
        df[f'{afsc}_Cadet'] = utility_matrix[:, j]

    # Add in AFSC data
    for j, afsc in enumerate(afscs):
        df[f'{afsc}_AFSC'] = dfs['au24'][afsc]

    # Convert to integer
    df['YEAR'] = df['YEAR'].astype(int)

    # Add volunteer columns
    rated_afscs = ['11XX_R', '11XX_U', '12XX', '13B', '18X']
    df['Rated Vol'] = df[[f'{afsc}_Cadet' for afsc in rated_afscs]].sum(axis=1) != 0.5
    df['USSF Vol'] = df[[f'{afsc}_Cadet' for afsc in ['USSF_R', 'USSF_U']]].sum(axis=1) != 0.2

    # Combine AFSCs segmented by SOC (11XX/USSF)
    df = fix_soc_afscs_to_generic(df=df, afscs=afscs)
    return df


def prepare_year_data(year: str, dfs: dict, afscs: np.ndarray):
    yr = year[2:]

    # Initialize CTGAN data for the given year
    df = pd.DataFrame()
    df['YEAR'] = [year for _ in range(len(dfs[f'c{yr}']))]
    df['CIP1'] = 'c' + dfs[f'c{yr}']['CIP1'].fillna('').astype(str)
    df['CIP2'] = dfs[f'c{yr}']['CIP2'].fillna('None').astype(str)
    df.loc[df["CIP2"] != 'None', 'CIP2'] = 'c' + df.loc[df["CIP2"] != 'None', 'CIP2']
    df['Merit'] = dfs[f'c{yr}']['Merit']
    df['SOC'] = 'USAFA'
    df.loc[dfs[f'c{yr}']['USAFA'] == 0, 'SOC'] = 'ROTC'

    # Assume your original column is named 'AFSCs' in DataFrame `df`
    # Adjust the column name as needed
    df['Last Choice'] = dfs[f'c{yr}']['Least Desired AFSC']
    df[['2nd-Last Choice', '3rd-Last Choice']] = dfs[f'c{yr}']['Second Least Desired AFSCs'].str.split(',', expand=True)

    # Optionally strip whitespace around values
    df['3rd-Last Choice'] = df['3rd-Last Choice'].str.strip()
    df['2nd-Last Choice'] = df['2nd-Last Choice'].str.strip()

    # Add in cadet data
    for afsc in afscs:
        df[f'{afsc}_Cadet'] = dfs[f'cuf{yr}'][afsc]

    # Add in AFSC data
    for afsc in afscs:
        df[f'{afsc}_AFSC'] = dfs[f'au{yr}'][afsc]

    # Add volunteer columns
    rated_afscs = ['11XX_R', '11XX_U', '12XX', '13B', '18X']
    df['Rated Vol'] = df[[f'{afsc}_Cadet' for afsc in rated_afscs]].sum(axis=1) != 0.5
    df['USSF Vol'] = df[[f'{afsc}_Cadet' for afsc in ['USSF_R', 'USSF_U']]].sum(axis=1) != 0.2

    # Combine AFSCs segmented by SOC (11XX/USSF)
    df = fix_soc_afscs_to_generic(df=df, afscs=afscs)
    return df


def fix_soc_afscs_to_generic(df: pd.DataFrame, afscs: np.ndarray):

    # Make a generic AFSC merging in ROTC/USAFA segmented AFSCs
    for afsc in ['11XX', 'USSF']:
        for col in ["Cadet", "AFSC"]:
            df[f'{afsc}_{col}'] = df[f'{afsc}_R_{col}']
            df.loc[df['SOC'] == 'USAFA', f'{afsc}_{col}'] = df.loc[df['SOC'] == 'USAFA', f'{afsc}_U_{col}']
        for col in ['Last Choice', '2nd-Last Choice', '3rd-Last Choice']:
            df[col] = df[col].replace(f'{afsc}_U', afsc)
            df[col] = df[col].replace(f'{afsc}_R', afsc)

    # Strip out the columns we don't need anymore
    afscs_new = np.hstack((['USSF', '11XX'], afscs[4:]))
    back_cols = [f'{afsc}_Cadet' for afsc in afscs_new] + [f'{afsc}_AFSC' for afsc in afscs_new]
    front_cols = [col for col in df.columns if '_Cadet' not in col and '_AFSC' not in col]
    cols = front_cols + back_cols
    return df[cols]


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


def process_rare_afscs_degrees_data(full_data=None):

    # Load in original data
    if full_data is None:
        print('Loading in data...')
        filepath = afccp.globals.paths["support"] + 'data/ctgan_data.csv'
        full_data = pd.read_csv(filepath)

    # Create degree dataframe and KDE samplers for each rare AFSC
    i = 0
    deg_df = pd.DataFrame()
    afscs_rare_eligible = ['13H', '32EXA', '32EXC', '32EXE', '32EXF',
                           '32EXJ', '61C', '61D', '62EXC', '62EXE', '62EXH', '62EXI']
    for afsc in afscs_rare_eligible:

        # Filter the real data on people who wanted this AFSC, and the AFSC wanted them
        conditions = (full_data[f'{afsc}_AFSC'] > 0.6) & (full_data[f'{afsc}_Cadet'] > 0.6)
        columns = ['YEAR', 'CIP1', 'CIP2', 'Merit', 'SOC', f'{afsc}_Cadet', f'{afsc}_AFSC']

        # Get the degrees of these people and save them to the degree proportions dataframe
        d = full_data.loc[conditions][columns]['CIP1'].value_counts()
        proportions = d ** 5 / (d ** 5).sum()
        for degree, val in proportions.items():
            deg_df.loc[i, 'AFSC'] = afsc
            deg_df.loc[i, 'Degree'] = degree
            deg_df.loc[i, 'Proportion'] = val
            i += 1

        # Save the KDE sampler functions for this AFSC (AFSC & Cadet Utility)
        a_data = list(full_data.loc[conditions][columns][f'{afsc}_AFSC'])
        sampler = KDESampler(a_data)
        sampler.save(f"support/kde_samplers/{afsc}_AFSC_KDESampler.pkl")
        c_data = list(full_data.loc[conditions][columns][f'{afsc}_Cadet'])
        sampler = KDESampler(c_data)
        sampler.save(f"support/kde_samplers/{afsc}_Cadet_KDESampler.pkl")

    # Export rare AFSC-degree data
    deg_df.to_csv('support/data/afscs_rare_eligibility.csv', index=False)


# ____________________________________CTGAN MODEL TRAINING AND IMPLEMENTATION___________________________________________
def train_ctgan(epochs=1000, printing=True, name='CTGAN_Full'):
    """
    Train CTGAN to produce realistic data based on the current "ctgan_data" file in the support sub-folder. This
    function then saves the ".pkl" file back to the support sub-folder
    """

    # Import data
    data = afccp.globals.import_csv_data(afccp.globals.paths['support'] + 'data/ctgan_data.csv')
    data = data[[col for col in data.columns if col not in ['YEAR']]]
    metadata = SingleTableMetadata()  # SDV requires this now
    metadata.detect_from_dataframe(data=data)  # get the metadata from dataframe

    # Create the synthesizer model
    model = CTGANSynthesizer(metadata, epochs=epochs, verbose=True)

    # List of constraints for CTGAN
    constraints = []

    # Get list of columns that must be between 0 and 1
    zero_to_one_columns = ["Merit"]
    for col in data.columns:
        if "_Cadet" in col or "_AFSC" in col:
            zero_to_one_columns.append(col)

    # Create the "zero to one" constraints and add them to our list of constraints
    for col in zero_to_one_columns:
        zero_to_one_constraint = {"constraint_class": "ScalarRange",
                                  "constraint_parameters": {
                                      'column_name': col,
                                      'low_value': 0,
                                      'high_value': 1,
                                      'strict_boundaries': False
                                  }}
        constraints.append(zero_to_one_constraint)

    # Add the constraints to the model
    model.add_constraints(constraints)

    # Train the model
    if printing:
        print("Training the model...")
    model.fit(data)

    # Save the model
    filepath = afccp.globals.paths["support"] + name + '.pkl'
    model.save(filepath)
    if printing:
        print("Model saved to", filepath)


def generate_ctgan_instance(N=1600, name='CTGAN_Full', pilot_condition=False, rare_degrees_adjust=True,
                            degree_qual_type='Consistent'):
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
    filepath = afccp.globals.paths["support"] + 'data/afscs_data.csv'
    afscs_data = afccp.globals.import_csv_data(filepath)

    # Initialize parameter dictionary (initially only AFSCs data)
    afscs = np.array(afscs_data['AFSC'])  # List of AFSCs
    p = {'afscs': afscs, 'M': len(afscs), 'acc_grp': np.array(afscs_data['Accessions Group']),
         'Deg Tiers': np.array(afscs_data.loc[:, 'Deg Tier 1': 'Deg Tier 4']), 'P': len(afscs),
         'J': np.arange(len(afscs)), 'num_util': 10, 'Qual Type': degree_qual_type,

         # Some cadet data needed too
         'N': N, 'cadets': np.arange(N), 'I': np.arange(N)}

    # Add AFSC features to parameters
    p = generate_afscs_data(p, afscs_data)

    # Generate the cadet data and add it to the parameters
    data = generate_cadet_data(p, model, pilot_condition, rare_degrees_adjust)
    p = initialize_cadet_parameters_in_dictionary(p, data)
    p = determine_cadet_preference_information(p, data)
    p = generate_afsc_eligibility_and_preferences_data(p, data)
    p = generate_rated_om_matrices(p)

    # Return parameters
    return p


def generate_afscs_data(p, afscs_data):

    # Edit Deg Tiers null values to make string null
    p['Deg Tiers'][pd.isnull(p["Deg Tiers"])] = ''  # TODO

    # Determine AFSCs by Accessions Group
    p['afscs_acc_grp'] = {}
    if 'acc_grp' in p:
        for acc_grp in ['Rated', 'USSF', 'NRL']:
            p['J^' + acc_grp] = np.where(p['acc_grp'] == acc_grp)[0]
            p['afscs_acc_grp'][acc_grp] = p['afscs'][p['J^' + acc_grp]]

    # Useful data elements to help us generate PGL targets
    usafa_prop, rotc_prop, pgl_prop = np.array(afscs_data['USAFA Proportion']), \
                                      np.array(afscs_data['ROTC Proportion']), \
                                      np.array(afscs_data['PGL Proportion'])

    # Total targets needed to distribute
    total_targets = int(p['N'] * min(0.95, np.random.normal(0.93, 0.08)))

    # PGL targets
    p['pgl'] = np.zeros(p['M']).astype(int)
    p['usafa_quota'] = np.zeros(p['M']).astype(int)
    p['rotc_quota'] = np.zeros(p['M']).astype(int)
    for j in p['J']:

        # Create the PGL target by sampling from the PGL proportion triangular distribution
        p_min = max(0, 0.8 * pgl_prop[j])
        p_max = 1.2 * pgl_prop[j]
        prop = np.random.triangular(p_min, pgl_prop[j], p_max)
        p['pgl'][j] = int(max(1, prop * total_targets))

        # Get the ROTC proportion of this PGL target to allocate
        if rotc_prop[j] in [1, 0]:
            prop = rotc_prop[j]
        else:
            rotc_p_min = max(0, 0.8 * rotc_prop[j])
            rotc_p_max = min(1, 1.2 * rotc_prop[j])
            prop = np.random.triangular(rotc_p_min, rotc_prop[j], rotc_p_max)

        # Create the SOC-specific targets
        p['rotc_quota'][j] = int(prop * p['pgl'][j])
        p['usafa_quota'][j] = p['pgl'][j] - p['rotc_quota'][j]

    # Initialize the other pieces of information here
    for param in ['quota_e', 'quota_d', 'quota_min', 'quota_max']:
        p[param] = p['pgl']

    return p


def generate_cadet_data(p, model, pilot_condition, rare_degrees_adjust=True):

    # Sample the cadets to fill the AFSCs with rare degrees
    if rare_degrees_adjust:
        data_rare_degrees = sample_rare_afscs_degrees(model=model, p=p)
    else:
        data_rare_degrees = pd.DataFrame()  # Empty dataframe (don't care about it)

    # Sample the majority of cadets
    if pilot_condition:
        data_all_else = sample_cadet_data_with_pilot_condition(N=p['N'] - len(data_rare_degrees), model=model)
    else:
        data_all_else = model.sample(p['N'] - len(data_rare_degrees))

    # Combine majority cadets and rare degree cadets
    data = pd.concat((data_rare_degrees, data_all_else), ignore_index=True)

    # Break up USSF and 11XX AFSC by SOC
    for afsc in ['USSF', '11XX']:
        for col in ['Cadet', 'AFSC']:
            for soc in ['USAFA', 'ROTC']:
                data[f'{afsc}_{soc[0]}_{col}'] = 0
                data.loc[data['SOC'] == soc, f'{afsc}_{soc[0]}_{col}'] = data.loc[data['SOC'] == soc, f'{afsc}_{col}']

    return data


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
            df_gen = model.sample_from_conditions([afsc_cip_conditions[afsc][cip]])
            data = pd.concat((data, df_gen), ignore_index=True)
            i += count
            print(f'\n{afsc} {cip}: ({int(i)}/{int(total_gen)}) {round((i / total_gen) * 100, 2)}% complete.')

    # Modify the utilities for the cadet/AFSC pairs
    i = 0
    for afsc in afscs_rare_eligible:
        for cip, count in afsc_cip_data[afsc].items():
            data.loc[i:i + count - 1, f'{afsc}_Cadet'] = cadet_util_samplers[afsc].sample(count)
            data.loc[i:i + count - 1, f'{afsc}_AFSC'] = afsc_util_samplers[afsc].sample(count)
            i += count
    return data


def sample_cadet_data_with_pilot_condition(N, model):

    # Split up the number of ROTC/USAFA cadets
    N_usafa = round(np.random.triangular(0.25, 0.33, 0.4) * N)
    N_rotc = N - N_usafa

    # Pilot is by far the #1 desired career field, let's make sure this is represented here
    N_usafa_pilots = round(np.random.triangular(0.3, 0.4, 0.43) * N_usafa)
    N_usafa_generic = N_usafa - N_usafa_pilots
    N_rotc_pilots = round(np.random.triangular(0.25, 0.3, 0.33) * N_rotc)
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


def initialize_cadet_parameters_in_dictionary(p, data):

    # Add cadet data features to parameter dictionary
    p['merit'] = np.array(data['Merit'])
    p['usafa'] = np.array(data['SOC'] == 'USAFA') * 1
    p['rotc'] = np.array(data['SOC'] == 'ROTC') * 1
    p['cip1'] = np.array(data['CIP1'])
    p['cip2'] = np.array(data['CIP2'])

    # Clean up degree columns (remove the leading "c" I put there if it's there)
    for i in p['I']:
        if p['cip1'][i][0] == 'c':
            p['cip1'][i] = p['cip1'][i][1:]
        if p['cip2'][i][0] == 'c':
            p['cip2'][i] = p['cip2'][i][1:]

    # Create "SOC" variable
    p['soc'] = np.array(['USAFA' if p['usafa'][i] == 1 else "ROTC" for i in p['I']])

    # Fix percentiles for USAFA and ROTC
    re_scaled_om = p['merit']
    for soc in ['usafa', 'rotc']:
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
            volunteer_col = np.array(data['Rated Vol'])
            volunteers = np.where(volunteer_col)[0]
            not_volunteers = np.where(volunteer_col == False)[0]
            ranked = np.where(p['afsc_utility'][:, j] > 0)[0]
            unranked = np.where(p['afsc_utility'][:, j] == 0)[0]

            # Fill in utility values with OM for rated folks who don't have an AFSC score
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
    dataset_dict = {'rotc': 'rr_om_matrix', 'usafa': 'ur_om_matrix'}
    cadets_dict = {'rotc': 'rr_om_cadets', 'usafa': 'ur_om_cadets'}
    p["Rated Cadets"] = {}

    # Create rated OM matrices for each SOC
    for soc in ['usafa', 'rotc']:

        # Rated AFSCs for this SOC
        if soc == 'rotc':
            rated_J_soc = np.array([j for j in p['J^Rated'] if '_U' not in p['afscs'][j]])
        else:  # usafa
            rated_J_soc = np.array([j for j in p['J^Rated'] if '_R' not in p['afscs'][j]])

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


# _________________________________________OTS INTRODUCTION ANALYSIS____________________________________________________
def augment_2026_data_with_ots(N: int = 3000, import_name: str = '2026_0', export_name: str = '2026O'):
    """
    Augment a base instance with a synthetic OTS cohort and export a new, fully wired instance.

    This pipeline loads the trained CTGAN model and historical CTGAN training table, samples **N**
    realistic cadets (with extra emphasis on degree‑scarce AFSCs), converts them to OTS,
    re-computes OM and AFSC utilities under AFCCP rules (eligibility & volunteer logic), and
    stitches the new cohort into all downstream CSVs (Cadets, Preferences, Utilities, AFSC
    Preferences, CASTLE input, etc.). The result is written to
    `instances/{export_name}/4. Model Input/`.

    Parameters
    ----------
    N : int, optional
        Number of OTS cadets to generate (default 3000).
    import_name : str, optional
        Name of the *source* instance to copy/extend (e.g., `'2026_0'`).
        Reads input CSVs from `instances/{import_name}/4. Model Input/`.
    export_name : str, optional
        Name of the *destination* instance to create (e.g., `'2026O'`).
        Writes outputs to `instances/{export_name}/4. Model Input/`.

    Workflow
    --------
    1) Load CTGAN training data (`<support>/data/ctgan_data.csv`) and AFSCs for the source instance.
    2) Load CTGAN model (`<support>/CTGAN_Full.pkl`).
    3) Targeted sampling for degree‑scarce AFSCs via
       `generate_data_with_degree_preference_fixes` (with KDE utility bootstrapping), then
       sample the remainder from the CTGAN.
    4) Force SOC to `OTS`, re‑scale OM and blend AFSC utilities with OM / cadet utility using
       `re_calculate_ots_om_and_afsc_rankings`.
    5) Align volunteers and degree fields for OTS with `align_ots_preferences_and_degrees_somewhat`
       (USSF turned off for OTS).
    6) Build AFCCP parameter dict and eligibility‑aware AFSC utilities with
       `construct_parameter_dictionary_and_augment_data` (zero for ineligible/non‑volunteer;
       OM backfill where appropriate).
    7) Rebuild AFSC preference rankings and matrices with
       `construct_full_afsc_preferences_data`, and cadet‑side preferences/utilities with
       `construct_full_cadets_data`.
    8) Merge everything with existing source CSVs via `compile_new_dataframes` and export.

    Files Read
    ----------
    - `<support>/data/ctgan_data.csv`
    - `<support>/CTGAN_Full.pkl`
    - `instances/{import_name}/4. Model Input/{import_name} AFSCs.csv`
    - `instances/{import_name}/4. Model Input/{import_name} AFSCs Preferences.csv`
    - `instances/{import_name}/4. Model Input/{import_name} Cadets.csv`
    - `instances/{import_name}/4. Model Input/{import_name} Castle Input.csv`

    Files Written (to `instances/{export_name}/4. Model Input/`)
    ------------------------------------------------------------
    - `{export_name} Cadets.csv`
    - `{export_name} AFSCs Preferences.csv`
    - `{export_name} AFSCs.csv` (copied base AFSCs, unchanged schema)
    - `{export_name} Raw Data.csv` (the assembled OTS sampling table)
    - `{export_name} Castle Input.csv`
    - Plus augmented matrices produced by `compile_new_dataframes`
      (e.g., Cadets Preferences, Cadets Utility, Cadets Selected, AFSCs Buckets, OTS Rated OM).

    Returns
    -------
    None
        Side‑effects only. Progress is printed to stdout; artifacts are saved to disk.

    Notes
    -----
    - Assumes the CTGAN model is saved as `<support>/CTGAN_Full.pkl`.
    - Assumes the source instance (`import_name`) contains the standard AFCCP CSVs under
      `4. Model Input/` with 2026‑style schemas.
    - OTS candidates are excluded from USSF by construction (`USSF Vol = False`, utilities set to 0).
    - Rated OM for OTS is derived from OM where needed, filtered by eligibility.

    Examples
    --------
    >>> augment_2026_data_with_ots(N=3000, import_name='2026_0', export_name='2026O')
    """

    # Load in original data
    print('Loading in data...')
    filepath = afccp.globals.paths["support"] + 'data/ctgan_data.csv'
    full_data = pd.read_csv(filepath)
    cadet_cols = np.array([col for col in full_data.columns if '_Cadet' in col])

    # Import 'AFSCs' data
    filepath = f'instances/{import_name}/4. Model Input/{import_name} AFSCs.csv'
    afscs_df = afccp.globals.import_csv_data(filepath)
    afscs = np.array([col.split('_')[0] for col in cadet_cols])

    # Load in the model
    print('Loading in model...')
    filepath = afccp.globals.paths["support"] + 'CTGAN_Full.pkl'
    model = CTGANSynthesizer.load(filepath)

    # Sample the data
    print('Sampling data...')
    data_degrees = generate_data_with_degree_preference_fixes(model, full_data, afscs_df)
    data_all_else = model.sample(N - len(data_degrees))
    data = pd.concat((data_degrees, data_all_else), ignore_index=True)

    # These are all OTS candidates now!
    data['SOC'] = 'OTS'

    # Determine AFSCs by accessions group
    rated = np.array([np.where(cadet_cols == f'{afsc}_Cadet')[0][0] for afsc in ['11XX', '12XX', '13B', '18X']])
    afscs_acc_grp = {'Rated': rated, 'USSF': np.array([0])}

    # Re-calculate OM/AFSC Rankings for OTS
    print('Modifying data...')
    data = re_calculate_ots_om_and_afsc_rankings(data)

    # OTS isn't going to USSF
    data['USSF Vol'], data['USSF_Cadet'], data['USSF_AFSC'] = False, 0, 0
    data = align_ots_preferences_and_degrees_somewhat(data, afscs_acc_grp)

    # Non-rated AFSC indices
    nrl_indices = np.array(
        [np.where(afscs == afsc)[0][0] for afsc in afscs if afsc not in ['USSF', '11XX', '12XX', '13B', '18X']])

    # Construct the parameter dictionary and adjust AFSC utilities
    data, p = construct_parameter_dictionary_and_augment_data(
        data, afscs, afscs_df, afscs_acc_grp, nrl_indices=nrl_indices)

    # Import AFSCs Preferences data
    filepath = f'instances/{import_name}/4. Model Input/{import_name} AFSCs Preferences.csv'
    a_pref_df = afccp.globals.import_csv_data(filepath)

    # Construct the full AFSC preference data
    full_a_pref_df = construct_full_afsc_preferences_data(p, a_pref_df, afscs, nrl_indices)

    # Import 'Cadets' dataframe
    filepath = f'instances/{import_name}/4. Model Input/{import_name} Cadets.csv'
    cadets_df = afccp.globals.import_csv_data(filepath)

    # Construct the cadets data
    full_cadets_df = construct_full_cadets_data(p, cadets_df, data, afscs)

    # Import CASTLE data
    filepath = f'instances/{import_name}/4. Model Input/{import_name} Castle Input.csv'
    castle_df = afccp.globals.import_csv_data(filepath)

    # Dictionary of dataframes to export with new OTS 2026 instance
    print('Compiling current 2026 data...')
    new_dfs = {'Cadets': full_cadets_df, 'AFSCs Preferences': full_a_pref_df, 'AFSCs': afscs_df, 'Raw Data': data,
               'Castle Input': castle_df}
    new_dfs = compile_new_dataframes(new_dfs, p, cadets_df, afscs, rated, data, import_name)

    # Export new dataframes for new instance
    print('Export new data instance...')
    folder_path = f'instances/{export_name}/4. Model Input/'
    os.makedirs(folder_path, exist_ok=True)
    for df_name, df in new_dfs.items():
        print(f'Data: "{df_name}", Shape: {np.shape(df)}')
        filepath = f'{folder_path}{export_name} {df_name}.csv'
        df.to_csv(filepath, index=False)


def generate_data_with_degree_preference_fixes(model, full_data, afscs_df):
    """
    Generate synthetic cadet data for rare AFSCs, preserving degree distribution
    preferences and realistic cadet/AFSC utilities.

    This function focuses on AFSCs that are difficult to fill ("rare" AFSCs),
    generating synthetic cadets in a way that matches observed degree patterns
    (CIP1) from historical data. It uses
    [`extract_afsc_cip_sampling_information`](../../../reference/data/generation/#data.generation.extract_afsc_cip_sampling_information)
    to determine sampling quotas and conditions, and
    [`sample_cadets_for_degree_conditions`](../../../reference/data/generation/#data.generation.sample_cadets_for_degree_conditions)
    to produce matching synthetic records. Cadet and AFSC utility values are
    then resampled for realism.

    Parameters
    ----------
    model : object
        A generative model instance (e.g., CTGAN) implementing
        `sample_from_conditions(conditions)` to produce synthetic cadets.
    full_data : pandas.DataFrame
        Full dataset containing historical cadet and AFSC information.
    afscs_df : pandas.DataFrame
        DataFrame containing AFSC metadata, including 'OTS Target' values.

    Returns
    -------
    pandas.DataFrame
        Synthetic dataset containing cadets for rare AFSCs with realistic
        degree distributions and utility values.

    Notes
    -----
    - Rare AFSC eligibility is hardcoded as a list of AFSC strings in this
      function.
    - Degree sampling is biased toward more common CIPs by cubic weighting
      (proportions ∝ frequency³).
    - Cadet and AFSC utilities are drawn from kernel density estimators
      (KDEs) fitted on historical data.

    See Also
    --------
    - [`extract_afsc_cip_sampling_information`](../../../afccp/reference/data/generation/#data.generation.extract_afsc_cip_sampling_information)
    - [`sample_cadets_for_degree_conditions`](../../../afccp/reference/data/generation/#data.generation.sample_cadets_for_degree_conditions)
    """

    # Filter dataframe to rare AFSCs (degree-wise)
    afscs_rare_eligible = ['13H', '32EXA', '32EXC', '32EXE', '32EXF',
                           '32EXJ', '61C', '61D', '62EXC', '62EXE', '62EXH', '62EXI']
    afscs_rare_df = afscs_df.set_index('AFSC')['OTS Target'].loc[afscs_rare_eligible]

    # Extract data generating parameters
    total_gen, afsc_cip_data, afsc_cip_conditions, afsc_util_samplers, cadet_util_samplers = \
        extract_afsc_cip_sampling_information(full_data, afscs_rare_eligible, afscs_rare_df)

    # Generate the data
    data = sample_cadets_for_degree_conditions(model, total_gen, afscs_rare_eligible, afsc_cip_data,
                                               afsc_cip_conditions)

    # Modify the utilities for the cadet/AFSC pairs
    i = 0
    for afsc in afscs_rare_eligible:
        for cip, count in afsc_cip_data[afsc].items():
            count = int(count)
            data.loc[i:i + count - 1, f'{afsc}_Cadet'] = cadet_util_samplers[afsc](count)
            data.loc[i:i + count - 1, f'{afsc}_AFSC'] = afsc_util_samplers[afsc](count)
            i += count

    return data


def extract_afsc_cip_sampling_information(full_data, afscs_rare_eligible, afscs_rare_df):
    """
    Extract degree distribution and utility sampling information for rare AFSCs.

    This function identifies cadets who have strong mutual preference with specific
    rare AFSCs (both the AFSC ranks the cadet highly and the cadet ranks the AFSC
    highly), determines the distribution of primary degrees (CIP1) for those cadets,
    and constructs constraints to ensure proportional representation in generated
    synthetic data. It also fits kernel density estimators (KDEs) to model cadet and
    AFSC utility scores for each AFSC-degree combination.

    Parameters
    ----------
    full_data : pandas.DataFrame
        Full dataset containing cadet records with columns for degree codes (`CIP1`),
        AFSC utilities (`<AFSC>_AFSC`), and cadet preferences (`<AFSC>_Cadet`).
    afscs_rare_eligible : list of str
        List of AFSC codes considered rare and eligible for targeted sampling.
    afscs_rare_df : pandas.DataFrame or pandas.Series
        Data structure mapping each AFSC to the number of cadets needed to meet
        quotas for that AFSC.

    Returns
    -------
    total_gen : int
        Total number of synthetic cadets to generate across all rare AFSCs.
    afsc_cip_data : dict
        Mapping of `{afsc: pandas.Series}` where the Series index is degree codes
        (CIP1) and values are the number of cadets to generate for each degree.
    afsc_cip_conditions : dict
        Mapping `{afsc: {cip: Condition}}` specifying generation constraints for each
        AFSC-degree combination.
    afsc_util_samplers : dict
        Mapping `{afsc: callable}` returning AFSC utility samples for a given AFSC.
    cadet_util_samplers : dict
        Mapping `{afsc: callable}` returning cadet utility samples for a given AFSC.

    Notes
    -----
    - Only cadets with mutual interest scores > 0.6 for a given AFSC are considered.
    - Degree frequencies are cubed to overweight common degrees, then scaled to match
      target generation counts using [`safe_round`](../../../reference/data/processing/#data.processing.safe_round).
    - For AFSC `62EXE`, target counts are halved due to quota filling difficulty.
    - Generation quotas are inflated by 40% or at least 3 extra cadets to ensure
      adequate representation.
    """

    afsc_cip_data = {}
    afsc_util_samplers = {}
    cadet_util_samplers = {}
    afsc_cip_conditions = {}
    total_gen = 0
    for afsc in afscs_rare_eligible:

        # Filter the real data on people who wanted this AFSC, and the AFSC wanted them
        conditions = (full_data[f'{afsc}_AFSC'] > 0.6) & (full_data[f'{afsc}_Cadet'] > 0.6)
        columns = ['YEAR', 'CIP1', 'CIP2', 'Merit', 'SOC', f'{afsc}_Cadet', f'{afsc}_AFSC']

        # Get the degrees of these people
        d = full_data.loc[conditions][columns]['CIP1'].value_counts()
        degrees = np.array(d.index)

        # Figure out how many degrees we have to ensure are present in this newly created dataset
        val = int(afscs_rare_df.loc[afsc])
        if afsc == '62EXE':  # We struggle to fill this quota!!
            val = val / 2
        num_gen = np.ceil(max(val * 1.4, val + 3))
        proportions = np.array(d ** 3) / np.array(d ** 3).sum()  # Tip the scales in favor of the more common CIP
        counts = safe_round(proportions * num_gen)
        afsc_cip_data[afsc] = pd.Series(counts, index=degrees)  # Save the degree information for this AFSC
        afsc_cip_data[afsc] = afsc_cip_data[afsc][afsc_cip_data[afsc] > 0]

        # Save functions to sample cadet/AFSC utilities for the ones with these degrees
        afsc_util_samplers[afsc] = fit_kde_sampler(list(full_data.loc[conditions][columns][f'{afsc}_AFSC']))
        cadet_util_samplers[afsc] = fit_kde_sampler(list(full_data.loc[conditions][columns][f'{afsc}_Cadet']))

        afsc_cip_conditions[afsc] = {}
        for cip, count in afsc_cip_data[afsc].items():
            condition = Condition(num_rows=int(count), column_values={"CIP1": cip})
            afsc_cip_conditions[afsc][cip] = condition
            total_gen += count

    return total_gen, afsc_cip_data, afsc_cip_conditions, afsc_util_samplers, cadet_util_samplers


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


def fit_kde_sampler(data):
    kde = gaussian_kde(data, bw_method='scott')

    def sampler(n):
        samples = kde.resample(n).flatten()
        # clip to ensure values stay between 0 and 1
        return np.clip(samples, 0, 1)

    return sampler


def sample_cadets_for_degree_conditions(model, total_gen, afscs_rare_eligible, afsc_cip_data, afsc_cip_conditions):
    """
    Generate synthetic cadets matching AFSC-degree sampling conditions.

    Iterates over rare AFSCs and their associated degree quotas to generate
    synthetic cadets using the provided generative model. For each AFSC-degree
    combination, the function samples cadets that meet the degree condition
    constraints, appending them to a cumulative dataset.

    Parameters
    ----------
    model : object
        A generative model instance (e.g., CTGAN) implementing
        `sample_from_conditions(conditions)` to produce synthetic cadets.
    total_gen : int
        Total number of cadets to generate across all AFSC-degree combinations.
    afscs_rare_eligible : list of str
        List of AFSC codes considered rare and eligible for targeted generation.
    afsc_cip_data : dict
        Mapping `{afsc: pandas.Series}` where the Series index is degree codes (CIP1)
        and values are the number of cadets to generate for each degree.
    afsc_cip_conditions : dict
        Mapping `{afsc: {cip: Condition}}` specifying generation constraints for each
        AFSC-degree combination.

    Returns
    -------
    pandas.DataFrame
        A concatenated dataset of synthetic cadets meeting all AFSC-degree constraints.

    Notes
    -----
    - This function logs progress to the console, showing both the number and
      percentage of cadets generated so far.
    - Sampling order is AFSC-major, iterating over all degrees within each AFSC
      before moving to the next AFSC.
    - The `count` values in `afsc_cip_data` are expected to be integers or
      convertible to integers.
    """

    # Generate dataframe
    data = pd.DataFrame()
    i = 0
    for afsc in afscs_rare_eligible:
        for cip, count in afsc_cip_data[afsc].items():
            print(f'{afsc} {cip}: {int(count)}...')
            df_gen = model.sample_from_conditions([afsc_cip_conditions[afsc][cip]])
            data = pd.concat((data, df_gen), ignore_index=True)
            i += count
            print(f'{afsc} {cip}: ({int(i)}/{int(total_gen)}) {round((i / total_gen) * 100, 2)}% complete.')

    return data


def re_calculate_ots_om_and_afsc_rankings(data: pd.DataFrame):

    # Re-scale OM!
    N = len(data)
    percentiles = np.array(data['Merit'])
    sorted_indices = np.argsort(percentiles)[::-1]  # Sort these percentiles (descending)
    new_percentiles = (np.arange(N)) / (N - 1)  # New percentiles we want to replace these with
    magic_indices = np.argsort(sorted_indices)  # Indices that let us put the new percentiles in right place
    new_percentiles = new_percentiles[::-1][magic_indices]  # Put the new percentiles back in the right place
    data['Merit'] = new_percentiles  # Load back into data

    # Re-bake in OM and Cadet Utility to AFSC rankings
    for col in [col for col in data.columns if '_AFSC' in col]:
        afsc = col.split('_')[0]
        scalar = np.random.triangular(0.1, 0.5, 0.9, len(data))
        data[col] = scalar * data[col] + (1 - scalar) * data['Merit']
        scalar = np.random.triangular(0.3, 0.7, 0.9, len(data))
        data[col] = scalar * data[col] + (1 - scalar) * data[f'{afsc}_Cadet']
    return data


def align_ots_preferences_and_degrees_somewhat(data: pd.DataFrame, afscs_acc_grp):

    # Clean up degree columns (remove the leading "c" I put there if it's there)
    for i in data.index:
        if data.loc[i, 'CIP1'][0] == 'c':
            data.loc[i, 'CIP1'] = str(int(data.loc[i, 'CIP1'][1:].replace('.0', '')))
        if data.loc[i, 'CIP2'][0] == 'c':
            data.loc[i, 'CIP2'] = str(int(data.loc[i, 'CIP2'][1:].replace('.0', '')))

    # Convert info to numpy arrays
    cadet_cols = np.array([col for col in data.columns if '_Cadet' in col])
    util_original = np.array(data[cadet_cols])

    # Loop through each cadet to tweak their preferences
    for i in data.index:

        # Fix rated/USSF volunteer situation
        for acc_grp in ['Rated', 'USSF']:
            if data.loc[i, f'{acc_grp} Vol']:
                if np.max(util_original[i, afscs_acc_grp[acc_grp]]) < 0.6:
                    util_original[i, afscs_acc_grp[acc_grp]] = 0
                    data.loc[i, f'{acc_grp} Vol'] = False
            else:  # Not a volunteer

                # We have a higher preference for these kinds of AFSCs
                if np.max(util_original[i, afscs_acc_grp[acc_grp]]) >= 0.6:
                    data.loc[i, f'{acc_grp} Vol'] = True  # Make them a volunteer now

    # Save utility information back to data
    data[cadet_cols] = util_original
    return data


def construct_parameter_dictionary_and_augment_data(data: pd.DataFrame, afscs: np.ndarray, afscs_df: pd.DataFrame,
                                                    afscs_acc_grp: dict, nrl_indices: np.ndarray):

    # Construct parameter dictionary for OTS cadets
    N = len(data)
    p = {'cip1': np.array(data['CIP1']), 'cip2': np.array(data['CIP2']), 'afscs': afscs, 'M': len(afscs),
         'Qual Type': 'Consistent', 'N': N, 'P': len(afscs), 'I': np.arange(len(data)), 'J': np.arange(len(afscs)),
         'merit': np.array(data['Merit']), 'Deg Tiers': np.array(afscs_df.loc[3:, 'Deg Tier 1': 'Deg Tier 4'])}

    p['Deg Tiers'][pd.isnull(p["Deg Tiers"])] = ''  # TODO
    p = afccp.data.adjustments.gather_degree_tier_qual_matrix(cadets_df=None, parameters=p)

    # Get the qual matrix to know what people are eligible for
    ineligible = (np.core.defchararray.find(p['qual'], "I") != -1) * 1
    eligible = (ineligible == 0) * 1
    I_E = [np.where(eligible[:, j])[0] for j in p['J']]  # set of cadets that are eligible for AFSC j

    # Modify AFSC utilities based on eligibility
    a_pref_cols = [f'{afsc}_AFSC' for afsc in afscs]
    p['afsc_utility'] = np.around(np.array(data[a_pref_cols]), 2)
    for acc_grp in ['Rated', 'USSF']:
        for j in afscs_acc_grp[acc_grp]:
            volunteer_col = np.array(data['Rated Vol'])
            volunteers = np.where(volunteer_col)[0]
            not_volunteers = np.where(volunteer_col == False)[0]
            ranked = np.where(p['afsc_utility'][:, j] > 0)[0]
            unranked = np.where(p['afsc_utility'][:, j] == 0)[0]

            # Fill in utility values with OM for rated folks who don't have an AFSC score
            volunteer_unranked = np.intersect1d(volunteers, unranked)
            p['afsc_utility'][volunteer_unranked, j] = p['merit'][volunteer_unranked]

            # If the cadet didn't actually volunteer, they should have utility of 0
            non_volunteer_ranked = np.intersect1d(not_volunteers, ranked)
            p['afsc_utility'][non_volunteer_ranked, j] = 0

    # Remove cadets from this AFSC's preferences if the cadet is not eligible
    for j in nrl_indices:
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

    # Put new calculated utilities back into dataframe
    data[a_pref_cols] = p['afsc_utility']
    return data, p


def construct_full_afsc_preferences_data(p, a_pref_df, afscs, nrl_indices):

    # Load in data and extra non-rated AFSC preferences/rankings
    nrl_rankings_current = np.array(a_pref_df[afscs[nrl_indices]])

    # Create full utility matrix for NRL AFSCs
    shape = (nrl_rankings_current.shape[0] + len(p['afsc_utility']), nrl_rankings_current.shape[1])
    nrl_afsc_utility_full = np.zeros(shape)

    # Loop through each AFSC to calculate utility from preferences (Current 2026 info)
    afsc_utility_curr = np.zeros(nrl_rankings_current.shape)
    for j, afsc in enumerate(afscs[nrl_indices]):

        # Indices of eligible folks for this AFSC
        indices = np.where(nrl_rankings_current[:, j])[0]
        num_nonzero = len(indices)  # How many are eligible?
        sorted_indices = np.argsort(nrl_rankings_current[:, j])  # Sort them!

        # Turn the 1, 2, 3, ... 10 list to 1, 0.9, 0.8, ..., 0.1
        utils = 1 - (np.arange(1, num_nonzero + 1) / num_nonzero) + 1 / num_nonzero

        # Place the utilities into the matrix in the correct spots
        afsc_utility_curr[sorted_indices[-num_nonzero:], j] = utils

        # Combine OTS utilities into full utility matrix for NRL AFSCs
        if afsc in afscs:
            j_o = np.where(afscs == afsc)[0][0]
            nrl_afsc_utility_full[:len(afsc_utility_curr), j] = afsc_utility_curr[:, j]
            nrl_afsc_utility_full[len(afsc_utility_curr):, j] = p['afsc_utility'][:, j_o]
        else:
            print(afsc, 'not in AFSCs')

    # Convert utilities to preferences combining USAFA, ROTC, OTS cadets for NRL
    nrl_a_pref_matrix_full = np.zeros(shape).astype(int)
    nrl_afsc_preferences = {}
    full_N = len(nrl_a_pref_matrix_full)
    for j, afsc in enumerate(afscs[nrl_indices]):
        # Sort the utilities to get the preference list
        utilities = nrl_afsc_utility_full[:, j]
        ineligible_indices = np.where(utilities == 0)[0]
        sorted_indices = np.argsort(utilities)[::-1][:full_N - len(ineligible_indices)]
        nrl_afsc_preferences[j] = sorted_indices

        # Since 'afsc_preferences' is an array of AFSC indices, we can do this
        nrl_a_pref_matrix_full[nrl_afsc_preferences[j], j] = np.arange(1, len(nrl_afsc_preferences[j]) + 1)

    # Get list of columns used in the cadet-AFSC matrices
    standard_afsc_df_columns = np.array(a_pref_df.columns)
    i = np.where(standard_afsc_df_columns == '11XX_U')[0][0]  # Add OTS pilot AFSC
    standard_afsc_df_columns = np.insert(standard_afsc_df_columns, i + 1, '11XX_O')

    full_a_pref_df = pd.DataFrame({'Cadet': np.arange(len(nrl_a_pref_matrix_full))})
    for col in standard_afsc_df_columns[1:]:
        if col in afscs[nrl_indices]:
            j = np.where(afscs[nrl_indices] == col)[0][0]
            full_a_pref_df[col] = nrl_a_pref_matrix_full[:, j]
        elif 'USSF' in col:
            full_a_pref_df.loc[:len(a_pref_df), col] = a_pref_df[col]
        else:
            full_a_pref_df[col] = 1
    full_a_pref_df = full_a_pref_df.fillna(0).astype(int)
    return full_a_pref_df


def construct_full_cadets_data(p: dict, cadets_df: pd.DataFrame, data: pd.DataFrame, afscs: np.ndarray):

    # Get new OTS cadet indices
    N = len(data)
    ots_cadets = np.arange(len(cadets_df), len(cadets_df) + N)
    cadet_cols = np.array([col for col in data.columns if '_Cadet' in col])

    # Initialize cadet preference information
    p['c_utilities'] = np.zeros((p['N'], 10))
    p['c_preferences'] = np.array([[' ' * 6 for _ in range(p['M'])] for _ in range(p['N'])])
    p['cadet_preferences'] = {}
    p['c_pref_matrix'] = np.zeros((p['N'], p['M'])).astype(int)
    p['utility'] = np.zeros((p['N'], p['M']))
    util_original = np.around(np.array(data[cadet_cols]), 2)

    # Loop through each cadet to fix the preference information
    for i in p['I']:

        # Save cadet preference information
        ordered_list = np.argsort(util_original[i])[::-1]
        num_pref = int(np.random.triangular(3, 9, 18))
        p['c_utilities'][i, :min(10, num_pref)] = util_original[i, ordered_list[:min(10, num_pref)]]
        p['cadet_preferences'][i] = ordered_list[:num_pref]
        p['c_preferences'][i, :num_pref] = afscs[p['cadet_preferences'][i]]
        p['c_pref_matrix'][i, p['cadet_preferences'][i]] = np.arange(1, len(p['cadet_preferences'][i]) + 1)
        p['utility'][i, p['cadet_preferences'][i][:min(10, num_pref)]] = p['c_utilities'][i, :min(10, num_pref)]

        # Determine bottom choice AFSCs
        bottom_choices = []
        for col in ['Last Choice', '2nd-Last Choice', '3rd-Last Choice']:
            condition = data.loc[i, col] not in afscs[p['cadet_preferences'][i]]  # can't be a preference
            condition *= data.loc[i, col] in afscs  # has to be real AFSC (not NaN)
            condition *= (data.loc[i, col] not in bottom_choices)  # can't already be a bottom choice
            if condition:
                bottom_choices.append(data.loc[i, col])

        # Add in bottom choices data
        if len(bottom_choices) == 1:
            data.loc[i, 'Least Desired AFSC'] = bottom_choices[0]
        elif len(bottom_choices) >= 2:
            data.loc[i, 'Second Least Desired AFSCs'] = ', '.join(bottom_choices[1:])
    p['selected'] = (p['c_pref_matrix'] > 0) * 1  # Create "selected" array

    # Create the OTS cadet dataframe
    ots_cadets_df = pd.DataFrame({'Cadet': ots_cadets, 'SOC': 'OTS', 'USAFA': 0,
                                  'CIP1': data['CIP1'], 'CIP2': data['CIP2'],
                                  'Merit': data['Merit'], 'Real Merit': data['Merit'],
                                  'Least Desired AFSC': data['Least Desired AFSC'],
                                  'Second Least Desired AFSCs': data['Second Least Desired AFSCs']})

    # Add in preferences and utilities
    for i in np.arange(10):
        ots_cadets_df[f'Util_{i + 1}'] = p['c_utilities'][:, i]
    for i in np.arange(20):
        ots_cadets_df[f'Pref_{i + 1}'] = p['c_preferences'][:, i]
    ots_cadets_df = ots_cadets_df.replace('11XX', '11XX_O')  # Add OTS pilot

    # Add in qual matrix
    for j, afsc in enumerate(afscs):
        if afsc == 'USSF':
            continue
        if afsc == '11XX':
            afsc = '11XX_O'
        ots_cadets_df[f'qual_{afsc}'] = p['qual'][:, j]

    # Add OTS cadets information to current cadets info
    new_cadets_df = pd.concat((cadets_df, ots_cadets_df))

    # Rearrange columns
    col = new_cadets_df.pop('CIP2')
    new_cadets_df.insert(5, 'CIP2', col)
    col = new_cadets_df.pop('CIP1')
    new_cadets_df.insert(5, 'CIP1', col)
    idx = np.where(new_cadets_df.columns == 'qual_11XX_U')[0][0]
    col = new_cadets_df.pop('qual_11XX_O')
    new_cadets_df.insert(int(idx + 1), 'qual_11XX_O', col)

    return new_cadets_df


def compile_new_dataframes(new_dfs, p, cadets_df, afscs, rated, data, import_name):

    N = len(data)
    ots_cadets = np.arange(len(cadets_df), len(cadets_df) + N)

    # Create dummy buckets for OTS (I don't care about this stuff rn)
    p['afsc_buckets'] = np.ones(p['c_pref_matrix'].shape)

    # Add additional cadets data for OTS
    df_arr_dict = {"Cadets Preferences": "c_pref_matrix", "Cadets Utility": "utility",
                   "Cadets Selected": "selected", "AFSCs Buckets": 'afsc_buckets',
                   'ROTC Rated OM': '', 'USAFA Rated OM': ''}
    for df_name, arr_name in df_arr_dict.items():

        # Import current dataframe
        filepath = f'instances/{import_name}/4. Model Input/{import_name} {df_name}.csv'
        df_i = afccp.globals.import_csv_data(filepath)

        # USAFA/ROTC Rated data get pulled over directly
        if 'USAFA' in df_name or 'ROTC' in df_name:
            new_dfs[df_name] = df_i
            continue

        # Initialize OTS' dataframe
        o_df = pd.DataFrame({'Cadet': ots_cadets})

        # Add in array information
        for j, afsc in enumerate(afscs):
            if afsc == 'USSF':
                continue
            if afsc == '11XX':
                afsc = '11XX_O'
            o_df[afsc] = p[arr_name][:, j]

        # Add OTS cadets information to current cadets info
        new_df = pd.concat((df_i, o_df))
        col = new_df.pop('11XX_O')
        new_df.insert(5, '11XX_O', col)
        new_dfs[df_name] = new_df.fillna(0)

    # Create OTS Rated OM dataframe (Use OM as rated rankings for OTS)
    rated_cadets = np.where(p['afsc_utility'][:, rated[2]])[0]
    eligible_rated = (p['afsc_utility'][rated_cadets][:, rated] > 0) * 1
    om_arr = np.around(np.array([p['merit'][rated_cadets] for _ in rated]).T, 3) * eligible_rated
    rated_om_df = pd.DataFrame({'Cadet': rated_cadets + len(cadets_df)})
    for idx, afsc in enumerate(afscs[rated]):
        if afsc == '11XX':
            afsc = '11XX_O'
        rated_om_df[afsc] = om_arr[:, idx]
    new_dfs['OTS Rated OM'] = rated_om_df
    return new_dfs