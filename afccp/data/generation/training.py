import random
import numpy as np
import pandas as pd
import pickle
from scipy.stats import beta

# afccp modules
import afccp.globals
import afccp.data.preferences
import afccp.data.adjustments
import afccp.data.values
import afccp.data.support

# SDV modules
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


# _______________________________________MAIN CTGAN DATA GENERATION_____________________________________________________
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


# ________________________________AUGMENTED SUPPORT FOR BASE/TRAINING ASSIGNMENTS_______________________________________
def train_base_ist_cadet_generator(filepath_2025_experimental=None, epochs=100, train_ctgan=True,
                                   ctgan_model_name='Base_Preferences_CTGAN', printing=True):

    if printing:
        print('Building cadet preference generators...')

    # Load in the data
    df = pd.read_excel(filepath_2025_experimental)

    # Process dataframe and break it out into chunks
    df1, df2 = clean_break_out_df1_df2(df)

    # Construct the probability samplers
    prob_sampler = construct_prob_samplers(df1, df2)

    # Save the samplers
    filepath = afccp.globals.paths["support"] + 'prob_samplers.pkl'
    with open(filepath, "wb") as f:
        pickle.dump(prob_sampler, f)

    # Fit model
    if train_ctgan:

        if printing:
            print(f'Training base preferences CTGAN model on {epochs} epochs...')
        ctgan_model = fit_ctgan_base_preferences(df2, epochs=epochs)

        # Save the model
        filepath = afccp.globals.paths["support"] + ctgan_model_name + '.pkl'
        ctgan_model.save(filepath)
        if printing:
            print("Model saved to", filepath)


def clean_break_out_df1_df2(df):

    # Just take the columns of interest
    first_base = 'MAXWELL-GUNTER AFB'
    last_base = 'MILDENHALL AFB'
    base_pref_columns = list(
        df.loc[:, df.columns[df.columns.get_loc(first_base):df.columns.get_loc(last_base) + 1]].columns)
    other_base_ist_columns = ['Start Preference', 'Base Affect AFSC', 'Base Affect AFSC Num', 'IST Affect AFSC',
                              'IST Affect AFSC Num', 'AFSC Weight', 'Base Weight', 'IST Weight']
    important_columns = ['DOC', 'Experimental Data', 'Experimental Feedback']
    columns = important_columns + other_base_ist_columns + base_pref_columns
    df.index.name = 'Cadet'
    df = df[columns]

    # Change values
    df = df.sort_values(by='Cadet', inplace=False).reset_index(drop=True)
    df.loc[df['Start Preference'] == 'Earliest', 'Start Preference'] = 'Early'
    df.loc[df['Start Preference'] == 'Latest', 'Start Preference'] = 'Late'

    # Extract base pref data
    base_prefs = df.loc[df['Experimental Data'] == 1][base_pref_columns].fillna(0)
    df2 = convert_rank_df_to_linear_utility(base_prefs)
    df2["num_base_preferences"] = (base_prefs > 0).sum(axis=1)

    # Extract other experimental pref columns
    df1 = df.loc[df['Experimental Data'] == 1][other_base_ist_columns]

    # Clean Up IST Start Preferences (Validate them)
    df1.loc[df1['Start Preference'] == 'None', 'IST Affect AFSC'] = 'Never'
    df1.loc[df1['Start Preference'] == 'None', 'IST Affect Num'] = np.nan
    df1.loc[df1['Start Preference'] == 'None', 'IST Weight'] = np.nan
    return df1, df2


def construct_prob_samplers(df1, df2):

    # Create probability dataframes for sampling
    prob_sampler = {'Start Pref': df1.loc[df1['Start Preference'].isin(['Early', 'None', 'Late'])][
        'Start Preference'].value_counts(normalize=True),
                    'Base Affect AFSC': df1.loc[df1['Base Affect AFSC'].isin(['Yes', 'Never'])][
                        'Base Affect AFSC'].value_counts(normalize=True),
                    'Base Affect AFSC Num': df1.loc[~pd.isnull(df1['Base Affect AFSC Num'])][
                        'Base Affect AFSC Num'].value_counts(normalize=True),
                    'IST Affect AFSC': df1.loc[df1['IST Affect AFSC'].isin(['Yes', 'Never'])][
                        'IST Affect AFSC'].value_counts(normalize=True),
                    'IST Affect AFSC Num': df1.loc[~pd.isnull(df1['Base Affect AFSC Num'])][
                        'Base Affect AFSC Num'].value_counts(normalize=True)}

    # Create sampler for AFSC, Base, Course Weights
    afsc_weights = df1.loc[~pd.isnull(df1['AFSC Weight'])]['AFSC Weight']
    base_weights = df1.loc[~pd.isnull(df1['Base Weight'])]['Base Weight']
    course_weights = df1.loc[~pd.isnull(df1['IST Weight'])]['IST Weight']
    prob_sampler['AFSC Weight'] = fit_one_inflated_beta(afsc_weights)
    prob_sampler['Base Weight'] = fit_one_inflated_beta(base_weights)
    prob_sampler['Course Weight'] = fit_one_inflated_beta(course_weights)

    # Number of Base Preferences
    prob_sampler['num_base_preferences'] = df2.loc[df2['num_base_preferences'] > 0][
        'num_base_preferences'].value_counts(normalize=True)
    return prob_sampler


def convert_rank_df_to_linear_utility(df):
    """
    Convert base rank matrix (1..N, 0 for unranked)
    into linear utility scaled from 1 to 1/N.

    Returns a new dataframe.
    """

    # Count how many bases each cadet ranked
    N = (df > 0).sum(axis=1)

    # Avoid divide-by-zero
    N_safe = N.replace(0, np.nan)

    # Apply linear transformation
    utility_df = df.copy()

    utility_df = utility_df.where(
        df == 0,
        (N_safe[:, None] - df + 1) / N_safe[:, None]
    )

    utility_df = utility_df.fillna(0)

    return utility_df


def fit_one_inflated_beta(series):
    x = series.values / 100

    # Identify spike at 1
    is_one = (x == 1)
    p_one = is_one.mean()

    # Continuous portion
    x_cont = x[~is_one]

    # Avoid exact 0 or 1 in continuous portion
    eps = 1e-6
    x_cont = np.clip(x_cont, eps, 1 - eps)

    a, b, _, _ = beta.fit(x_cont, floc=0, fscale=1)

    return {"p_one": p_one, "a": a, "b": b}


def fit_ctgan_base_preferences(df2, epochs=400):
    """
    Fit CTGAN model to base preference utility dataframe.
    """

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df2)

    # Force correct types
    for col in df2.columns:
        metadata.update_column(column_name=col, sdtype='numerical')

    ctgan = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        batch_size=256,
        # generator_dim=(256, 256),
        # discriminator_dim=(256, 256),
        pac=1,
        verbose=True
    )

    ctgan.fit(df2)

    return ctgan