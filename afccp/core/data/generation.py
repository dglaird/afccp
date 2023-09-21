import random
import numpy as np
import pandas as pd
import string

# afccp modules
import afccp.core.globals
import afccp.core.data.preferences
import afccp.core.data.adjustments
import afccp.core.data.values
import afccp.core.data.support
import warnings
warnings.filterwarnings('ignore')  # prevent red warnings from printing

# Import sdv if it is installed
if afccp.core.globals.use_sdv:
    import sdv

def generate_random_instance(N=1600, P=6, M=32, generate_only_nrl=False):
    """
    This procedure takes in the specified parameters (defined below) and then simulates new random "fixed" cadet/AFSC
    input parameters. These parameters are then returned and can be used to solve the VFT model.
    :param generate_only_nrl: Only generate NRL AFSCs (default to False)
    :param N: number of cadets
    :param P: number of preferences allowed
    :param M: number of AFSCs
    :return: model fixed parameters
    """

    # Initialize parameter dictionary
    # noinspection PyDictCreation
    p = {'N': N, 'P': P, 'M': M, 'num_util': P, 'cadets': np.arange(N),
         'minority': np.random.choice([0, 1], size=N, p=[1 / 3, 2 / 3]),
         'male': np.random.choice([0, 1], size=N, p=[1 / 3, 2 / 3]),
         'usafa': np.random.choice([0, 1], size=N, p=[1 / 3, 2 / 3]), 'merit': np.random.rand(N)}

    # Generate various features of the cadets
    p['merit_all'] = p['merit']
    p['assigned'] = np.array(['' for _ in range(N)])

    # Calculate quotas for each AFSC
    p['pgl'], p['usafa_quota'], p['rotc_quota'] = np.zeros(M), np.zeros(M), np.zeros(M)
    p['quota_min'], p['quota_max'] = np.zeros(M), np.zeros(M)
    p['quota_e'], p['quota_d'] = np.zeros(M), np.zeros(M)
    for j in range(M):

        # Get PGL target
        p['pgl'][j] = max(10, np.random.normal(1000 / M, 100))

    # Scale PGL and force integer values and minimum of 1
    p['pgl'] = np.around((p['pgl'] / np.sum(p['pgl'])) * N * 0.8)
    indices = np.where(p['pgl'] == 0)[0]
    p['pgl'][indices] = 1

    # Sort PGL by size
    p['pgl'] = np.sort(p['pgl'])[::-1]

    # USAFA/ROTC Quotas
    p['usafa_quota'] = np.around(np.random.rand(M) * 0.3 + 0.1 * p['pgl'])
    p['rotc_quota'] = p['pgl'] - p['usafa_quota']

    # Min/Max
    p['quota_min'], p['quota_max'] = p['pgl'], np.around(p['pgl'] * (1 + np.random.rand(M) * 0.9))

    # Target is a random integer between the minimum and maximum targets
    target = np.around(p['quota_min'] + np.random.rand(M) * (p['quota_max'] - p['quota_min']))
    p['quota_e'], p['quota_d'] = target, target

    # Generate AFSCs
    p['afscs'] = np.array(['R' + str(j + 1) for j in range(M)])

    # Determine what "accessions group" each AFSC is in
    if generate_only_nrl:
        p['acc_grp'] = np.array(["NRL" for _ in range(M)])
    else:

        # If there are 3 or more AFSCs, we want all three accessions groups represented
        if M >= 3:
            invalid = True
            while invalid:
                p['acc_grp'] = np.array([np.random.choice(['NRL', 'Rated', 'USSF']) for _ in range(M)])

                # Make sure we have at least one AFSC from each accession's group
                invalid = False  # "Innocent until proven guilty"
                for grp in ['NRL', 'Rated', 'USSF']:
                    if grp not in p['acc_grp']:
                        invalid = True
                        break

        # If we only have one or two AFSCs, they'll all be NRL
        else:
            p['acc_grp'] = np.array(["NRL" for _ in range(M)])

    # Add an "*" to the list of AFSCs to be considered the "Unmatched AFSC"
    p["afscs"] = np.hstack((p["afscs"], "*"))

    # Add degree tier qualifications to the set of parameters
    def generate_degree_tier_qualifications():
        """
        I made this nested function, so I could have a designated section to generate degree qualifications and such
        """

        # Determine degree tiers and qualification information
        p['qual'] = np.array([['P1' for _ in range(M)] for _ in range(N)])
        p['Deg Tiers'] = np.array([[' ' * 10 for _ in range(4)] for _ in range(M)])
        for j in range(M):

            if p['acc_grp'][j] == 'Rated':  # All Degrees eligible for Rated
                p['qual'][:, j] = np.array(['P1' for _ in range(N)])
                p['Deg Tiers'][j, :] = ['P = 1', 'I = 0', '', '']

                # Pick 20% of the cadets at random to be ineligible for this Rated AFSC
                indices = random.sample(list(np.arange(N)), k=int(0.2 * N))
                p['qual'][indices, j] = 'I2'
            else:
                # Determine what tiers to use on this AFSC
                if N < 100:
                    random_number = np.random.rand()
                    if random_number < 0.2:
                        tiers = ['M1', 'I2']
                        p['Deg Tiers'][j, :] = ['M = 1', 'I = 0', '', '']
                    elif 0.2 < random_number < 0.4:
                        tiers = ['D1', 'P2']
                        target_num = round(np.random.rand(), 2)
                        p['Deg Tiers'][j, :] = ['D > ' + str(target_num), 'P < ' + str(1 - target_num), '', '']
                    elif 0.4 < random_number < 0.6:
                        tiers = ['P1']
                        p['Deg Tiers'][j, :] = ['P = 1', '', '', '']
                    else:
                        tiers = ['M1', 'P2']
                        target_num = round(np.random.rand(), 2)
                        p['Deg Tiers'][j, :] = ['M > ' + str(target_num), 'P < ' + str(1 - target_num), '', '']
                else:
                    random_number = np.random.rand()
                    if random_number < 0.1:
                        tiers = ['M1', 'I2']
                        p['Deg Tiers'][j, :] = ['M = 1', 'I = 0', '', '']
                    elif 0.1 < random_number < 0.2:
                        tiers = ['D1', 'P2']
                        target_num = round(np.random.rand(), 2)
                        p['Deg Tiers'][j, :] = ['D > ' + str(target_num), 'P < ' + str(1 - target_num), '', '']
                    elif 0.2 < random_number < 0.3:
                        tiers = ['P1']
                        p['Deg Tiers'][j, :] = ['P = 1', '', '', '']
                    elif 0.3 < random_number < 0.4:
                        tiers = ['M1', 'P2']
                        target_num = round(np.random.rand(), 2)
                        p['Deg Tiers'][j, :] = ['M > ' + str(target_num), 'P < ' + str(1 - target_num), '', '']
                    elif 0.4 < random_number < 0.5:
                        tiers = ['M1', 'D2', 'P3']
                        target_num_1 = round(np.random.rand() * 0.7, 2)
                        target_num_2 = round(np.random.rand() * (1 - target_num_1) * 0.8, 2)
                        target_num_3 = round(1 - target_num_1 - target_num_2, 2)
                        p['Deg Tiers'][j, :] = ['M > ' + str(target_num_1), 'D > ' + str(target_num_2),
                                                'P < ' + str(target_num_3), '']
                    elif 0.5 < random_number < 0.6:
                        tiers = ['D1', 'D2', 'P3']
                        target_num_1 = round(np.random.rand() * 0.7, 2)
                        target_num_2 = round(np.random.rand() * (1 - target_num_1) * 0.8, 2)
                        target_num_3 = round(1 - target_num_1 - target_num_2, 2)
                        p['Deg Tiers'][j, :] = ['D > ' + str(target_num_1), 'D > ' + str(target_num_2),
                                                'P < ' + str(target_num_3), '']
                    elif 0.6 < random_number < 0.7:
                        tiers = ['M1', 'D2', 'I3']
                        target_num = round(np.random.rand(), 2)
                        p['Deg Tiers'][j, :] = ['M > ' + str(target_num), 'D < ' + str(1 - target_num), 'I = 0', '']
                    elif 0.7 < random_number < 0.8:
                        tiers = ['M1', 'P2', 'I3']
                        target_num = round(np.random.rand(), 2)
                        p['Deg Tiers'][j, :] = ['M > ' + str(target_num), 'P < ' + str(1 - target_num), 'I = 0', '']
                    else:
                        tiers = ['M1', 'D2', 'P3', 'I4']
                        target_num_1 = round(np.random.rand() * 0.7, 2)
                        target_num_2 = round(np.random.rand() * (1 - target_num_1) * 0.8, 2)
                        target_num_3 = round(1 - target_num_1 - target_num_2, 2)
                        p['Deg Tiers'][j, :] = ['M > ' + str(target_num_1), 'D > ' + str(target_num_2),
                                                'P < ' + str(target_num_3), 'I = 0']

                # Generate the tiers for the cadets
                c_tiers = np.random.randint(0, len(tiers), N)
                p['qual'][:, j] = np.array([tiers[c_tiers[i]] for i in range(N)])

        # NxM qual matrices with various features
        p["ineligible"] = (np.core.defchararray.find(p['qual'], "I") != -1) * 1
        p["eligible"] = (p["ineligible"] == 0) * 1
        for t in [1, 2, 3, 4]:
            p["tier " + str(t)] = (np.core.defchararray.find(p['qual'], str(t)) != -1) * 1
        p["mandatory"] = (np.core.defchararray.find(p['qual'], "M") != -1) * 1
        p["desired"] = (np.core.defchararray.find(p['qual'], "D") != -1) * 1
        p["permitted"] = (np.core.defchararray.find(p['qual'], "P") != -1) * 1

        # NEW: Exception to degree qualification based on CFM ranks
        p["exception"] = (np.core.defchararray.find(p['qual'], "E") != -1) * 1

        # Initialize information for AFSC degree tiers
        p["t_count"] = np.zeros(p['M']).astype(int)
        p["t_proportion"] = np.zeros([p['M'], 4])
        p["t_leq"] = (np.core.defchararray.find(p["Deg Tiers"], "<") != -1) * 1
        p["t_geq"] = (np.core.defchararray.find(p["Deg Tiers"], ">") != -1) * 1
        p["t_eq"] = (np.core.defchararray.find(p["Deg Tiers"], "=") != -1) * 1
        p["t_mandatory"] = (np.core.defchararray.find(p["Deg Tiers"], "M") != -1) * 1
        p["t_desired"] = (np.core.defchararray.find(p["Deg Tiers"], "D") != -1) * 1
        p["t_permitted"] = (np.core.defchararray.find(p["Deg Tiers"], "P") != -1) * 1

        # Loop through each AFSC
        for j, afsc in enumerate(p["afscs"][:p['M']]):

            # Loop through each potential degree tier
            for t in range(4):
                val = p["Deg Tiers"][j, t]

                # Empty degree tier
                if 'M' not in val and 'D' not in val and 'P' not in val and 'I' not in val:
                # if val in ["nan", "", ''] or pd.isnull(val):
                    t -= 1
                    break

                # Degree Tier Proportion
                p["t_proportion"][j, t] = val.split(" ")[2]

            # Num tiers
            p["t_count"][j] = t + 1

        return p   # Return updated parameters
    p = generate_degree_tier_qualifications()

    # Cadet preferences
    utility = np.random.rand(N, M)  # random utility matrix
    max_util = np.max(utility, axis=1)
    p['utility'] = np.around(utility / np.array([[max_util[i]] for i in range(N)]), 2)
    p['c_preferences'], p['c_utilities'] = afccp.core.data.preferences.get_utility_preferences(p)
    p['c_preferences'] = p['c_preferences'][:, :P]
    p['c_utilities'] = p['c_utilities'][:, :P]

    # Get cadet preferences
    p["c_pref_matrix"] = np.zeros([p["N"], p["M"]]).astype(int)
    for i in range(p['N']):

        # Sort the utilities to get the preference list
        utilities = p["utility"][i, :p["M"]]
        sorted_indices = np.argsort(utilities)[::-1]
        preferences = np.argsort(
            sorted_indices) + 1  # Add 1 to change from python index (at 0) to rank (start at 1)
        p["c_pref_matrix"][i, :] = preferences

    # Update set of parameters
    p = afccp.core.data.adjustments.parameter_sets_additions(p)

    return p  # Return updated parameters

def generate_random_value_parameters(parameters, num_breakpoints=24):
    """
    Generates value parameters for a given problem instance from scratch
    """

    # Shorthand
    p = parameters

    # Objective to parameters lookup dictionary (if the parameter is in "p", we include the objective)
    objective_lookups = {'Norm Score': 'a_pref_matrix', 'Merit': 'merit', 'USAFA Proportion': 'usafa',
                         'Combined Quota': 'quota_d', 'USAFA Quota': 'usafa_quota', 'ROTC Quota': 'rotc_quota',
                         'Utility': 'utility', 'Male': 'male', 'Minority': 'minority', 'Mandatory': 'mandatory',
                         'Desired': 'desired', 'Permitted': 'permitted'}
    for t in ["1", "2", "3", "4"]:  # Add in AFOCD Degree tiers
        objective_lookups["Tier " + t] = "tier " + t

    # Add the AFSC objectives that are included in this instance (check corresponding parameters using dict above)
    objectives = np.array([objective for objective in objective_lookups if objective_lookups[objective] in p])

    # Initialize set of value parameters
    vp = {'objectives': objectives, 'cadets_overall_weight': np.random.rand(), 'O': len(objectives),
          'K': np.arange(len(objectives)), 'num_breakpoints': num_breakpoints, 'cadets_overall_value_min': 0,
          'afscs_overall_value_min': 0}
    vp['afscs_overall_weight'] = 1- vp['cadets_overall_weight']

    # Generate AFSC and cadet weights
    weight_functions = ['Linear', 'Direct', 'Curve_1', 'Curve_2', 'Equal']
    vp['cadet_weight_function'] = np.random.choice(weight_functions)
    vp['afsc_weight_function'] = np.random.choice(weight_functions)
    vp['cadet_weight'] = afccp.core.data.values.cadet_weight_function(p['merit_all'], func= vp['cadet_weight_function'])
    vp['afsc_weight'] = afccp.core.data.values.afsc_weight_function(p['pgl'], func = vp['afsc_weight_function'])

    # Stuff that doesn't matter here
    vp['cadet_value_min'], vp['afsc_value_min'] = np.zeros(p['N']), np.zeros(p['N'])
    vp['USAFA-Constrained AFSCs'], vp['Cadets Top 3 Constraint'] = '', ''
    vp['USSF OM'] = False

    # Initialize arrays
    vp['objective_weight'], vp['objective_target'] = np.zeros([p['M'], vp['O']]), np.zeros([p['M'], vp['O']])
    vp['constraint_type'] = np.zeros([p['M'], vp['O']])
    vp['objective_value_min'] = np.array([[' ' * 20 for _ in vp['K']] for _ in p['J']])
    vp['value_functions'] = np.array([[' ' * 200 for _ in vp['K']] for _ in p['J']])

    # Initialize breakpoints
    vp['a'] = [[[] for _ in vp['K']] for _ in p["J"]]
    vp['f^hat'] = [[[] for _ in vp['K']] for _ in p["J"]]

    # Initialize objective set
    vp['K^A'] = {}

    # Get AFOCD Tier objectives
    vp = afccp.core.data.values.generate_afocd_value_parameters(p, vp)
    vp['constraint_type'] = np.zeros([p['M'], vp['O']])  # Turn off all the constraints again

    # Loop through all AFSCs
    for j in p['J']:

        # Loop through all AFSC objectives
        for k, objective in enumerate(vp['objectives']):

            maximum, minimum, actual = None, None, None
            if objective == 'Norm Score':
                vp['objective_weight'][j, k] = (np.random.rand() * 0.2 + 0.3) * 100  # Scale up to 100
                vp['value_functions'][j, k] = 'Min Increasing|0.3'
                vp['objective_target'][j, k] = 1

            if objective == 'Merit':
                vp['objective_weight'][j, k] = (np.random.rand() * 0.4 + 0.05) * 100
                vp['value_functions'][j, k] = 'Min Increasing|-0.3'
                vp['objective_target'][j, k] = p['sum_merit'] / p['N']
                actual = np.mean(p['merit'][p['I^E'][j]])

            elif objective == 'USAFA Proportion':
                vp['objective_weight'][j, k] = (np.random.rand() * 0.3 + 0.05) * 100
                vp['value_functions'][j, k] = 'Balance|0.15, 0.15, 0.1, 0.08, 0.08, 0.1, 0.6'
                vp['objective_target'][j, k] = p['usafa_proportion']
                actual = len(p['I^D'][objective][j]) / len(p['I^E'][j])

            elif objective == 'Combined Quota':
                vp['objective_weight'][j, k] = (np.random.rand() * 0.8 + 0.2) * 100
                vp['value_functions'][j, k] = 'Quota_Normal|0.2, 0.25, 0.2'
                vp['objective_target'][j, k] = p['quota_d'][j]

                # Get bounds and turn on this constraint
                minimum, maximum = p['quota_min'][j], p['quota_max'][j]
                vp['objective_value_min'][j, k] = str(int(minimum)) + ", " + str(int(maximum))
                vp['constraint_type'][j, k] = 2

            elif objective == 'USAFA Quota':
                vp['objective_weight'][j, k] = 0
                vp['value_functions'][j, k] = 'Min Increasing|0.3'
                vp['objective_target'][j, k] = p['usafa_quota'][j]

                # Bounds on this constraint (but leave it off)
                vp['objective_value_min'][j, k] = str(int(p['usafa_quota'][j])) + ", " + \
                                                  str(int(p['quota_max'][j]))

            elif objective == 'ROTC Quota':
                vp['objective_weight'][j, k] = 0
                vp['value_functions'][j, k] = 'Min Increasing|0.3'
                vp['objective_target'][j, k] = p['rotc_quota'][j]

                # Bounds on this constraint (but leave it off)
                vp['objective_value_min'][j, k] = str(int(p['rotc_quota'][j])) + ", " + \
                                                  str(int(p['quota_max'][j]))

            elif objective == 'Male':
                vp['objective_weight'][j, k] = (np.random.rand() * 0.25) * 100
                vp['value_functions'][j, k] = 'Balance|0.15, 0.15, 0.1, 0.08, 0.08, 0.1, 0.6'
                vp['objective_target'][j, k] = p['male_proportion']
                actual = len(p['I^D'][objective][j]) / len(p['I^E'][j])

            elif objective == 'Minority':
                vp['objective_weight'][j, k] = (np.random.rand() * 0.25) * 100
                vp['value_functions'][j, k] = 'Balance|0.15, 0.15, 0.1, 0.08, 0.08, 0.1, 0.6'
                vp['objective_target'][j, k] = p['minority_proportion']
                actual = len(p['I^D'][objective][j]) / len(p['I^E'][j])

            # If we care about this objective, we load in its value function breakpoints
            if vp['objective_weight'][j, k] != 0:

                # Create the non-linear piecewise exponential segment dictionary
                segment_dict = afccp.core.data.values.create_segment_dict_from_string(
                    vp['value_functions'][j, k], vp['objective_target'][j, k],
                    minimum=minimum, maximum=maximum, actual=actual)

                # Linearize the non-linear function using the specified number of breakpoints
                vp['a'][j][k], vp['f^hat'][j][k] = afccp.core.data.values.value_function_builder(
                    segment_dict, num_breakpoints=num_breakpoints)

        # Scale the objective weights for this AFSC, so they sum to 1
        vp['objective_weight'][j] = vp['objective_weight'][j] / sum(vp['objective_weight'][j])
        vp['K^A'][j] = np.where(vp['objective_weight'][j] != 0)[0]

    return vp

# SDV functions (we may not have the SDV library!)
if afccp.core.globals.use_sdv:

    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    from sdv.sampling import Condition

    def train_ctgan(epochs=1000, printing=True, name='CTGAN_2024'):
        """
        Train CTGAN to produce realistic data based on the current "ctgan_data" file in the support sub-folder. This
        function then saves the ".pkl" file back to the support sub-folder
        """

        # Import data
        data = afccp.core.globals.import_csv_data(afccp.core.globals.paths['support'] + 'data/ctgan_data.csv')
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

        # load the constraints from the file
        model.load_custom_constraint_classes(
            filepath='afccp/core/data/custom_ctgan_constraints.py',
            class_names=['IfROTCNo11XX_U', 'IfUSAFANo11XX_R']
        )

        # create ROTC constraint using the class
        rotc_no_11XX_U = {
            'constraint_class': 'IfROTCNo11XX_U',
            'constraint_parameters': {
                'column_names': ['SOC', '11XX_U_Cadet'],
            }
        }
        constraints.append(rotc_no_11XX_U)

        # create USAFA constraint using the class
        usafa_no_11XX_R = {
            'constraint_class': 'IfUSAFANo11XX_R',
            'constraint_parameters': {
                'column_names': ['SOC', '11XX_R_Cadet'],
            }
        }
        constraints.append(usafa_no_11XX_R)

        # Add the constraints to the model
        model.add_constraints(constraints)

        # Train the model
        if printing:
            print("Training the model...")
        model.fit(data)

        # Save the model
        filepath = afccp.core.globals.paths["support"] + name + '.pkl'
        model.save(filepath)
        if printing:
            print("Model saved to", filepath)

    def generate_ctgan_instance(N=1600, name='CTGAN_2024', pilot_condition=False):
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
        filepath = afccp.core.globals.paths["support"] + name + '.pkl'
        model = CTGANSynthesizer.load(filepath)

        # Split up the number of ROTC/USAFA cadets
        N_usafa = round(np.random.triangular(0.25, 0.33, 0.4) * N)
        N_rotc = N - N_usafa

        # Pilot is by far the #1 desired career field, let's make sure this is represented here
        N_usafa_pilots = round(np.random.triangular(0.3, 0.4, 0.43) * N_usafa)
        N_usafa_generic = N_usafa - N_usafa_pilots
        N_rotc_pilots = round(np.random.triangular(0.25, 0.3, 0.33) * N_rotc)
        N_rotc_generic = N_rotc - N_rotc_pilots

        # Condition the data generated to produce the right composition of pilot first choice preferences
        usafa_pilot_first_choice = Condition(num_rows = N_usafa_pilots, column_values={'SOC': 'USAFA', '11XX_U_Cadet': 1})
        usafa_generic_cadets = Condition(num_rows=N_usafa_generic, column_values={'SOC': 'USAFA'})
        rotc_pilot_first_choice = Condition(num_rows=N_rotc_pilots, column_values={'SOC': 'ROTC', '11XX_R_Cadet': 1})
        rotc_generic_cadets = Condition(num_rows=N_rotc_generic, column_values={'SOC': 'ROTC'})

        # Sample data  (Sampling from conditions may take too long!)
        if pilot_condition:
            data = model.sample_from_conditions(conditions=[usafa_pilot_first_choice, usafa_generic_cadets,
                                                            rotc_pilot_first_choice, rotc_generic_cadets])
        else:
            data = model.sample(N)

        # Extract information from the generated data
        cadet_utility_cols = [col for col in data.columns if '_Cadet' in col]

        # Get list of AFSCs
        afscs = np.array([col[:-6] for col in cadet_utility_cols])

        # Initialize parameter dictionary
        p = {'afscs': afscs, 'N': N, 'P': len(afscs), 'M': len(afscs), 'race': np.array(data['Race']),
             'ethnicity': np.array(data['Ethnicity']), 'merit': np.array(data['Merit']), 'cadets': np.arange(N),
             'usafa': np.array(data['SOC'] == 'USAFA') * 1, 'male': np.array(data['Gender'] == 'Male') * 1,
             'cip1': np.array(data['CIP1']), 'cip2': np.array(data['CIP2']), 'num_util': 10,  # 10 utilities taken
             'rotc': np.array(data['SOC'] == 'ROTC'), 'I': np.arange(N), 'J': np.arange(len(afscs))}

        # Clean up degree columns (remove the leading "c" I put there if it's there)
        for i in p['I']:
            if p['cip1'][i][0] == 'c':
                p['cip1'][i] = p['cip1'][i][1:]
            if p['cip2'][i][0] == 'c':
                p['cip2'][i] = p['cip2'][i][1:]

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

        # Load in AFSCs data
        filepath = afccp.core.globals.paths["support"] + 'data/afscs_data.csv'
        afscs_data = afccp.core.globals.import_csv_data(filepath)

        # Add AFSC features to parameters
        p['acc_grp'] = np.array(afscs_data['Accessions Group'])
        p['afscs_stem'] = np.array(afscs_data['STEM'])
        p['Deg Tiers'] = np.array(afscs_data.loc[:, 'Deg Tier 1': 'Deg Tier 4'])

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

        # Cadet/AFSC initial utility matrices
        cadet_utility = np.array(data.loc[:, afscs[0] + '_Cadet':afscs[p['M'] - 1] + '_Cadet'])
        afsc_utility = np.array(data.loc[:, afscs[0] + '_AFSC':afscs[p['M'] - 1] + '_AFSC'])

        # Rated eligibility needs to match from both sources
        for i in p['I']:
            for j in p['J^Rated']:
                if cadet_utility[i, j] == 0:
                    afsc_utility[i, j] = 0
                if afsc_utility[i, j] == 0:
                    cadet_utility[i, j] = 0

        # Get the qual matrix to know what people are eligible for
        qual = afccp.core.data.support.cip_to_qual_tiers(p['afscs'], p['cip1'], p['cip2'])
        ineligible = (np.core.defchararray.find(qual, "I") != -1) * 1
        eligible = (ineligible == 0) * 1
        J_E = [np.where(eligible[i, :])[0] for i in p['I']]  # set of AFSCs that cadet i is eligible for
        I_E = [np.where(eligible[:, j])[0] for j in p['J']]  # set of cadets that are eligible for AFSC j

        # Create a default eligibility matrix including everyone
        p['eligible'] = np.ones((p['N'], p['M']))

        # Create the cadet preferences by sorting the utilities
        p['cadet_preferences'] = {}
        p['c_pref_matrix'] = np.zeros([p['N'], p['M']]).astype(int)  # Cadet preference matrix
        for i in p['I']:

            # Remove AFSC preferences if the cadet isn't eligible for them
            for j in np.where(cadet_utility[i, :] > 0)[0]:
                if j not in J_E[i]:
                    cadet_utility[i, j] = 0

            # Get cadet preferences (list of AFSC indices in order)
            ineligibles = np.where(cadet_utility[i, :] == 0)[0]
            num_ineligible = len(ineligibles)  # Ineligibles are going to be at the bottom of the list
            p['cadet_preferences'][i] = np.argsort(cadet_utility[i, :])[::-1][:p['M'] - num_ineligible]

            # Add AFSCs that this cadet is eligible for if they're not in this cadet's preference list
            for j in J_E[i]:
                if j in p['J^NRL']:  # USSF/Rated AFSCs require volunteers
                    if j not in p['cadet_preferences'][i]:
                        p['cadet_preferences'][i] = np.hstack((p['cadet_preferences'][i], j))

            # Create cadet preference matrix
            p['c_pref_matrix'][i, p['cadet_preferences'][i]] = np.arange(1, len(p['cadet_preferences'][i]) + 1)

        # Create the AFSC preferences by sorting the utilities
        p['afsc_preferences'] = {}
        p['a_pref_matrix'] = np.zeros([p['N'], p['M']]).astype(int)  # AFSC preference matrix
        for j in p['J']:

            # Remove cadets from this AFSC's preferences if the cadet is not eligible
            for i in np.where(afsc_utility[:, j] > 0)[0]:
                if i not in I_E[j]:
                    afsc_utility[i, j] = 0

            # Get AFSC preferences (list of cadet indices in order)
            ineligibles = np.where(afsc_utility[:, j] == 0)[0]
            num_ineligible = len(ineligibles)  # Ineligibles are going to be at the bottom of the list
            p['afsc_preferences'][j] = np.argsort(afsc_utility[:, j])[::-1][:p['N'] - num_ineligible]

            # Add cadets that are eligible for this AFSC if they're not on the AFSC's preference list
            if j in p['J^NRL']:  # USSF/Rated AFSCs require volunteers
                for i in I_E[j]:
                    if i not in p['afsc_preferences'][j]:
                        p['afsc_preferences'][j] = np.hstack((p['afsc_preferences'][j], i))

            # Create AFSC preference matrix
            p['a_pref_matrix'][p['afsc_preferences'][j], j] = np.arange(1, len(p['afsc_preferences'][j]) + 1)

        # Create "initial" cadet utility matrix from generated utility matrix
        p['utility'] = np.zeros((p['N'], p['M']))
        for i in p['I']:

            # Gather top 10 utilities from generated matrix
            p['utility'][i, p['cadet_preferences'][i][:10]] = np.around(
                cadet_utility[i, p['cadet_preferences'][i][:10]], 2)

            # First choice is a value of 1
            p['utility'][i, p['cadet_preferences'][i][0]] = 1

        # Create cadet preference and utility columns for Cadets.csv
        p['c_preferences'], p['c_utilities'] = \
            afccp.core.data.preferences.get_utility_preferences_from_preference_array(p)

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

        # Return parameters
        return p

