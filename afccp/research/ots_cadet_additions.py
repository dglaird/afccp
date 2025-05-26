# Import basic libraries
import numpy as np
import pandas as pd
import os

from afccp.data.generation import augment_2026_data_with_ots
from afccp import CadetCareerProblem


def conduct_ots_ctgan_instance_analysis(num_instances=1, N=3000, gen_data=True, correct_data=True, solve_model=True,
                                        import_name: str = '2026_0', solve_ots_guo_addition=False, instances=None):

    # Determine which instances to analyze
    if instances is None:
        instances = [f'2026O{i}' for i in np.arange(num_instances)]

    # Loop through each data instance and generate the data
    if gen_data:
        for instance_name in instances:

            # Check to see if we've already generated the data
            if f'{instance_name}' in os.listdir('instances/'):
                print(f'Data already generated for {instance_name}.')
                continue

            # We haven't generated the data yet, so do it now
            print(f'Generating data for {instance_name}...')
            augment_2026_data_with_ots(export_name=instance_name, N=N, import_name=import_name)

    # Loop through each data instance and correct the data
    if correct_data:
        for instance_name in instances:

            # Check to see if we've already corrected the data
            file_to_check = f'{instance_name} Cadets Utility (Final).csv'
            if file_to_check in os.listdir(f'instances/{instance_name}/4. Model Input/'):
                print(f'Data already corrected for {instance_name}.')
                continue

            # We haven't corrected the data already, so do it now
            print(f'Correcting data for {instance_name}...')
            instance = CadetCareerProblem(instance_name, printing=False)
            instance.make_all_initial_real_instance_modifications(  # Start with the VPs needed to fill OTS on top
                vp_defaults_filename='Value_Parameters_Defaults_2026O_OTS_Fill.xlsx')  # of OG One Market solution
            instance.parameter_sanity_check()
            instance.export_data()

    # Loop through each data instance and solve the different models!
    if solve_model:
        for instance_name in instances:
            solve_on_top_of_original_one_market_solution(instance_name, import_name, solve_ots_guo_addition)
            solve_full_one_market_potential(instance_name, import_name)
            solve_one_market_classification_only(instance_name)

    build_analysis_dataframe(instances=instances)


def solve_on_top_of_original_one_market_solution(instance_name, import_name, solve_ots_guo_addition):

    instance = CadetCareerProblem(instance_name, printing=False)
    instance.set_value_parameters()

    # Solve USAFA/ROTC rated algorithms for their components (If we haven't already)
    run_algorithm = True
    if instance.solutions is not None:
        if 'Rated USAFA HR (Matches)' in instance.solutions.keys():
            run_algorithm = False

            # If we've already solved on top of the original One Market solution, we break out of the function
            if 'One Market OTS Addition-GUO' in instance.solutions.keys():
                print(f'Data already solved (A) for {instance_name}.')
                return

    # Run the SOC rated algorithms (and print out status)
    instance.printing = True  # Turn printing back on
    print(f'Solving for OTS candidates on top of original One Market solution for {instance_name}...')
    if run_algorithm:
        instance.soc_rated_matching_algorithm({'soc': 'usafa', 'rated_alternates': True})
        instance.soc_rated_matching_algorithm({'soc': 'rotc', 'rated_alternates': True})
    instance.incorporate_rated_algorithm_results({'socs_to_use': ['usafa', 'rotc']})

    # Load in original AFSC solution vector (If we haven't already)
    if 'One Market ROTC_USAFA' not in instance.solutions.keys():
        filepath = f'instances/{import_name}/4. Model Input/{import_name} Solution.csv'
        solution_df = pd.read_csv(filepath)
        solution_df = solution_df.reindex(instance.parameters['I'])
        solution_df.loc[instance.parameters['I^OTS'], 'Original'] = '*'
        afsc_array_original = np.array(solution_df['Original'])

        # Incorporate original "One Market" solution from 2026
        instance.add_solution(afsc_array=afsc_array_original, method='One Market ROTC_USAFA')

    # Allocate the OTS candidates according to the "status quo" method of selection/assignment
    if 'One Market OTS Addition-Status Quo' not in instance.solutions.keys():  # (If we haven't already)
        instance.allocate_ots_candidates_original_method()  # New "complete" solution created here

    # Solve OTS rated algorithm (If we haven't already)
    if 'Rated OTS HR (Matches)' not in instance.solutions.keys():
        instance.soc_rated_matching_algorithm({'soc': 'ots', 'rated_alternates': True})
    instance.incorporate_rated_algorithm_results()

    # Solve the GUO model with the rated algorithm results! (If we haven't already)
    if 'One Market OTS Addition-GUO' not in instance.solutions.keys() and solve_ots_guo_addition:

        # Constrain ROTC/USAFA cadets to their One Market matches
        j_array = instance.solutions['One Market ROTC_USAFA']['j_array']
        for i, j in enumerate(j_array):
            if i not in instance.parameters['I^OTS']:
                instance.parameters['J^Fixed'][i] = j

        # Solve the model
        instance.solve_guo_pyomo_model(
            {'rated_alternates': True, 'pyomo_max_time': None, 'solve_castle_guo': False,
             'rated_alternate_afscs': ['11XX_R', '11XX_U', '11XX_O', '12XX', '13B', '18X'],
             'w^G': 0.8, 'usafa_soc_pilot_cross_in': False, 'solution_method': 'One Market OTS Addition-GUO'})

    # Export results
    instance.export_data(datasets='Solutions')


def solve_full_one_market_potential(instance_name, import_name):

    # Load in instance and determine if we need to solve the model
    instance = CadetCareerProblem(instance_name, printing=False)
    if 'Castle Market All SOCs-Pilot Flow' in instance.solutions.keys():
        print(f'Data already solved (B) for {instance_name}.')
        return

    # Print statements! We will solve the models
    print(f'Solving for full One Market model w/CASTLE input {instance_name}...')
    instance.printing = True

    # Import default value parameters to run the full One Market model for all USAFA/ROTC/OTS cadets
    instance.import_default_value_parameters(
        printing=True, vp_defaults_filename='Value_Parameters_Defaults_2026O_Full.xlsx')

    # Import cadets utility dataframe and set the original USAFA/ROTC utility constraints (Top 20% get Top 3)
    filepath = f'instances/{import_name}/4. Model Input/{import_name} Cadets Utility Constraints.csv'
    cadets_utility_df = pd.read_csv(filepath)
    cadet_value_min_og = np.array(cadets_utility_df['VP'])
    instance.value_parameters['cadet_value_min'][:len(cadet_value_min_og)] = cadet_value_min_og
    instance.update_value_parameters()
    instance.parameter_sanity_check()

    # Incorporate rated algorithm results
    instance.incorporate_rated_algorithm_results({'rated_alternates': True, 'usafa_soc_pilot_cross_in': True})

    # Solve the GUO model with the rated algorithm results!
    try:
        instance.solve_guo_pyomo_model(
            {'rated_alternates': True, 'pyomo_max_time': 600, 'solve_castle_guo': True,
             'rated_alternate_afscs': ['11XX_R', '11XX_U', '11XX_O', '12XX', '13B', '18X'],
             'ots_selected_preferences_only': True, 'ots_constrain_must_match': False,
             'w^G': 0.8, 'usafa_soc_pilot_cross_in': True, 'solution_method': 'Castle Market All SOCs-Pilot Flow'})
    except:
        print(f'MODEL "Castle Market All SOCs-Pilot Flow" infeasible for {instance_name}')

    # Export instance data
    instance.export_data()


def solve_one_market_classification_only(instance_name):

    # Load in instance and determine if we need to solve the model
    instance = CadetCareerProblem(instance_name, printing=False)
    if 'One Market All SOCs-Classification Only' in instance.solutions.keys():
        print(f'Data already solved (C) for {instance_name}.')
        return

    # Print statements! We will solve the models
    print(f'Solving for One Market model (Classification Only) {instance_name}...')
    instance.printing = True

    # Set Full One Market value parameters
    instance.set_value_parameters('VP2')
    instance.update_value_parameters()
    instance.parameter_sanity_check()

    # Incorporate rated algorithm results
    instance.incorporate_rated_algorithm_results({'rated_alternates': True, 'usafa_soc_pilot_cross_in': True})

    # Update the "Must Match" information
    instance.set_ots_must_matches()

    # Solve the GUO model with the rated algorithm results! (
    try:
        instance.solve_guo_pyomo_model(
            {'rated_alternates': True, 'pyomo_max_time': 600, 'solve_castle_guo': True,
             'rated_alternate_afscs': ['11XX_R', '11XX_U', '11XX_O', '12XX', '13B', '18X'],
             'ots_selected_preferences_only': False, 'ots_constrain_must_match': True,
             'w^G': 0.8, 'usafa_soc_pilot_cross_in': True,
             'solution_method': 'One Market All SOCs-Classification Only'})
    except:
        print(f'MODEL "One Market All SOCs-Classification Only" infeasible for {instance_name}')

    # Export instance data
    instance.export_data()


def build_analysis_dataframe(instances):

    # Initialize dataframe
    df = pd.DataFrame()
    i = 0

    # Loop through each data instance we have and load it in
    for instance_name in instances:
        instance = CadetCareerProblem(instance_name, printing=True)

        # Set Full One Market value parameters
        instance.set_value_parameters('VP2')
        instance.update_value_parameters()
        instance.parameter_sanity_check()
        instance.set_ots_must_matches()

        # Incorporate rated algorithm results
        instance.incorporate_rated_algorithm_results({'rated_alternates': True, 'usafa_soc_pilot_cross_in': True})

        # Loop through each solution
        solution_names = ['One Market OTS Addition-Status Quo',
                          'One Market All SOCs-Classification Only',
                          'Castle Market All SOCs-Pilot Flow']
        for solution_name in solution_names:
            df.loc[i, 'Instance'] = instance.data_name
            df.loc[i, 'Solution'] = solution_name
            if solution_name in instance.solutions.keys():
                instance.set_solution(solution_name)
                s = instance.solution
                p = instance.parameters
                df.loc[i, 'Feasible'] = True
                df.loc[i, 'CASTLE'] = round(s['z^CASTLE (Values)'], 4)
                df.loc[i, 'GUO'] = round(s['z^gu'], 4)
                df.loc[i, 'CASTLE-GUO'] = round(s['z^CASTLE'], 4)
                df.loc[i, 'Top 3 Cadet Choice %'] = round(s['top_3_choice_percent'], 4)
                df.loc[i, 'AFSC Norm Score'] = round(s['weighted_average_afsc_score'], 4)
                df.loc[i, 'AFSC NRL Norm Score'] = round(s['weighted_average_nrl_afsc_score'], 4)
                df.loc[i, 'Num OTS Non-Vols'] = len(s['I^Match-OTS']) - s['OTS Selected Pref Count']
                df.loc[i, "Num 'Must Matches'"] = len(np.intersect1d(p['I^Must_Match'], s['I^Match']))
                df.loc[i, 'OTS Average Merit'] = round(s['OTS Average Merit'], 4)
            else:
                df.loc[i, 'Feasible'] = False
            i += 1
    df.to_csv('OTS Analysis Data Results.csv', index=False)  # Export to csv!


