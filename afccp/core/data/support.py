import os
import pandas as pd
import numpy as np
import datetime
import glob
import copy


# Data/Instance supporting functions
def initialize_instance_functional_parameters(N):
    """
    Initializes the various instance parameters for the CadetCareerProblem object.

    Parameters:
        N (int): The number of cadets in the problem instance.

    Returns:
        dict: A dictionary containing the initialized instance parameters.

    This function initializes the hyperparameters and toggles for the CadetCareerProblem object.
    It sets default values for various parameters that control the behavior and visualization of the problem instance.

    The parameters are organized into different sections, including graph parameters, AFSC chart versions,
    colors for different bar types, animation colors, value function chart parameters, and more.

    The function returns a dictionary containing all the initialized instance parameters.

    Note: The analyst can modify the default parameter values by specifying new values in this initialization
    function or by passing them as arguments when calling the CadetCareerProblem object methods.
    """

    # Parameters for the graphs
    mdl_p = {

        # Generic Solution Handling (for multiple algorithms/models)
        "initial_solutions": None, "solution_names": None, "add_to_dict": True, "set_to_instance": True,
        "initialize_new_heuristics": False, 'gather_all_metrics': True, 're-calculate x': True,

        # Matching Algorithm Parameters
        'ma_printing': False, 'capacity_parameter': 'quota_max', 'rotc_rated_board_afsc_order': None,
        'collect_solution_iterations': True, 'soc': 'usafa', 'incorporate_rated_results': True,
        'create_new_rated_solutions': True,

        # Genetic Matching Algorithm Parameters
        "gma_pop_size": 4, 'gma_max_time': 20, 'gma_num_crossover_points': 2, 'gma_mutations': 1,
        'gma_mutation_rate': 1, 'gma_printing': False, 'stopping_conditions': 'Time', 'gma_num_generations': 200,

        # Genetic Algorithm Parameters
        "pop_size": 12, "ga_max_time": 60, "num_crossover_points": 3, "initialize": True, "ga_printing": True,
        "mutation_rate": 0.05, "num_time_points": 100, "num_mutations": int(np.ceil(N / 75)), "time_eval": False,
        "percent_step": 10, "ga_constrain_first_only": False, 'mutation_function': 'cadet_choice',
        'preference_mutation_rate': 0.5,

        # Pyomo General Parameters
        "real_usafa_n": 960, "solver_name": "cbc", "pyomo_max_time": 10, "provide_executable": False,
        "executable": None, "exe_extension": False, 'alternate_list_iterations_printing': False,

        # Additional Constraints/Modeling
        "assignment_model_obj": "Global Utility", 'ussf_merit_bound': 0.03, 'ussf_soc_pgl_constraint': False,
        'ussf_soc_pgl_constraint_bound': 0.01, 'rated_alternates': True, 'USSF OM': False,
        'USAFA-Constrained AFSCs': None, 'BIG M': 100, 'solve_extra_components': False, 'rated_alternate_afscs': None,

        # VFT Model Parameters
        "pyomo_constraint_based": True, "constraint_tolerance": 0.95, "warm_start": None, "init_from_X": False,
        "obtain_warm_start_variables": True, "add_breakpoints": True, "approximate": True,

        # VFT Population Generation Parameters
        "populate": True, "iterate_from_quota": True, "max_quota_iterations": 5, "population_additions": 10,
        "population_generation_model": "Assignment",

        # Model Constraint Placement Algorithm parameters
        'constraint_model_to_use': 'Assignment', "skip_quota_constraint": False,

        # Sensitivity Analysis
        "pareto_step": 10,

        # Goal Programming Parameters
        "get_reward": False, "con_term": None, "get_new_rewards_penalties": False, "use_gp_df": True,

        # Value Parameter Generation
        "new_vp_weight": 100, "num_breakpoints": 24,

        # BubbleChart Parameters
        'b_figsize': (19, 10), 's': 1, 'fw': 100, 'circle_radius_percent': 0.8,
        'fh_ratio': 0.5, 'bw^t_ratio': 0.05, 'bw^l_ratio': 0, 'bw^r_ratio': 0, 'b_title': None,
        'bw^b_ratio': 0, 'bw^u_ratio': 0.02, 'abw^lr_ratio': 0.01, 'abw^ud_ratio': 0.02, 'b_title_size': 30,
        'lh_ratio': 0.1, 'lw_ratio': 0.1, 'dpi': 200, 'pgl_linestyle': '-', 'pgl_color': 'gray',
        'pgl_alpha': 0.5, 'surplus_linestyle': '--', 'surplus_color': 'white', 'surplus_alpha': 1,
        'usafa_pgl_color': 'blue', 'rotc_pgl_color': 'red', 'usafa_bubble': 'blue', 'rotc_bubble': 'red',
        'cb_edgecolor': 'black', 'save_board_default': True, 'circle_color': 'black', 'focus': 'Cadet Choice',
        'save_iteration_frames': True, 'afsc_title_size': 20, 'afsc_names_sized_box': False,
        'b_solver_name': 'couenne', 'b_pyomo_max_time': None, 'row_constraint': False, 'n^rows': 3,
        'simplified_model': True, 'use_pyomo_model': True, 'sort_cadets_by': 'AFSC Preferences', 'add_legend': False,
        'draw_containers': False, 'figure_color': 'black', 'text_color': 'white', 'afsc_text_to_show': 'Norm Score',
        'use_rainbow_hex': True, 'build_orientation_slides': True, 'b_legend': True, 'b_legend_size': 20,
        'b_legend_marker_size': 20, 'b_legend_title_size': 20, 'x_ext_left': 0, 'x_ext_right': 0, 'y_ext_left': 0,
        'y_ext_right': 0, 'show_rank_text': False, 'rank_text_color': 'white', 'fontsize_single_digit_adj': 0.6,
        'b_legend_loc': 'upper right', 'redistribute_x': True, 'cadets_solved_for': None,

        # These parameters pertain to the AFSCs that will ultimately show up in the visualizations
        'afscs_solved_for': 'All', 'afscs_to_show': 'All',

        # Generic Chart Handling
        "save": True, "figsize": (19, 10), "facecolor": "white", "title": None, "filename": None, "display_title": True,
        "label_size": 25, "afsc_tick_size": 20, "yaxis_tick_size": 25, "bar_text_size": 15, "xaxis_tick_size": 20,
        "afsc_rotation": None, "bar_color": "#3287cd", "alpha": 1, "legend_size": 25, "title_size": 25,
        "text_size": 15, 'text_bar_threshold': 400, 'dot_size': 35, 'legend_dot_size': 15, 'ncol': 1,
        "color_afsc_text_by_grp": True, "proportion_legend_size": 15, 'proportion_text_bar_threshold': 10,
        "square_figsize": (11, 10),

        # AFSC Chart Elements
        "eligibility": True, "eligibility_limit": None, "skip_afscs": None, "all_afscs": True, "y_max": 1.1,
        "y_exact_max": None, "preference_chart": False, "preference_proportion": False, "dot_chart": False,
        "sort_by_pgl": True, "solution_in_title": True, "afsc": None, "only_desired_graphs": True,
        'add_legend_afsc_chart': True, 'legend_loc': 'upper right', "add_bound_lines": False,
        "large_afsc_distinction": False,

        # Cadet Utility Chart Elements
        "cadet": 0, "util_type": "Final Utility",

        # Accessions Chart Elements
        "label_size_acc": 25, "acc_text_size": 25, "acc_bar_text_size": 25, "acc_legend_size": 15,
        "acc_text_bar_threshold": 10,

        # Macro Chart Controls
        "cadets_graph": True, "data_graph": "AFOCD Data", "results_graph": "Measure", "objective": "Merit",
        "version": "bar", "macro_chart_kind": "AFSC Chart",

        # Similarity Chart Elements
        "sim_dot_size": 220, "new_similarity_matrix": True, 'default_sim_color': 'black',
        'default_sim_marker': 'o',

        # Value Function Chart Elements
        "smooth_value_function": False,

        # Solution Comparison Chart Information
        "compare_solutions": False, "vp_name": None,
        "color_choices": ["red", "blue", "green", "orange", "purple", "black", "magenta"],
        "marker_choices": ['o', 'D', '^', 'P', 'v', '*', 'h'], "marker_size": 20, "comparison_afscs": None,
        "zorder_choices": [2, 3, 2, 2, 2, 2, 2], "num_solutions": None,

        # Multi-Criteria Chart
        "num_afscs_to_compare": 8, "comparison_criteria": ["Utility", "Merit", "AFOCD"],

        # Slide Parameters
        "ch_top": 2.35, "ch_left": 0.59, "ch_height": 4.64, "ch_width": 8.82,

        # Subset of charts I actually really care about
        "desired_charts": [("Combined Quota", "quantity_bar"), ("Norm Score", "quantity_bar_proportion"),
                           ("Norm Score", "bar"),
                           ("Utility", "quantity_bar_proportion"), ("Utility", "quantity_bar_choice"),
                           ("Merit", "bar"), ("USAFA Proportion", "quantity_bar_proportion"),
                           ("USAFA Proportion", "preference_chart"), ("Male", "preference_chart"),
                           ('Extra', 'Race Chart'),
                           ('Extra', 'Race Chart_proportion'), ('Extra', 'Ethnicity Chart'),
                           ('Extra', 'Ethnicity Chart_proportion'), ('Extra', 'Gender Chart'),
                           ('Extra', 'Gender Chart_proportion'), ('Extra', 'SOC Chart'),
                           ('Extra', 'SOC Chart_proportion')],

        "desired_comparison_charts": [('Utility', 'median_preference'), ('Combined Quota', 'dot'), ('Utility', 'dot'),
                                      ('Norm Score', 'dot'), ('Merit', 'dot'), ('Tier 1', 'dot'), ('Extra', 'Race Chart'),
                                      ('USAFA Proportion', 'dot'), ('Male', 'dot'), ('Utility', 'mean_preference')],

        'desired_other_charts': [("Accessions Group", "Race Chart"), ("Accessions Group", "Gender Chart"),
                                 ("Accessions Group", "SOC Chart"), ("Accessions Group", "Ethnicity Chart")]

    }

    # AFSC Measure Chart Versions
    afsc_chart_versions = {"Merit": ["large_only_bar", "bar", "quantity_bar_gradient", "quantity_bar_proportion"],
                           "USAFA Proportion": ["large_only_bar", "bar", "preference_chart", "quantity_bar_proportion"],
                           "Male": ["bar", "preference_chart", "quantity_bar_proportion"],
                           "Combined Quota": ["dot", "quantity_bar"],
                           "Minority": ["bar", "preference_chart", "quantity_bar_proportion"],
                           "Utility": ["bar", "quantity_bar_gradient", "quantity_bar_proportion", "quantity_bar_choice"],
                           "Mandatory": ["dot"], "Desired": ["dot"], "Permitted": ["dot"],
                           "Tier 1": ["dot"], "Tier 2": ["dot"], "Tier 3": ["dot"], "Tier 4": ["dot"],
                           "Norm Score": ["dot", "quantity_bar_proportion", "bar"],
                           'Extra': ['Race Chart', 'Race Chart_proportion', 'Gender Chart', 'SOC Chart',
                                     'Ethnicity Chart', 'Gender Chart_proportion', 'SOC Chart_proportion',
                                     'Ethnicity Chart_proportion']}

    # Colors for the various bar types:
    colors = {

        # Cadet Preference Charts
        "top_choices": "#5490f0", "mid_choices": "#eef09e", "bottom_choices": "#f25d50",
        "Volunteer": "#5490f0", "Non-Volunteer": "#f25d50", "Top 6 Choices": "#5490f0", "7+ Choice": "#f25d50",

        # Quartile Charts
        "quartile_1": "#373aed", "quartile_2": "#0b7532", "quartile_3": "#d1bd4d", "quartile_4": "#cc1414",

        # AFOCD Charts
        "Mandatory": "#311cd4", "Desired": "#085206", "Permitted": "#bda522", "Ineligible": "#f25d50",

        # Cadet Demographics
        "male": "#6531d6", "female": "#73d631", "usafa": "#5ea0bf", "rotc": "#cc9460", "minority": "#eb8436",
        "non-minority": "#b6eb6c",

        # Misc. AFSC Criteria  #cdddf7
        "large_afscs": "#060d47", "small_afscs": "#3287cd", "merit_above": "#c7b93a", "merit_within": "#3d8ee0",
        "merit_below": "#bf4343", "large_within": "#3d8ee0", "large_else": "#c7b93a",

        # Utility Chart Colors
        "Utility Ascribed": "#4793AF", "Normalized Rank": "#FFC470", "Not Bottom 3": "#DD5746",
        "Not Last Choice": "#8B322C",

        # PGL Charts
        "pgl": "#5490f0", "surplus": "#eef09e", "failed_pgl": "#f25d50",

        # Race Colors
        "American Indian/Alaska Native": "#d46013", "Asian": "#3ad413",
        "Black or African American": "#1100ff", "Native Hawaiian/Pacific Islander": "#d4c013",
        "Two or more races": "#ff0026", "Unknown": "#27dbe8", "White": "#a3a3a2",

        # Gender/SOC written differently (need to fix this later)
        "Male": "#6531d6", "Female": "#73d631", "USAFA": "#5ea0bf", "ROTC": "#cc9460",

        # Accessions group colors
        "All Cadets": "#000000", "Rated": "#ff0011", "USSF": "#0015ff", "NRL": "#000000",

        "Hispanic or Latino": "#66d4ce", "Not Hispanic": "#e09758", "Unknown Ethnicity": "#9e9e9e"

    }

    # Animation Colors
    choice_colors = {1: '#3700ff', 2: '#008dff', 3: '#00e7ff', 4: '#00FF93', 5: '#17FF00',  # #00ff04
                     6: '#BDFF00', 7: '#FFFF00', 8: '#FFCD00', 9: '#FF8700', 10: '#FF5100'}
    mdl_p['all_other_choice_colors'] = '#FF0000'
    mdl_p['choice_colors'] = choice_colors
    mdl_p['interest_colors'] = {'High': '#3700ff', 'Med': '#dad725', 'Low': '#ff9100', 'None': '#ff000a'}
    mdl_p['reserved_slot_color'] = "#ac9853"
    mdl_p['matched_slot_color'] = "#3700ff"
    mdl_p['unfocused_color'] = "#aaaaaa"
    mdl_p['unmatched_color'] = "#aaaaaa"
    mdl_p['exception_edge'] = "#FFD700"
    mdl_p['base_edge'] = 'black'

    # Add these elements to the main dictionary
    mdl_p["afsc_chart_versions"] = afsc_chart_versions
    mdl_p["bar_colors"] = colors

    # Value Function Chart parameters
    mdl_p['ValueFunctionChart'] = {'x_pt': None, 'y_pt': None, 'title': None, 'display_title': True, 'figsize': (10, 10),
                                   'facecolor': 'white', 'save': True, 'breakpoints': None, 'x_ticks': None,
                                   'crit_point': None, 'label_size': 25, 'yaxis_tick_size': 25, 'xaxis_tick_size': 25,
                                   'x_label': None, 'filepath': None, 'graph_color': 'black', 'breakpoint_color': 'black'}

    return mdl_p


def determine_afsc_plot_details(instance, results_chart=False):
    """
    Takes in the problem instance object and alters the plot parameters based on the kind of chart
    being generated as well as the type of data we're looking at
    """

    # Shorthand
    p, mdl_p = instance.parameters, instance.mdl_p

    # Get list of AFSCs we're showing
    mdl_p = determine_afscs_in_image(p, mdl_p)

    # Determine if we are "skipping" AFSC labels in the x-axis
    if mdl_p["skip_afscs"] is None:
        if instance.data_variant == "Year" or "CTGAN" in instance.data_name:  # Real AFSCs don't get skipped!
            mdl_p["skip_afscs"] = False
        else:
            if mdl_p["M"] < p['M']:
                mdl_p["skip_afscs"] = False
            else:  # "C2, C4, C6" could be appropriate
                mdl_p["skip_afscs"] = True

    # Determine if we are rotating the AFSC labels in the x-axis
    if mdl_p["afsc_rotation"] is None:
        if mdl_p["skip_afscs"]:  # If we skip the AFSCs, then we don't rotate them
            mdl_p["afsc_rotation"] = 0
        else:

            # We're not skipping the AFSCs, but we could rotate them
            if instance.data_variant == "Year" or "CTGAN" in instance.data_name:
                if mdl_p['M'] > 32:
                    mdl_p["afsc_rotation"] = 80
                elif mdl_p["M"] > 18:
                    mdl_p["afsc_rotation"] = 45
                else:
                    mdl_p["afsc_rotation"] = 0
            else:
                if mdl_p["M"] < 25:
                    mdl_p["afsc_rotation"] = 0
                else:
                    mdl_p["afsc_rotation"] = 45

    # Get AFSC
    if mdl_p["afsc"] is None:
        mdl_p["afsc"] = p['afscs'][0]

    # Get AFSC index
    j = np.where(p["afscs"] == mdl_p["afsc"])[0][0]

    # Get objective
    if mdl_p["objective"] is None:
        k = instance.value_parameters["K^A"][j][0]
        mdl_p["objective"] = instance.value_parameters['objectives'][k]

    # Figure out which solutions to show, what the colors/markers are going to be, and some error data
    if results_chart:

        # Only applies to AFSC charts
        if mdl_p["results_graph"] in ["Measure", "Value"]:
            if mdl_p["objective"] not in mdl_p["afsc_chart_versions"]:
                raise ValueError("Objective " + mdl_p["objective"] +
                                 " does not have any charts available")

            if mdl_p["version"] not in mdl_p["afsc_chart_versions"][mdl_p["objective"]]:

                if not mdl_p["compare_solutions"]:
                    raise ValueError("Objective '" + mdl_p["objective"] +
                                     "' does not have chart version '" + mdl_p["version"] + "'.")

            if mdl_p["objective"] == "Norm Score" and mdl_p["version"] == "quantity_bar_proportion":
                if "afsc_utility" not in p:
                    raise ValueError("The AFSC Utility matrix is needed for the Norm Score "
                                     "'quantity_bar_proportion' chart. ")

        if mdl_p["solution_names"] is None:

            # Default to the current active solutions
            mdl_p["solution_names"] = list(instance.solutions.keys())
            num_solutions = len(mdl_p["solution_names"])  # Number of solutions in dictionary

            # Get number of solutions
            if mdl_p["num_solutions"] is None:
                mdl_p["num_solutions"] = num_solutions

            # Can't have more than 4 solutions
            if num_solutions > 4:
                mdl_p["num_solutions"] = 4
                mdl_p["solution_names"] = list(instance.solutions.keys())[:4]

        else:
            mdl_p["num_solutions"] = len(mdl_p["solution_names"])
            if mdl_p["num_solutions"] > 4 and mdl_p["results_graph"] != "Multi-Criteria Comparison":
                raise ValueError("Error. Can't have more than 4 solutions shown in the results plot.")

        # Don't need to do this for the Multi-Criteria Comparison chart
        if mdl_p["results_graph"] != "Multi-Criteria Comparison":

            # Load in the colors and markers for each of the solutions
            mdl_p["colors"], mdl_p["markers"], mdl_p["zorder"] = {}, {}, {}
            for s, solution in enumerate(mdl_p["solution_names"]):
                mdl_p["colors"][solution] = mdl_p["color_choices"][s]
                mdl_p["markers"][solution] = mdl_p["marker_choices"][s]
                mdl_p["zorder"][solution] = mdl_p["zorder_choices"][s]

            # This only applies to the Merit and USAFA proportion objectives
            if mdl_p["version"] == "large_only_bar":
                mdl_p["all_afscs"] = False
            else:
                mdl_p["all_afscs"] = True

        # Value Parameters
        if mdl_p["vp_name"] is None:
            mdl_p["vp_name"] = instance.vp_name

    return mdl_p


def determine_afscs_in_image(p, mdl_p):
    """
    This function determines which AFSCs were solved for and which AFSCs should be in the visualization
    """

    # Determine what AFSCs we solved for
    if mdl_p['afscs_solved_for'] == 'All':
        mdl_p['afscs_in_solution'] = p['afscs'][:p['M']]  # All AFSCs (without unmatched)
    else:

        # We solved for some subset of Rated, NRL, or USSF AFSCs
        mdl_p['afscs_in_solution'] = []
        for acc_grp in ['Rated', 'NRL', 'USSF']:

            # If this accessions group was specified by the user
            if acc_grp in mdl_p['afscs_solved_for']:

                # Make sure this is an accessions group for which we have data
                if acc_grp in p['afscs_acc_grp']:

                    # Add each of these AFSCs to the list
                    for afsc in p['afscs_acc_grp'][acc_grp]:
                        mdl_p['afscs_in_solution'].append(afsc)

                # We don't have data on this group
                else:
                    raise ValueError(
                        "Error. Accessions group '" + str(acc_grp) + "' not found in this problem instance.")
        mdl_p['afscs_in_solution'] = np.array(mdl_p['afscs_in_solution'])  # Convert to numpy array

    # Now Determine what AFSCs we want to show in this visualization (must be a subset of AFSCs in the solution)
    if mdl_p['afscs_to_show'] == 'All':
        mdl_p['afscs'] = mdl_p['afscs_in_solution']  # All AFSCs that we solved for

    # If the user still supplied a string, we know we're looking for the three accessions groups
    elif type(mdl_p['afscs_to_show']) == str:

        # Loop through each of the three groups
        mdl_p['afscs'] = []
        for acc_grp in ['Rated', 'NRL', 'USSF']:

            # If this accessions group was specified by the user
            if acc_grp in mdl_p['afscs_to_show']:

                # Make sure this is an accessions group for which we have data
                if acc_grp in p['afscs_acc_grp']:

                    # Add each of these AFSCs to the list if they were also in the list of AFSCs we solved for
                    for afsc in p['afscs_acc_grp'][acc_grp]:
                        if afsc in mdl_p['afscs_in_solution']:
                            mdl_p['afscs'].append(afsc)

                # We don't have data on this group
                else:
                    raise ValueError(
                        "Error. Accessions group '" + str(acc_grp) + "' not found in this problem instance.")
        mdl_p['afscs'] = np.array(mdl_p['afscs'])  # Convert to numpy array

    # The user supplied a list of AFSCs
    else:

        # Loop through each AFSC in the supplied list and add it if it is also in the list of AFSCs we solved for
        mdl_p['afscs'] = []
        for afsc in mdl_p['afscs_to_show']:
            if afsc in mdl_p['afscs_in_solution']:
                mdl_p['afscs'].append(afsc)
        mdl_p['afscs'] = np.array(mdl_p['afscs'])  # Convert to numpy array

    # New set of AFSC indices
    mdl_p['J'] = np.array([np.where(p['afscs'] == afsc)[0][0] for afsc in mdl_p['afscs']])
    mdl_p['M'] = len(mdl_p['J'])

    # Determine if we only want to view smaller AFSCs (those with fewer eligible cadets than the specified limit)
    if mdl_p["eligibility_limit"] is not None:

        # Update set of AFSCs
        mdl_p['J'] = np.array([j for j in mdl_p['J'] if len(p['I^E'][j]) < mdl_p["eligibility_limit"]])
        mdl_p['afscs'] = np.array([p['afscs'][j] for j in mdl_p['J']])
        mdl_p['M'] = len(mdl_p['J'])
    else:
        mdl_p["eligibility_limit"] = p['N']

    return mdl_p


def pick_most_changed_afscs(instance):
    """
    Checks the specified solutions for the instance "Multi-Criteria Comparison" chart and determines which
    AFSCs change the most in the solution based on the cadets that are assigned.
    """

    # Get necessary info
    p = instance.parameters
    assigned_cadets = {}
    max_assigned = np.zeros(p["M"])
    solution_names = instance.mdl_p["solution_names"]

    # Loop through each solution to get the max number of cadets assigned to each AFSC across each solution
    for solution_name in solution_names:
        solution = instance.solutions[solution_name]
        assigned_cadets[solution_name] = {j: np.where(solution['j_array'] == j)[0] for j in p["J"]}
        num_assigned = np.array([len(assigned_cadets[solution_name][j]) for j in p["J"]])
        max_assigned = np.array([max(max_assigned[j], num_assigned[j]) for j in p["J"]])

    # Loop through each AFSC to get the number of cadets shared across each solution for each AFSC
    shared_cadet_count = np.zeros(p["M"])
    for j in p["J"]:

        # Pick the first solution as a baseline
        baseline_cadets = assigned_cadets[solution_names[0]][j]

        # Loop through each cadet assigned to this AFSC in the "baseline" solution
        for i in baseline_cadets:

            # Check if the cadet is assigned to this AFSC in all solutions
            cadet_in_each_solution = True
            for solution_name in solution_names:
                if i not in assigned_cadets[solution_name][j]:
                    cadet_in_each_solution = False
                    break

            # If the cadet is assigned in all solutions, we add one to the shared count
            if cadet_in_each_solution:
                shared_cadet_count[j] += 1

    # The difference between the maximum number of cadets assigned to a given AFSC across all solutions and the number
    # of cadets that are common to said AFSC across all solutions is our "Delta"
    delta_afsc = max_assigned - shared_cadet_count

    # Return the AFSCs that change the most
    indices = np.argsort(delta_afsc)[::-1]
    afscs = p["afsc_vector"][indices][:instance.mdl_p["num_afscs_to_compare"]]
    return afscs


# Important function! This is the CIP -> Qual Matrix function!!
def cip_to_qual_tiers(afscs, cip1, cip2=None, cip3=None, business_hours=None, true_tiers=True):
    """
    Generate qualification tiers for cadets based on their CIP codes and AFSCs.

    AFOCD c/ao Oct '23

    This function calculates qualification tiers for a list of cadets based on their CIP (Classification of Instructional
    Programs) codes and the specified AFSCs. The qualification tiers consider both tier and requirement (e.g., M1, D2).

    Args:
        afscs (list of str): A list of Air Force Specialty Codes (AFSCs) to determine cadet qualifications for.
        cip1 (numpy array): A numpy array containing CIP codes for the primary degrees of the cadets.
        cip2 (numpy array, optional): A numpy array containing CIP codes for the secondary degrees of the cadets.
        cip3 (numpy array, optional): A numpy array containing CIP codes for a third "source of truth".
        business_hours (numpy array, optional): An array indicating the number of business hours each cadet works.
        true_tiers (bool, optional): Set to True to use more accurate qualification tiers (as of June 2023).

    Returns:
        numpy array: A qualification matrix representing the qualification tiers for each cadet and AFSC.

    Details:
    - The function calculates qualification tiers for both primary and, if specified, secondary degrees of the cadets.
    - The qualification tiers are generated based on the CIP codes and the specified AFSCs.
    - The `true_tiers` parameter allows you to choose between more accurate tiers (as of June 2023) or the
      official tiers defined in the AFOCD (Air Force Officer Classification Directory).
    ```
    """

    # AFOCD CLASSIFICATION
    N = len(cip1)  # Number of Cadets
    M = len(afscs)  # Number of AFSCs we're building the qual matrix for

    # Dictionary of CIP codes
    cips = {1: cip1, 2: cip2, 3: cip3}

    # List of CIP degrees that are not empty
    degrees = []
    for d in cips:
        if cips[d] is not None:
            degrees.append(d)

    # Initialize qual dictionary
    qual = {}

    # Loop through both sets of degrees (if applicable)
    for d in degrees:

        # Initialize qual matrix (for this set of degrees)
        qual[d] = np.array([["I5" for _ in range(M)] for _ in range(N)])

        # Loop through each cadet and AFSC pair
        for i in range(N):
            cip = str(cips[d][i])
            cip = "0" * (6 - len(cip)) + cip
            for j, afsc in enumerate(afscs):

                # Rated Career Fields
                if afsc in ["11U", "11XX", "12XX", "13B", "18X", "92T0", "92T1", "92T2", "92T3",
                            "11XX_R", "11XX_U", "USSF", "USSF_U", "USSF_R"]:
                    qual[d][i, j] = "P1"

                # Aerospace Physiologist
                elif afsc == '13H':  # Proportions/Degrees Updated Oct '23
                    if cip in ['302701', '260912', '310505', '260908', '260707', '260403']:
                        qual[d][i, j] = 'M1'
                    elif cip in ['290402', '261501', '422813'] or cip[:4] in ['2609']:
                        qual[d][i, j] = 'P2'
                    else:
                        qual[d][i, j] = 'I3'

                # Airfield Ops
                elif afsc == '13M':  # Proportions Updated Oct '23
                    if cip == '290402' or cip[:4] == '4901':
                        qual[d][i, j] = 'D1'
                    elif cip[:4] in ['5201', '5202', '5206', '5207', '5211', '5212', '5213',
                                     '5214', '5218', '5219', '5220']:
                        qual[d][i, j] = 'D2'
                    else:
                        qual[d][i, j] = 'P3'

                # Nuclear and Missile Operations
                elif afsc == '13N':  # Updated Oct '23
                    qual[d][i, j] = 'P1'

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

                # Weather and Environmental Sciences (Current a/o Apr '24 AFOCD)
                elif afsc == '15W':
                    if cip[:4] == '4004':
                        qual[d][i, j] = 'M1'
                    elif cip in ['270301', '270303', '270304', '303501', '303001', '140802',
                                 '303801', '141201', '141301', '400601', '400605', '400607', '400801', '400805',
                                 '400806', '400807', '400809']:
                        qual[d][i, j] = 'P2'
                    elif cip[:2] in ['40'] or cip in ['040999', '030104', '110102', '110101', '110803', '110201',
                                                      '110701', '110802', '110104', '110804']:
                        qual[d][i, j] = 'P3'

                    else:
                        qual[d][i, j] = 'I4'

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
                elif afsc == '21A':  # Proportions Updated Oct '23
                    d_list4 = ['5202', '5206', '1101', '1102', '1103', '1104', '1107', '1110', '5212']
                    if cip[:2] == '14':
                        qual[d][i, j] = 'D1'
                    elif cip[:4] in d_list4 or cip[:2] == '40' or cip in ['151501', '520409', '490104', '490101']:
                        qual[d][i, j] = 'D2'
                    else:
                        qual[d][i, j] = 'P3'

                # Munitions and Missile Maintenance
                elif afsc == '21M':  # Proportions Updated Oct '23
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

                # Logistics Readiness: Conversations w/CFMs changed this!
                elif afsc == '21R':
                    if true_tiers:  # More accurate than current AFOCD a/o Oct '23
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
                elif afsc == '31P':  # Updated Oct '23
                    qual[d][i, j] = 'P1'

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

                # Civil Engineering: Electrical Engineer  *added 1447 per CFM conversation 2 Jun '23
                elif afsc == '32EXE':
                    if cip[:4] in ['1410', '1447']:
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Civil Engineering: Mechanical Engineer
                elif afsc == '32EXF':
                    if cip == '141901':
                        qual[d][i, j] = 'M1'
                    else:
                        qual[d][i, j] = 'I2'

                # Civil Engineering: General Engineer  *Updated AFOCD a/o 30 Apr '23 w/further adjustments a/o 2 Jun '23
                elif afsc == '32EXG':
                    if cip[:4] in ['1408', '1410'] or cip in ['140401', '141401', '141901', '143301', '143501',
                                                              '144701']:
                        qual[d][i, j] = 'M1'
                    elif cip in ["140701"] or cip[:4] in ["1405", "1425", "1402", "5220", '1510']:  # added 1510
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
                    if cip[:4] == '1402':
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
                    if cip[:2] == '14' and cip != '140102' and cip[:4] != "1437":
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

                # This shouldn't happen... but here's some error checking!
                else:
                    raise ValueError("Error. AFSC '" + str(afsc) + "' not a valid AFSC that this code recognizes.")

    # If no other CIP list is specified, we just take the qual matrix from the first degrees
    if len(degrees) == 1:
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

                if cip3 is not None:
                    qual_3 = qual[3][i, j]
                else:
                    qual_3 = "I9"  # Dummy value

                # Determine which qualification is best
                if int(qual_1[1]) < int(qual_2[1]):  # D1 beats D2
                    if int(qual_1[1]) < int(qual_3[1]): # D1 beats D3
                        qual_matrix[i, j] = qual_1  # Qual 1 wins!
                    else:  # D3 beats D1
                        qual_matrix[i, j] = qual_3  # Qual 3 wins!
                else:  # D2 beats D1
                    if int(qual_2[1]) < int(qual_3[1]):  # D2 beats D3
                        qual_matrix[i, j] = qual_2  # Qual 2 wins!
                    else:  # D3 beats D2
                        qual_matrix[i, j] = qual_3  # Qual 3 wins!

    return qual_matrix


