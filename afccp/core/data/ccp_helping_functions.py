# Import libraries
import os

import pandas as pd
import numpy as np
import datetime
import glob
import copy


def determine_afsc_plot_details(instance, results_chart=False):
    """
    Takes in the problem instance object and alters the plot parameters based on the kind of chart
    being generated as well as the type of data we're looking at
    """

    # Determine if we only want to view smaller AFSCs (those with fewer eligible cadets than the specified limit)
    if instance.mdl_p["eligibility_limit"] is not None:
        instance.mdl_p["num_afscs"] = sum([1 if len(
            instance.parameters['I^E'][j]) < instance.mdl_p["eligibility_limit"] else 0 for j in
                                           instance.parameters['J']])
    else:
        instance.mdl_p["eligibility_limit"] = instance.parameters['N']
        instance.mdl_p["num_afscs"] = instance.parameters['M']

    # Determine if we are "skipping" AFSC labels in the x-axis
    if instance.mdl_p["skip_afscs"] is None:
        if instance.data_variant == "Year":  # Real AFSCs don't get skipped!
            instance.mdl_p["skip_afscs"] = False
        else:
            if instance.mdl_p["num_afscs"] < instance.parameters['M']:
                instance.mdl_p["skip_afscs"] = False
            else:  # "C2, C4, C6" could be appropriate
                instance.mdl_p["skip_afscs"] = True

    # Determine if we are rotating the AFSC labels in the x-axis
    if instance.mdl_p["afsc_rotation"] is None:
        if instance.mdl_p["skip_afscs"]:  # If we skip the AFSCs, then we don't rotate them
            instance.mdl_p["afsc_rotation"] = 0
        else:

            # We're not skipping the AFSCs, but we could rotate them
            if instance.data_variant == "Year":
                if instance.mdl_p["num_afscs"] > 18:
                    instance.mdl_p["afsc_rotation"] = 45
                else:
                    instance.mdl_p["afsc_rotation"] = 0
            else:
                if instance.mdl_p["num_afscs"] < 25:
                    instance.mdl_p["afsc_rotation"] = 0
                else:
                    instance.mdl_p["afsc_rotation"] = 45

    # Get AFSC
    if instance.mdl_p["afsc"] is None:
        instance.mdl_p["afsc"] = instance.parameters['afscs'][0]

    # Get AFSC index
    j = np.where(instance.parameters["afscs"] == instance.mdl_p["afsc"])[0][0]

    # Get objective
    if instance.mdl_p["objective"] is None:
        k = instance.value_parameters["K^A"][j][0]
        instance.mdl_p["objective"] = instance.value_parameters['objectives'][k]

    # Figure out which solutions to show, what the colors/markers are going to be, and some error data
    if results_chart:

        # Only applies to AFSC charts
        if instance.mdl_p["results_graph"] in ["Measure", "Value"]:
            if instance.mdl_p["objective"] not in instance.mdl_p["afsc_chart_versions"]:
                raise ValueError("Objective " + instance.mdl_p["objective"] +
                                 " does not have any charts available")

            if instance.mdl_p["version"] not in instance.mdl_p["afsc_chart_versions"][instance.mdl_p["objective"]]:

                if not instance.mdl_p["compare_solutions"]:
                    raise ValueError("Objective '" + instance.mdl_p["objective"] +
                                     "' does not have chart version '" + instance.mdl_p["version"] + "'.")

            if instance.mdl_p["objective"] == "Norm Score" and instance.mdl_p["version"] == "quantity_bar_proportion":
                if "afsc_utility" not in instance.parameters:
                    raise ValueError("The AFSC Utility matrix is needed for the Norm Score "
                                     "'quantity_bar_proportion' chart. ")

        if instance.mdl_p["solution_names"] is None:

            # Default to the current active solutions
            instance.mdl_p["solution_names"] = list(instance.solution_dict.keys())
            num_solutions = len(instance.mdl_p["solution_names"])  # Number of solutions in dictionary

            # Get number of solutions
            if instance.mdl_p["num_solutions"] is None:
                instance.mdl_p["num_solutions"] = num_solutions

            # Can't have more than 4 solutions
            if num_solutions > 4:
                instance.mdl_p["num_solutions"] = 4
                instance.mdl_p["solution_names"] = list(instance.solution_dict.keys())[:4]

        else:
            instance.mdl_p["num_solutions"] = len(instance.mdl_p["solution_names"])
            if instance.mdl_p["num_solutions"] > 4 and instance.mdl_p["results_graph"] != "Multi-Criteria Comparison":
                raise ValueError("Error. Can't have more than 4 solutions shown in the results plot.")

        # Don't need to do this for the Multi-Criteria Comparison chart
        if instance.mdl_p["results_graph"] != "Multi-Criteria Comparison":

            # Load in the colors and markers for each of the solutions
            instance.mdl_p["colors"], instance.mdl_p["markers"], instance.mdl_p["zorder"] = {}, {}, {}
            for s, solution in enumerate(instance.mdl_p["solution_names"]):
                instance.mdl_p["colors"][solution] = instance.mdl_p["color_choices"][s]
                instance.mdl_p["markers"][solution] = instance.mdl_p["marker_choices"][s]
                instance.mdl_p["zorder"][solution] = instance.mdl_p["zorder_choices"][s]

            # This only applies to the Merit and USAFA proportion objectives
            if instance.mdl_p["version"] == "large_only_bar":
                instance.mdl_p["all_afscs"] = False
            else:
                instance.mdl_p["all_afscs"] = True

        # Value Parameters
        if instance.mdl_p["vp_name"] is None:
            instance.mdl_p["vp_name"] = instance.vp_name

    return instance.mdl_p


def initialize_instance_functional_parameters(N):
    """
    This function initializes all the various instance parameters including graphs,
    solving the different models, and much more. If the analyst wants to change the defaults, they just need
    to specify the new parameter in this initialization function or when passed in the method it is need
    """

    # Parameters for the graphs
    mdl_p = {

        # Parameters for the animation (CadetBoardFigure)
        'board_kind': 'Solution', 'b_figsize': (20, 10), 's': 1, 'fw': 100, 'animation_type': None,
        'fh_ratio': 0.5, 'bw^t_ratio': 0.05, 'bw^l_ratio': 0, 'bw^r_ratio': 0, 'b_title': None,
        'bw^b_ratio': 0, 'bw^u_ratio': 0.02, 'abw^lr_ratio': 0.01, 'abw^ud_ratio': 0.02, 'b_title_size': 30,
        'lh_ratio': 0.1, 'lw_ratio': 0.1, 'dpi': 200, 'pgl_linestyle': '-', 'pgl_color': 'gray',
        'pgl_alpha': 0.5, 'surplus_linestyle': '--', 'surplus_color': 'white', 'surplus_alpha': 1,
        'cb_edgecolor': 'black', 'save_board_default': True, 'circle_color': 'black', 'focus': 'Cadet Choice',
        'save_iteration_frames': True, 'afsc_title_size': 20, 'afscs_solved_for': 'All', 'afsc_names_sized_box': False,
        'b_solver_name': 'couenne', 'b_pyomo_max_time': None, 'row_constraint': False, 'n^rows': 3,
        'simplified_model': True, 'use_pyomo_model': True, 'cadets_solved_for': 'All', 'afscs_to_show': 'All',
        'sort_cadets_by': 'AFSC Preferences', 'add_legend': False, 'draw_containers': False, 'figure_color': 'black',
        'text_color': 'white', 'afsc_text_to_show': 'Norm Score', 'use_rainbow_hex': True,

        # Generic Chart Handling
        "save": True, "figsize": (19, 10), "facecolor": "white", "title": None, "filename": None, "display_title": True,
        "label_size": 25, "afsc_tick_size": 20, "yaxis_tick_size": 25, "bar_text_size": 15, "xaxis_tick_size": 20,
        "afsc_rotation": None, "bar_color": "black", "alpha": 1, "legend_size": 25,  "title_size": 25,
        "text_size": 20,

        # AFSC Chart Elements
        "eligibility": True, "eligibility_limit": None, "skip_afscs": None, "all_afscs": True, "y_max": 1.1,
        "y_exact_max": None, "preference_chart": False, "preference_proportion": False, "dot_chart": False,
        "sort_by_pgl": True, "solution_in_title": True, "afsc": None, "use_useful_charts": True,
        "desired_charts": [("Combined Quota", "quantity_bar"), ("Norm Score", "quantity_bar_proportion"),
                           ("Utility", "quantity_bar_proportion"), ("Extra", "AFOCD_proportion"),
                           ("USAFA Proportion", "bar"), ("Merit", "bar")],

        # Macro Chart Controls
        "cadets_graph": True, "data_graph": "AFOCD Data", "results_graph": "Measure", "objective": "Merit",
        "version": "1",

        # Similarity Chart Elements
        "sim_dot_size": 220, "new_similarity_matrix": True,

        # Value Function Chart Elements
        "x_point": None, "dot_size": 100, "smooth_value_function": False,

        # Solution Comparison Chart Information
        "compare_solutions": False, "vp_name": None,
        "color_choices": ["red", "blue", "green", "orange", "purple", "black", "magenta"],
        "marker_choices": ['o', 'D', '^', 'P', 'v', '*', 'h'], "marker_size": 20, "comparison_afscs": None,
        "zorder_choices": [2, 3, 2, 2, 2, 2, 2], "num_solutions": None,

        # Multi-Criteria Chart
        "num_afscs_to_compare": 8, "comparison_criteria": ["Utility", "Merit", "AFOCD"],

        # Generic Solution Handling (for multiple algorithms/models)
        "initial_solutions": None, "solution_names": None, "add_to_dict": True, "set_to_instance": True,
        "initialize_new_heuristics": False, 'gather_all_metrics': True,

        # Matching Algorithm Parameters
        'ma_printing': False, 'capacity_parameter': 'quota_max',

        # Genetic Algorithm Parameters
        "pop_size": 12, "ga_max_time": 60, "num_crossover_points": 3, "initialize": True, "ga_printing": True,
        "mutation_rate": 0.05, "num_time_points": 100, "num_mutations": int(np.ceil(N / 75)), "time_eval": False,
        "percent_step": 10, "ga_constrain_first_only": False,

        # Pyomo General Parameters
        "real_usafa_n": 960, "solver_name": "cbc", "pyomo_max_time": 10, "provide_executable": False,
        "executable": None, "exe_extension": False,

        # VFT Model Parameters
        "pyomo_constraint_based": True, "constraint_tolerance": 0.95, "warm_start": None, "init_from_X": False,
        "obtain_warm_start_variables": False, "add_breakpoints": True, "approximate": True,

        # VFT Population Generation Parameters (Including Pareto)
        "populate": False, "iterate_from_quota": True, "max_quota_iterations": 5, "population_additions": 5,
        "skip_quota_constraint": False, "pareto_step": 10,

        # Goal Programming Parameters
        "get_reward": False, "con_term": None, "get_new_rewards_penalties": False, "use_gp_df": True,

        # Value Parameter Generation
        "new_vp_weight": 100, "num_breakpoints": 24,
    }

    # AFSC Measure Chart Versions
    afsc_chart_versions = {"Merit": ["large_only_bar", "bar", "quantity_bar_gradient", "quantity_bar_proportion"],
                           "USAFA Proportion": ["large_only_bar", "bar", "preference_chart", "quantity_bar_proportion"],
                           "Male": ["bar", "preference_chart", "quantity_bar_proportion"],
                           "Combined Quota": ["dot", "quantity_bar"],
                           "Minority": ["bar", "preference_chart", "quantity_bar_proportion"],
                           "Utility": ["bar", "quantity_bar_gradient", "quantity_bar_proportion"],
                           "Mandatory": ["dot"], "Desired": ["dot"], "Permitted": ["dot"],
                           "Norm Score": ["dot", "quantity_bar_proportion"],
                           "Extra": ["AFOCD_proportion", "gender_preference"]}

    # Colors for the various bar types:
    colors = {

        # Cadet Preference Charts
        "top_choices": "#5490f0", "mid_choices": "#eef09e", "bottom_choices": "#f25d50",
        "Volunteer": "#5490f0", "Non-Volunteer": "#f25d50",

        # Quartile Charts
        "quartile_1": "#373aed", "quartile_2": "#0b7532", "quartile_3": "#d1bd4d", "quartile_4": "#cc1414",

        # AFOCD Charts
        "Mandatory": "#311cd4", "Desired": "#085206", "Permitted": "#bda522", "Ineligible": "#f25d50",

        # Cadet Demographics
        "male": "#6531d6", "female": "#73d631", "usafa": "#5ea0bf", "rotc": "#cc9460", "minority": "#eb8436",
        "non-minority": "#b6eb6c",

        # Misc. AFSC Criteria
        "large_afscs": "#060d47", "small_afscs": "#cdddf7", "merit_above": "#c7b93a", "merit_within": "#3d8ee0",
        "merit_below": "#bf4343", "large_within": "#3d8ee0", "large_else": "#c7b93a",

        # PGL Charts
        "pgl": "#5490f0", "surplus": "#eef09e", "failed_pgl": "#f25d50",
    }

    # Animation Colors
    choice_colors = {1: '#3700ff', 2: '#008dff', 3: '#00e7ff', 4: '#00ff8a', 5: '#00ff10',
                     6: '#b4ff00', 7: '#f8fe01', 8: '#ffa100', 9: '#ff5d00', 10: '#ff1c00'}
    mdl_p['all_other_choice_colors'] = '#ff000a'
    mdl_p['choice_colors'] = choice_colors

    # Add these elements to the main dictionary
    mdl_p["afsc_chart_versions"] = afsc_chart_versions
    mdl_p["bar_colors"] = colors

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
        solution = instance.solution_dict[solution_name]
        assigned_cadets[solution_name] = {j: np.where(solution == j)[0] for j in p["J"]}
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


