# Import libraries
import os

import pandas as pd

from afccp.core.comprehensive_functions import *
import datetime
import glob
import copy


def determine_afsc_plot_details(instance, results_chart=False):
    """
    Takes in the problem instance object and alters the plot parameters based on the kind of chart
    being generated as well as the type of data we're looking at
    """

    # Determine if we only want to view smaller AFSCs (those with fewer eligible cadets than the specified limit)
    if instance.plt_p["eligibility_limit"] is not None:
        instance.plt_p["num_afscs"] = sum([1 if len(
            instance.parameters['I^E'][j]) < instance.plt_p["eligibility_limit"] else 0 for j in
                                           instance.parameters['J']])
    else:
        instance.plt_p["eligibility_limit"] = instance.parameters['N']
        instance.plt_p["num_afscs"] = instance.parameters['M']

    # Determine if we are "skipping" AFSC labels in the x-axis
    if instance.plt_p["skip_afscs"] is None:
        if instance.data_variant == "Year":  # Real AFSCs don't get skipped!
            instance.plt_p["skip_afscs"] = False
        else:
            if instance.plt_p["num_afscs"] < instance.parameters['M']:
                instance.plt_p["skip_afscs"] = False
            else:  # "C2, C4, C6" could be appropriate
                instance.plt_p["skip_afscs"] = True

    # Determine if we are rotating the AFSC labels in the x-axis
    if instance.plt_p["afsc_rotation"] is None:
        if instance.plt_p["skip_afscs"]:  # If we skip the AFSCs, then we don't rotate them
            instance.plt_p["afsc_rotation"] = 0
        else:

            # We're not skipping the AFSCs, but we could rotate them
            if instance.data_variant == "Year":
                if instance.plt_p["num_afscs"] > 18:
                    instance.plt_p["afsc_rotation"] = 45
                else:
                    instance.plt_p["afsc_rotation"] = 0
            else:
                if instance.plt_p["num_afscs"] < 25:
                    instance.plt_p["afsc_rotation"] = 0
                else:
                    instance.plt_p["afsc_rotation"] = 45

    # Figure out which solutions to show, what the colors/markers are going to be, and some error handling
    if results_chart:

        # Only applies to AFSC charts
        if instance.plt_p["results_graph"] in ["Measure", "Value"]:
            if instance.plt_p["objective"] not in instance.plt_p["afsc_chart_versions"]:
                raise ValueError("Objective " + instance.plt_p["objective"] +
                                 " does not have any charts available")

            if instance.plt_p["version"] not in instance.plt_p["afsc_chart_versions"][instance.plt_p["objective"]]:

                if not instance.plt_p["compare_solutions"]:
                    raise ValueError("Objective '" + instance.plt_p["objective"] +
                                     "' does not have chart version '" + instance.plt_p["version"] + "'.")

            if instance.plt_p["objective"] == "Norm Score" and instance.plt_p["version"] == "quantity_bar_proportion":
                if "afsc_utility" not in instance.parameters:
                    raise ValueError("The AFSC Utility matrix is needed for the Norm Score "
                                     "'quantity_bar_proportion' chart. ")

        if instance.plt_p["solution_names"] is None:

            # Default to the current active solutions
            instance.plt_p["solution_names"] = list(instance.solution_dict.keys())
            num_solutions = len(instance.plt_p["solution_names"])  # Number of solutions in dictionary

            # Get number of solutions
            if instance.plt_p["num_solutions"] is None:
                instance.plt_p["num_solutions"] = num_solutions

            # Can't have more than 4 solutions
            if num_solutions > 4:
                instance.plt_p["num_solutions"] = 4
                instance.plt_p["solution_names"] = list(instance.solution_dict.keys())[:4]

        else:
            instance.plt_p["num_solutions"] = len(instance.plt_p["solution_names"])
            if instance.plt_p["num_solutions"] > 4 and instance.plt_p["results_graph"] != "Multi-Criteria Comparison":
                raise ValueError("Error. Can't have more than 4 solutions shown in the results plot.")

        # Don't need to do this for the Multi-Criteria Comparison chart
        if instance.plt_p["results_graph"] != "Multi-Criteria Comparison":

            # Load in the colors and markers for each of the solutions
            instance.plt_p["colors"], instance.plt_p["markers"], instance.plt_p["zorder"] = {}, {}, {}
            for s, solution in enumerate(instance.plt_p["solution_names"]):
                instance.plt_p["colors"][solution] = instance.plt_p["color_choices"][s]
                instance.plt_p["markers"][solution] = instance.plt_p["marker_choices"][s]
                instance.plt_p["zorder"][solution] = instance.plt_p["zorder_choices"][s]

            # This only applies to the Merit and USAFA proportion objectives
            if instance.plt_p["version"] == "large_only_bar":
                instance.plt_p["all_afscs"] = False
            else:
                instance.plt_p["all_afscs"] = True

        # Value Parameters
        if instance.plt_p["vp_name"] is None:
            instance.plt_p["vp_name"] = instance.vp_name

    return instance.plt_p


def initialize_instance_functional_parameters(N):
    """
    This function initializes all the various instance parameters including graphs,
    solving the different models, and much more. If the analyst wants to change the defaults, they just need
    to specify the new parameter in this initialization function or when passed in the method it is need
    """

    # Parameters for the graphs
    plt_p = {"save": True, "figsize": (19, 10), "facecolor": "white", "eligibility": True, "title": None,
             "filename": None, "display_title": True, "eligibility_limit": None, "label_size": 25,
             "afsc_tick_size": 20, "data_graph": "AFOCD Data", "yaxis_tick_size": 25, "bar_text_size": 15,
             "xaxis_tick_size": 20, "afsc_rotation": None, "dpi": 100, "bar_color": "black", "alpha": 1,
             "legend_size": 25, "skip_afscs": None, "results_graph": "Measure", "y_max": 1.1, "y_exact_max": None,
             "objective": "Merit", "compare_solutions": False, "solution_names": None, "vp_name": None,
             "color_choices": ["red", "blue", "green", "orange", "purple", "black", "magenta"],
             "marker_choices": ['o', 'D', '^', 'P', 'v', '*', 'h'], "sim_dot_size": 220,
             "dot_size": 100, "marker_size": 20, "all_afscs": True, "title_size": 25, "comparison_afscs": None,
             "zorder_choices": [2, 3, 2, 2, 2, 2, 2], "preference_chart": False, "preference_proportion": False,
             "dot_chart": False, "sort_by_pgl": True, "version": None, "solution_in_title": True, "afsc": None,
             "num_afscs_to_compare": 8, "comparison_criteria": ["Utility", "Merit", "AFOCD"], "text_size": 20,
             "num_solutions": None, "use_useful_charts": True, "new_similarity_matrix": True,
             "desired_charts": [("Combined Quota", "quantity_bar"), ("Norm Score", "quantity_bar_proportion"),
                                ("Utility", "quantity_bar_proportion"), ("Extra", "AFOCD_proportion"),
                                ("USAFA Proportion", "bar"), ("Merit", "bar")]}

    # AFSC Measure Chart Versions
    afsc_chart_versions = {"Merit": ["large_only_bar", "bar", "quantity_bar_gradient",
                                     "quantity_bar_proportion"],
                           "USAFA Proportion": ["large_only_bar", "bar", "preference_chart", "quantity_bar_proportion"],
                           "Male": ["bar", "preference_chart", "quantity_bar_proportion"],
                           "Combined Quota": ["dot", "quantity_bar"], "Minority": ["bar", "preference_chart",
                                                                                   "quantity_bar_proportion"],
                           "Utility": ["bar", "quantity_bar_gradient", "quantity_bar_proportion"],
                           "Mandatory": ["dot"], "Desired": ["dot"], "Permitted": ["dot"],
                           "Norm Score": ["dot", "quantity_bar_proportion"],
                           "Extra": ["AFOCD_proportion", "gender_preference"]}

    # Colors for the various bar types:
    colors = {"top_choices": "#5490f0", "mid_choices": "#eef09e", "bottom_choices": "#f25d50",
              "quartile_1": "#373aed", "quartile_2": "#0b7532", "quartile_3": "#d1bd4d", "quartile_4": "#cc1414",
              "Mandatory": "#311cd4", "Desired": "#085206", "Permitted": "#bda522", "Ineligible": "#f25d50",
              "male": "#6531d6", "female": "#73d631", "usafa": "#5ea0bf", "rotc": "#cc9460", "large_afscs": "#060d47",
              "small_afscs": "#cdddf7", "merit_above": "#c7b93a", "merit_within": "#3d8ee0", "minority": "#eb8436",
              "non-minority": "#b6eb6c", "merit_below": "#bf4343", "large_within": "#3d8ee0",
              "large_else": "#c7b93a", "pgl": "#5490f0", "surplus": "#eef09e", "failed_pgl": "#f25d50",
              "Volunteer": "#5490f0", "Non-Volunteer": "#f25d50"}

    # Add these elements to the main dictionary
    plt_p["afsc_chart_versions"] = afsc_chart_versions
    plt_p["bar_colors"] = colors

    # Parameters to solve the models
    mdl_p = {"pop_size": 12, "ga_max_time": 60, "num_crossover_points": 3, "initialize": True,
             "initial_solutions": None, "solution_names": None, "ga_printing": True,
             "mutation_rate": 0.05, "num_time_points": 100, "time_eval": False, "real_usafa_n": 960,
             "num_mutations": int(np.ceil(N / 75)), "percent_step": 10, "pareto_step": 10,
             "add_to_dict": True, "constraint_tolerance": 0.95, "ga_constrain_first_only": False,
             "pyomo_constraint_based": True, "set_to_instance": True, "initialize_new_heuristics": False,
             "solver_name": "cbc", "approximate": True, "pyomo_max_time": 10, "warm_start": None, "init_from_X": False,
             "report": False, "add_breakpoints": True, "populate": False, "iterate_from_quota": True,
             "population_additions": 5, "provide_executable": True, "executable": None, "get_reward": False,
             "con_term": False, "get_new_rewards_penalties": False, "use_gp_df": True, "exe_extension": False}

    return plt_p, mdl_p


def pick_most_changed_afscs(instance):
    """
    Checks the specified solutions for the instance "Multi-Criteria Comparison" chart and determines which
    AFSCs change the most in the solution based on the cadets that are assigned.
    """

    # Get necessary info
    p = instance.parameters
    assigned_cadets = {}
    max_assigned = np.zeros(p["M"])
    solution_names = instance.plt_p["solution_names"]
    num_solutions = len(solution_names)

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
    afscs = p["afsc_vector"][indices][:instance.plt_p["num_afscs_to_compare"]]
    return afscs


