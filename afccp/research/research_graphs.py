import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import random
from math import *
from afccp.core.globals import *

# Set matplotlib default font to Times New Roman
import matplotlib as mpl

mpl.rc('font', family='Times New Roman')


# Older Functions
def Pyomo_Model_Approx_Exact_Time_Steps_Chart(results_df_dict, title=None, figsize=(19, 7), facecolor='white',
                                              save=False):
    """
    This is the chart that compares pyomo models and solvers according to VFT objective values and times
    :param results_df_dict: dictionary of different dataframes containing times/z columns
    :param save: If we should save the figure
    :param title: the title of the chart
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :return: figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)
    colors = ['blue', 'green', 'yellow', 'orange', 'cyan', 'red', 'purple', 'lime', 'pink', 'brown', 'magenta',
              'darkblue']
    i = 0
    legend_elements = []
    for df_name in results_df_dict.keys():
        df = results_df_dict[df_name]
        data_names = []
        for column_name in df.keys():
            data_name = column_name.split()[0]
            if data_name not in data_names:
                data_names.append(data_name)

        for data_name in data_names:
            times = np.array(df[data_name + ' Time'])
            z = np.array(df[data_name + ' Value'])
            ax.plot(times, z, color=colors[i], linewidth=5)
            legend_elements.append(Patch(facecolor=colors[i], label=data_name + ' ' + df_name))
            i += 1

    if title is None:
        title = 'Approximate & Exact Model Solver Performance'

    ax.set_title(title)
    ax.set_ylabel('Real Objective Value')
    ax.set_xlabel('Time')
    ax.legend(handles=legend_elements, edgecolor='black')
    if save:
        fig.savefig(title + '.png')
    return fig


def Pyomo_Model_Approx_Exact_Performance_Chart(results_df, colors=None, title=None, figsize=(19, 7), facecolor='white',
                                               save=False):
    """
    This is the chart that compares the exact and approximate models in solve time and objective value
    :param colors: colors of problem instances
    :param results_df: dataframe of approximate and exact model times and values
    :param title: the title of the chart
    :param save: If we should save the figure
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :return: figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)
    if colors is None:
        colors = ['green', 'purple', 'orange', 'yellow', 'red', 'darkgrey', 'magenta', 'lime', 'turquoise', 'darkblue',
                  'sandybrown', 'cyan', 'lightgreen', 'coral', 'pink', 'black']
        colors = colors[:len(results_df)]
    markers = ['o', 's']
    data_names = np.array(results_df.loc[:, 'Data'])
    legend_elements = []
    for m, model_type in enumerate(['Approximate', 'Exact']):
        x = np.array(results_df.loc[:, model_type + ' Model Times'])
        y = np.array(results_df.loc[:, model_type + ' Model Z'])
        ax.scatter(x, y, c=colors, marker=markers[m], s=70)

    for i, name in enumerate(data_names):
        legend_elements.append(Patch(facecolor=colors[i], label=name))

    if title is None:
        title = 'Approximate vs Exact Model Performance'

    ax.set_title(title)
    ax.set_ylabel('Objective Value (Z)')
    ax.set_xlabel('Time (seconds)')
    ax.legend(handles=legend_elements, edgecolor='black')
    if save:
        fig.savefig(title + '.png')
    return fig


# Thesis: Data Generation
def CIP_Hist_Chart(cip_proportions, colors=None, save=False, figsize=(16, 10), facecolor="white",
                   display_title=False, title=None):
    """
    This procedure takes a dictionary of CIP bin proportions and creates a histogram of this
    information
    :param display_title:
    :param cip_proportions: dictionary of CIP bin proportions
    :param colors: colors for the different datasets
    :param save: Whether we should save the graph
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :param title: If we should include a title or not
    :return: figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)

    names = list(cip_proportions.keys())
    num_data = len(names)
    num_groups = len(cip_proportions[names[0]])

    if colors is None:
        if 'Base' not in names:
            colors = ['blue', 'green', 'magenta', 'orange', 'red', 'cyan', 'red', 'lime', 'pink', 'chocolate',
                      'deepskyblue', 'gold', 'deeppink', 'sandybrown', 'olive', 'maroon', 'navy', 'coral', 'teal',
                      'darkorange']
        else:
            colors = ['green', 'yellow', 'orange', 'purple', 'red', 'cyan', 'magenta', 'lime', 'pink', 'chocolate',
                      'deepskyblue', 'gold', 'deeppink', 'sandybrown', 'olive', 'maroon', 'navy', 'coral', 'teal',
                      'darkorange']
            colors = colors[:num_data - 1]
            colors.append('blue')

    x = np.arange(num_groups)
    width = 1

    for i, name in enumerate(names):
        ax.bar(x=x, height=cip_proportions[name], width=width, label=name, edgecolor='black',
               color=colors[i], alpha=0.5, align='edge', linewidth=0.2)
    # Text
    ax.set_ylabel('Cadet CIP Proportions')
    ax.yaxis.label.set_size(35)
    ax.set(xlim=(0, num_groups + 1))
    ax.set_xlabel('CIP Bins')
    ax.xaxis.label.set_size(35)
    if title is None:
        title = 'CIP Proportions Histogram'

    # Adjust axes and ticks
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    y_ticks = [0.02, 0.04, 0.06, 0.08, 0.10]
    ax.set_yticks(y_ticks)

    ax.legend(edgecolor='black', fontsize=35, loc='upper right',
              ncol=2, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.5, borderpad=0.2)

    if display_title:
        ax.set_title(title)

    if save:
        fig.savefig(title + '.png')

    return fig


def Average_Utility_Data_Graph(data_dict=None, years=False, colors=None, save=False, figsize=(16, 10),
                               facecolor='white', averaged=False, title=None, display_title=False,
                               parameters=None, afscs=None):
    """
    This procedure takes in either a dataframe or a set of parameters and calculates the average utility placed
    for each AFSC in the data
    :param display_title:
    :param years: if we're comparing years or not
    :param averaged: if we want to average across the data sets
    :param colors: colors of data
    :param data_dict: dictionary of data sets
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :param save: Whether we should save the graph
    :param parameters: fixed cadet/AFSC data
    :param title: If we should include a title or not
    :return: figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)

    if data_dict is not None:

        # Data variables
        names = list(data_dict.keys())
        num_data = len(names)

        if colors is None:

            if years:
                colors = ['blue', 'green', 'magenta', 'orange', 'red', 'cyan', 'red', 'lime', 'pink', 'chocolate',
                          'deepskyblue', 'gold', 'deeppink', 'sandybrown', 'olive', 'maroon', 'navy', 'coral', 'teal',
                          'darkorange']
            else:
                colors = ['green', 'yellow', 'orange', 'purple', 'red', 'cyan', 'magenta', 'lime', 'pink', 'chocolate',
                          'deepskyblue', 'gold', 'deeppink', 'sandybrown', 'olive', 'maroon', 'navy', 'coral', 'teal',
                          'darkorange']

            if 'Base' in names:
                colors = colors[:num_data - 1]
                colors.append('blue')

        # Baseline Data
        baseline_data = data_dict[names[num_data - 1]]
        N = len(baseline_data)  # number of cadets

        # Get Utility Matrix
        base_preferences = np.array(baseline_data.loc[:, 'NRat1':'NRat6']).astype(str)
        base_utilities = np.array(baseline_data.loc[:, 'NrWgt2':'NrWgt6']).astype(float)
        base_utilities = np.hstack((np.ones((N, 1)), base_utilities))
        if afscs is None:
            afscs = np.unique(base_preferences)[1:]
        M = len(afscs)  # number of AFSCs
        base_utility = np.zeros([N, M])
        for i in range(N):
            for p in range(6):
                j = np.where(base_preferences[i, p] == afscs)[0]
                if len(j) != 0:
                    base_utility[i, j[0]] = base_utilities[i, p]

        # Get Baseline Averages
        base_averages = np.zeros(M)
        for j in range(M):
            base_averages[j] = round(np.mean(base_utility[:, j]), 2)

        if averaged:
            synthetic_averages = []

        # Loop through all sets of data (except baseline) to compare against real
        if years:
            d_nums = range(num_data)
        else:
            d_nums = range(num_data - 1)
        for d_num in d_nums:

            # Get Utility Matrix
            name = names[d_num]
            N = len(data_dict[name])
            preferences = np.array(data_dict[name].loc[:, 'NRat1':'NRat6']).astype(str)
            utilities = np.array(data_dict[name].loc[:, 'NrWgt2':'NrWgt6']).astype(float)
            utilities = np.hstack((np.ones((N, 1)), utilities))
            utility = np.zeros([N, M])
            for i in range(N):
                for p in range(6):
                    j = np.where(preferences[i, p] == afscs)[0]
                    if len(j) != 0:
                        utility[i, j[0]] = utilities[i, p]

            # Get Averages
            averages = np.zeros(M)
            for j in range(M):
                averages[j] = round(np.mean(utility[:, j]), 2)

            if averaged:
                synthetic_averages.append(averages)

            # Bar Chart
            width = 0.4

            if years:
                locations = np.arange(M)
            else:
                locations = np.arange(M) + width

            if not averaged:
                ax.bar(locations, averages, width, align='edge', label=name, alpha=0.5,
                       color=colors[d_num], linewidth=0.2, edgecolor='black')

        if averaged:
            synthetic_averages = np.array(synthetic_averages)
            synthetic_averages = np.mean(synthetic_averages, axis=0)

            # Bar Chart
            width = 0.4
            locations = np.arange(M) + width
            ax.bar(locations, synthetic_averages, width, align='edge', label='CTGAN',
                   alpha=0.5, color=colors[0], linewidth=0.2, edgecolor='black')

        # Baseline Data Bar Chart
        if not years:
            locations = np.arange(M)
            ax.bar(locations, base_averages, width, align='edge',
                   label=names[num_data - 1], alpha=0.5, color=colors[num_data - 1],
                   linewidth=0.2, edgecolor='black')

        # Title and Labels
        if title is None:
            title = 'Average Utilities Across Data Types'

        if display_title:
            ax.set_title(title)

    # Axis labels
    ax.set_ylabel('Average Utility')
    ax.yaxis.label.set_size(35)
    ax.set_xlabel('AFSCs')
    ax.xaxis.label.set_size(35)

    # Axis ticks
    tick_indices = np.arange(1, M, 2).astype(int)
    ax.set(xticks=tick_indices + width, xticklabels=afscs[tick_indices])
    ax.set_xticklabels(afscs[tick_indices], rotation=0)
    ax.set(xlim=(-width, M))
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)

    ax.legend(edgecolor='black', fontsize=35, loc='upper right',
              ncol=2, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.5, borderpad=0.2)

    if save:
        fig.savefig(title + '.png', bbox_inches='tight')

    return fig


# Thesis: Performance - Solver Results
def Solver_Size_Performance_Chart(time_dict, z_dict, title=None, figsize=(19, 7), averaged=False,
                                  display_title=True, facecolor='white', y_ax_zero=False, save=False):
    """
    This is the chart that compares problem sizes on time and optimality
    :param averaged: if the chart is averaged data or not
    :param y_ax_zero: if we want to start the y axis at 0 or not
    :param display_title: if we want to display a title or not for the chart
    :param time_dict: dictionary of times
    :param z_dict: dictionary of objective values
    :param title: the title of the chart
    :param save: If we should save the figure
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :return: figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)

    if averaged:
        colors = ['black']
        models = ['Averaged']
    else:
        colors = ['blue', 'green', 'orange', 'purple', 'red', 'yellow']
        models = list(time_dict.keys())
        colors = colors[:len(models)]

    legend_elements = []
    for m, model in enumerate(models):
        x = list(time_dict[model])
        y = list(z_dict[model])
        ax.scatter(x, y, c=colors[m], s=50)
        ax.plot(x, y, c=colors[m])
        legend_elements.append(Patch(facecolor=colors[m], label=model))

    if title is None:
        title = 'Problem Size Solution Performance'

    if display_title:
        ax.set_title(title)

    ax.set_ylabel('Objective Value (Z)')
    ax.set_xlabel('Time (seconds)')

    ax.legend(handles=legend_elements, edgecolor='black')
    if save:
        fig.savefig(title + '.png')

    return fig


# Thesis: Performance - Solution Technique Results
def Solution_Analysis_Performance_Chart(results_df, time_tables_dict, colors=None, title=None, figsize=(16, 10),
                                        display_title=True, facecolor='white', y_ax_zero=False, save=False):
    """
    This is the chart that compares the approximate and exact models (with genetic algorithm) in solve time and
    objective value
    :param y_ax_zero: if we want to start the y axis at 0 or not
    :param display_title: if we want to display a title or not for the chart
    :param time_tables_dict: dictionary of time tables
    :param colors: colors of problem instances
    :param results_df: dataframe of approximate and exact model times and values
    :param title: the title of the chart
    :param save: If we should save the figure
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :return: figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)
    if colors is None:
        colors = ['blue', 'green', 'orange', 'purple', 'red', 'yellow']
        colors = colors[:len(results_df)]
    markers = ['o', '^', 's', 'v']
    data_names = np.array(results_df.loc[:, 'Instances'])
    vps = []
    for data_name in data_names:
        vps.append(data_name.split(' ')[1])

    y_max = 0
    x_max = 0
    legend_elements = []
    for m, model in enumerate(['Approximate', 'Approximate_GA', 'Exact', 'Full_GA']):
        x = np.array(results_df.loc[:, model + ' Model Time']) / 60
        y = np.array(results_df.loc[:, model + ' Model Z'])
        x_max = max(x_max, max(x))
        if max(y) > y_max:
            y_max = max(y)
        ax.scatter(x, y, c=colors, marker=markers[m], s=70)
        element = mlines.Line2D([], [], color='black', marker=markers[m], linestyle='None',
                                markeredgecolor='black', markersize=10, label=model)
        legend_elements.append(element)

    for model in ['Approximate', 'Full']:
        for i, vp in enumerate(vps):
            x = time_tables_dict[model][vp].loc[:, 'Time']
            y = time_tables_dict[model][vp].loc[:, 'Objective Value']
            ax.plot(x, y, c=colors[i])

    # legend_elements = []
    # for i, name in enumerate(data_names):
    #     legend_elements.append(Patch(facecolor=colors[i], label=str(i + 1)))

    if title is None:
        data_name = data_names[0].split(' ')[0]
        title = data_name + ' Solution Performance'

    if display_title:
        ax.set_title(title)

    ax.set_ylabel('Objective Value (Z)')
    ax.yaxis.label.set_size(35)
    ax.set_xlabel('Time (minutes)')
    ax.xaxis.label.set_size(35)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    # ax.set(xlim=[-0.1, x_max + 3])
    if y_ax_zero:
        ax.set(ylim=[0, y_max + 0.1])

    ax.legend(handles=legend_elements, edgecolor='black', fontsize=20, ncol=4, loc='lower right', columnspacing=0.8,
              handletextpad=0.25, borderaxespad=0.5, borderpad=0.4)
    if save:
        fig.savefig(title + '.png')
    return fig


def Solution_Technique_Full_Results_Chart(results_df=None, figsize=(16, 10),
                                          display_title=False, facecolor='white', y_ax_zero=False,
                                          lines=True, save=True):
    """
    This is the chart that compares the approximate and exact models (with genetic algorithm) in solve time and
    objective value on all 60 instances
    :param lines: if we want to plot lines connecting the instances
    :param y_ax_zero: if we want to start the y axis at 0 or not
    :param display_title: if we want to display a title or not for the chart
    :param results_df: dataframe of approximate and exact model times and values
    :param save: If we should save the figure
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :return: figure
    """

    # Gather graph data
    if results_df is None:
        results_df = pd.read_excel(paths['Tables'] + 'Thesis_Appendix_full_solution_technique.xlsx',
                                   sheet_name='Results Averaged',
                                   engine='openpyxl')
        size = 300

    # Get colors and models
    colors = ['black', 'blue', 'red', 'green', 'magenta']
    models = ['Random', 'Approximate', 'Approximate_GA', 'Exact', 'Full_GA']
    markers = ['s', 'o', '*', '>', '<']

    # Create figure object
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)

    # Get coordinates
    y_max = 0
    # legend_elements = []
    x_dict = {}
    y_dict = {}
    for m, model in enumerate(models):

        x_dict[model] = np.array(results_df.loc[:, model + ' Model Time']) / 60
        y_dict[model] = np.array(results_df.loc[:, model + ' Model Z'])
        if max(y_dict[model]) > y_max:
            y_max = max(y_dict[model])
        # element = mlines.Line2D([], [], color=colors[m], marker=markers[m], linestyle='None',
        #                         markeredgecolor='black', markersize=20, label=model)
        # legend_elements.append(element)

    # Plot lines
    if lines:
        for i in range(len(results_df)):
            for m in [1, 3]:
                model1 = models[m]
                model2 = models[m + 1]
                point1 = [x_dict[model1][i], y_dict[model1][i]]
                point2 = [x_dict[model2][i], y_dict[model2][i]]
                x_values, y_values = [point1[0], point2[0]], [point1[1], point2[1]]
                ax.plot(x_values, y_values, color='black', linestyle='--', zorder=1)

    # Plot points
    for m, model in enumerate(models):
        ax.scatter(x_dict[model], y_dict[model], c=colors[m], marker=markers[m], edgecolor='black', zorder=2,
                   s=size)
        if model == 'Random':
            ax.text(x_dict[model], y_dict[model] + 0.03, model, fontsize=30)
        elif model == 'Approximate':
            ax.text(x_dict[model], y_dict[model] - 0.05, model, fontsize=30)
        elif model == 'Approximate_GA':
            ax.text(x_dict[model] - 1, y_dict[model] - 0.05, "GA w/Approximate", fontsize=30)
        elif model == 'Exact':
            ax.text(x_dict[model] - 1, y_dict[model] - 0.05, model, fontsize=30)
        elif model == 'Full_GA':
            ax.text(x_dict[model] - 3.5, y_dict[model] - 0.05, "GA w/Exact", fontsize=30)

    # Title
    title = 'Solution Technique Performance Results Averaged'
    if display_title:
        ax.set_title(title)

    # Axis labels
    ax.set_ylabel('Objective Value (Z)')
    ax.set_xlabel('Time (minutes)')
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.yaxis.label.set_size(35)
    ax.xaxis.label.set_size(35)

    if y_ax_zero:
        ax.set(ylim=[0, y_max + 0.1])

    # ax.legend(handles=legend_elements, edgecolor='black', fontsize=35)
    if save:
        fig.savefig(paths['Charts & Figures'] + title + '.png')
    fig.show()
    return fig


