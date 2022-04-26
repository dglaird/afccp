# Import libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import numpy as np
from afccp.core.globals import *

# Set matplotlib default font to Times New Roman
import matplotlib as mpl
mpl.rc('font', family='Times New Roman')


def data_graph(parameters, save=False, figsize=(19, 7), facecolor='white', eligibility=True,
               title=None, display_title=True, num=None, label_size=25, afsc_tick_size=15, graph='AFOCD Data',
               yaxis_tick_size=25, afsc_rotation=80, dpi=100, bar_color='black',
               alpha=0.5, title_size=None, legend_size=None, skip_afscs=True):
    """
    Creates the fixed parameter data plots. These figures show the characteristics of the cadets as they pertain
    to each AFSC as well as each AFSC objective
    :param skip_afscs:  Whether we should label every other AFSC
    :param parameters:  fixed cadet/AFSC parameters
    :param save: If we should save the figure or not
    :param figsize: size of the figure
    :param facecolor: color of the figure
    :param eligibility: if we want to plot average utility across eligible cadets for each AFSC, or all cadets
    :param title: title of the figure
    :param display_title: if we want to show the title of the figure
    :param num: number of eligible cadets for each AFSC (determines which AFSCs to show)
    :param label_size: font size of the labels
    :param afsc_tick_size: font size of the AFSC labels
    :param graph: Which graph to show
    :param yaxis_tick_size: font size of the y axis tick marks
    :param afsc_rotation: how much to rotate the AFSC labels
    :param dpi: dots per inch for figure
    :param bar_color: color of bars for figure (for certain kinds of graphs)
    :param alpha: alpha parameter for the bars of the figure
    :param title_size: font size of the title
    :param legend_size: font size of the legend
    :return: figure
    """

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True, dpi=dpi)

    # Set defaults
    if title_size is None:
        title_size = label_size
    if legend_size is None:
        legend_size = label_size

    # Load the data
    N = parameters['N']
    M = parameters['M']
    if num is None:
        num = N
    eligible_count = np.array([len(parameters['I^E'][j]) for j in range(M)])

    # Get applicable afscs
    indices = np.where(eligible_count <= num)[0]
    afscs = parameters['afsc_vector'][indices]
    M = len(afscs)

    # We can skip AFSCs
    if skip_afscs:
        tick_indices = np.arange(1, M, 2).astype(int)
    else:
        tick_indices = np.arange(M).astype(int)

    # Create figures
    add_legend = False
    if graph in ['Average Merit', 'USAFA Proportion', 'Average Utility']:

        # Get correct metric
        if graph == 'Average Merit':
            metric = np.array([np.mean(parameters['merit'][parameters['I^E'][j]]) for j in indices])
            target = 0.5
        elif graph == 'USAFA Proportion':
            metric = np.array([len(parameters['I^D'][graph][j]) / len(parameters['I^E'][j]) for j in indices])
            target = parameters['usafa_proportion']
        else:
            if eligibility:
                metric = np.array([np.mean(parameters['utility'][parameters['I^E'][j], j]) for j in indices])
            else:
                metric = np.array([np.mean(parameters['utility'][:, j]) for j in indices])
            target = None

        # Bar Chart
        ax.bar(afscs, metric, color=bar_color, alpha=alpha, edgecolor='black')

        if target is not None:
            ax.axhline(y=target, color='black', linestyle='--', alpha=alpha)

        # Get correct text
        y_label = graph
        if title is None:
            title = graph + ' Across Eligible Cadets for AFSCs with less than ' + str(num) + ' Eligible Cadets'
            if graph == 'Average Utility' and not eligibility:
                title = graph + ' Across All Cadets for AFSCs with less than ' + str(num) + ' Eligible Cadets'

    elif graph == 'AFOCD Data':

        # Legend
        add_legend = True
        legend_elements = [Patch(facecolor='yellow', label='Permitted', edgecolor='black'),
                           Patch(facecolor='green', label='Desired', edgecolor='black'),
                           Patch(facecolor='blue', label='Mandatory', edgecolor='black')]

        # Get metrics
        mandatory_count = np.array([np.sum(parameters['mandatory'][:, j]) for j in indices])
        desired_count = np.array([np.sum(parameters['desired'][:, j]) for j in indices])
        permitted_count = np.array([np.sum(parameters['permitted'][:, j]) for j in indices])

        # Bar Chart
        ax.bar(afscs, mandatory_count, color='blue', edgecolor='black')
        ax.bar(afscs, desired_count, bottom=mandatory_count, color='green', edgecolor='black')
        ax.bar(afscs, permitted_count, bottom=mandatory_count + desired_count, color='yellow', edgecolor='black')

        # Axis Adjustments
        ax.set(ylim=(0, num + num / 100))

        # Get correct text
        y_label = "Number of Cadets"
        if title is None:
            title = 'AFOCD Degree Tier Breakdown for AFSCs with less than ' + str(num) + ' Eligible Cadets'

    elif graph == 'Eligible Quota':

        # Legend
        add_legend = True
        legend_elements = [Patch(facecolor='blue', label='Eligible Cadets', edgecolor='black'),
                           Patch(facecolor='black', alpha=0.5, label='AFSC Quota', edgecolor='black')]

        # Get metrics
        eligible_count = np.array([len(parameters['I^E'][j]) for j in indices])
        quota = parameters['quota'][indices]

        # Bar Chart
        ax.bar(afscs, eligible_count, color='blue', edgecolor='black')
        ax.bar(afscs, quota, color='black', edgecolor='black', alpha=0.5)

        # Axis Adjustments
        ax.set(ylim=(0, num + num / 100))

        # Get correct text
        y_label = "Number of Cadets"
        if title is None:
            title = 'Eligible Cadets and Quotas for AFSCs with less than ' + str(num) + ' Eligible Cadets'

    # Display title
    if display_title:
        fig.suptitle(title, fontsize=title_size)

    # Labels
    ax.set_ylabel(y_label)
    ax.yaxis.label.set_size(label_size)
    ax.set_xlabel('AFSCs')
    ax.xaxis.label.set_size(label_size)

    # X axis
    ax.tick_params(axis='x', labelsize=afsc_tick_size)
    ax.set_xticklabels(afscs[tick_indices], rotation=afsc_rotation)
    ax.set_xticks(tick_indices)
    ax.set(xlim=(-0.8, M))

    # Y axis
    ax.tick_params(axis='y', labelsize=yaxis_tick_size)

    # Legend
    if add_legend:
        ax.legend(handles=legend_elements, edgecolor='black', loc="upper right", fontsize=legend_size, ncol=1,
                  labelspacing=1, handlelength=0.8, handletextpad=0.2, borderpad=0.2, handleheight=2)

    # Fix Colors
    if facecolor == 'black':
        ax.set_facecolor('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['right'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        if display_title:
            fig.suptitle(title, fontsize=label_size, color='white')

    if save:
        fig.savefig(paths['figures'] + title + '.png', bbox_inches='tight')

    return fig


# Value Parameters
def value_function_graph(x, y, x_point=None, f_x_point=None, title=None, display_title=True, figsize=(11, 10),
                         facecolor='white', save=False, breakpoints=None, x_ticks=None, crit_point=None,
                         label_size=25, yaxis_tick_size=25, xaxis_tick_size=25, x_label=None):
    """
    Displays the value function for the chosen function parameters
    :param x_label: label of the x variable (objective measure)
    :param xaxis_tick_size: font size of the x axis tick marks
    :param yaxis_tick_size: font size of the y axis tick marks
    :param label_size: font size of the labels
    :param crit_point: critical measure point to show
    :param x_ticks: x tick markers
    :param breakpoints: optional breakpoints to plot
    :param f_x_point: the value of that single x point
    :param x_point: a single x point that we can plot
    :param x: x coordinates for graph
    :param y: y coordinates for graph
    :param display_title: If we should display a title or not
    :param title: title
    :param figsize: size of figure
    :param facecolor: color of figure
    :param save: if we should save the figure
    :return: fig
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)

    ax.plot(x, y, color="black", linewidth=3)
    if breakpoints is not None:
        if breakpoints is True:
            ax.scatter(x, y, color='black', s=100)
        elif breakpoints is not False:
            ax.scatter(breakpoints[0], breakpoints[1], color='black', s=100)
    if crit_point is not None:
        if type(crit_point) == list:
            for point in crit_point:
                ax.plot((point, point), (0, 1), c="black", linestyle="--", linewidth=3)
        else:
            ax.plot((crit_point, crit_point), (0, 1), c="black", linestyle="--", linewidth=3)
    elif x_point is not None:
        ax.scatter(x_point, f_x_point, color="blue", s=50)
        ax.plot((x_point, x_point), (0, f_x_point), c="blue", linestyle="--", linewidth=3)
        ax.plot((0, x_point), (f_x_point, f_x_point), c="blue", linestyle="--", linewidth=3)
        ax.text(x=x_point, y=f_x_point, s=str(round(x_point, 2)) + ", " + str(round(f_x_point, 2)))

    if title is None:
        title = "Example Value Function Graph"

    if display_title:
        fig.suptitle(title, fontsize=label_size)

    # Set ticks and labels
    # y_ticks = [0.2, 0.4, 0.6, 0.8, 1]
    y_ticks = [1]
    ax.set_yticks(y_ticks)

    if x_ticks is not None:
        ax.set_xticks(x_ticks)

    # ax.set_xticks([0, 0.5, 1])
    # ax.plot((0.5, 0.5), (0, 1), c='black', linewidth=3, linestyle='--')

    # Adjust axes and ticks
    ax.tick_params(axis='x', labelsize=xaxis_tick_size)
    ax.tick_params(axis='y', labelsize=yaxis_tick_size)
    ax.set_facecolor(facecolor)
    ax.yaxis.label.set_size(label_size)
    ax.set_ylabel('Value')
    ax.xaxis.label.set_size(label_size)
    if x_label is None:
        x_label = 'Measure'
    ax.set_xlabel(x_label)
    ax.margins(x=0)
    ax.margins(y=0)
    plt.ylim(0, 1.05)

    if save:
        fig.savefig(paths['figures'] + title + '.png')
    return fig


def individual_weight_graph(parameters, value_parameters, cadets=True, save=False, figsize=(19, 7), facecolor='white',
                            display_title=True, title=None, label_size=25, afsc_tick_size=15, gui_graph=False, dpi=100,
                            yaxis_tick_size=25, afsc_rotation=80, xaxis_tick_size=25, skip_afscs=False):
    """
    This function creates the chart for either the individual weight function for cadets or the actual
    individual weights on the AFSCs
    :param dpi: dot per inch parameter for figure
    :param gui_graph: if this is the weight chart used in the GUI
    :param title: title of chart
    :param cadets: if the weight chart is for cadets or AFSCs
    :param xaxis_tick_size: x axis tick size
    :param value_parameters: value parameters
    :param skip_afscs: if we want to skip AFSCs
    :param afsc_rotation: rotation of the AFSCs
    :param yaxis_tick_size: y axis tick sizes
    :param afsc_tick_size: x axis tick sizes for AFSCs
    :param label_size: size of labels
    :param display_title: if we want to show a title
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :param save: Whether we should save the graph
    :param parameters: fixed cadet/AFSC data
    :return: chart
    """

    if title is None:
        if cadets:
            title = 'Individual Weight on Cadets'
        else:
            title = 'Individual Weight on AFSCs'

    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, dpi=dpi, tight_layout=True)

    if cadets:

        x = parameters['merit']
        y = value_parameters['cadet_weight']

        # Plot
        ax.plot(x, y, color='black', alpha=1, linewidth=3)

        # Labels
        ax.set_ylabel('Cadet Weight', fontname='Times New Roman')
        ax.yaxis.label.set_size(label_size)
        ax.set_xlabel('Percentile', fontname='Times New Roman')
        ax.xaxis.label.set_size(label_size)

        if gui_graph:
            ax.set_facecolor(facecolor)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            ax.spines['right'].set_color(facecolor)
            ax.spines['top'].set_color(facecolor)
        else:
            # Ticks
            y_ticks = [0.5, 1]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks, fontname='Times New Roman')
            x_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
            ax.set_xticklabels(x_ticks, fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=xaxis_tick_size)
            ax.tick_params(axis='y', labelsize=yaxis_tick_size)

        # Margins
        ax.margins(x=0)
        ax.margins(y=0)

    else:

        # Get data
        weights = value_parameters['afsc_weight']
        afscs = parameters['afsc_vector']
        M = parameters['M']

        # Labels
        ax.set_ylabel('AFSC Weight')
        ax.yaxis.label.set_size(label_size)
        ax.set_xlabel('AFSCs')
        ax.xaxis.label.set_size(label_size)

        # We can skip AFSCs
        if skip_afscs:
            tick_indices = np.arange(1, M, 2).astype(int)
        else:
            tick_indices = np.arange(M)

        if gui_graph:
            ax.set_facecolor(facecolor)
            ax.set_yticks([])
            # ax.xaxis.label.set_color('black')
            # ax.yaxis.label.set_color('black')
            ax.spines['right'].set_color(facecolor)
            ax.spines['top'].set_color(facecolor)
            ax.set(xlim=(-0.8, M))
            ax.set_xticklabels(afscs[tick_indices], rotation=afsc_rotation)
            ax.tick_params(axis='x', labelsize=afsc_tick_size)
            ax.set_xticks(tick_indices)

            # Plot
            ax.bar(afscs, weights, color='black')
        else:

            # Plot
            ax.bar(afscs, weights, color='black', alpha=0.5)

            # Labels
            ax.set_ylabel('AFSC Weight', fontname='Times New Roman')
            ax.yaxis.label.set_size(label_size)
            ax.set_xlabel('AFSCs')
            ax.xaxis.label.set_size(label_size)

            # Ticks
            ax.set(xlim=(-0.8, M))
            ax.tick_params(axis='x', labelsize=afsc_tick_size)
            # ax.tick_params(axis='y', labelsize=yaxis_tick_size)
            ax.set_yticks([])
            ax.set_xticklabels(afscs[tick_indices], rotation=afsc_rotation)
            ax.set_xticks(tick_indices)

    if display_title:
        ax.set_title(title, fontsize=label_size)

    if save:
        fig.savefig(paths['figures'] + title + '.png', bbox_inches='tight')

    return fig


# Results
def afsc_value_results_graph(parameters, value_parameters, metrics=None, metrics_dict=None, save=None, figsize=(19, 7),
                             facecolor='white', title=None, display_title=True, label_size=25, afsc_tick_size=15,
                             yaxis_tick_size=25, afsc_rotation=80, dpi=100, y_max=1.1, skip_afscs=False,
                             value_type="Overall", gui_chart=False, legend_size=None, title_size=None, colors=None,
                             alpha=0.5, bar_color='black', dot_size=100):
    """
    Builds the AFSC Value Results Bar Chart
    :param y_max: multiplier of the maximum y value for the graph to extend the window above
    :param value_type: type of value objective to show (overall for AFSC, or can specify each AFSC objective)
    :param gui_chart: if this graph is used in the GUI
    :param colors: colors of the different solutions
    :param dot_size: size of the scatter points
    :param afsc_rotation: rotation of the AFSCs
    :param yaxis_tick_size: y axis tick sizes
    :param afsc_tick_size: x axis tick sizes for AFSCs
    :param label_size: size of labels
    :param title: title of chart
    :param display_title: if we should show the title
    :param metrics_dict: dictionary of solution metrics
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :param save: Whether we should save the graph
    :param parameters: fixed cadet/AFSC data
    :param value_parameters: value parameters
    :param metrics: solution metrics
    :param skip_afscs:  Whether we should label every other AFSC
    :param dpi: dots per inch for figure
    :param bar_color: color of bars for figure (for certain kinds of graphs)
    :param alpha: alpha parameter for the bars of the figure
    :param title_size: font size of the title
    :param legend_size: font size of the legend
    :return: figure
    """
    if legend_size is None:
        legend_size = label_size
    if title_size is None:
        title_size = label_size

    if save is None:
        save = False

    if title is None:
        title = value_type + ' Value Chart'

    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, dpi=dpi, tight_layout=True)

    # Load the data
    afscs = parameters['afsc_vector']
    M = len(afscs)
    indices = np.arange(M)

    # We can skip AFSCs
    if skip_afscs:
        tick_indices = np.arange(1, M, 2).astype(int)
    else:
        tick_indices = indices

    if metrics_dict is None:

        if value_type == "Overall":
            value = metrics['afsc_value']
        else:
            try:
                k = np.where(value_parameters['objectives'] == value_type)[0][0]
                indices = np.where(value_parameters['objective_weight'][:, k] != 0)[0]
                value = metrics['objective_value'][indices, k]
            except:
                indices = np.array([])
                value = np.array([])

            if len(indices) < M:
                afscs = afscs[indices]
                M = len(afscs)
                tick_indices = np.arange(M)

        # Bar Chart
        ax.bar(afscs, value, color=bar_color, alpha=alpha, edgecolor='black')

    else:

        num_solutions = len(list(metrics_dict.keys()))
        if colors is None:
            color_choices = ['red', 'blue', 'green', 'orange']
            marker_choices = ['o', 'D', '^', 'P']
            colors = {}
            markers = {}
            for s, solution in enumerate(list(metrics_dict.keys())):

                if s < len(color_choices):
                    colors[solution] = color_choices[s]
                    markers[solution] = marker_choices[s]
                else:
                    colors[solution] = color_choices[0]
                    markers[solution] = marker_choices[0]
        max_value = np.zeros(M)
        min_value = np.repeat(10000, M)
        max_solution = np.zeros(M).astype(int)
        legend_elements = []
        for s, solution in enumerate(list(metrics_dict.keys())):

            if value_type == "Overall":
                value = metrics_dict[solution]['afsc_value']
            else:
                k = np.where(value_parameters['objectives'] == value_type)[0][0]
                indices = np.where(value_parameters['objective_weight'][:, k] != 0)[0]
                value = metrics_dict[solution]['objective_value'][indices, k]
                if len(indices) < M:
                    afscs = afscs[indices]
                    M = len(afscs)
                    tick_indices = np.arange(M)

            ax.scatter(afscs, value, color=colors[solution], marker=markers[solution], edgecolor='black',
                       s=dot_size, zorder=2)

            max_value = np.array([max(max_value[j], value[j]) for j in range(M)])
            min_value = np.array([min(min_value[j], value[j]) for j in range(M)])
            for j in range(M):
                if max_value[j] == value[j]:
                    max_solution[j] = s
            element = mlines.Line2D([], [], color=colors[solution], marker=markers[solution], linestyle='None',
                                    markeredgecolor='black', markersize=20, label=solution)
            legend_elements.append(element)

        for j in range(M):
            ax.plot((j, j), (0, max_value[j]), color='black', linestyle='--', zorder=1, alpha=0.4, linewidth=2)

        ax.legend(handles=legend_elements, edgecolor='black', fontsize=legend_size, loc='upper left',
                  ncol=num_solutions, columnspacing=0.5, handletextpad=0.05, borderaxespad=0.5, borderpad=0.2)

    if gui_chart:

        # Labels
        ax.set_ylabel('Value')
        ax.yaxis.label.set_size(label_size)

        # Axis
        ax.set_facecolor('white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')

        # Ticks
        y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
        ax.margins(y=0)
        ax.set(ylim=(0, y_max))
        ax.set_xticklabels(afscs[tick_indices], rotation=afsc_rotation)
        ax.set_xticks(tick_indices)
        ax.tick_params(axis='x', labelsize=afsc_tick_size, colors='white')
        ax.tick_params(axis='y', labelsize=yaxis_tick_size, colors='white')
        ax.set(xlim=(-1, M))
    else:

        # Labels
        ax.set_ylabel(value_type + ' Value')
        ax.yaxis.label.set_size(label_size)
        ax.set_xlabel('AFSCs')
        ax.xaxis.label.set_size(label_size)

        # Y ticks
        y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='y', labelsize=yaxis_tick_size)
        ax.set_yticklabels(y_ticks)
        ax.margins(y=0)
        ax.set(ylim=(0, y_max))

        # X ticks
        ax.set_xticklabels(afscs[tick_indices], rotation=afsc_rotation)
        ax.set_xticks(tick_indices)
        ax.tick_params(axis='x', labelsize=afsc_tick_size)
        ax.set(xlim=(-1, M))

        if display_title:
            ax.set_title(title, fontsize=title_size)

    if save:
        fig.savefig(paths['figures'] + title + '.png', bbox_inches='tight')

    return fig


def afsc_objective_values_graph(parameters, value_parameters, metrics, afsc, save=False, dpi=100, gui_chart=False,
                                figsize=(19, 7), facecolor="white", title=None, display_title=True,
                                label_size=25, xaxis_tick_size=15, yaxis_tick_size=25, legend_size=None, y_max=1.1,
                                title_size=None, metrics_dict=None, alpha=0.5, bar_color='black', dot_size=100):
    """
    This chart presents the values for the AFSC objectives for the given afsc
    :param value_parameters: value parameters
    :param metrics: solution metrics
    :param dpi: dots per inch for figure
    :param gui_chart: if this graph is used in the GUI
    :param legend_size: font size of the legend
    :param y_max: multiplier of the maximum y value for the graph to extend the window above
    :param title_size: font size of the title
    :param metrics_dict: dictionary of solution metrics
    :param alpha: alpha parameter for the bars of the figure
    :param bar_color: color of bars for figure (for certain kinds of graphs)
    :param dot_size: size of the scatter points
    :param yaxis_tick_size: y axis tick sizes
    :param xaxis_tick_size: x axis tick sizes
    :param label_size: size of labels
    :param display_title: if we should show the title
    :param title: title of chart
    :param parameters: fixed cadet/AFSC parameters
    :param afsc: which AFSC we should plot
    :param save: Whether we should save the graph
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :return: figure
    """
    if legend_size is None:
        legend_size = label_size
    if title_size is None:
        title_size = label_size

    if title is None:
        title = afsc + ' Objective Values'

    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, dpi=dpi, tight_layout=True)

    # Get chart specs
    j = np.where(parameters['afsc_vector'] == afsc)[0][0]
    objectives = value_parameters['objectives'][value_parameters['K^A']]
    for k, objective in enumerate(objectives):
        if objective == 'USAFA Proportion':
            objectives[k] = 'USAFA\nProportion'
        elif objective == 'Combined Quota':
            objectives[k] = 'Combined\nQuota'

    if metrics_dict is None:

        # Plot values
        values = metrics['objective_value'][j, value_parameters['K^A']]
        ax.bar(objectives, values, edgecolor='black', color=bar_color, alpha=alpha)

    else:

        num_solutions = len(list(metrics_dict.keys()))
        if colors is None:
            color_choices = ['red', 'blue', 'green', 'orange']
            marker_choices = ['o', 'D', '^', 'P']
            colors = {}
            markers = {}
            for s, solution in enumerate(list(metrics_dict.keys())):

                if s < len(color_choices):
                    colors[solution] = color_choices[s]
                    markers[solution] = marker_choices[s]
                else:
                    colors[solution] = color_choices[0]
                    markers[solution] = marker_choices[0]
        O = len(objectives)
        max_value = np.zeros(O)
        min_value = np.repeat(10000, O)
        max_solution = np.zeros(O).astype(int)
        legend_elements = []

        for s, solution in enumerate(list(metrics_dict.keys())):

            # Plot points
            value = metrics_dict[solution]['objective_value'][j, value_parameters['K^A']]
            ax.scatter(objectives, value, color=colors[s], marker=markers[s], edgecolor='black', s=dot_size, zorder=2)

            max_value = np.array([max(max_value[k], value[k]) for k in range(O)])
            min_value = np.array([min(min_value[k], value[k]) for k in range(O)])
            for k in range(O):
                if max_value[k] == value[k]:
                    max_solution[k] = s
            element = mlines.Line2D([], [], color=colors[solution], marker=markers[solution], linestyle='None',
                                    markeredgecolor='black', markersize=20, label=solution)
            legend_elements.append(element)

        for k in range(O):
            ax.plot((k, k), (0, max_value[k]), color='black', linestyle='--', zorder=1, alpha=0.4, linewidth=2)
            # ax.plot((j, j), (min_value[j], max_value[j]), color=colors[max_solution[j]], linestyle='-', zorder=1,
            #         alpha=1, linewidth=2)

        ax.legend(handles=legend_elements, edgecolor='black', fontsize=legend_size, loc='upper left',
                  ncol=num_solutions, columnspacing=0.5, handletextpad=0.05, borderaxespad=0.5, borderpad=0.2)
    # Labels
    ax.set_ylabel('Objective Value')
    ax.yaxis.label.set_size(label_size)
    ax.set_xlabel(afsc + ' Objectives')
    ax.xaxis.label.set_size(label_size)

    # X ticks
    ax.tick_params(axis='x', labelsize=xaxis_tick_size)
    ax.set_xticklabels(objectives)
    ax.set_xticks(np.arange(len(objectives)))
    ax.set(xlim=(-1, len(objectives)))

    # Y ticks
    ax.tick_params(axis='y', labelsize=yaxis_tick_size)
    ax.margins(y=0)
    ax.set(ylim=(0, y_max))

    # GUI Chart
    if gui_chart:
        ax.yaxis.label.set_color('white')
        # ax.xaxis.label.set_color('white')
        ax.tick_params(axis='x', labelsize=xaxis_tick_size, colors='white')
        ax.tick_params(axis='y', labelsize=yaxis_tick_size, colors='white')
        ax.spines['right'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.set_title(title, fontsize=label_size, color='white')
    else:
        if display_title:
            ax.set_title(title, fontsize=title_size)

    if save:
        fig.savefig(paths['figures'] + title + '.png')

    return fig


def average_merit_results_graph(parameters, value_parameters, metrics=None, metrics_dict=None, save=False,
                                figsize=(19, 7), facecolor='white', title=None, display_title=True, label_size=25,
                                afsc_tick_size=15, yaxis_tick_size=25, afsc_rotation=80, skip_afscs=False,
                                legend_size=None, title_size=None, y_max=1.1, alpha=0.5,
                                dot_size=100, colors=None):
    """
    Builds the Average Merit Results Bar Chart
    :param colors: solution colors
    :param skip_afscs: Whether we should label every other AFSC
    :param legend_size: font size of the legend
    :param title_size: font size of the title
    :param y_max: multiplier of the maximum y value for the graph to extend the window above
    :param alpha: alpha parameter for the bars of the figure
    :param dot_size: size of the scatter points
    :param afsc_rotation: rotation of the AFSCs
    :param yaxis_tick_size: y axis tick sizes
    :param afsc_tick_size: x axis tick sizes for AFSCs
    :param label_size: size of labels
    :param title: title of chart
    :param display_title: if we should show the title
    :param metrics_dict: dictionary of solution metrics
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :param save: Whether we should save the graph
    :param parameters: fixed cadet/AFSC data
    :param value_parameters: value parameters
    :param metrics: solution metrics
    :return: figure
    """
    if legend_size is None:
        legend_size = label_size
    if title_size is None:
        title_size = label_size

    if title is None:
        title = 'Average Merit of Assigned Cadets for Each AFSC'

    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)

    # Load the data
    afscs = parameters['afsc_vector']
    merit_k = np.where(value_parameters['objectives'] == 'Merit')[0][0]
    M = parameters['M']
    indices = np.arange(M)

    # We can skip AFSCs
    if skip_afscs:
        tick_indices = np.arange(1, M, 2).astype(int)
    else:
        tick_indices = indices

    if metrics_dict is None:

        colors = np.array([" " * 10 for _ in range(len(afscs))])
        legend_elements = [Patch(facecolor='#bf4343', label='Measure < 0.35'),
                           Patch(facecolor='#3d8ee0', label='0.35 < Measure < 0.65'),
                           Patch(facecolor='#c7b93a', label='Measure > 0.65')]
        merit = metrics['objective_measure'][:, merit_k]
        max_merit = merit

        # Assign the right color to the AFSCs
        for j in range(len(afscs)):
            if 0.35 <= merit[j] <= 0.65:
                colors[j] = '#3d8ee0'
            elif merit[j] > 0.65:
                colors[j] = '#c7b93a'
            else:
                colors[j] = '#bf4343'

        # Add lines for the ranges
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=0.35, color='blue', linestyle='-', alpha=0.5)
        ax.axhline(y=0.65, color='blue', linestyle='-', alpha=0.5)

        # Bar Chart
        ax.bar(afscs, merit, color=colors, edgecolor='black', alpha=alpha)
        ax.legend(handles=legend_elements, edgecolor='black', fontsize=legend_size,
                  ncol=1, columnspacing=0.5, handletextpad=0.05, borderaxespad=0.5, borderpad=0.2)

    else:

        num_solutions = len(list(metrics_dict.keys()))
        if colors is None:
            color_choices = ['red', 'blue', 'green', 'orange']
            marker_choices = ['o', 'D', '^', 'P']
            colors = {}
            markers = {}
            for s, solution in enumerate(list(metrics_dict.keys())):

                if s < len(color_choices):
                    colors[solution] = color_choices[s]
                    markers[solution] = marker_choices[s]
                else:
                    colors[solution] = color_choices[0]
                    markers[solution] = marker_choices[0]
        max_merit = np.zeros(M)
        legend_elements = []
        for s, solution in enumerate(list(metrics_dict.keys())):
            merit = metrics_dict[solution]['objective_measure'][:, merit_k]
            ax.scatter(indices, merit, color=colors[solution], marker=markers[solution], edgecolor='black',
                       s=dot_size, zorder=2)

            max_merit = np.array([max(max_merit[j], merit[j]) for j in range(M)])
            element = mlines.Line2D([], [], color=colors[solution], marker=markers[solution], linestyle='None',
                                    markeredgecolor='black', markersize=20, label=solution)
            legend_elements.append(element)

        for j in range(M):
            ax.plot((j, j), (0, max_merit[j]), color='black', linestyle='--', zorder=1, alpha=0.5, linewidth=2)

        # Constraint Lines
        ax.plot((-1, 50), (0.65, 0.65), color='black', linestyle='-', zorder=1, alpha=1, linewidth=1.5)
        ax.plot((-1, 50), (0.5, 0.5), color='black', linestyle='--', zorder=1, alpha=1, linewidth=1.5)
        ax.plot((-1, 50), (0.35, 0.35), color='black', linestyle='-', zorder=1, alpha=1, linewidth=1.5)

        ax.legend(handles=legend_elements, edgecolor='black', fontsize=legend_size, loc='upper left',
                  ncol=num_solutions, columnspacing=0.5, handletextpad=0.05, borderaxespad=0.5, borderpad=0.2)

    # Labels
    ax.set_ylabel('Average Merit')
    ax.yaxis.label.set_size(label_size)
    ax.set_xlabel('AFSCs')
    ax.xaxis.label.set_size(label_size)

    # Y ticks
    y_ticks = [0, 0.35, 0.50, 0.65, 0.80, 1]
    y_tick_labels = ['0', '0.35', '0.50', '0.65', '0.80', '1']
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='y', labelsize=yaxis_tick_size)
    ax.set_yticklabels(y_tick_labels)
    ax.margins(y=0)
    y_top = max(max_merit)
    ax.set(ylim=(0, y_top * y_max))

    # X ticks
    ax.set_xticklabels(afscs[tick_indices], rotation=afsc_rotation)
    ax.set_xticks(tick_indices)
    ax.tick_params(axis='x', labelsize=afsc_tick_size)
    ax.set(xlim=(-1, M))

    if display_title:
        ax.set_title(title, fontsize=title_size)

    if save:
        fig.savefig(paths['figures'] + title + '.png', bbox_inches='tight')

    return fig


def quota_fill_results_graph(parameters, value_parameters, metrics=None, metrics_dict=None, save=False,
                             figsize=(19, 7), facecolor='white', title=None, display_title=True, label_size=25,
                             afsc_tick_size=15, yaxis_tick_size=25, afsc_rotation=80, skip_afscs=False, colors=None,
                             xaxis_tick_size=15, legend_size=None, title_size=None, y_max=1.1, dot_size=100):
    """
    Builds the Combined Quota Results Bar Chart
    :param colors: colors of the solutions
    :param skip_afscs: Whether we should label every other AFSC
    :param legend_size: font size of the legend
    :param title_size: font size of the title
    :param y_max: multiplier of the maximum y value for the graph to extend the window above
    :param dot_size: size of the scatter points
    :param afsc_rotation: rotation of the AFSCs
    :param yaxis_tick_size: y axis tick sizes
    :param xaxis_tick_size: in this context, this is the font size for the count text
    :param afsc_tick_size: x axis tick sizes for AFSCs
    :param label_size: size of labels
    :param title: title of chart
    :param display_title: if we should show the title
    :param metrics_dict: dictionary of solution metrics
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :param save: Whether we should save the graph
    :param parameters: fixed cadet/AFSC data
    :param value_parameters: value parameters
    :param metrics: solution metrics
    :return: figure
    """
    if legend_size is None:
        legend_size = label_size
    if title_size is None:
        title_size = label_size

    if title is None:
        title = 'Percentage of Quota Filled for each AFSC'

    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)

    # Load the data
    afscs = parameters['afsc_vector']
    M = len(afscs)
    indices = np.arange(M)
    quota_k = np.where(value_parameters['objectives'] == 'Combined Quota')[0][0]

    # We can skip AFSCs
    if skip_afscs:
        tick_indices = np.arange(1, M, 2).astype(int)
    else:
        tick_indices = indices

    x_under = []
    x_over = []
    quota_percent_filled = np.zeros(M)
    max_quota_percent = np.zeros(M)
    num_cadets = np.zeros(M)
    if metrics_dict is None:
        colors = np.repeat(' ' * 20, M)
        for j in range(M):

            value_list = value_parameters['objective_value_min'][j, quota_k].split(",")
            quota = parameters['quota'][j]
            min_measure = float(value_list[0].strip())
            max_measure = float(value_list[1].strip())
            num_cadets[j] = metrics['objective_measure'][j, quota_k]
            if min_measure > num_cadets[j]:
                colors[j] = "red"
                x_under.append(j)
            elif min_measure <= num_cadets[j] <= max_measure:
                colors[j] = "blue"
            else:
                colors[j] = "orange"
                x_over.append(j)

            quota_percent_filled[j] = num_cadets[j] / quota
            max_quota_percent[j] = max_measure / quota

        # Points
        ax.scatter(indices, quota_percent_filled, c=colors, linewidths=4, s=dot_size)

        # Lines
        y_mins = np.repeat(1, M)
        y_maxs = max_quota_percent
        y = [(y_mins[i], y_maxs[i]) for i in range(M)]
        y_under = [(quota_percent_filled[j], 1) for j in x_under]
        y_over = [(max_quota_percent[j], quota_percent_filled[j]) for j in x_over]

        # Plot Bounds
        ax.scatter(indices, y_mins, c=np.repeat('blue', M), marker="_", linewidth=2)
        ax.scatter(indices, y_maxs, c=np.repeat('blue', M), marker="_", linewidth=2)

        # Plot Range Lines
        ax.plot((indices, indices), ([i for (i, j) in y], [j for (i, j) in y]), c='blue')
        ax.plot((x_under, x_under), ([i for (i, j) in y_under], [j for (i, j) in y_under]), c='red', linestyle='--')
        ax.plot((x_over, x_over), ([i for (i, j) in y_over], [j for (i, j) in y_over]), c='orange', linestyle='--')
        ax.plot((indices, indices), (np.zeros(M), np.ones(M)), c='black', linestyle='--', alpha=0.3)

        # Quota Line
        ax.axhline(y=1, color='black', linestyle='-', alpha=0.5)

        # Put quota text
        y_top = round(max(quota_percent_filled))
        y_spacing = (y_top / 80)
        for j in indices:
            if int(num_cadets[j]) >= 100:
                ax.text(j, quota_percent_filled[j] + 1.4 * y_spacing, int(num_cadets[j]), fontsize=xaxis_tick_size,
                        multialignment='right')
            elif int(num_cadets[j]) >= 10:
                ax.text(j + 0.2, quota_percent_filled[j] + y_spacing, int(num_cadets[j]), fontsize=xaxis_tick_size,
                        multialignment='right')
            else:
                ax.text(j + 0.2, quota_percent_filled[j] + y_spacing, int(num_cadets[j]), fontsize=xaxis_tick_size,
                        multialignment='right')
    else:

        num_solutions = len(list(metrics_dict.keys()))
        y_top = 0
        if colors is None:
            color_choices = ['red', 'blue', 'green', 'orange']
            marker_choices = ['o', 'D', '^', 'P']
            colors = {}
            markers = {}
            for s, solution in enumerate(list(metrics_dict.keys())):

                if s < len(color_choices):
                    colors[solution] = color_choices[s]
                    markers[solution] = marker_choices[s]
                else:
                    colors[solution] = color_choices[0]
                    markers[solution] = marker_choices[0]
        zorder = [2, 3]
        legend_elements = []
        for s, solution in enumerate(list(metrics_dict.keys())):
            for j in range(M):

                value_list = value_parameters['objective_value_min'][j, quota_k].split(",")
                quota = parameters['quota'][j]
                min_measure = float(value_list[0].strip())
                max_measure = float(value_list[1].strip())
                num_cadets[j] = metrics_dict[solution]['objective_measure'][j, quota_k]
                if min_measure > num_cadets[j]:
                    x_under.append(j)
                elif num_cadets[j] > max_measure:
                    x_over.append(j)

                quota_percent_filled[j] = num_cadets[j] / quota
                max_quota_percent[j] = max(max_quota_percent[j], max_measure / quota)

            y_top = max(y_top, max(quota_percent_filled))

            # Points
            ax.scatter(indices, quota_percent_filled, c=colors[solution], marker=markers[solution],
                       linewidths=4, s=dot_size, zorder=zorder[s], edgecolor='black', linewidth=0.75)

            # Legend
            element = mlines.Line2D([], [], color=colors[solution], marker=markers[solution], linestyle='None',
                                    markeredgecolor='black', markersize=20, label=solution)
            legend_elements.append(element)

            # Lines
            y_mins = np.repeat(1, M)
            y_maxs = max_quota_percent
            y = [(y_mins[i], y_maxs[i]) for i in range(M)]
            y_under = [(quota_percent_filled[j], 1) for j in x_under]
            y_over = [(max_quota_percent[j], quota_percent_filled[j]) for j in x_over]

            # Plot Bounds
            ax.scatter(indices, y_mins, c=np.repeat('black', M), marker="_", linewidth=2, zorder=1)
            ax.scatter(indices, y_maxs, c=np.repeat('black', M), marker="_", linewidth=2, zorder=1)

            # Plot Range Lines
            ax.plot((indices, indices), ([i for (i, j) in y], [j for (i, j) in y]), c='black', zorder=1)
            ax.plot((x_under, x_under), ([i for (i, j) in y_under], [j for (i, j) in y_under]), c='black',
                    linestyle='--', zorder=1)
            ax.plot((x_over, x_over), ([i for (i, j) in y_over], [j for (i, j) in y_over]), c='black', linestyle='--',
                    zorder=1)
            ax.plot((indices, indices), (np.zeros(M), np.ones(M)), c='black', linestyle='--', alpha=0.3,
                    zorder=1)

            # Quota Line
            ax.axhline(y=1, color='black', linestyle='-', alpha=0.5, zorder=1)

        # Legend
        ax.legend(handles=legend_elements, edgecolor='black', fontsize=legend_size, loc='upper left',
                  ncol=num_solutions, columnspacing=0.5, handletextpad=0.05, borderaxespad=0.5, borderpad=0.2)

    if display_title:
        ax.set_title(title, fontsize=title_size)

    ax.set_ylabel('Percent of Quota Achieved')
    ax.yaxis.label.set_size(label_size)
    ax.set_xlabel('AFSCs')
    ax.xaxis.label.set_size(label_size)

    # Y ticks
    ticks = list(np.arange(0, int(round(y_top)) + 0.5, 0.5))
    ax.set_yticks(ticks)
    ax.tick_params(axis='y', labelsize=yaxis_tick_size)

    # X ticks
    ax.set(xticks=tick_indices, xticklabels=afscs[tick_indices])
    ax.tick_params(axis='x', labelsize=afsc_tick_size)
    ax.set_xticklabels(afscs[tick_indices], rotation=afsc_rotation)
    ax.set(xlim=(-1, M))
    ax.set(ylim=(0, y_top * y_max))

    if save:
        fig.savefig(paths['figures'] + title + '.png', bbox_inches='tight')

    return fig


def usafa_proportion_results_graph(parameters, value_parameters, metrics=None, metrics_dict=None, save=False,
                                   figsize=(19, 7), facecolor='white', title=None, display_title=True, label_size=25,
                                   afsc_tick_size=15, yaxis_tick_size=25, afsc_rotation=80, skip_afscs=False,
                                   legend_size=None, title_size=None, y_max=1.1, dot_size=100, colors=None):
    """
    Builds the USAFA Proportion Results Bar Chart
    :param skip_afscs: Whether we should label every other AFSC
    :param legend_size: font size of the legend
    :param title_size: font size of the title
    :param y_max: multiplier of the maximum y value for the graph to extend the window above
    :param dot_size: size of the scatter points
    :param afsc_rotation: rotation of the AFSCs
    :param yaxis_tick_size: y axis tick sizes
    :param afsc_tick_size: x axis tick sizes for AFSCs
    :param label_size: size of labels
    :param title:
    :param display_title:
    :param metrics_dict: dictionary of solution metrics
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :param save: Whether we should save the graph
    :param parameters: fixed cadet/AFSC data
    :param value_parameters: value parameters
    :param metrics: solution metrics
    :return: figure
    """
    if legend_size is None:
        legend_size = label_size
    if title_size is None:
        title_size = label_size

    if title is None:
        title = 'USAFA Proportion of Assigned Cadets for Each AFSC'

    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)

    # Load the data
    afscs = parameters['afsc_vector']
    M = len(afscs)
    indices = range(M)
    usafa_k = np.where(value_parameters['objectives'] == 'USAFA Proportion')[0][0]
    usafa_target = value_parameters['objective_target'][:, usafa_k]
    usafa_proportion = round(parameters['usafa_proportion'], 2)

    # We can skip AFSCs
    if skip_afscs:
        tick_indices = np.arange(1, M, 2).astype(int)
    else:
        tick_indices = indices
    if metrics_dict is None:

        usafa = metrics['objective_measure'][:, usafa_k]

        # Plot Points
        ax.scatter(indices, usafa, color='black', edgecolor='black', s=dot_size, zorder=2)

        # Get range acceptable for each AFSC for USAFA proportion
        usafa_ranges = []
        for j in range(M):

            if value_parameters['constraint_type'][j, usafa_k] != 0:
                value_list = value_parameters['objective_value_min'][j, usafa_k].split(",")
                min_measure = float(value_list[0].strip())
                max_measure = float(value_list[1].strip())
                usafa_ranges.append((min_measure, max_measure))
            else:

                if usafa_target[j] == 0:
                    usafa_ranges.append((0, 0.2))
                elif usafa_target[j] == 1:
                    usafa_ranges.append((0.8, 1))
                else:
                    usafa_ranges.append((usafa_proportion - 0.15, usafa_proportion + 0.15))

        # Plot Range lines and bounds
        for j in range(M):
            ax.plot((j, j), usafa_ranges[j], color='black', zorder=1)
            ax.scatter((j, j), usafa_ranges[j], color='black', zorder=1, marker='_', linewidth=2)
            if usafa[j] < usafa_ranges[j][0]:
                ax.plot((j, j), (usafa[j], usafa_ranges[j][0]), color='red', zorder=1, linestyle='--')
            elif usafa[j] > usafa_ranges[j][1]:
                ax.plot((j, j), (usafa_ranges[j][1], usafa[j]), color='red', zorder=1, linestyle='--')
    else:

        num_solutions = len(list(metrics_dict.keys()))
        if colors is None:
            color_choices = ['red', 'blue', 'green', 'orange']
            marker_choices = ['o', 'D', '^', 'P']
            colors = {}
            markers = {}
            for s, solution in enumerate(list(metrics_dict.keys())):

                if s < len(color_choices):
                    colors[solution] = color_choices[s]
                    markers[solution] = marker_choices[s]
                else:
                    colors[solution] = color_choices[0]
                    markers[solution] = marker_choices[0]
        legend_elements = []
        value_dict = {}
        for s, solution in enumerate(list(metrics_dict.keys())):
            usafa = metrics_dict[solution]['objective_measure'][:, usafa_k]
            value_dict[solution] = usafa

            # Plot Points
            ax.scatter(indices, usafa, color=colors[solution], marker=markers[solution], edgecolor='black', s=dot_size,
                       zorder=2)
            element = mlines.Line2D([], [], color=colors[solution], marker=markers[solution], linestyle='None',
                                    markeredgecolor='black', markersize=20, label=solution)
            legend_elements.append(element)

            # Get range acceptable for each AFSC for USAFA proportion
            usafa_ranges = []
            for j in range(M):

                if value_parameters['constraint_type'][j, usafa_k] != 0:
                    value_list = value_parameters['objective_value_min'][j, usafa_k].split(",")
                    min_measure = float(value_list[0].strip())
                    max_measure = float(value_list[1].strip())
                    usafa_ranges.append((min_measure, max_measure))
                else:

                    if usafa_target[j] == 0:
                        usafa_ranges.append((0, 0.2))
                    elif usafa_target[j] == 1:
                        usafa_ranges.append((0.8, 1))
                    else:
                        usafa_ranges.append((usafa_proportion - 0.15, usafa_proportion + 0.15))

            usafa_ranges = np.array(usafa_ranges)

            # Plot Range lines and bounds
            for j in range(M):
                ax.plot((j, j), usafa_ranges[j], color='black', zorder=1)
                ax.scatter((j, j), usafa_ranges[j], color='black', zorder=1, marker='_', linewidth=2)
                if usafa[j] < usafa_ranges[j][0]:
                    ax.plot((j, j), (usafa[j], usafa_ranges[j][0]), color='red', zorder=1, linestyle='--')
                elif usafa[j] > usafa_ranges[j][1]:
                    ax.plot((j, j), (usafa_ranges[j][1], usafa[j]), color='red', zorder=1, linestyle='--')

        # Plot more lines
        for j in range(M):
            real_min = usafa_ranges[j][0]
            for solution in metrics_dict:
                val = value_dict[solution][j]
                real_min = min(val, real_min)
            ax.plot((j, j), (0, real_min), color='black', linestyle='--', zorder=1, alpha=0.5,
                    linewidth=2)

        ax.legend(handles=legend_elements, edgecolor='black', fontsize=legend_size, loc='upper left',
                  ncol=num_solutions, columnspacing=0.5, handletextpad=0.05, borderaxespad=0.5, borderpad=0.2)

    # Labels
    ax.set_ylabel('USAFA Proportion')
    ax.yaxis.label.set_size(label_size)
    ax.set_xlabel('AFSCs')
    ax.xaxis.label.set_size(label_size)

    # Y ticks
    y_ticks = [0, round(usafa_proportion - 0.15, 2), usafa_proportion, round(usafa_proportion + 0.15, 2), 1]
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='y', labelsize=yaxis_tick_size)
    ax.set_yticklabels(y_ticks)
    ax.margins(y=0)
    ax.set(ylim=(0, y_max))

    # Usafa Proportion Line
    ax.axhline(y=usafa_proportion, color='black', linestyle='--', alpha=0.5)

    # X ticks
    ax.set_xticklabels(afscs[tick_indices], rotation=afsc_rotation)
    ax.set_xticks(tick_indices)
    ax.tick_params(axis='x', labelsize=afsc_tick_size)
    ax.set(xlim=(-1, M))

    if display_title:
        ax.set_title(title, fontsize=title_size)

    if save:
        fig.savefig(paths['figures'] + title + '.png', bbox_inches='tight')

    return fig


def afocd_degree_proportions_results_graph(parameters, value_parameters, metrics=None, metrics_dict=None, save=False,
                                           figsize=(19, 7), facecolor='white', title=None, display_title=True,
                                           degree="Mandatory", label_size=25, afsc_tick_size=15, yaxis_tick_size=25,
                                           afsc_rotation=80, legend_size=None, title_size=None,
                                           y_max=1.1, dot_size=100):
    """
    Builds the AFOCD Proportion Results Bar Chart
    :param legend_size: font size of the legend
    :param title_size: font size of the title
    :param y_max: multiplier of the maximum y value for the graph to extend the window above
    :param dot_size: size of the scatter points
    :param degree: degree tier to show
    :param afsc_rotation: rotation of the AFSCs
    :param yaxis_tick_size: y axis tick sizes
    :param afsc_tick_size: x axis tick sizes for AFSCs
    :param label_size: size of labels
    :param title: title of chart
    :param display_title: if we should display the title
    :param metrics_dict: dictionary of solution metrics
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :param save: Whether we should save the graph
    :param parameters: fixed cadet/AFSC data
    :param value_parameters: value parameters
    :param metrics: solution metrics
    :return: figure
    """
    if legend_size is None:
        legend_size = label_size
    if title_size is None:
        title_size = label_size

    if title is None:
        title = degree + ' AFOCD Tier Proportion of Assigned Cadets for Each AFSC'

    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)

    # Load the data
    afscs = parameters['afsc_vector']
    M = len(afscs)
    k = np.where(value_parameters['objectives'] == degree)[0][0]
    tick_indices = np.arange(M)

    indices = np.where(value_parameters['objective_weight'][:, k] != 0)[0]
    if len(indices) < M:
        afscs = afscs[indices]
        M = len(afscs)
        tick_indices = np.arange(M)

    # More data
    if metrics_dict is None:

        proportions = metrics['objective_measure'][indices, k]
        minimums = np.zeros(M)
        maximums = np.zeros(M)
        colors = np.array([" " * 10 for _ in range(M)])
        x_under = []
        x_over = []
        x_within = []
        for j, loc in enumerate(indices):
            if degree == 'Mandatory':
                minimums[j] = value_parameters['objective_target'][loc, k]
                maximums[j] = 1
            else:
                minimums[j] = 0
                maximums[j] = value_parameters['objective_target'][loc, k]

            if minimums[j] <= proportions[j] <= maximums[j]:
                colors[j] = "blue"
                x_within.append(j)
            else:
                colors[j] = "red"
                if proportions[j] < minimums[j]:
                    x_under.append(j)
                else:
                    x_over.append(j)

        # Plot points
        ax.scatter(tick_indices, proportions, c=colors, linewidths=4, s=dot_size)

        # Calculate ranges
        y_within = [(minimums[j], maximums[j]) for j in x_within]
        y_under_ranges = [(minimums[j], maximums[j]) for j in x_under]
        y_over_ranges = [(minimums[j], maximums[j]) for j in x_over]
        y_under = [(proportions[j], minimums[j]) for j in x_under]
        y_over = [(maximums[j], proportions[j]) for j in x_over]

        # Plot bounds
        ax.scatter(afscs, minimums, c=colors, marker="_", linewidth=2)
        ax.scatter(afscs, maximums, c=colors, marker="_", linewidth=2)

        # Plot Ranges
        ax.plot((x_within, x_within), ([i for (i, j) in y_within], [j for (i, j) in y_within]), c="blue")
        ax.plot((x_under, x_under), ([i for (i, j) in y_under_ranges], [j for (i, j) in y_under_ranges]), c="black")
        ax.plot((x_over, x_over), ([i for (i, j) in y_over_ranges], [j for (i, j) in y_over_ranges]), c="black")

        # How far off
        ax.plot((x_under, x_under), ([i for (i, j) in y_under], [j for (i, j) in y_under]), c='red', linestyle='--')
        ax.plot((x_over, x_over), ([i for (i, j) in y_over], [j for (i, j) in y_over]), c='red', linestyle='--')

    else:
        num_solutions = len(list(metrics_dict.keys()))
        if colors is None:
            color_choices = ['red', 'blue', 'green', 'orange']
            marker_choices = ['o', 'D', '^', 'P']
            colors = {}
            markers = {}
            for s, solution in enumerate(list(metrics_dict.keys())):

                if s < len(color_choices):
                    colors[solution] = color_choices[s]
                    markers[solution] = marker_choices[s]
                else:
                    colors[solution] = color_choices[0]
                    markers[solution] = marker_choices[0]
        legend_elements = []
        above = np.zeros(M)
        below = np.repeat(10000, M)
        value_dict = {}
        minimums = np.zeros(M)
        maximums = np.zeros(M)
        for s, solution in enumerate(list(metrics_dict.keys())):

            x_under = []
            x_over = []
            proportions = np.zeros(M)
            minimums = np.zeros(M)
            maximums = np.zeros(M)
            for j, loc in enumerate(indices):
                proportions[j] = metrics_dict[solution]['objective_measure'][loc, k]
                afsc = parameters['afsc_vector'][loc]

                if value_parameters['constraint_type'][loc, k] != 0:
                    value_list = value_parameters['objective_value_min'][loc, k].split(",")
                    minimums[j] = float(value_list[0].strip())
                    maximums[j] = min(1, float(value_list[1].strip()))
                else:
                    if degree == 'Mandatory':
                        minimums[j] = value_parameters['objective_target'][loc, k]
                        maximums[j] = 1
                    elif degree == 'Desired' and afsc not in ['15A', '61A', '14F', '17X', '17D', '17S', '17DXS']:
                        minimums[j] = value_parameters['objective_target'][loc, k]
                        maximums[j] = 1
                    else:
                        minimums[j] = 0
                        maximums[j] = value_parameters['objective_target'][loc, k]
                if minimums[j] > proportions[j]:
                    x_under.append(j)
                elif proportions[j] > maximums[j]:
                    x_over.append(j)
            value_dict[solution] = proportions

            # further solutions away from the range
            below = np.array([min(proportions[j], below[j]) for j in range(M)])
            above = np.array([max(proportions[j], above[j]) for j in range(M)])

            # Plot Points
            ax.scatter(tick_indices, proportions, color=colors[solution], marker=markers[solution], edgecolor='black',
                       s=dot_size, zorder=2)

            element = mlines.Line2D([], [], color=colors[solution], marker=markers[solution], linestyle='None',
                                    markeredgecolor='black', markersize=20, label=solution)
            legend_elements.append(element)

        # Plot more lines
        for j in range(M):
            real_min = minimums[j]
            for solution in metrics_dict:
                val = value_dict[solution][j]
                real_min = min(val, real_min)
            ax.plot((j, j), (0, real_min), color='black', linestyle='--', zorder=1, alpha=0.5,
                    linewidth=2)

        # Calculate ranges
        y = [(minimums[j], maximums[j]) for j in tick_indices]
        y_under = [(below[j], minimums[j]) for j in x_under]
        y_over = [(maximums[j], above[j]) for j in x_over]

        # Plot bounds
        ax.scatter(afscs, minimums, c='black', marker="_", linewidth=2, zorder=1)
        ax.scatter(afscs, maximums, c='black', marker="_", linewidth=2, zorder=1)

        # Plot Ranges
        ax.plot((tick_indices, tick_indices), ([i for (i, j) in y], [j for (i, j) in y]), c="black", zorder=1)

        # How far off
        ax.plot((x_under, x_under), ([i for (i, j) in y_under], [j for (i, j) in y_under]), c='red',
                linestyle='--', zorder=1)
        ax.plot((x_over, x_over), ([i for (i, j) in y_over], [j for (i, j) in y_over]), c='red', linestyle='--',
                zorder=1)

        # Legend
        ax.legend(handles=legend_elements, edgecolor='black', fontsize=legend_size, loc='upper left',
                  ncol=num_solutions, columnspacing=0.5, handletextpad=0.05, borderaxespad=0.5, borderpad=0.2)
    # Y ticks
    y_ticks = np.arange(6) / 5
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='y', labelsize=yaxis_tick_size)
    ax.set_yticklabels(y_ticks)
    ax.margins(y=0)
    ax.set(ylim=(0, y_max))

    # X ticks
    ax.set(xticks=tick_indices, xticklabels=afscs[tick_indices])
    ax.tick_params(axis='x', labelsize=afsc_tick_size)
    ax.set_xticklabels(afscs[tick_indices], rotation=afsc_rotation)
    ax.set(xlim=(-1, M))

    # Titles and Labels
    if display_title:
        ax.set_title(title, fontsize=title_size)

    ax.set_ylabel(degree + ' Degree Tier Proportion')
    ax.yaxis.label.set_size(label_size)
    ax.set_xlabel('AFSCs')
    ax.xaxis.label.set_size(label_size)

    if save:
        fig.savefig(paths['figures'] + title + '.png', bbox_inches='tight')

    return fig


def average_utility_results_graph(parameters, value_parameters, metrics=None, metrics_dict=None, save=False,
                                  figsize=(19, 7), facecolor='white', title=None, display_title=True, label_size=25,
                                  afsc_tick_size=15, yaxis_tick_size=25, afsc_rotation=80, skip_afscs=False,
                                  legend_size=None, title_size=None, y_max=1.1, alpha=0.5, bar_color='black',
                                  dot_size=100, colors=None):
    """
    Builds the Average Utility Results Chart
    :param colors: colors of the solutions
    :param alpha: alpha parameter for the bars of the figure
    :param bar_color: color of bars for figure (for certain kinds of graphs)
    :param skip_afscs: Whether we should label every other AFSC
    :param legend_size: font size of the legend
    :param title_size: font size of the title
    :param y_max: multiplier of the maximum y value for the graph to extend the window above
    :param dot_size: size of the scatter points
    :param afsc_rotation: rotation of the AFSCs
    :param yaxis_tick_size: y axis tick sizes
    :param afsc_tick_size: x axis tick sizes for AFSCs
    :param label_size: size of labels
    :param title: title of the chart
    :param display_title: if we should display the title
    :param metrics_dict: dictionary of solution metrics
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :param save: Whether we should save the graph
    :param parameters: fixed cadet/AFSC data
    :param value_parameters: value parameters
    :param metrics: solution metrics
    :return: figure
    """
    if legend_size is None:
        legend_size = label_size
    if title_size is None:
        title_size = label_size

    if title is None:
        title = 'Average Utility of Assigned Cadets for Each AFSC'

    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)

    # Load the data
    afscs = parameters['afsc_vector']
    M = len(afscs)
    indices = np.arange(M)

    # We can skip AFSCs
    if skip_afscs:
        tick_indices = np.arange(1, M, 2).astype(int)
    else:
        tick_indices = indices

    utility_k = np.where(value_parameters['objectives'] == 'Utility')[0][0]
    if metrics_dict is None:

        utility = metrics['objective_measure'][:, utility_k]

        # Bar Chart
        ax.bar(indices, utility, color=bar_color, alpha=alpha, edgecolor='black')

    else:

        num_solutions = len(list(metrics_dict.keys()))
        if colors is None:
            color_choices = ['red', 'blue', 'green', 'orange']
            marker_choices = ['o', 'D', '^', 'P']
            colors = {}
            markers = {}
            for s, solution in enumerate(list(metrics_dict.keys())):

                if s < len(color_choices):
                    colors[solution] = color_choices[s]
                    markers[solution] = marker_choices[s]
                else:
                    colors[solution] = color_choices[0]
                    markers[solution] = marker_choices[0]
        legend_elements = []
        max_value = np.zeros(M)
        min_value = np.repeat(10000, M)
        for s, solution in enumerate(list(metrics_dict.keys())):
            utility = metrics_dict[solution]['objective_measure'][:, utility_k]
            ax.scatter(indices, utility, color=colors[solution], marker=markers[solution], edgecolor='black',
                       s=dot_size, zorder=2)

            element = mlines.Line2D([], [], color=colors[solution], marker=markers[solution], linestyle='None',
                                    markeredgecolor='black', markersize=20, label=solution)
            legend_elements.append(element)
            max_value = np.array([max(max_value[j], utility[j]) for j in range(M)])
            min_value = np.array([min(min_value[j], utility[j]) for j in range(M)])

        for j in range(M):
            ax.plot((j, j), (0, max_value[j]), color='black', linestyle='--', zorder=1, alpha=0.4, linewidth=2)

        ax.legend(handles=legend_elements, edgecolor='black', fontsize=legend_size, loc='upper left',
                  ncol=num_solutions, columnspacing=0.5, handletextpad=0.05, borderaxespad=0.5, borderpad=0.2)

    # Labels
    ax.set_ylabel('Average Utility')
    ax.yaxis.label.set_size(label_size)
    ax.set_xlabel('AFSCs')
    ax.xaxis.label.set_size(label_size)

    # Y ticks
    y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='y', labelsize=yaxis_tick_size)
    ax.set_yticklabels(y_ticks)
    ax.margins(y=0)
    ax.set(ylim=(0, y_max))

    # X ticks
    ax.set(xticks=tick_indices, xticklabels=afscs[tick_indices])
    ax.tick_params(axis='x', labelsize=afsc_tick_size)
    ax.set_xticklabels(afscs[tick_indices], rotation=afsc_rotation)
    ax.set(xlim=(-1, M))

    # # top line
    # ax.plot((-1, M), (1, 1), color='black', linestyle='-', zorder=1, alpha=0.5, linewidth=2)

    if display_title:
        ax.set_title(title, fontsize=title_size)

    if save:
        fig.savefig(paths['figures'] + title + '.png', bbox_inches='tight')

    return fig


def cadet_utility_histogram(metrics=None, metrics_dict=None, save=False, figsize=(19, 7), facecolor='white',
                            title=None, display_title=True, label_size=25, xaxis_tick_size=15, yaxis_tick_size=25,
                            dpi=100, gui_chart=False, legend_size=None, title_size=None):
    """
    Builds the Cadet Utility histogram
    :param dpi: dots per inch for figure
    :param gui_chart: if this graph is used in the GUI
    :param legend_size: font size of the legend
    :param title_size: font size of the title
    :param yaxis_tick_size: y axis tick sizes
    :param xaxis_tick_size: x axis tick sizes
    :param label_size: size of labels
    :param title: title of chart
    :param display_title: if we should show the title
    :param metrics_dict: dictionary of solution metrics
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :param save: Whether we should save the graph
    :param metrics: solution metrics
    :return: figure
    """
    if legend_size is None:
        legend_size = label_size
    if title_size is None:
        title_size = label_size

    if title is None:
        title = 'Cadet Utility Results Histogram'

    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, dpi=dpi, tight_layout=True)

    bins = np.arange(21) / 20
    if metrics_dict is None:
        value = metrics['cadet_value']
        ax.hist(value, bins=bins, edgecolor='white', color='black', alpha=1)

    else:

        num_solutions = len(list(metrics_dict.keys()))
        if colors is None:
            color_choices = ['red', 'blue', 'green', 'orange']
            colors = {}
            for s, solution in enumerate(list(metrics_dict.keys())):

                if s < len(color_choices):
                    colors[solution] = color_choices[s]
                else:
                    colors[solution] = color_choices[0]
        legend_elements = []
        for s, solution in enumerate(list(metrics_dict.keys())):
            value = metrics_dict[solution]['cadet_value']
            ax.hist(value, bins=bins, edgecolor='black', color=colors[solution], alpha=0.5)
            legend_elements.append(Patch(facecolor=colors[solution], label=solution, alpha=0.5, edgecolor='black'))

        ax.legend(handles=legend_elements, edgecolor='black', fontsize=legend_size, loc='upper left',
                  ncol=num_solutions, columnspacing=0.8, handletextpad=0.25, borderaxespad=0.5, borderpad=0.4)

    # Labels
    ax.set_ylabel('Number of Cadets')
    ax.yaxis.label.set_size(label_size)
    ax.set_xlabel('Utility Received')
    ax.xaxis.label.set_size(label_size)

    # Axis
    if gui_chart:
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        x_ticks = np.arange(11) / 10
        ax.set_xticks(x_ticks)
        ax.tick_params(axis='x', labelsize=xaxis_tick_size, colors='white')
        ax.tick_params(axis='y', labelsize=yaxis_tick_size, colors='white')
        ax.spines['right'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.set_title('Cadet Utility Distribution', fontsize=title_size, color='white')
    else:
        x_ticks = np.arange(11) / 10
        ax.set_xticks(x_ticks)
        ax.tick_params(axis='x', labelsize=xaxis_tick_size)
        ax.tick_params(axis='y', labelsize=yaxis_tick_size)
        # y_ticks = [200, 400, 600, 800, 1000, 1200]
        # ax.set_yticks(y_ticks)

        if display_title:
            ax.set_title(title, fontsize=title_size)

    if save:
        fig.savefig(paths['figures'] + title + '.png', bbox_inches='tight')

    return fig


def holistic_color_graph(parameters, value_parameters, metrics, figsize=(11, 10), save=False, facecolor='white'):
    """
    Builds the Holistic Color Chart
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :param save: Whether we should save the graph
    :param parameters: fixed cadet/AFSC data
    :param value_parameters: value parameters
    :param metrics: solution metrics
    :return: figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)
    ax.set_aspect('equal', adjustable='box')
    N = parameters['N']
    title = 'Attribute Weights and Values Color Chart. Z = ' + str(round(metrics['z'], 2))
    ax.set_title(title)

    # Cadets
    length = value_parameters['cadets_overall_weight']
    values = metrics['cadet_value']
    weights = value_parameters['cadet_weight']
    sorted_indices = values.argsort()[::-1]
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    y = 0
    for i in range(N):
        xy = (0, y)
        height = sorted_weights[i]
        objective_value = sorted_values[i]
        c = (1 - objective_value, 0, objective_value)
        rect = Rectangle(xy, length, height, edgecolor='none', color=c)
        ax.add_patch(rect)
        y += height

    # AFSCs
    full_length = value_parameters['afscs_overall_weight']
    afsc_values = metrics['afsc_value']
    afsc_weights = value_parameters['afsc_weight']
    sorted_afsc_indices = afsc_values.argsort()[::-1]
    sorted_afsc_weights = afsc_weights[sorted_afsc_indices]
    y = 0
    for j in sorted_afsc_indices:
        weights = value_parameters['objective_weight'][j, :]
        values = metrics['objective_value'][j, :]
        zeros = np.where(weights == 0)[0]
        weights = np.delete(weights, zeros)
        values = np.delete(values, zeros)
        sorted_indices = values.argsort()[::-1]
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        x = value_parameters['cadets_overall_weight']
        height = sorted_afsc_weights[j]
        for k in range(len(sorted_indices)):
            xy = (x, y)
            objective_value = sorted_values[k]
            c = (1 - objective_value, 0, objective_value)
            rect = Rectangle(xy, sorted_weights[k], height, edgecolor='none', color=c)
            ax.add_patch(rect)
            x += sorted_weights[k] * full_length
        height = sorted_afsc_weights[j]
        y += height

    if save:
        fig.savefig(paths['figures'] + title + '.png', bbox_inches='tight')

    return fig


# Sensitivity Analysis
def pareto_graph(pareto_df, dimensions=None, save=False, title=None, figsize=(10, 6), facecolor='white',
                 thesis_chart=False, display_title=False):
    """
    Builds the Pareto Frontier Chart for adjusting the overall weight on cadets
    :param display_title: if we should display a title or not
    :param thesis_chart:
    :param save: If we should save the figure
    :param title: If we should include a title or not
    :param pareto_df: data frame of pareto analysis
    :param dimensions: N and M
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :return: figure
    """
    if thesis_chart:
        figsize = (12, 10)
        display_title = False
        if title is None:
            title = 'Results_Pareto'
        save = True
        # cm = plt.cm.get_cmap('binary')
        cm = plt.cm.get_cmap('RdYlBu')
        label_size = 35
        xaxis_tick_size = 30
        yaxis_tick_size = 30
    else:
        cm = plt.cm.get_cmap('RdYlBu')
        label_size = 35
        xaxis_tick_size = 30
        yaxis_tick_size = 30

    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)
    ax.set_aspect('equal', adjustable='box')

    x_data = pareto_df['Value on AFSCs']
    y_data = pareto_df['Value on Cadets']

    sc = ax.scatter(pareto_df['Value on AFSCs'], pareto_df['Value on Cadets'], c=pareto_df['Weight on Cadets'],
                    s=150, cmap=cm, edgecolor='black', zorder=1)
    # ax.plot(pareto_df['Value on AFSCs'], pareto_df['Value on Cadets'], c='black', linestyle='--', alpha=0.3, zorder=1)
    min_value = min(min(pareto_df['Value on Cadets']), min(pareto_df['Value on AFSCs'])) - 0.005
    max_value = max(max(pareto_df['Value on Cadets']), max(pareto_df['Value on AFSCs'])) + 0.005
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    c_bar = plt.colorbar(sc)
    c_bar.set_label('Weight on Cadets', fontsize=label_size)
    c_bar.ax.tick_params(labelsize=xaxis_tick_size)
    if title is None:
        if dimensions is not None:
            N = dimensions[0]
            M = dimensions[1]
            title = 'Pareto Frontier for Weight on Cadets (N=' + str(N) + ', M=' + str(M) + ')'
        else:
            title = 'Pareto Frontier for Weight on Cadets'
    if display_title:
        ax.set_title(title)

    # Labels
    ax.set_ylabel('Value on Cadets')
    ax.yaxis.label.set_size(label_size)
    ax.set_xlabel('Value on AFSCs')
    ax.xaxis.label.set_size(label_size)

    # Axis
    ax.tick_params(axis='y', labelsize=yaxis_tick_size)
    ax.tick_params(axis='x', labelsize=xaxis_tick_size)

    if save:
        fig.savefig(paths['figures'] + title + '.png')

    return fig


def afsc_objective_weights_graph(parameters, value_parameters_dict, afsc, colors=None, save=False, figsize=(19, 7),
                                 facecolor="white", title=None, display_title=True, label_size=25, bar_color=None,
                                 xaxis_tick_size=15, yaxis_tick_size=25, legend_size=25, title_size=25):
    """
    This chart compares the weights under different value parameters for AFSC objectives for a particular AFSC
    :param bar_color: color of bars for figure (for certain kinds of graphs)
    :param title_size: font size of the title
    :param legend_size: font size of the legend
    :param yaxis_tick_size: y axis tick sizes
    :param xaxis_tick_size: x axis tick sizes
    :param label_size: size of labels
    :param display_title: if we should show the title
    :param title: title of chart
    :param parameters: fixed cadet/AFSC parameters
    :param value_parameters_dict: dictionary of value parameters
    :param afsc: which AFSC we should plot
    :param colors: colors for the kinds of weights
    :param save: Whether we should save the graph
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :return: figure
    """
    if colors is None:
        colors = ['blue', 'black', 'orange', 'magenta']

    if title is None:
        title = afsc + ' Objective Weights For Different Value Parameters'

    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)

    # Get chart specs
    num_weights = len(value_parameters_dict)
    j = np.where(parameters['afsc_vector'] == afsc)[0][0]
    first_key = list(value_parameters_dict.keys())[0]
    K_A = value_parameters_dict[first_key]['K^A'][j].astype(int)
    objectives = value_parameters_dict[first_key]['objectives'][K_A]
    for k, objective in enumerate(objectives):
        if objective == 'USAFA Proportion':
            objectives[k] = 'USAFA\nProportion'
        elif objective == 'Combined Quota':
            objectives[k] = 'Combined\nQuota'

    if colors is None:
        colors = ['blue', 'lime', 'orange', 'magenta', 'yellow', 'cyan', 'green', 'deeppink', 'red']
        colors = colors[:num_weights]

    if bar_color is not None:
        colors = [bar_color for _ in range(num_weights)]

    # Set the bar chart structure parameters
    label_locations = np.arange(len(K_A))
    bar_width = 0.8 / num_weights

    # Loop through each set of value parameters
    max_weight = 0
    for w_num, weight_name in enumerate(value_parameters_dict):

        # Plot weights
        weights = value_parameters_dict[weight_name]['objective_weight'][j, K_A]
        max_weight = max(max_weight, max(weights))
        ax.bar(label_locations + bar_width * w_num, weights, bar_width, edgecolor='black',
               label=weight_name, color=colors[w_num], alpha=0.5)

    # Labels
    ax.set_ylabel('Objective Weight')
    ax.yaxis.label.set_size(label_size)
    if bar_color is not None:
        ax.set_xlabel('Objectives')
        ax.xaxis.label.set_size(label_size)

    # X ticks
    ax.set(xticks=label_locations + (bar_width / 2) * (num_weights - 1),
           xticklabels=[value_parameters_dict[first_key]['objectives'][i] for i in K_A])
    ax.tick_params(axis='x', labelsize=xaxis_tick_size)
    ax.set_xticklabels(objectives)

    # Y ticks
    ax.tick_params(axis='y', labelsize=yaxis_tick_size)
    ax.margins(y=0)
    ax.set(ylim=(0, max_weight * 1.2))

    if display_title:
        ax.set_title(title, fontsize=title_size)

    if bar_color is None:
        ax.legend(edgecolor='black', fontsize=legend_size, loc='upper right',
                  ncol=num_weights, columnspacing=0.8, handletextpad=0.25, borderaxespad=0.5, borderpad=0.4)

    if save:
        fig.savefig(paths['figures'] + title + '.png')

    return fig


def solution_parameter_comparison_graph(z_dict, colors=None, save=False, figsize=(19, 7), facecolor="white"):
    """
        This chart compares the solutions' objective values under different value parameters
        :param z_dict: dictionary of solution objective values for each set of value parameters
        :param value_parameters_dict: dictionary of value parameters
        :param colors: colors for the kinds of weights
        :param save: Whether we should save the graph
        :param facecolor: color of the background of the graph
        :param figsize: size of the figure
        :return: figure
        """
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)

    # Get chart specs
    solution_names = list(z_dict.keys())
    vp_names = list(z_dict[solution_names[0]].keys())
    num_solutions = len(solution_names)
    num_vps = len(vp_names)

    if colors is None:
        colors = ['blue', 'lime', 'orange', 'magenta', 'yellow', 'cyan', 'green', 'deeppink', 'red']
        colors = colors[:num_solutions]

    # Set the bar chart structure parameters
    label_locations = np.arange(num_vps)
    bar_width = 0.8 / num_solutions

    # Loop through each set of solutions
    legend_elements = []
    for s_num, solution_name in enumerate(solution_names):
        legend_elements.append(Patch(facecolor=colors[s_num], label=solution_name, alpha=0.5, edgecolor='black'))
        for vp_num, vp_name in enumerate(vp_names):

            # Plot solutions
            ax.bar(label_locations[vp_num] + bar_width * s_num, z_dict[solution_name][vp_name], bar_width,
                   edgecolor='black', color=colors[s_num])

    # Text
    ax.set_ylabel('Z')
    ax.set(xticks=label_locations + bar_width / 2, xlim=[2 * bar_width - 1, num_vps],
           xticklabels=vp_names)

    title = 'Objective Values for Different Solutions and Value Parameters'
    ax.set_title(title)
    ax.legend(handles=legend_elements, edgecolor='black', loc='upper right', columnspacing=0.8, handletextpad=0.25,
              borderaxespad=0.5, borderpad=0.4)
    if save:
        fig.savefig(paths['figures'] + title + '.png')

    return fig


def solution_results_graph(parameters, value_parameters, metrics_dict, vp_name, k, save=False, colors=None,
                           figsize=(19, 7), facecolor='white'):
    """
    Builds the Graph to show how well we meet each of the objectives
    :param colors: colors of the solutions
    :param vp_name: value parameter name (to access from metrics_dict)
    :param k: objective index
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :param save: Whether we should save the graph
    :param parameters: fixed cadet/AFSC data
    :param value_parameters: value parameters
    :param metrics_dict: solution metrics dictionary
    :return: figure
    """

    # Load the data
    indices = value_parameters['J^E'][k]
    afscs = parameters['afsc_vector'][indices]
    minimums = np.zeros(len(indices))
    maximums = np.zeros(len(indices))
    num_solutions = len(metrics_dict.keys())
    if colors is None:
        colors = ['blue', 'lime', 'orange', 'magenta', 'yellow', 'cyan', 'green', 'deeppink', 'red']
        colors = colors[:num_solutions]

    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)
    for j, loc in enumerate(indices):
        if k == 0:
            minimums[j], maximums[j] = 0.35, 0.65
        elif k == 1:
            minimums[j], maximums[j] = 0.20, 0.40
        elif k == 2:
            minimums[j] = parameters['quota_min'][j] / parameters['quota'][j]
            maximums[j] = parameters['quota_max'][j] / parameters['quota'][j]
        elif k == 3:
            if parameters['usafa_quota'][j] / parameters['quota'][j] == 0:
                minimums[j], maximums[j] = 0, 0
            elif parameters['usafa_quota'][j] / parameters['quota'][j] == 1:
                minimums[j], maximums[j] = 1, 1
            else:
                minimums[j] = parameters['usafa_quota'][j] / parameters['quota'][j] - 0.1
                maximums[j] = parameters['usafa_quota'][j] / parameters['quota'][j] + 0.1
        elif k == 4:
            if parameters['usafa_quota'][j] / parameters['quota'][j] == 0:
                minimums[j], maximums[j] = 1, 1
            elif parameters['usafa_quota'][j] / parameters['quota'][j] == 1:
                minimums[j], maximums[j] = 0, 0
            else:
                minimums[j] = (parameters['quota'][j] - parameters['usafa_quota'][j]) / parameters['quota'][j] - 0.1
                maximums[j] = (parameters['quota'][j] - parameters['usafa_quota'][j]) / parameters['quota'][j] + 0.1
        elif k in [5, 6, 7]:
            if k == 5 or (k == 6 and afscs[j] not in ["14F", "15A", "17D"]):
                minimums[j] = value_parameters['objective_target'][loc, k]
                maximums[j] = 1
            else:
                minimums[j] = 0
                maximums[j] = value_parameters['objective_target'][loc, k]
        elif k == 8:
            minimums[j], maximums[j] = 0.8, 1
        elif k == 9:
            male_proportion = np.mean(parameters['male'])
            minimums[j], maximums[j] = male_proportion - 0.1, male_proportion + 0.1
        elif k == 10:
            minority_proportion = np.mean(parameters['minority'])
            minimums[j], maximums[j] = minority_proportion - 0.1, minority_proportion + 0.1

    # ticks = list(np.arange(0, 1.1, 0.1))
    # ax.set_yticks(ticks)
    for s_num, solution_name in enumerate(metrics_dict.keys()):
        if k == 2:
            measures = metrics_dict[solution_name][vp_name]['objective_measure'][indices, k] / \
                       parameters['quota'][indices]
        elif k == 3:
            measures = metrics_dict[solution_name][vp_name]['objective_measure'][indices, k] / \
                       parameters['usafa_quota'][indices]
        elif k == 4:
            measures = metrics_dict[solution_name][vp_name]['objective_measure'][indices, k] / \
                       (parameters['quota'][indices] - parameters['usafa_quota'][indices])
        else:
            measures = metrics_dict[solution_name][vp_name]['objective_measure'][indices, k]

        ax.scatter(afscs, measures, c=colors[s_num], linewidths=4)

    # Ranges
    y = [(minimums[i], maximums[i]) for i in range(len(afscs))]
    x = range(len(afscs))
    for i in x:
        plt.axvline(x=i, color='black', linestyle='--', alpha=0.2)
    ax.plot((x, x), ([i for (i, j) in y], [j for (i, j) in y]), c='black')
    ax.scatter(afscs, minimums, c='black', marker="_", linewidth=2)
    ax.scatter(afscs, maximums, c='black', marker="_", linewidth=2)

    # Titles and Labels
    objective = value_parameters['objectives'][k]
    ax.set_title(objective + ' Solution Comparison')
    ax.set_ylabel(objective + ' Measure')

    if save:
        fig.savefig(paths['figures'] + 'Solution_Results.png', bbox_inches='tight')

    return fig


def solution_similarity_graph(coords, solution_names, title=None, colors=None, figsize=(10, 7), thesis_chart=False,
                              display_title=True, use_models=False, facecolor='white', save=False, year=None,
                              _plt=None):
    """
    This is the chart that compares the approximate and exact models (with genetic algorithm) in solve time and
    objective value
    :param _plt: this was used for the defense slides to add some connecting red lines for some slides
    :param year: year of the data
    :param thesis_chart: if this is for the thesis paper or not
    :param use_models: if we want to take advantage of value parameters/models and plot using colors
    and markers (colors for VPs, markers for models)
    :param coords: coordinates to plot
    :param solution_names: names of the solutions
    :param display_title: if we want to display a title or not for the chart
    :param colors: colors of problem instances
    :param title: the title of the chart
    :param save: If we should save the figure
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :return: figure
    """
    if thesis_chart:
        figsize = (12.5, 10)
        display_title = False
        if title is None:
            if _plt is not None:
                title = 'Similarity_' + str(year) + '_' + str(_plt)
            else:
                title = 'Similarity_' + str(year)
        save = True
        use_models = False
    size = 220
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)
    ax.set_aspect('equal', adjustable='box')

    if colors is None:
        colors = ['green', 'blue', 'orange', 'purple', 'red', 'cyan', 'magenta', 'lime', 'pink', 'yellow',
                  'deepskyblue', 'gold', 'deeppink', 'sandybrown', 'olive', 'maroon', 'navy', 'coral', 'teal',
                  'darkorange']
        colors = colors[:len(coords)]

    if thesis_chart:
        for i, solution_name in enumerate(solution_names):
            x, y = coords[i, 0], coords[i, 1]
            if solution_name == 'Original':
                ax.scatter(x, y, c='red', marker='X', edgecolor='black', s=size, zorder=2)
            elif solution_name == 'VFT New':
                ax.scatter(x, y, c='green', marker='s', edgecolor='black', s=size, zorder=2)
            else:
                ax.scatter(x, y, c='blue', marker='o', edgecolor='black', s=size, zorder=2)

        solution_1 = '2018_VP_30'
        if _plt == 1:
            solutions = ['2018_VP_14']
        elif _plt == 2:
            solutions = ['2018_VP_14', '2018_VP_13']
        elif _plt == 3:
            solutions = ['2018_VP_14', '2018_VP_13', '2018_VP_20']
        elif _plt == 4:
            solutions = ['2018_VP_14', '2018_VP_13', '2018_VP_20', '2018_VP_9']
        elif _plt == 5:
            solutions = solution_names
        elif _plt == 6:
            solution_1 = '2018_VP_14'
            solutions = solution_names
        elif _plt == 7:
            solution_1 = '2018_VP_13'
            solutions = solution_names
        elif _plt == 8:
            solution_1 = 'Original'
            solutions = solution_names

        if _plt is not None:
            for solution in solutions:
                if solution != solution_1:
                    s_1 = np.where(solution_names == solution_1)[0][0]
                    s_2 = np.where(solution_names == solution)[0][0]
                    x1, x2 = coords[s_1, 0], coords[s_2, 0]
                    y1, y2 = coords[s_1, 1], coords[s_2, 1]
                    ax.plot((x1, x2), (y1, y2), linestyle='-', linewidth=3, color='red', zorder=1)

        # Create legend
        if 'VFT New' in solution_names:
            legend_elements = [mlines.Line2D([], [], color='black', marker='X', linestyle='None',
                                             markeredgecolor='black', markersize=20, label='Original'),
                               mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                                             markeredgecolor='black', markersize=20, label='VFT'),
                               mlines.Line2D([], [], color='green', marker='s', linestyle='None',
                                             markeredgecolor='black', markersize=20, label='VFT New')]
        else:
            legend_elements = [mlines.Line2D([], [], color='red', marker='X', linestyle='None',
                                             markeredgecolor='black', markersize=20, label='Original'),
                               mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                                             markeredgecolor='black', markersize=20, label='VFT')]

        legend = ax.legend(handles=legend_elements, edgecolor='black', fontsize=35, loc='upper left',
                           ncol=len(legend_elements),
                           columnspacing=0.4, handletextpad=0.1, borderaxespad=0.5, borderpad=0.2)
        legend.get_title().set_fontsize('35')

        # Remove tick marks
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        if save:
            fig.savefig(paths['figures'] + title + '.png')

        return fig
    if use_models:
        markers = {'Original': 'X', 'Approximate': 'o', 'Approximate_GA': '^', 'Exact': 's',
                   'Full_GA': 'v', 'VFT': 'o'}
        colors = {'A': 'blue', 'B': 'green', 'C': 'orange', 'D': 'magenta', 'E': 'yellow',
                  'F': 'cyan', 'G': 'red', 'H': 'lime', 'I': 'deeppink', 'J': 'purple', 'Original': 'black'}
        vps = []
        models = []
        vp_legend = []
        model_legend = []
        for i, solution_name in enumerate(solution_names):

            # Get correct model and vp
            if thesis_chart:

                # Get names
                if len(solution_name) == 1:
                    vp = solution_name
                    model = 'VFT'
                else:
                    vp = 'Original'
                    model = 'Original'

                # Add legend elements
                element = mlines.Line2D([], [], color=colors[vp], marker=markers[model], linestyle='None',
                                        markeredgecolor='black', markersize=20, label=vp)
                vp_legend.append(element)

            else:

                # Get names
                name_split = solution_name.split('_')
                if len(name_split) == 1:
                    model = name_split[0]
                    vp = None
                elif len(name_split) == 2:
                    model = name_split[0]
                    vp = name_split[1]
                else:
                    model = name_split[0] + '_' + name_split[1]
                    vp = name_split[2]

                # Add legend elements
                if vp not in vps:
                    if vp is not None:
                        vps.append(vp)
                        vp_legend.append(Patch(facecolor=colors[vp], label=vp, edgecolor='black'))
                if model not in models:
                    models.append(model)
                    if model == 'Original':
                        fill_color = 'black'
                    else:
                        fill_color = facecolor
                    element = mlines.Line2D([], [], color=fill_color, marker=markers[model], linestyle='None',
                                            markeredgecolor='black', markersize=20, label=model)
                    model_legend.append(element)

            # Plot solution
            x = coords[i, 0]
            y = coords[i, 1]
            ax.scatter(x, y, c=colors[vp], marker=markers[model], edgecolor='black', s=size)

        if title is None:
            title = 'Solution Similarity Graph'

        # Remove tick marks
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Create legend
        if thesis_chart:
            legend = ax.legend(handles=vp_legend, edgecolor='black', fontsize=25, title='Solutions', loc='upper left',
                               ncol=2)
            legend.get_title().set_fontsize('25')
        # else:
        #     ax2.scatter(np.NaN, np.NaN)
        #     ax2.get_yaxis().set_visible(False)
        #     ax.legend(handles=vp_legend, edgecolor='black', loc=2)
        #     ax2.legend(handles=model_legend, edgecolor='black', loc=4)

        if display_title:
            ax.set_title(title)

        if save:
            fig.savefig(paths['figures'] + title + '.png')

        return fig
