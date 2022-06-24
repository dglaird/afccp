# Import libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import numpy as np
from afccp.core.globals import *
from afccp.core.handling.data_handling import get_utility_preferences

# Set matplotlib default font to Times New Roman
import matplotlib as mpl

mpl.rc('font', family='Times New Roman')


def data_graph(instance):
    """
    Creates the fixed parameter data plots. These figures show the characteristics of the cadets as they pertain
    to each AFSC as well as each AFSC objective
    :return: figure
    """

    # Shorthand
    ip = instance.plt_p
    p = instance.parameters

    # Create figure
    fig, ax = plt.subplots(figsize=ip['figsize'], facecolor=ip['facecolor'], tight_layout=True, dpi=ip['dpi'])

    # Load the data
    eligible_count = np.array([len(p['I^E'][j]) for j in range(p['M'])])

    # Get applicable AFSCs
    indices = np.where(eligible_count <= ip["eligibility_limit"])[0]
    afscs = p['afsc_vector'][indices]
    M = len(afscs)

    # We can skip AFSCs
    if ip['skip_afscs']:
        tick_indices = np.arange(1, M, 2).astype(int)
    else:
        tick_indices = np.arange(M).astype(int)

    # Create figures
    add_legend = False
    if ip['data_graph'] in ['Average Merit', 'USAFA Proportion', 'Average Utility']:

        # Get correct metric
        if ip['data_graph'] == 'Average Merit':
            metric = np.array([np.mean(p['merit'][p['I^E'][j]]) for j in indices])
            target = 0.5
        elif ip['data_graph'] == 'USAFA Proportion':
            metric = np.array([len(p['I^D'][ip['data_graph']][j]) / len(p['I^E'][j]) for j in indices])
            target = p['usafa_proportion']
        else:
            if ip["eligibility"]:
                metric = np.array([np.mean(p['utility'][p['I^E'][j], j]) for j in indices])
            else:
                metric = np.array([np.mean(p['utility'][:, j]) for j in indices])
            target = None

        # Bar Chart
        ax.bar(afscs, metric, color=ip["bar_color"], alpha=ip["alpha"], edgecolor='black')

        if target is not None:
            ax.axhline(y=target, color='black', linestyle='--', alpha=ip["alpha"])

        # Get correct text
        y_label = ip['data_graph']
        if ip['title'] is None:
            ip['title'] = ip['data_graph'] + ' Across Eligible Cadets for AFSCs with less than ' + \
                          str(ip['eligibility_limit']) + ' Eligible Cadets'
            if ip['data_graph'] == 'Average Utility' and not ip['eligibility']:
                ip['title'] = ip['data_graph'] + ' Across All Cadets for AFSCs with less than ' + \
                              str(ip['eligibility_limit']) + ' Eligible Cadets'

    elif ip['data_graph'] == 'AFOCD Data':

        # Legend
        add_legend = True
        legend_elements = [Patch(facecolor=ip["bar_colors"]["Permitted"], label='Permitted', edgecolor='black'),
                           Patch(facecolor=ip["bar_colors"]["Desired"], label='Desired', edgecolor='black'),
                           Patch(facecolor=ip["bar_colors"]["Mandatory"], label='Mandatory', edgecolor='black')]

        # Get metrics
        mandatory_count = np.array([np.sum(p['mandatory'][:, j]) for j in indices])
        desired_count = np.array([np.sum(p['desired'][:, j]) for j in indices])
        permitted_count = np.array([np.sum(p['permitted'][:, j]) for j in indices])

        # Bar Chart
        ax.bar(afscs, mandatory_count, color=ip["bar_colors"]["Mandatory"], edgecolor='black')
        ax.bar(afscs, desired_count, bottom=mandatory_count, color=ip["bar_colors"]["Desired"], edgecolor='black')
        ax.bar(afscs, permitted_count, bottom=mandatory_count + desired_count, color=ip["bar_colors"]["Permitted"],
               edgecolor='black')

        # Axis Adjustments
        ax.set(ylim=(0, ip['eligibility_limit'] + ip['eligibility_limit'] / 100))

        # Get correct text
        y_label = "Number of Cadets"
        if ip['title'] is None:
            ip['title'] = 'AFOCD Degree Tier Breakdown for AFSCs with less than ' + \
                          str(ip['eligibility_limit']) + ' Eligible Cadets'

    elif ip['data_graph'] == 'Cadet Preference':

        # Legend
        add_legend = True

        # Convert utility matrix to utility columns
        preferences, utilities_array = get_utility_preferences(p)

        # Counts
        top_3_count = np.zeros(M)
        bottom_3_count = np.zeros(M)

        for i in p["I"]:
            for j in p["J"]:
                afsc = p["afsc_vector"][j]
                if afsc in preferences[i, 0:3]:
                    top_3_count[j] += 1
                elif afsc in preferences[i, 3:6]:
                    bottom_3_count[j] += 1

        # Get the title and filename
        ip["title"] = "Cadet Preferences Placed on Each AFSC (Before Match)"
        ip["filename"] = "1_Cadet_Preference_Choices_Placed_Across_AFSCs"

        # Legend
        if ip["version"] is None:
            legend_elements = [Patch(facecolor=ip['bar_colors']["top_3_choices"], label='Top 3 Choices',
                                     edgecolor='black'),
                               Patch(facecolor=ip['bar_colors']["bottom_3_choices"],
                                     label='Bottom 3 Choices', edgecolor='black')]

            # Bar Chart
            ax.bar(afscs, bottom_3_count, color=ip['bar_colors']["bottom_3_choices"], edgecolor='black')
            ax.bar(afscs, top_3_count, bottom=bottom_3_count, color=ip['bar_colors']["top_3_choices"],
                   edgecolor='black')

            y_axis_count = top_3_count + bottom_3_count
            y_max = np.max(y_axis_count)

        elif ip["version"] == "Top 3":
            legend_elements = [Patch(facecolor=ip['bar_colors']["top_3_choices"], label='Top 3 Choices',
                                     edgecolor='black')]
            ip["title"] = "Cadet Top 3 Preferences Placed on Each AFSC (Before Match)"
            ip["filename"] = "2_Cadet_Preference_Top3_Choices_Placed_Across_AFSCs"

            # Bar Chart
            ax.bar(afscs, top_3_count, color=ip['bar_colors']["top_3_choices"], edgecolor='black')

            y_axis_count = top_3_count
            y_max = np.max(y_axis_count)

        else:
            raise ValueError("Version '" + str(ip["version"]) + "' is not valid for Preference Data graph.")

        # Axis Adjustments
        print(ip["y_max"], y_max, ip["y_max"] * y_max)
        ax.set(ylim=(0, ip["y_max"] * y_max))

        # Get correct text
        y_label = "Number of Cadets"
        if ip["solution_in_title"]:
            ip['title'] = ip['title']

    elif ip['data_graph'] == 'Cadet Preference Analysis':

        # Legend
        add_legend = True

        # Convert utility matrix to utility columns
        preferences, utilities_array = get_utility_preferences(p)

        # Counts
        top_3_count = np.zeros(M)
        eligible_pref_count = np.zeros(M)
        pref_cadets = {}
        cadets = {}
        for j in p["J"]:

            afsc = p["afsc_vector"][j]
            pref_cadets[j] = []  # List of cadets that have this AFSC in their top 3 choices
            cadets[j] = []  # List of cadets that have this AFSC in their top 3 choices and are also eligible for it
            for i in p["I"]:
                if afsc in preferences[i, 0:3]:
                    top_3_count[j] += 1
                    pref_cadets[j].append(i)

                    # If the cadet is eligible for the AFSC, we add them to this list
                    if p["ineligible"][i, j] == 0:
                        cadets[j].append(i)

            # Convert to numpy arrays
            pref_cadets[j] = np.array(pref_cadets[j])
            cadets[j] = np.array(cadets[j])
            eligible_pref_count[j] = len(cadets[j])

        # Used for the y-axis
        y_max = np.max(top_3_count)

        # Get the title and filename
        ip["title"] = "Cadet Degree Eligibility of Preferred Cadets on Each AFSC (Before Match)"
        ip["filename"] = "3_Cadet_Degree_Eligibility_Preference_Cadets_Across_AFSCs"

        # Get metrics
        mandatory_count = np.array([np.sum(p['mandatory'][pref_cadets[j], j]) for j in p["J"]])
        desired_count = np.array([np.sum(p['desired'][pref_cadets[j], j]) for j in p["J"]])
        permitted_count = np.array([np.sum(p['permitted'][pref_cadets[j], j]) for j in p["J"]])
        ineligible_count = np.array([np.sum(p['ineligible'][pref_cadets[j], j]) for j in p["J"]])

        # Legend
        if ip["version"] is None:

            # Legend
            legend_elements = [Patch(facecolor=ip['bar_colors'][objective], label=objective) for
                               objective in ["Ineligible", "Permitted", "Desired", "Mandatory"]]

            # Bar Chart
            ax.bar(afscs, mandatory_count, color=ip['bar_colors']["Mandatory"], edgecolor='black')
            ax.bar(afscs, desired_count, bottom=mandatory_count, color=ip['bar_colors']["Desired"],
                   edgecolor='black')
            ax.bar(afscs, permitted_count, bottom=desired_count + mandatory_count, color=ip['bar_colors']["Permitted"],
                   edgecolor='black')
            ax.bar(afscs, ineligible_count, bottom=permitted_count + desired_count + mandatory_count,
                   color=ip['bar_colors']["Ineligible"], edgecolor='black')

        elif ip["version"] == "AFOCD_Eligible":

            # Legend
            legend_elements = [Patch(facecolor=ip['bar_colors'][objective], label=objective) for
                               objective in ["Permitted", "Desired", "Mandatory"]]

            # Bar Chart
            ax.bar(afscs, mandatory_count, color=ip['bar_colors']["Mandatory"], edgecolor='black')
            ax.bar(afscs, desired_count, bottom=mandatory_count, color=ip['bar_colors']["Desired"],
                   edgecolor='black')
            ax.bar(afscs, permitted_count, bottom=desired_count + mandatory_count, color=ip['bar_colors']["Permitted"],
                   edgecolor='black')

            ip["filename"] = "4_Cadet_Degree_Preference_Cadets_Across_AFSCs"

        elif ip["version"] == "Merit":

            add_legend = False

            for j in p["J"]:
                merit = p["merit"][cadets[j]]
                uq = np.unique(merit)
                count_sum = 0
                for val in uq:
                    count = len(np.where(merit == val)[0])
                    c = str(val)  # Grayscale
                    ax.bar([j], count, bottom=count_sum, color=c, zorder=2)
                    count_sum += count

                # Add the text and an outline
                ax.text(j - 0.3, eligible_pref_count[j] + 2, round(np.mean(merit), 2))
                ax.bar([j], eligible_pref_count[j], color="black", zorder=1, edgecolor="black")

            # DIY Colorbar
            h = (100 / 245) * y_max
            w1 = 0.8
            w2 = 0.74
            vals = np.arange(101) / 100
            current_height = (150 / 245) * y_max
            ax.add_patch(Rectangle((M - 2, current_height), w1, h, edgecolor='black', facecolor='black',
                                   fill=True, lw=2))
            ax.text(M - 3.3, (245 / 245) * y_max, '100%', fontsize=ip["xaxis_tick_size"])
            ax.text(M - 2.8, current_height, '0%', fontsize=ip["xaxis_tick_size"])
            ax.text((M - 0.95), (166 / 245) * y_max, 'Cadet Percentile', fontsize=ip["xaxis_tick_size"],
                    rotation=270)
            for val in vals:
                c = str(val)  # Grayscale
                ax.add_patch(Rectangle((M - 1.95, current_height), w2, h / 101, color=c, fill=True))
                current_height += h / 101

            ip["title"] = "Merit of Preferred Cadets on Each AFSC (Before Match)"
            ip["filename"] = "5_Cadet_Merit_Preference_Cadets_Across_AFSCs"

        elif ip["version"] == "USAFA":

            # Legend
            legend_elements = [Patch(facecolor=ip['bar_colors']["usafa"], label='USAFA'),
                               Patch(facecolor=ip['bar_colors']["rotc"], label='ROTC'),
                               mlines.Line2D([], [], color="red", linestyle='-', linewidth=3, label="Baseline")]

            rotc_baseline = eligible_pref_count - (eligible_pref_count * p["usafa_proportion"])

            usafa_count = np.array([np.sum(p['usafa'][cadets[j]]) for j in p["J"]])
            rotc_count = np.array([eligible_pref_count[j] - usafa_count[j] for j in p["J"]])

            # Bar Chart
            ax.bar(afscs, rotc_count, color=ip['bar_colors']["rotc"], edgecolor='black')
            ax.bar(afscs, usafa_count, bottom=rotc_count, color=ip['bar_colors']["usafa"],
                   edgecolor='black')

            for j in p["J"]:

                # Add the PGL lines
                ax.plot((j - 0.4, j + 0.4), (rotc_baseline[j], rotc_baseline[j]),
                        color="red", linestyle="-", zorder=2, linewidth=3)

            ip["title"] = "USAFA/ROTC Breakdown of Preferred Cadets on Each AFSC (Before Match)"
            ip["filename"] = "6_Cadet_USAFA_Preference_Cadets_Across_AFSCs"

        elif ip["version"] == "Gender":

            # Legend
            legend_elements = [Patch(facecolor=ip['bar_colors']["male"], label='Male'),
                               Patch(facecolor=ip['bar_colors']["female"], label='Female'),
                               mlines.Line2D([], [], color="red", linestyle='-', linewidth=3, label="Baseline")]

            female_baseline = eligible_pref_count - (eligible_pref_count * p["male_proportion"])

            male_count = np.array([np.sum(p['male'][cadets[j]]) for j in p["J"]])
            female_count = np.array([eligible_pref_count[j] - male_count[j] for j in p["J"]])

            # Bar Chart
            ax.bar(afscs, female_count, color=ip['bar_colors']["female"], edgecolor='black')
            ax.bar(afscs, male_count, bottom=female_count, color=ip['bar_colors']["male"],
                   edgecolor='black')

            for j in p["J"]:

                # Add the PGL lines
                ax.plot((j - 0.4, j + 0.4), (female_baseline[j], female_baseline[j]),
                        color="red", linestyle="-", zorder=2, linewidth=3)

            ip["title"] = "Male/Female Breakdown of Preferred Cadets on Each AFSC (Before Match)"
            ip["filename"] = "7_Cadet_Male_Preference_Cadets_Across_AFSCs"

        else:
            raise ValueError("Version '" + str(ip["version"]) + "' is not valid for Preference Analysis Data graph.")

        # Axis Adjustments
        ax.set(ylim=(0, ip["y_max"] * y_max))

        # Get correct text
        y_label = "Number of Cadets"
        if ip["solution_in_title"]:
            ip['title'] = ip['title']

    elif ip['data_graph'] == 'Eligible Quota':

        # Legend
        add_legend = True
        legend_elements = [Patch(facecolor='blue', label='Eligible Cadets', edgecolor='black'),
                           Patch(facecolor='black', alpha=0.5, label='AFSC Quota', edgecolor='black')]

        # Get metrics
        eligible_count = np.array([len(p['I^E'][j]) for j in indices])
        quota = p['quota'][indices]

        # Bar Chart
        ax.bar(afscs, eligible_count, color='blue', edgecolor='black')
        ax.bar(afscs, quota, color='black', edgecolor='black', alpha=0.5)

        # Axis Adjustments
        ax.set(ylim=(0, ip['eligibility_limit'] + ip['eligibility_limit'] / 100))

        # Get correct text
        y_label = "Number of Cadets"
        if ip['title'] is None:
            ip['title'] = 'Eligible Cadets and Quotas for AFSCs with less than ' + \
                          str(ip['eligibility_limit']) + ' Eligible Cadets'

    # Display title
    if ip['display_title']:
        fig.suptitle(ip['title'], fontsize=ip['title_size'])

    # Labels
    ax.set_ylabel(y_label)
    ax.yaxis.label.set_size(ip['label_size'])
    ax.set_xlabel('AFSCs')
    ax.xaxis.label.set_size(ip['label_size'])

    # X axis
    ax.tick_params(axis='x', labelsize=ip['afsc_tick_size'])
    ax.set_xticklabels(afscs[tick_indices], rotation=ip['afsc_rotation'])
    ax.set_xticks(tick_indices)
    ax.set(xlim=(-0.8, M))

    # Y axis
    ax.tick_params(axis='y', labelsize=ip['yaxis_tick_size'])

    # Legend
    if add_legend:
        ax.legend(handles=legend_elements, edgecolor='black', loc="upper right", fontsize=ip['legend_size'], ncol=1,
                  labelspacing=1, handlelength=0.8, handletextpad=0.2, borderpad=0.2, handleheight=2)

    # Fix Colors
    if ip['facecolor'] == 'black':
        ax.set_facecolor('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['right'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        if ip['display_title']:
            fig.suptitle(ip['title'], fontsize=ip['title_size'], color='white')

    if ip["filename"] is None:
        ip["filename"] = ip["title"]

    if ip['save']:
        fig.savefig(paths_out['figures'] + instance.data_name + "/data/" + ip['filename'] + '.png', bbox_inches='tight')

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
        fig.savefig(paths_out['figures'] + title + '.png')
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

        if 'merit_all' in parameters:
            x = parameters['merit_all']
        else:
            x = parameters['merit']
        y = value_parameters['cadet_weight'] / np.max(value_parameters['cadet_weight'])

        # Plot
        ax.scatter(x, y, color='black', alpha=1, linewidth=3)

        # Labels
        ax.set_ylabel('Cadet Weight', fontname='Times New Roman')
        ax.yaxis.label.set_size(label_size)
        ax.set_xlabel('Percentile', fontname='Times New Roman')
        ax.xaxis.label.set_size(label_size)

        # Ticks
        # y_ticks = [0.5, 1]
        # ax.set_yticks(y_ticks)
        # ax.set_yticklabels(y_ticks, fontname='Times New Roman')
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
        fig.savefig(paths_out['figures'] + title + '.png', bbox_inches='tight')

    return fig


# Results
def afsc_results_graph(instance):
    """
    Displays an AFSC-based results chart
    """

    # Shorthand
    ip = instance.plt_p
    p = instance.parameters
    vp = instance.value_parameters

    # Create figure
    fig, ax = plt.subplots(figsize=ip['figsize'], facecolor=ip['facecolor'], tight_layout=True, dpi=ip['dpi'])

    # Initially assume we have a tick mark for every AFSC (unless we're skipping them)
    afscs = p["afsc_vector"]
    M = len(afscs)
    indices = np.arange(M)  # Indices of AFSCs we will plot
    if ip["skip_afscs"]:  # Indices of AFSCs we will label (can skip sometimes)
        tick_indices = np.arange(1, M, 2).astype(int)
    else:
        tick_indices = indices

    # Check if we're comparing solutions or just showing one solution
    if ip["compare_solutions"]:  # Comparing two or more solutions (Dot charts)

        # Shorthand
        m_dict = instance.metrics_dict[ip["vp_name"]]
        vp = instance.vp_dict[ip["vp_name"]]  # In case the set of value parameters doesn't match current selected one

        # We're plotting the value(s) for a particular objective (or overall Value)
        if ip["results_graph"] == "Value":

            # Initialize some chart elements
            y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
            max_value = np.zeros(M)
            min_value = np.repeat(1000, M)
            max_solution = np.zeros(M).astype(int)
            legend_elements = []

            # Loop through each solution
            for s, solution in enumerate(ip["solution_names"]):

                # Determine what kind of objective we're showing
                if ip["objective"] == "Overall":
                    value = m_dict[solution]["afsc_value"]
                    y_label = "Overall Value"
                else:
                    if ip["objective"] not in vp["objectives"]:
                        raise ValueError("Error, objective '" + str(ip["objective"]) + "' is not in list of objectives")
                    else:
                        k = np.where(vp["objectives"] == ip["objective"])[0][0]
                        indices = np.where(vp["objective_weight"][:, k] != 0)[0]
                        value = m_dict[solution]["objective_value"][indices, k]
                        afscs = p["afsc_vector"][indices]
                        M = len(afscs)
                        y_label = ip["objective"] + " Value"
                        if M < p["M"]:
                            tick_indices = np.arange(len(afscs))  # Make sure we're not skipping AFSCs at this point

                # Dot Chart
                ax.scatter(afscs, value, color=ip["colors"][solution], marker=ip["markers"][solution],
                           alpha=ip["alpha"], edgecolor='black', s=ip["dot_size"], zorder=2)

                max_value = np.array([max(max_value[j], value[j]) for j in range(M)])
                min_value = np.array([min(min_value[j], value[j]) for j in range(M)])
                for j in range(M):
                    if max_value[j] == value[j]:
                        max_solution[j] = s
                element = mlines.Line2D([], [], color=ip["colors"][solution], marker=ip["markers"][solution],
                                        linestyle='None', markeredgecolor='black', markersize=ip["marker_size"],
                                        label=solution, alpha=ip["alpha"])
                legend_elements.append(element)

            # Add dot lines
            for j in range(M):
                ax.plot((j, j), (0, max_value[j]), color='black', linestyle='--', zorder=1, alpha=0.4, linewidth=2)

            ax.legend(handles=legend_elements, edgecolor='black', fontsize=ip["legend_size"], loc='upper left',
                      ncol=ip["num_solutions"], columnspacing=0.5, handletextpad=0.05, borderaxespad=0.5, borderpad=0.2)

            if ip["title"] is None:

                # Create the title!
                if ip["num_solutions"] == 1:
                    ip['title'] = ip["solution_names"][0] + " " + ip["objective"] + " AFSC Value Chart"
                elif ip["num_solutions"] == 2:
                    ip['title'] = ip["solution_names"][0] + " and " + ip["solution_names"][1] + \
                                  ip["objective"] + " AFSC Value Chart"
                elif ip["num_solutions"] == 3:
                    ip['title'] = ip["solution_names"][0] + ", " + ip["solution_names"][1] + \
                                  ", and " + ip["solution_names"][2] + ip["objective"] + " AFSC Value Chart"
                else:
                    ip['title'] = ip["solution_names"][0] + ", " + ip["solution_names"][1] + \
                                  ", " + ip["solution_names"][2] + ", and " + ip["solution_names"][3] + \
                                  " " + ip["objective"] + " AFSC Value Chart"

        # We're plotting the measure(s) for a particular objective
        else:
            label_dict = {"Merit": "Average Merit", "USAFA Proportion": "USAFA Proportion",
                          "Combined Quota": "Percent of PGL Target Met", "USAFA Quota": "Number of USAFA Cadets",
                          "ROTC Quota": "Number of ROTC Cadets", "Mandatory": "Mandatory Degree Tier Proportion",
                          "Desired": "Desired Degree Tier Proportion", "Permitted":
                              "Permitted Degree Tier Proportion", "Male": "Proportion of Male Cadets",
                          "Minority": "Proportion of Non-Caucasian Cadets", "Utility": "Average Utility"}

            # Get the correct objective elements
            k = np.where(vp["objectives"] == ip["objective"])[0][0]
            y_label = label_dict[ip["objective"]]
            if ip["objective"] in ["Merit", "USAFA Proportion", "Male", "Minority"] and ip["all_afscs"]:
                indices = np.arange(M)
            else:
                indices = np.where(vp["objective_weight"][:, k] != 0)[0]
                afscs = p["afsc_vector"][indices]
                M = len(afscs)  # Number of AFSCs being plotted
                if M < p["M"]:
                    tick_indices = np.arange(len(afscs))  # Make sure we're not skipping AFSCs at this point

            # Make sure at least one AFSC has this objective selected
            if len(afscs) == 0:
                raise ValueError("Error. No AFSCs have objective '" + ip["objective"] + "'.")

            # Shared elements
            legend_elements = []
            max_measure = np.zeros(M)
            if ip["title"] is None:

                # Create the title!
                if ip["num_solutions"] == 1:
                    ip['title'] = ip["solution_names"][0] + " " + label_dict[ip["objective"]] + " Across Each AFSC"
                elif ip["num_solutions"] == 2:
                    ip['title'] = ip["solution_names"][0] + " and " + ip["solution_names"][1] + \
                                  " " + label_dict[ip["objective"]] + " Across Each AFSC"
                elif ip["num_solutions"] == 3:
                    ip['title'] = ip["solution_names"][0] + ", " + ip["solution_names"][1] + \
                                  ", and " + ip["solution_names"][2] + " " + \
                                  label_dict[ip["objective"]] + " Across Each AFSC"
                else:
                    ip['title'] = ip["solution_names"][0] + ", " + ip["solution_names"][1] + \
                                  ", " + ip["solution_names"][2] + ", and " + ip["solution_names"][3] + \
                                  " " + label_dict[ip["objective"]] + " Across Each AFSC"

            # Matters for Quota and AFOCD objectives
            x_under = []
            x_over = []
            quota_percent_filled = np.zeros(M)
            max_quota_percent = np.zeros(M)
            y_top = 0

            # Loop through each solution
            for s, solution in enumerate(ip["solution_names"]):

                # Calculate objective measure
                measure = m_dict[solution]["objective_measure"][indices, k]
                if ip["objective"] == "Combined Quota":

                    # Assign the right color to the AFSCs
                    for j in range(M):

                        # Get bounds
                        value_list = vp['objective_value_min'][j, k].split(",")
                        ub = float(value_list[1].strip())  # upper bound
                        if "pgl" in p:
                            quota = p["pgl"][j]
                        else:
                            quota = p['quota'][j]

                        # Constraint violations
                        if quota > measure[j]:
                            x_under.append(j)
                        elif measure[j] > ub:
                            x_over.append(j)

                        quota_percent_filled[j] = measure[j] / quota
                        max_quota_percent[j] = ub / quota

                    # Top dot location
                    y_top = max(y_top, max(quota_percent_filled))

                    # Plot the points
                    ax.scatter(afscs, quota_percent_filled, color=ip["colors"][solution],
                               marker=ip["markers"][solution], alpha=ip["alpha"], edgecolor='black',
                               s=ip["dot_size"], zorder=ip["zorder"][solution])

                else:

                    # Plot the points
                    ax.scatter(afscs, measure, color=ip["colors"][solution], marker=ip["markers"][solution],
                               alpha=ip["alpha"], edgecolor='black', s=ip["dot_size"],
                               zorder=ip["zorder"][solution])

                max_measure = np.array([max(max_measure[j], measure[j]) for j in range(M)])
                element = mlines.Line2D([], [], color=ip["colors"][solution], marker=ip["markers"][solution],
                                        linestyle='None', markeredgecolor='black', markersize=15, label=solution,
                                        alpha=ip["alpha"])
                legend_elements.append(element)

            # Lines to the top solution's dot
            if ip["objective"] not in ["Combined Quota", "Mandatory", "Desired", "Permitted"]:
                for j in range(M):
                    ax.plot((j, j), (0, max_measure[j]), color='black', linestyle='--', zorder=1, alpha=0.5,
                            linewidth=2)

            # Objective Specific elements
            if ip["objective"] == "Merit":

                # Tick marks and extra lines
                y_ticks = [0, 0.35, 0.50, 0.65, 0.80, 1]
                ax.plot((-1, 50), (0.65, 0.65), color='black', linestyle='-', zorder=1, alpha=1, linewidth=1.5)
                ax.plot((-1, 50), (0.5, 0.5), color='black', linestyle='--', zorder=1, alpha=1, linewidth=1.5)
                ax.plot((-1, 50), (0.35, 0.35), color='black', linestyle='-', zorder=1, alpha=1, linewidth=1.5)

                # Set the max for the y-axis
                ip["y_max"] = ip["y_max"] * np.max(max_measure)

            elif ip["objective"] in ["USAFA Proportion", "Male", "Minority"]:

                # Demographic Proportion elements
                prop_dict = {"USAFA Proportion": "usafa_proportion", "Male": "male_proportion",
                             "Minority": "minority_proportion"}
                up_lb = round(p[prop_dict[ip["objective"]]] - 0.15, 2)
                up_ub = round(p[prop_dict[ip["objective"]]] + 0.15, 2)
                up = round(p[prop_dict[ip["objective"]]], 2)
                y_ticks = [0, up_lb, up, up_ub, 1]

                # Add lines for the ranges
                ax.axhline(y=up, color='black', linestyle='--', alpha=0.5)
                ax.axhline(y=up_lb, color='blue', linestyle='-', alpha=0.5)
                ax.axhline(y=up_ub, color='blue', linestyle='-', alpha=0.5)

                # Set the max for the y-axis
                ip["y_max"] = ip["y_max"] * np.max(max_measure)

            elif ip["objective"] in ["Mandatory", "Desired", "Permitted"]:

                # Degree Tier elements
                y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
                minimums = np.zeros(M)
                maximums = np.zeros(M)

                # Assign the right color to the AFSCs
                for j, loc in enumerate(indices):
                    if "Increasing" in vp["value_functions"][loc, k]:
                        minimums[j] = vp['objective_target'][loc, k]
                        maximums[j] = 1
                    else:
                        minimums[j] = 0
                        maximums[j] = vp['objective_target'][loc, k]

                # Calculate ranges
                y = [(minimums[j], maximums[j]) for j in range(M)]
                y_lines = [(0, minimums[j]) for j in range(M)]

                # Plot bounds
                ax.scatter(range(M), minimums, c="black", marker="_", linewidth=2, zorder=1)
                ax.scatter(range(M), maximums, c="black", marker="_", linewidth=2, zorder=1)

                # Constraint Range
                ax.plot((range(M), range(M)), ([i for (i, j) in y], [j for (i, j) in y]),
                        c="black", zorder=1)

                # Line from x-axis to constraint range
                ax.plot((range(M), range(M)), ([i for (i, j) in y_lines], [j for (i, j) in y_lines]),
                        c="black", zorder=1, alpha=0.5, linestyle='--', linewidth=2)

            elif ip["objective"] == "Combined Quota":

                # Y axis adjustments
                y_ticks = [0, 0.5, 1, 1.5, 2]
                ip["y_max"] = ip["y_max"] * y_top

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
                ax.plot((x_over, x_over), ([i for (i, j) in y_over], [j for (i, j) in y_over]), c='black',
                        linestyle='--',
                        zorder=1)
                ax.plot((indices, indices), (np.zeros(M), np.ones(M)), c='black', linestyle='--', alpha=0.3,
                        zorder=1)

                # Quota Line
                ax.axhline(y=1, color='black', linestyle='-', alpha=0.5, zorder=1)

            elif ip["objective"] == "Utility":

                # Tick marks and extra lines
                y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]

            # Create a legend
            if legend_elements is not None:
                ax.legend(handles=legend_elements, edgecolor='black', fontsize=ip["legend_size"],
                          ncol=2, columnspacing=0.5, handletextpad=0.05, borderaxespad=0.5, borderpad=0.2)

    else:  # Viewing one solution (Can be bar charts)

        # We're plotting the value for a particular objective (or overall Value)
        if ip["results_graph"] == "Value":

            # Initialize some chart specifics
            y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]

            # Determine what kind of objective we're showing
            if ip["objective"] == "Overall":
                value = instance.metrics["afsc_value"]
                y_label = "Overall Value"
            else:
                if ip["objective"] not in vp["objectives"]:
                    raise ValueError("Error, objective '" + str(ip["objective"]) + "' is not in list of objectives")
                else:
                    k = np.where(vp["objectives"] == ip["objective"])[0][0]
                    indices = np.where(vp["objective_weight"][:, k] != 0)[0]
                    value = instance.metrics["objective_value"][indices, k]
                    afscs = p["afsc_vector"][indices]
                    y_label = ip["objective"] + " Value"
                    if len(afscs) < p["M"]:
                        tick_indices = np.arange(len(afscs))  # Make sure we're not skipping AFSCs at this point

            if ip["title"] is None:
                ip['title'] = instance.solution_name + " " + ip["objective"] + " AFSC Value Chart"

            # Bar Chart
            ax.bar(afscs, value, color=ip["bar_color"], alpha=ip["alpha"], edgecolor='black')

        # We're plotting the measure for a particular objective
        else:

            legend_elements = None  # Assume there is no legend until there is one
            if ip["objective"] != "Extra":

                label_dict = {"Merit": "Average Merit", "USAFA Proportion": "USAFA Proportion",
                              "Combined Quota": "Percent of PGL Target Met", "USAFA Quota": "Number of USAFA Cadets",
                              "ROTC Quota": "Number of ROTC Cadets", "Mandatory": "Mandatory Degree Tier Proportion",
                              "Desired": "Desired Degree Tier Proportion", "Permitted":
                                  "Permitted Degree Tier Proportion", "Male": "Proportion of Male Cadets",
                              "Minority": "Proportion of Non-Caucasian Cadets", "Utility": "Average Utility"}

                # Get the correct objective elements
                k = np.where(vp["objectives"] == ip["objective"])[0][0]
                y_label = label_dict[ip["objective"]]

                if ip["objective"] in ["Merit", "USAFA Proportion", "Male", "Minority"] and ip["all_afscs"]:
                    measure = instance.metrics["objective_measure"][:, k]
                else:
                    indices = np.where(vp["objective_weight"][:, k] != 0)[0]
                    measure = instance.metrics["objective_measure"][indices, k]
                    afscs = p["afsc_vector"][indices]
                    M = len(afscs)  # Number of AFSCs being plotted
                    if M < p["M"]:
                        tick_indices = np.arange(len(afscs))  # Make sure we're not skipping AFSCs at this point

                # Make sure at least one AFSC has this objective selected
                if len(afscs) == 0:
                    raise ValueError("Error. No AFSCs have objective '" + ip["objective"] + "'.")

                # Get the correct quota
                if "pgl" in p:
                    quota = p["pgl"][indices]
                else:
                    quota = p["quota"][indices]

                # Shared elements
                colors = np.array([" " * 10 for _ in range(len(afscs))])

                # Objective specific elements
                if ip["objective"] == "Merit":

                    if ip["version"] == "large_only_bar":

                        # Get the title and filename
                        ip["title"] = "Average Merit Across Each Large AFSC"
                        ip["filename"] = instance.solution_name + " Merit_Average_Large_AFSCs"
                        if ip["solution_in_title"]:
                            ip['title'] = instance.solution_name + ": " + ip['title']

                        # Set the max for the y-axis
                        ip["y_max"] = ip["y_max"] * np.max(measure)

                        # Merit elements
                        y_ticks = [0, 0.35, 0.50, 0.65, 0.80, 1]
                        legend_elements = [Patch(facecolor=ip['bar_colors']["merit_below"], label='Merit < 0.35'),
                                           Patch(facecolor=ip['bar_colors']["merit_within"],
                                                 label='0.35 < Merit < 0.65'),
                                           Patch(facecolor=ip['bar_colors']["merit_above"], label='Merit > 0.65')]

                        # Assign the right color to the AFSCs
                        for j in range(len(afscs)):
                            if 0.35 <= measure[j] <= 0.65:
                                colors[j] = ip['bar_colors']["merit_within"]
                            elif measure[j] > 0.65:
                                colors[j] = ip['bar_colors']["merit_above"]
                            else:
                                colors[j] = ip['bar_colors']["merit_below"]

                        # Add lines for the ranges
                        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
                        ax.axhline(y=0.35, color='blue', linestyle='-', alpha=0.5)
                        ax.axhline(y=0.65, color='blue', linestyle='-', alpha=0.5)

                        # Bar Chart
                        ax.bar(afscs, measure, color=colors, edgecolor='black', alpha=ip["alpha"])

                    elif ip["version"] == "bar":

                        # Get the title and filename
                        ip["title"] = "Average Merit Across Each AFSC"
                        ip["filename"] = instance.solution_name + " Merit_Average_All_AFSCs"
                        if ip["solution_in_title"]:
                            ip['title'] = instance.solution_name + ": " + ip['title']

                        # Set the max for the y-axis
                        ip["y_max"] = ip["y_max"] * np.max(measure)

                        # Merit elements
                        y_ticks = [0, 0.35, 0.50, 0.65, 0.80, 1]
                        legend_elements = [Patch(facecolor=ip['bar_colors']["small_afscs"], label='Small AFSC'),
                                           Patch(facecolor=ip['bar_colors']["large_afscs"], label='Large AFSC'),
                                           mlines.Line2D([], [], color="blue", linestyle='-', label="Bound")]

                        # Assign the right color to the AFSCs
                        for j in range(len(afscs)):
                            if quota[j] >= 40:
                                colors[j] = ip['bar_colors']["large_afscs"]
                            else:
                                colors[j] = ip['bar_colors']["small_afscs"]

                        # Add lines for the ranges
                        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
                        ax.axhline(y=0.35, color='blue', linestyle='-', alpha=0.5)
                        ax.axhline(y=0.65, color='blue', linestyle='-', alpha=0.5)

                        # Bar Chart
                        ax.bar(afscs, measure, color=colors, edgecolor='black', alpha=ip["alpha"])

                    elif ip["version"] == "quantity_bar_gradient":

                        # Get the title and filename
                        ip["title"] = "Cadet Merit Breakdown Across Each AFSC"
                        ip["filename"] = instance.solution_name + " Merit_PGL_Sorted_Gradient"
                        if ip["solution_in_title"]:
                            ip['title'] = instance.solution_name + ": " + ip['title']
                        label_dict["Merit"] = "Number of Cadets"
                        y_label = label_dict[ip["objective"]]

                        # Plot the number of cadets and the PGL
                        if "pgl" in p:
                            quota = p["pgl"][indices]
                        else:
                            quota = p["quota"][indices]

                        # Need to know number of cadets assigned
                        quota_k = np.where(vp["objectives"] == "Combined Quota")[0][0]
                        total_count = instance.metrics["objective_measure"][:, quota_k]

                        # Sort the AFSCs by the PGL
                        if ip["sort_by_pgl"]:
                            indices = np.argsort(quota)[::-1]

                        # Sort the AFSCs by the number of cadets assigned
                        else:
                            indices = np.argsort(total_count)[::-1]

                        afscs = afscs[indices]
                        measure = measure[indices]
                        total_count = total_count[indices]

                        # Y max
                        y_max = max(total_count)
                        ip["y_max"] = ip["y_max"] * y_max
                        if 100 <= y_max < 150:
                            y_ticks = [50, 100, 150]
                        elif 150 <= y_max < 200:
                            y_ticks = [50, 100, 150, 200]
                        elif 200 <= y_max < 250:
                            y_ticks = [50, 100, 150, 200, 250]
                        elif 250 <= y_max < 300:
                            y_ticks = [50, 100, 150, 200, 250, 300]
                        elif y_max >= 300:
                            y_ticks = [50, 100, 150, 200, 250, 300, 350]
                        else:
                            y_ticks = [50]

                        for index, afsc in enumerate(afscs):
                            j = np.where(p["afsc_vector"] == afsc)[0][0]
                            cadets = np.where(instance.solution == j)[0]
                            merit = p["merit"][cadets]
                            uq = np.unique(merit)
                            count_sum = 0
                            for val in uq:
                                count = len(np.where(merit == val)[0])
                                # c = (1 - val, 0, val)  # Blue to Red
                                c = str(val)  # Grayscale
                                ax.bar([index], count, bottom=count_sum, color=c, zorder=2)
                                count_sum += count

                            # Add the text and an outline
                            ax.text(index - 0.3, total_count[index] + 2, round(measure[index], 2))
                            ax.bar([index], total_count[index], color="black", zorder=1, edgecolor="black")

                        # DIY Colorbar
                        h = (100 / 245) * y_max
                        w1 = 0.8
                        w2 = 0.74
                        vals = np.arange(101) / 100
                        current_height = (150 / 245) * y_max
                        ax.add_patch(Rectangle((M - 2, current_height), w1, h, edgecolor='black', facecolor='black',
                                               fill=True, lw=2))
                        ax.text(M - 3.3, (245 / 245) * y_max, '100%', fontsize=ip["xaxis_tick_size"])
                        ax.text(M - 2.8, current_height, '0%', fontsize=ip["xaxis_tick_size"])
                        ax.text((M - 0.95), (166 / 245) * y_max, 'Cadet Percentile', fontsize=ip["xaxis_tick_size"],
                                rotation=270)
                        for val in vals:
                            # c = (1 - val, 0, val)  # Blue to Red
                            c = str(val)  # Grayscale
                            ax.add_patch(Rectangle((M - 1.95, current_height), w2, h / 101, color=c, fill=True))
                            current_height += h / 101

                    else:  # quantity_bar_proportion  (Quartiles in this case)

                        # Get the title and filename
                        ip["title"] = "Cadet Quartiles Across Each AFSC"
                        ip["filename"] = instance.solution_name + " Merit_PGL_Sorted_Quartile"
                        if ip["solution_in_title"]:
                            ip['title'] = instance.solution_name + ": " + ip['title']
                        label_dict["Merit"] = "Number of Cadets"
                        y_label = label_dict[ip["objective"]]

                        # Legend
                        legend_elements = [Patch(facecolor=ip['bar_colors']["quartile_1"], label='1st Quartile'),
                                           Patch(facecolor=ip['bar_colors']["quartile_2"], label='2nd Quartile'),
                                           Patch(facecolor=ip['bar_colors']["quartile_3"], label='3rd Quartile'),
                                           Patch(facecolor=ip['bar_colors']["quartile_4"], label='4th Quartile')]

                        # Plot the number of cadets and the PGL
                        if "pgl" in p:
                            quota = p["pgl"][indices]
                        else:
                            quota = p["quota"][indices]

                        # Need to know number of cadets assigned
                        quota_k = np.where(vp["objectives"] == "Combined Quota")[0][0]
                        total_count = instance.metrics["objective_measure"][:, quota_k]

                        # Sort the AFSCs by the PGL
                        if ip["sort_by_pgl"]:
                            indices = np.argsort(quota)[::-1]

                        # Sort the AFSCs by the number of cadets assigned
                        else:
                            indices = np.argsort(total_count)[::-1]

                        afscs = afscs[indices]
                        total_count = total_count[indices]

                        # Y max
                        y_max = max(total_count)
                        ip["y_max"] = ip["y_max"] * y_max
                        if 100 <= y_max < 150:
                            y_ticks = [50, 100, 150]
                        elif 150 <= y_max < 200:
                            y_ticks = [50, 100, 150, 200]
                        elif 200 <= y_max < 250:
                            y_ticks = [50, 100, 150, 200, 250]
                        elif 250 <= y_max < 300:
                            y_ticks = [50, 100, 150, 200, 250, 300]
                        elif y_max >= 300:
                            y_ticks = [50, 100, 150, 200, 250, 300, 350]
                        else:
                            y_ticks = [50]

                        percentile_dict = {1: (0.75, 1), 2: (0.5, 0.75), 3: (0.25, 0.5), 4: (0, 0.25)}
                        for index, afsc in enumerate(afscs):
                            j = np.where(p["afsc_vector"] == afsc)[0][0]
                            cadets = np.where(instance.solution == j)[0]
                            merit = p["merit"][cadets]

                            # Loop through each quartile
                            count_sum = 0
                            for q in [4, 3, 2, 1]:
                                lb, ub = percentile_dict[q][0], percentile_dict[q][1]
                                count = len(np.where((merit <= ub) & (merit > lb))[0])
                                ax.bar([index], count, bottom=count_sum,
                                       color=ip['bar_colors']["quartile_" + str(q)], zorder=2)
                                count_sum += count

                                # Put a number on the bar
                                if count >= 10:
                                    if q == 1:
                                        c = "white"
                                    else:
                                        c = "black"
                                    ax.text(index - 0.2, (count_sum - count / 2 - 2), int(count), color=c,
                                            zorder=3, fontsize=ip["bar_text_size"])

                            # Add the text and an outline
                            ax.text(index - 0.3, total_count[index] + 2, int(total_count[index]),
                                    fontsize=ip["bar_text_size"])
                            ax.bar([index], total_count[index], color="black", zorder=1, edgecolor="black")

                elif ip["objective"] in ["USAFA Proportion", "Male", "Minority"]:

                    # Demographic Proportion elements
                    prop_dict = {"USAFA Proportion": "usafa_proportion", "Male": "male_proportion",
                                 "Minority": "minority_proportion"}
                    legend_dict = {"USAFA Proportion": "USAFA Proportion", "Male": "Male Proportion",
                                   "Minority": "Minority Proportion"}
                    up_lb = round(p[prop_dict[ip["objective"]]] - 0.15, 2)
                    up_ub = round(p[prop_dict[ip["objective"]]] + 0.15, 2)
                    up = round(p[prop_dict[ip["objective"]]], 2)

                    if ip["version"] == "large_only_bar":
                        y_ticks = [0, up_lb, up, up_ub, 1]
                        legend_elements = [Patch(facecolor=ip['bar_colors']["large_within"],
                                                 label=str(up_lb) + ' < ' + legend_dict[
                            ip["objective"]] + ' < ' + str(up_ub)),
                                           Patch(facecolor=ip['bar_colors']["large_else"], label='Otherwise')]

                        # Get the title and filename
                        ip["title"] = legend_dict[ip["objective"]] + " Across Large AFSCs"
                        split_words = legend_dict[ip["objective"]].split(" ")
                        ip["filename"] = instance.solution_name + " " + split_words[0] + "_" + \
                                         split_words[1] + "_Large_AFSCs"
                        if ip["solution_in_title"]:
                            ip['title'] = instance.solution_name + ": " + ip['title']

                        # Set the max for the y-axis
                        ip["y_max"] = ip["y_max"] * np.max(measure)

                        # Assign the right color to the AFSCs
                        for j in range(len(afscs)):
                            if up_lb <= measure[j] <= up_ub:
                                colors[j] = ip['bar_colors']["large_within"]
                            else:
                                colors[j] = ip['bar_colors']["large_else"]

                        # Add lines for the ranges
                        ax.axhline(y=up, color='black', linestyle='--', alpha=0.5)
                        ax.axhline(y=up_lb, color='blue', linestyle='-', alpha=0.5)
                        ax.axhline(y=up_ub, color='blue', linestyle='-', alpha=0.5)

                        # Bar Chart
                        ax.bar(afscs, measure, color=colors, edgecolor='black', alpha=ip["alpha"])

                    elif ip["version"] == "bar":

                        # Get the title and filename
                        ip["title"] = legend_dict[ip["objective"]] + " Across Each AFSC"
                        split_words = legend_dict[ip["objective"]].split(" ")
                        ip["filename"] = instance.solution_name + " " + split_words[0] + "_" + \
                                         split_words[1] + "_All_AFSCs"
                        if ip["solution_in_title"]:
                            ip['title'] = instance.solution_name + ": " + ip['title']

                        # Set the max for the y-axis
                        ip["y_max"] = ip["y_max"] * np.max(measure)

                        # Merit elements
                        y_ticks = [0, up_lb, up, up_ub, 1]
                        legend_elements = [Patch(facecolor=ip['bar_colors']["small_afscs"], label='Small AFSC'),
                                           Patch(facecolor=ip['bar_colors']["large_afscs"], label='Large AFSC'),
                                           mlines.Line2D([], [], color="blue", linestyle='-', label="Bound")]

                        # Assign the right color to the AFSCs
                        for j in range(len(afscs)):
                            if quota[j] >= 40:
                                colors[j] = ip['bar_colors']["large_afscs"]
                            else:
                                colors[j] = ip['bar_colors']["small_afscs"]

                        # Add lines for the ranges
                        ax.axhline(y=up, color='black', linestyle='--', alpha=0.5)
                        ax.axhline(y=up_lb, color='blue', linestyle='-', alpha=0.5)
                        ax.axhline(y=up_ub, color='blue', linestyle='-', alpha=0.5)

                        # Bar Chart
                        ax.bar(afscs, measure, color=colors, edgecolor='black', alpha=ip["alpha"])

                    else:  # Sorted Sized Bar Chart

                        # Get the title and filename
                        if ip["objective"] == "USAFA Proportion":
                            ip["title"] = "Source of Commissioning Breakdown Across Each AFSC"
                            ip["filename"] = instance.solution_name + " SOC_Sorted_Proportion"
                            label_dict["USAFA Proportion"] = "Number of Cadets"

                            class_1_color = ip['bar_colors']["usafa"]
                            class_2_color = ip['bar_colors']["rotc"]

                            # Legend
                            legend_elements = [Patch(facecolor=class_1_color, label='USAFA'),
                                               Patch(facecolor=class_2_color, label='ROTC')]
                        else:
                            ip["title"] = "Gender Breakdown Across Each AFSC"
                            label_dict["Male"] = "Number of Cadets"
                            ip["filename"] = instance.solution_name + " Gender_Sorted_Proportion"

                            class_1_color = ip['bar_colors']["male"]
                            class_2_color = ip['bar_colors']["female"]

                            # Legend
                            legend_elements = [Patch(facecolor=class_1_color, label='Male'),
                                               Patch(facecolor=class_2_color, label='Female')]

                        if ip["solution_in_title"]:
                            ip['title'] = instance.solution_name + ": " + ip['title']
                        y_label = label_dict[ip["objective"]]

                        # Plot the number of cadets and the PGL
                        if "pgl" in p:
                            quota = p["pgl"][indices]
                        else:
                            quota = p["quota"][indices]

                        # Need to know number of cadets assigned
                        quota_k = np.where(vp["objectives"] == "Combined Quota")[0][0]
                        total_count = instance.metrics["objective_measure"][:, quota_k]

                        # Sort the AFSCs by the PGL
                        if ip["sort_by_pgl"]:
                            indices = np.argsort(quota)[::-1]

                        # Sort the AFSCs by the number of cadets assigned
                        else:
                            indices = np.argsort(total_count)[::-1]

                        afscs = afscs[indices]
                        total_count = total_count[indices]
                        measure = measure[indices]

                        # Y max
                        y_max = max(total_count)
                        ip["y_max"] = ip["y_max"] * y_max
                        if 100 <= y_max < 150:
                            y_ticks = [50, 100, 150]
                        elif 150 <= y_max < 200:
                            y_ticks = [50, 100, 150, 200]
                        elif 200 <= y_max < 250:
                            y_ticks = [50, 100, 150, 200, 250]
                        elif 250 <= y_max < 300:
                            y_ticks = [50, 100, 150, 200, 250, 300]
                        elif y_max >= 300:
                            y_ticks = [50, 100, 150, 200, 250, 300, 350]
                        else:
                            y_ticks = [50]

                        class_1 = measure * total_count
                        class_2 = (1 - measure) * total_count
                        ax.bar(afscs, class_2, color=class_2_color, zorder=2)
                        ax.bar(afscs, class_1, bottom=class_2, color=class_1_color, zorder=2)
                        for j, afsc in enumerate(afscs):
                            if class_2[j] >= 10:
                                ax.text(j - 0.2, class_2[j] / 2, int(class_2[j]), color="black",
                                        zorder=3, fontsize=ip["bar_text_size"])
                            if class_1[j] >= 10:
                                ax.text(j - 0.2, class_2[j] + class_1[j] / 2, int(class_1[j]),
                                        color="white", zorder=3, fontsize=ip["bar_text_size"])

                            # Add the text and an outline
                            ax.text(j - 0.3, total_count[j] + 2, round(measure[j], 2),
                                    fontsize=ip["bar_text_size"])
                            ax.bar([j], total_count[j], color="black", zorder=1, edgecolor="black")

                elif ip["objective"] in ["Mandatory", "Desired", "Permitted"]:

                    # Get the title and filename
                    ip["title"] = ip["objective"] + " Proportion Across Each AFSC"
                    ip["filename"] = instance.solution_name + " " + ip["objective"] + "_Proportion_AFSCs"
                    if ip["solution_in_title"]:
                        ip['title'] = instance.solution_name + ": " + ip['title']

                    # Degree Tier elements
                    y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
                    minimums = np.zeros(M)
                    maximums = np.zeros(M)
                    x_under = []
                    x_over = []
                    x_within = []

                    # Assign the right color to the AFSCs
                    for j, loc in enumerate(indices):
                        if "Increasing" in vp["value_functions"][loc, k]:
                            minimums[j] = vp['objective_target'][loc, k]
                            maximums[j] = 1
                        else:
                            minimums[j] = 0
                            maximums[j] = vp['objective_target'][loc, k]

                        if minimums[j] <= measure[j] <= maximums[j]:
                            colors[j] = "blue"
                            x_within.append(j)
                        else:
                            colors[j] = "red"
                            if measure[j] < minimums[j]:
                                x_under.append(j)
                            else:
                                x_over.append(j)

                    # Plot points
                    ax.scatter(afscs, measure, c=colors, linewidths=4, s=ip["dot_size"], zorder=3)

                    # Calculate ranges
                    y_within = [(minimums[j], maximums[j]) for j in x_within]
                    y_under_ranges = [(minimums[j], maximums[j]) for j in x_under]
                    y_over_ranges = [(minimums[j], maximums[j]) for j in x_over]
                    y_under = [(measure[j], minimums[j]) for j in x_under]
                    y_over = [(maximums[j], measure[j]) for j in x_over]

                    # Plot bounds
                    ax.scatter(afscs, minimums, c="black", marker="_", linewidth=2)
                    ax.scatter(afscs, maximums, c="black", marker="_", linewidth=2)

                    # Plot Ranges
                    ax.plot((x_within, x_within), ([i for (i, j) in y_within], [j for (i, j) in y_within]),
                            c="black")
                    ax.plot((x_under, x_under),
                            ([i for (i, j) in y_under_ranges], [j for (i, j) in y_under_ranges]), c="black")
                    ax.plot((x_over, x_over),
                            ([i for (i, j) in y_over_ranges], [j for (i, j) in y_over_ranges]), c="black")

                    # How far off
                    ax.plot((x_under, x_under), ([i for (i, j) in y_under], [j for (i, j) in y_under]), c='red',
                            linestyle='--')
                    ax.plot((x_over, x_over), ([i for (i, j) in y_over], [j for (i, j) in y_over]), c='red',
                            linestyle='--')

                elif ip["objective"] == "Combined Quota":

                    if ip["version"] == "dot":

                        # Get the title and filename
                        ip["title"] = "Percent of PGL Target Met Across Each AFSC"
                        ip["filename"] = instance.solution_name + " Quota_PGL_Percent"
                        if ip["solution_in_title"]:
                            ip['title'] = instance.solution_name + ": " + ip['title']

                        # Degree Tier elements
                        y_ticks = [0, 0.5, 1, 1.5, 2]
                        x_under = []
                        x_over = []
                        x_within = []
                        quota_percent_filled = np.zeros(M)
                        max_quota_percent = np.zeros(M)

                        # Assign the right color to the AFSCs
                        for j in range(M):

                            # Get bounds
                            value_list = vp['objective_value_min'][j, k].split(",")

                            if "pgl" in p:
                                quota = p["pgl"][j]
                            else:
                                quota = p['quota'][j]
                            max_measure = float(value_list[1].strip())
                            if quota > measure[j]:
                                colors[j] = "red"
                                x_under.append(j)
                            elif quota <= measure[j] <= max_measure:
                                colors[j] = "blue"
                                x_within.append(j)
                            else:
                                colors[j] = "orange"
                                x_over.append(j)

                            quota_percent_filled[j] = measure[j] / quota
                            max_quota_percent[j] = max_measure / quota

                        # Plot points
                        ax.scatter(afscs, quota_percent_filled, c=colors, linewidths=4, s=ip["dot_size"], zorder=3)

                        # Set the max for the y-axis
                        ip["y_max"] = ip["y_max"] * np.max(quota_percent_filled)

                        # Lines
                        y_mins = np.repeat(1, M)
                        y_maxs = max_quota_percent
                        y = [(y_mins[j], y_maxs[j]) for j in range(M)]
                        y_under = [(quota_percent_filled[j], 1) for j in x_under]
                        y_over = [(max_quota_percent[j], quota_percent_filled[j]) for j in x_over]

                        # Plot Bounds
                        ax.scatter(afscs, y_mins, c=np.repeat('blue', M), marker="_", linewidth=2)
                        ax.scatter(afscs, y_maxs, c=np.repeat('blue', M), marker="_", linewidth=2)

                        # Plot Range Lines
                        ax.plot((indices, indices), ([i for (i, j) in y], [j for (i, j) in y]), c='blue')
                        ax.plot((x_under, x_under), ([i for (i, j) in y_under], [j for (i, j) in y_under]), c='red',
                                linestyle='--')
                        ax.plot((x_over, x_over), ([i for (i, j) in y_over], [j for (i, j) in y_over]), c='orange',
                                linestyle='--')
                        ax.plot((indices, indices), (np.zeros(M), np.ones(M)), c='black', linestyle='--', alpha=0.3)

                        # Quota Line
                        ax.axhline(y=1, color='black', linestyle='-', alpha=0.5)

                        # Put quota text
                        y_top = round(max(quota_percent_filled))
                        y_spacing = (y_top / 80)
                        for j in indices:
                            if int(measure[j]) >= 100:
                                ax.text(j, quota_percent_filled[j] + 1.4 * y_spacing, int(measure[j]),
                                        fontsize=ip["xaxis_tick_size"],
                                        multialignment='right')
                            elif int(measure[j]) >= 10:
                                ax.text(j + 0.2, quota_percent_filled[j] + y_spacing, int(measure[j]),
                                        fontsize=ip["xaxis_tick_size"],
                                        multialignment='right')
                            else:
                                ax.text(j + 0.2, quota_percent_filled[j] + y_spacing, int(measure[j]),
                                        fontsize=ip["xaxis_tick_size"],
                                        multialignment='right')

                    else:  # quantity_bar

                        # Get the title and filename
                        ip["title"] = "Number of Cadets Assigned to Each AFSC against PGL"
                        ip["filename"] = instance.solution_name + " Quota_PGL_Sorted_Size"
                        if ip["solution_in_title"]:
                            ip['title'] = instance.solution_name + ": " + ip['title']
                        label_dict["Combined Quota"] = "Number of Cadets"
                        y_label = label_dict[ip["objective"]]

                        # Plot the number of cadets and the PGL
                        if "pgl" in p:
                            quota = p["pgl"]
                        else:
                            quota = p["quota"]

                        # Sort the AFSCs by the PGL
                        if ip["sort_by_pgl"]:
                            indices = np.argsort(quota)[::-1]

                        # Sort the AFSCs by the number of cadets assigned
                        else:
                            indices = np.argsort(measure)[::-1]
                        afscs = afscs[indices]
                        measure = measure[indices]
                        quota = quota[indices]

                        # Legend
                        legend_elements = [Patch(facecolor=ip['bar_colors']["pgl"], label='PGL Target',
                                                 edgecolor="black"),
                                           Patch(facecolor=ip['bar_colors']["surplus"],
                                                 label='Cadets Exceeding PGL Target', edgecolor="black"),
                                           Patch(facecolor=ip['bar_colors']["failed_pgl"], label='PGL Target Not Met',
                                                 edgecolor="black")]

                        # Y max
                        y_max = max(measure)
                        ip["y_max"] = ip["y_max"] * y_max
                        if 100 <= y_max < 150:
                            y_ticks = [50, 100, 150]
                        elif 150 <= y_max < 200:
                            y_ticks = [50, 100, 150, 200]
                        elif 200 <= y_max < 250:
                            y_ticks = [50, 100, 150, 200, 250]
                        elif 250 <= y_max < 300:
                            y_ticks = [50, 100, 150, 200, 250, 300]
                        elif y_max >= 300:
                            y_ticks = [50, 100, 150, 200, 250, 300, 350]
                        else:
                            y_ticks = [50]

                        # Add the text and quota lines
                        for j in range(M):
                            if int(measure[j]) >= 100:
                                ax.text(j - 0.35, measure[j] + 2, int(measure[j]),
                                        fontsize=ip["bar_text_size"],
                                        multialignment='right')
                            elif int(measure[j]) >= 10:
                                ax.text(j - 0.27, measure[j] + 2, int(measure[j]),
                                        fontsize=ip["bar_text_size"],
                                        multialignment='right')
                            else:
                                ax.text(j - 0.2, measure[j] + 2, int(measure[j]),
                                        fontsize=ip["bar_text_size"],
                                        multialignment='right')

                            # Determine which category the AFSC falls into
                            line_color = "black"
                            if measure[j] > quota[j]:
                                ax.bar([j], quota[j], color=ip['bar_colors']["pgl"], edgecolor="black")
                                ax.bar([j], measure[j] - quota[j], bottom=quota[j],
                                       color=ip['bar_colors']["surplus"], edgecolor="black")
                            elif measure[j] < quota[j]:
                                ax.bar([j], measure[j], color=ip['bar_colors']["failed_pgl"], edgecolor="black")
                                ax.plot((j - 0.4, j - 0.4), (quota[j], measure[j]),
                                        color=ip['bar_colors']["failed_pgl"], linestyle="--", zorder=2)
                                ax.plot((j + 0.4, j + 0.4), (quota[j], measure[j]),
                                        color=ip['bar_colors']["failed_pgl"], linestyle="--", zorder=2)
                                line_color = ip['bar_colors']["failed_pgl"]
                            else:
                                ax.bar([j], measure[j], color=ip['bar_colors']["pgl"], edgecolor="black")

                            # Add the PGL lines
                            ax.plot((j - 0.4, j + 0.4), (quota[j], quota[j]), color=line_color, linestyle="-", zorder=2)

                elif ip["objective"] == "Utility":

                    if ip["version"] == "quantity_bar_proportion":

                        # Get the title and filename
                        ip["title"] = "Cadet Preference Breakdown Across Each AFSC"
                        ip["filename"] = instance.solution_name + " Cadet_Preference_Sorted"
                        if ip["solution_in_title"]:
                            ip['title'] = instance.solution_name + ": " + ip['title']
                        label_dict["Utility"] = "Number of Cadets"
                        y_label = label_dict[ip["objective"]]

                        # Convert utility matrix to utility columns
                        preferences, utilities_array = get_utility_preferences(p)

                        # Counts
                        top_3_count = np.zeros(M)
                        bottom_3_count = np.zeros(M)
                        non_vol_count = np.zeros(M)

                        for i, j in enumerate(instance.solution):
                            afsc = p["afsc_vector"][j]
                            if afsc in preferences[i, 0:3]:
                                top_3_count[j] += 1
                            elif afsc in preferences[i, 3:6]:
                                bottom_3_count[j] += 1
                            else:
                                non_vol_count[j] += 1

                        # Legend
                        legend_elements = [Patch(facecolor=ip['bar_colors']["top_3_choices"], label='Top 3 Choices',
                                                 edgecolor='black'),
                                           Patch(facecolor=ip['bar_colors']["bottom_3_choices"],
                                                 label='Bottom 3 Choices', edgecolor='black'),
                                           Patch(facecolor=ip['bar_colors']["non_volunteer"], label='Non-Volunteer',
                                                 edgecolor='black')]

                        # Need to know number of cadets assigned
                        quota_k = np.where(vp["objectives"] == "Combined Quota")[0][0]
                        total_count = instance.metrics["objective_measure"][:, quota_k]

                        # Plot the number of cadets and the PGL
                        if "pgl" in p:
                            quota = p["pgl"]
                        else:
                            quota = p["quota"]

                        # Sort the AFSCs by the PGL
                        if ip["sort_by_pgl"]:
                            indices = np.argsort(quota)[::-1]

                        # Sort the AFSCs by the number of cadets assigned
                        else:
                            indices = np.argsort(total_count)[::-1]

                        afscs = afscs[indices]
                        top_3_count, bottom_3_count, non_vol_count, total_count = \
                            top_3_count[indices], bottom_3_count[indices], non_vol_count[indices], total_count[
                                indices]

                        # Y max
                        y_max = max(total_count)
                        ip["y_max"] = ip["y_max"] * y_max
                        if 100 <= y_max < 150:
                            y_ticks = [50, 100, 150]
                        elif 150 <= y_max < 200:
                            y_ticks = [50, 100, 150, 200]
                        elif 200 <= y_max < 250:
                            y_ticks = [50, 100, 150, 200, 250]
                        elif 250 <= y_max < 300:
                            y_ticks = [50, 100, 150, 200, 250, 300]
                        elif y_max >= 300:
                            y_ticks = [50, 100, 150, 200, 250, 300, 350]
                        else:
                            y_ticks = [50]

                        # Bar Chart
                        ax.bar(afscs, non_vol_count, color=ip['bar_colors']["non_volunteer"], edgecolor='black')
                        ax.bar(afscs, bottom_3_count, bottom=non_vol_count,
                               color=ip['bar_colors']["bottom_3_choices"], edgecolor='black')
                        ax.bar(afscs, top_3_count, bottom=non_vol_count + bottom_3_count,
                               color=ip['bar_colors']["top_3_choices"], edgecolor='black')

                    elif ip["version"] == "bar":

                        # Get the title and filename
                        ip["title"] = "Average Utility Across Each AFSC"
                        ip["filename"] = instance.solution_name + " Utility_Average_AFSC"
                        if ip["solution_in_title"]:
                            ip['title'] = instance.solution_name + ": " + ip['title']

                        # Average Utility Chart (simple)
                        y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
                        ax.bar(afscs, measure, color="black", edgecolor='black', alpha=ip["alpha"])

                    elif ip["version"] == "quantity_bar_gradient":

                        # Get the title and filename
                        ip["title"] = "Cadet Satisfaction Breakdown Across Each AFSC"
                        ip["filename"] = instance.solution_name + " Cadet_Satisfaction_Gradient"
                        if ip["solution_in_title"]:
                            ip['title'] = instance.solution_name + ": " + ip['title']
                        label_dict["Utility"] = "Number of Cadets"
                        y_label = label_dict[ip["objective"]]

                        # Need to know number of cadets assigned
                        quota_k = np.where(vp["objectives"] == "Combined Quota")[0][0]
                        total_count = instance.metrics["objective_measure"][:, quota_k]

                        # Plot the number of cadets and the PGL
                        if "pgl" in p:
                            quota = p["pgl"]
                        else:
                            quota = p["quota"]

                        # Sort the AFSCs by the PGL
                        if ip["sort_by_pgl"]:
                            indices = np.argsort(quota)[::-1]

                        # Sort the AFSCs by the number of cadets assigned
                        else:
                            indices = np.argsort(total_count)[::-1]

                        afscs = afscs[indices]

                        # Y max
                        y_max = max(total_count)
                        ip["y_max"] = ip["y_max"] * y_max
                        if 100 <= y_max < 150:
                            y_ticks = [50, 100, 150]
                        elif 150 <= y_max < 200:
                            y_ticks = [50, 100, 150, 200]
                        elif 200 <= y_max < 250:
                            y_ticks = [50, 100, 150, 200, 250]
                        elif 250 <= y_max < 300:
                            y_ticks = [50, 100, 150, 200, 250, 300]
                        elif y_max >= 300:
                            y_ticks = [50, 100, 150, 200, 250, 300, 350]
                        else:
                            y_ticks = [50]

                        for index, afsc in enumerate(afscs):
                            j = np.where(p["afsc_vector"] == afsc)[0][0]
                            cadets = np.where(instance.solution == j)[0]
                            utility = p["utility"][cadets, j]
                            uq = np.unique(utility)
                            count_sum = 0
                            for val in uq:
                                count = len(np.where(utility == val)[0])
                                c = (1 - val, 0, val)
                                ax.bar([index], count, bottom=count_sum, color=c)
                                count_sum += count

                        # DIY Colorbar
                        h = (100 / 245) * y_max
                        w1 = 0.8
                        w2 = 0.74
                        vals = np.arange(101) / 100
                        current_height = (150 / 245) * y_max
                        ax.add_patch(Rectangle((M - 2, current_height), w1, h, edgecolor='black', facecolor='black',
                                               fill=True, lw=2))
                        ax.text(M - 3.3, (245 / 245) * y_max, '100%', fontsize=ip["xaxis_tick_size"])
                        ax.text(M - 2.8, current_height, '0%', fontsize=ip["xaxis_tick_size"])
                        ax.text((M - 0.95), (166 / 245) * y_max, 'Cadet Satisfaction', fontsize=ip["xaxis_tick_size"],
                                rotation=270)
                        for val in vals:
                            c = (1 - val, 0, val)
                            ax.add_patch(Rectangle((M - 1.95, current_height), w2, h / 101, facecolor=c, fill=True))
                            current_height += h / 101

            else:

                if ip["version"] == "AFOCD_proportion":

                    # Get the title and filename
                    ip["title"] = "AFOCD Degree Tier Proportion Across Each AFSC"
                    ip["filename"] = instance.solution_name + " AFOCD_Proportion_AFSCs"
                    if ip["solution_in_title"]:
                        ip['title'] = instance.solution_name + ": " + ip['title']
                    y_label = "Number of Cadets"

                    # Get the correct quota
                    if "pgl" in p:
                        quota = p["pgl"][indices]
                    else:
                        quota = p["quota"][indices]

                    # Need to know number of cadets assigned
                    quota_k = np.where(vp["objectives"] == "Combined Quota")[0][0]
                    total_count = instance.metrics["objective_measure"][:, quota_k]

                    # Sort the AFSCs by the PGL
                    if ip["sort_by_pgl"]:
                        indices = np.argsort(quota)[::-1]

                    # Sort the AFSCs by the number of cadets assigned
                    else:
                        indices = np.argsort(total_count)[::-1]

                    # AFOCD
                    afocd_objectives = ["Mandatory", "Desired", "Permitted"]
                    afocd_k = {objective: np.where(vp["objectives"] == objective)[0][0] for objective in
                               afocd_objectives}
                    afocd_count = {
                        objective: total_count * instance.metrics["objective_measure"][:, afocd_k[objective]]
                        for objective in afocd_objectives}
                    afocd_count = {objective: afocd_count[objective][indices] for objective in
                                   afocd_objectives}  # Re-sort

                    # Sort AFSCs
                    afscs = afscs[indices]
                    total_count = total_count[indices]

                    # Legend
                    legend_elements = [Patch(facecolor=ip['bar_colors'][objective], label=objective) for
                                       objective in ["Permitted", "Desired", "Mandatory"]]

                    # Labels and tick marks
                    y_max = max(total_count)
                    ip["y_max"] = ip["y_max"] * y_max
                    if 100 <= y_max < 150:
                        y_ticks = [50, 100, 150]
                    elif 150 <= y_max < 200:
                        y_ticks = [50, 100, 150, 200]
                    elif 200 <= y_max < 250:
                        y_ticks = [50, 100, 150, 200, 250]
                    elif 250 <= y_max < 300:
                        y_ticks = [50, 100, 150, 200, 250, 300]
                    elif y_max >= 300:
                        y_ticks = [50, 100, 150, 200, 250, 300, 350]
                    else:
                        y_ticks = [50]

                    # Loop through each AFSC to plot the bars
                    for index, afsc in enumerate(afscs):

                        # Plot the AFOCD bars
                        count_sum = 0
                        for objective in afocd_objectives:

                            # Plot AFOCD bars
                            count = afocd_count[objective][index]
                            ax.bar([index], count, bottom=count_sum, edgecolor='black',
                                   color=ip['bar_colors'][objective])
                            count_sum += count

                            # Put a number on the bar
                            if count >= 10:

                                prop = count / total_count[index]
                                if objective == "Permitted":
                                    color = "black"
                                else:
                                    color = "white"
                                ax.text(index, (count_sum - count / 2 - 2),
                                        round(prop, 2), color=color, zorder=2, fontsize=ip["bar_text_size"],
                                        horizontalalignment='center')

                        # Add the text
                        # ax.text(index, total_count[index] + 2, int(total_count[index]),
                        #         fontsize=ip["text_size"], horizontalalignment='center')

                else:
                    pass

        # Create a legend
        if legend_elements is not None:
            ax.legend(handles=legend_elements, edgecolor='black', fontsize=ip["legend_size"],
                      ncol=1, columnspacing=0.5, handletextpad=0.3, borderaxespad=0.5, borderpad=0.2)

    # Labels
    ax.set_ylabel(y_label)
    ax.yaxis.label.set_size(ip["label_size"])
    ax.set_xlabel("AFSCs")
    ax.xaxis.label.set_size(ip["label_size"])

    # Y ticks
    ax.set_yticks(y_ticks)
    ax.tick_params(axis="y", labelsize=ip["yaxis_tick_size"])
    ax.set_yticklabels(y_ticks)
    ax.margins(y=0)
    ax.set(ylim=(0, ip["y_max"]))

    # X ticks
    ax.set_xticklabels(afscs[tick_indices], rotation=ip["afsc_rotation"])
    ax.set_xticks(tick_indices)
    ax.tick_params(axis="x", labelsize=ip["afsc_tick_size"])
    ax.set(xlim=(-1, len(afscs)))

    # Title
    if ip["display_title"]:
        ax.set_title(ip["title"], fontsize=ip["title_size"])

    # Filename
    if ip["filename"] is None:
        ip["filename"] = ip["title"]

    # Save
    if ip['save']:
        fig.savefig(paths_out['figures'] + instance.data_name + "/results/" + ip['filename'] + '.png',
                    bbox_inches='tight')

    return fig


def afsc_multi_criteria_graph(instance, max_num=None):
    """
    This chart compares certain AFSCs in a solution according to multiple criteria
    """

    # Shorthand
    ip = instance.plt_p
    p = instance.parameters
    vp = instance.value_parameters

    # Create figure
    fig, ax = plt.subplots(figsize=ip['figsize'], facecolor=ip['facecolor'], tight_layout=True, dpi=ip['dpi'])

    # Get list of AFSCs we're considering
    afscs = np.array(ip["comparison_afscs"])
    used_indices = np.array([np.where(p["afsc_vector"] == afsc)[0][0] for afsc in afscs])
    criteria = ip["comparison_criteria"]
    num_afscs = len(afscs)
    num_criteria = len(criteria)

    # Get quota
    if "pgl" in p:
        quota = p["pgl"][used_indices]
    else:
        quota = p["quota"][used_indices]

    # Need to know number of cadets assigned
    quota_k = np.where(vp["objectives"] == "Combined Quota")[0][0]
    total_count = instance.metrics["objective_measure"][used_indices, quota_k]
    full_count = instance.metrics["objective_measure"][:, quota_k]

    # Sort the AFSCs by the PGL
    if ip["sort_by_pgl"]:
        indices = np.argsort(quota)[::-1]

    # Sort the AFSCs by the number of cadets assigned
    else:
        indices = np.argsort(total_count)[::-1]

    # Re-sort the AFSCs, quota, and total count
    afscs = afscs[indices]
    quota = quota[indices]
    total_count = total_count[indices]
    indices = used_indices[indices]

    # Set the bar chart structure parameters
    label_locations = np.arange(num_afscs)
    bar_width = 0.8 / num_criteria

    # Y max
    if max_num is None:
        y_max = ip["y_max"] * max(instance.metrics["objective_measure"][:, quota_k])
    else:
        y_max = ip["y_max"] * max_num

    # Y tick marks
    if 100 <= y_max < 150:
        y_ticks = [50, 100, 150]
    elif 150 <= y_max < 200:
        y_ticks = [50, 100, 150, 200]
    elif 200 <= y_max < 250:
        y_ticks = [50, 100, 150, 200, 250]
    elif 250 <= y_max < 300:
        y_ticks = [50, 100, 150, 200, 250, 300]
    elif y_max >= 300:
        y_ticks = [50, 100, 150, 200, 250, 300, 350]
    else:
        y_ticks = [50]

    # Convert utility matrix to utility columns
    preferences, utilities_array = get_utility_preferences(p)

    # AFOCD
    afocd_objectives = ["Mandatory", "Desired", "Permitted"]
    afocd_k = {objective: np.where(vp["objectives"] == objective)[0][0] for objective in afocd_objectives}
    afocd_count = {objective: full_count * instance.metrics["objective_measure"][:, afocd_k[objective]]
                   for objective in afocd_objectives}
    afocd_count = {objective: afocd_count[objective][indices] for objective in afocd_objectives}  # Re-sort

    # Counts
    top_3_count = np.zeros(p["M"])
    bottom_3_count = np.zeros(p["M"])
    non_vol_count = np.zeros(p["M"])
    for i, j in enumerate(instance.solution):

        # Preference Counts
        afsc = p["afsc_vector"][j]
        if afsc in preferences[i, 0:3]:
            top_3_count[j] += 1
        elif afsc in preferences[i, 3:6]:
            bottom_3_count[j] += 1
        else:
            non_vol_count[j] += 1

    # Re-sort preferences
    top_3_count, bottom_3_count = top_3_count[indices], bottom_3_count[indices]
    non_vol_count = non_vol_count[indices]

    # Percentile
    percentile_dict = {1: (0.75, 1), 2: (0.5, 0.75), 3: (0.25, 0.5), 4: (0, 0.25)}

    # Loop through each solution/AFSC bar
    M = len(indices)
    for index, j in enumerate(indices):
        cadets = np.where(instance.solution == j)[0]
        merit = p["merit"][cadets]
        utility = p["utility"][cadets, j]
        for c, obj in enumerate(criteria):

            if obj == "Preference":

                # Plot preference bars
                ax.bar(label_locations[index] + bar_width * c, non_vol_count[index], bar_width,
                       edgecolor='black', color=ip['bar_colors']["non_volunteer"])
                ax.bar(label_locations[index] + bar_width * c, bottom_3_count[index], bar_width,
                       bottom=non_vol_count[index], edgecolor='black',
                       color=ip['bar_colors']["bottom_3_choices"])
                ax.bar(label_locations[index] + bar_width * c, top_3_count[index], bar_width,
                       bottom=non_vol_count[index] + bottom_3_count[index], edgecolor='black',
                       color=ip['bar_colors']["top_3_choices"])

            elif obj == "Merit":

                # Plot the merit gradient bars
                uq = np.unique(merit)
                count_sum = 0
                for val in uq:
                    count = len(np.where(merit == val)[0])
                    color = str(val)  # Grayscale
                    ax.bar(label_locations[index] + bar_width * c, count, bar_width, bottom=count_sum, color=color,
                           zorder=2)
                    count_sum += count

                # Add the text
                ax.text(label_locations[index] + bar_width * c, total_count[index] + 2, round(np.mean(merit), 2),
                        fontsize=ip["text_size"], horizontalalignment='center')

            elif obj == "Utility":

                # Plot the utility gradient bars
                uq = np.unique(utility)
                count_sum = 0
                for val in uq:
                    count = len(np.where(utility == val)[0])
                    color = (1 - val, 0, val)  # Blue to Red
                    ax.bar(label_locations[index] + bar_width * c, count, bar_width, bottom=count_sum, color=color,
                           zorder=2)
                    count_sum += count

                # Add the text
                ax.text(label_locations[index] + bar_width * c, total_count[index] + 2, round(np.mean(utility), 2),
                        fontsize=ip["text_size"], horizontalalignment='center')

            elif obj == "Quartile":

                # Loop through each quartile
                count_sum = 0
                for q in [4, 3, 2, 1]:
                    lb, ub = percentile_dict[q][0], percentile_dict[q][1]
                    count = len(np.where((merit <= ub) & (merit > lb))[0])
                    ax.bar(label_locations[index] + bar_width * c, count, bar_width, bottom=count_sum,
                           edgecolor="black",
                           color=ip['bar_colors']["quartile_" + str(q)], zorder=2)
                    count_sum += count

                    # Put a number on the bar
                    if count >= 10:
                        if q == 1:
                            color = "white"
                        else:
                            color = "black"
                        ax.text(label_locations[index] + bar_width * c, (count_sum - count / 2 - 2), int(count),
                                color=color, zorder=3, fontsize=ip["bar_text_size"], horizontalalignment='center')

            elif obj == "AFOCD":

                # Plot the AFOCD bars
                count_sum = 0
                for objective in afocd_objectives:

                    # Plot AFOCD bars
                    count = afocd_count[objective][index]
                    ax.bar(label_locations[index] + bar_width * c, count, bar_width,
                           bottom=count_sum, edgecolor='black', color=ip['bar_colors'][objective])
                    count_sum += count

                    # Put a number on the bar
                    if count >= 10:

                        prop = count / total_count[index]
                        if objective == "Permitted":
                            color = "black"
                        else:
                            color = "white"
                        ax.text(label_locations[index] + bar_width * c, (count_sum - count / 2 - 2),
                                round(prop, 2), color=color, zorder=3, fontsize=ip["bar_text_size"],
                                horizontalalignment='center')

                # Add the text
                ax.text(label_locations[index] + bar_width * c, total_count[index] + 2, int(total_count[index]),
                        fontsize=ip["text_size"], horizontalalignment='center')

        # Add Lines to the bar chart
        left = label_locations[index] - bar_width / 2
        right = label_locations[index] + bar_width * (num_criteria - 1) + bar_width / 2
        ax.plot((left, right), (total_count[index], total_count[index]), linestyle="-", linewidth=1, zorder=2,
                color="black")

        # PGL Line
        ax.plot((right - 0.02, right + 0.02), (quota[index], quota[index]), linestyle="-", zorder=2, linewidth=4,
                color="black")  # PGL tick mark

        # Add the text
        ax.text(right + 0.04, quota[index], int(quota[index]), fontsize=ip["bar_text_size"], horizontalalignment='left',
                verticalalignment="center")
    # Labels
    ax.set_ylabel("Number of Cadets")
    ax.yaxis.label.set_size(ip["label_size"])
    ax.set_xlabel("AFSCs")
    ax.xaxis.label.set_size(ip["label_size"])

    # Y ticks
    ax.set_yticks(y_ticks)
    ax.tick_params(axis="y", labelsize=ip["yaxis_tick_size"])
    ax.set_yticklabels(y_ticks)
    ax.margins(y=0)
    ax.set(ylim=(0, y_max))

    # X ticks
    ax.set_xticklabels(afscs, rotation=0)
    ax.set_xticks(label_locations + (bar_width / 2) * (num_criteria - 1))
    ax.tick_params(axis="x", labelsize=ip["afsc_tick_size"])
    # ax.set(xlim=[2 * bar_width - 1, num_criteria])

    # Title
    if ip["display_title"]:
        ax.set_title(ip["title"], fontsize=ip["title_size"])

    # Filename
    if ip["filename"] is None:
        ip["filename"] = ip["title"]

    # Save
    if ip['save']:
        fig.savefig(paths_out['figures'] + instance.data_name + "/results/" + ip['filename'] + '.png',
                    bbox_inches='tight')

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
        fig.savefig(paths_out['figures'] + title + '.png')

    return fig


def cadet_utility_histogram(instance):
    """
    Builds the Cadet Utility histogram
    """

    # Shorthand
    ip = instance.plt_p

    # Shared elements
    fig, ax = plt.subplots(figsize=ip["figsize"], facecolor=ip["facecolor"], dpi=ip["dpi"], tight_layout=True)
    bins = np.arange(21) / 20

    if ip["compare_solutions"]:  # Comparing two or more solutions (Dot charts)
        if ip["title"] is None:

            # Create the title!
            if ip["num_solutions"] == 1:
                ip['title'] = ip["solution_names"][0] + " Cadet Utility Results Histogram"
            elif ip["num_solutions"] == 2:
                ip['title'] = ip["solution_names"][0] + " and " + ip["solution_names"][1] + \
                              " Cadet Utility Results Histogram"
            elif ip["num_solutions"] == 3:
                ip['title'] = ip["solution_names"][0] + ", " + ip["solution_names"][1] + \
                              ", and " + ip["solution_names"][2] + " Cadet Utility Results Histogram"
            else:
                ip['title'] = ip["solution_names"][0] + ", " + ip["solution_names"][1] + \
                              ", " + ip["solution_names"][2] + ", and " + ip["solution_names"][3] + \
                              " Cadet Utility Results Histogram"

        # Plot the results
        m_dict = instance.metrics_dict[ip["vp_name"]]
        legend_elements = []
        for s, solution in enumerate(ip["solution_names"]):
            value = m_dict[solution]['cadet_value']
            ax.hist(value, bins=bins, edgecolor='black', color=ip["colors"][solution], alpha=0.5)
            legend_elements.append(Patch(facecolor=ip["colors"][solution], label=solution,
                                         alpha=0.5, edgecolor='black'))

        ax.legend(handles=legend_elements, edgecolor='black', fontsize=ip["legend_size"], loc='upper left',
                  ncol=ip["num_solutions"], columnspacing=0.8, handletextpad=0.25, borderaxespad=0.5, borderpad=0.4)
    else:

        # Get the title and filename
        ip["title"] = "Cadet Utility Results Histogram"
        ip["filename"] = instance.solution_name + " Cadet_Utility_Histogram"
        if ip["solution_in_title"]:
            ip['title'] = instance.solution_name + ": " + ip['title']

        value = instance.metrics["cadet_value"]
        ax.hist(value, bins=bins, edgecolor='white', color='black', alpha=1)

    # Labels
    ax.set_ylabel('Number of Cadets')
    ax.yaxis.label.set_size(ip["label_size"])
    ax.set_xlabel('Utility Received')
    ax.xaxis.label.set_size(ip["label_size"])

    # Axis
    x_ticks = np.arange(11) / 10
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', labelsize=ip["xaxis_tick_size"])
    ax.tick_params(axis='y', labelsize=ip["yaxis_tick_size"])

    # Title
    if ip["display_title"]:
        ax.set_title(ip["title"], fontsize=ip["title_size"])

    # Filename
    if ip["filename"] is None:
        ip["filename"] = ip["title"]

    if ip["save"]:
        fig.savefig(paths_out['figures'] + instance.data_name + "/results/" + ip["filename"] + '.png',
                    bbox_inches='tight')

    return fig


def cadet_utility_merit_scatter(instance):
    """
    Scatter plot of cadet utility vs cadet merit
    """

    # Shorthand
    ip = instance.plt_p

    # Shared elements
    fig, ax = plt.subplots(figsize=ip["figsize"], facecolor=ip["facecolor"], dpi=ip["dpi"], tight_layout=True)

    # Get the title and filename
    ip["title"] = "Cadet Preference vs. Merit"
    ip["filename"] = instance.solution_name + " Cadet_Preference_Merit_Scatter"
    if ip["solution_in_title"]:
        ip['title'] = instance.solution_name + ": " + ip['title']

    y = instance.metrics["cadet_value"]

    if "merit_all" in instance.parameters:
        x = instance.parameters["merit_all"]
    else:
        x = instance.parameters["merit"]

    ax.scatter(x, y, s=ip["dot_size"], color='black', alpha=1)

    # Labels
    ax.set_ylabel('Cadet Utility Value')
    ax.yaxis.label.set_size(ip["label_size"])
    ax.set_xlabel('Cadet Merit')
    ax.xaxis.label.set_size(ip["label_size"])

    # Axis
    x_ticks = np.arange(11) / 10
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', labelsize=ip["xaxis_tick_size"])
    ax.tick_params(axis='y', labelsize=ip["yaxis_tick_size"])

    # Title
    if ip["display_title"]:
        ax.set_title(ip["title"], fontsize=ip["title_size"])

    # Filename
    if ip["filename"] is None:
        ip["filename"] = ip["title"]

    if ip["save"]:
        fig.savefig(paths_out['figures'] + instance.data_name + "/results/" + ip["filename"] + '.png',
                    bbox_inches='tight')

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
        fig.savefig(paths_out['figures'] + title + '.png', bbox_inches='tight')

    return fig


# Sensitivity Analysis
def pareto_graph(pareto_df, dimensions=None, save=True, title=None, figsize=(10, 8), facecolor='white',
                 display_title=False):
    """
    Builds the Pareto Frontier Chart for adjusting the overall weight on cadets
    :param display_title: if we should display a title or not
    :param save: If we should save the figure
    :param title: If we should include a title or not
    :param pareto_df: data frame of pareto analysis
    :param dimensions: N and M
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :return: figure
    """
    # Colors and Axis
    cm = plt.cm.get_cmap('RdYlBu')
    label_size = 20
    xaxis_tick_size = 20
    yaxis_tick_size = 20

    # Chart
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)
    ax.set_aspect('equal', adjustable='box')

    sc = ax.scatter(pareto_df['Value on AFSCs'], pareto_df['Value on Cadets'], c=pareto_df['Weight on Cadets'],
                    s=100, cmap=cm, edgecolor='black', zorder=1)
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
        fig.savefig(paths_out['figures'] + title + '.png')

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
        fig.savefig(paths_out['figures'] + title + '.png')

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
        fig.savefig(paths_out['figures'] + title + '.png')

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
        fig.savefig(paths_out['figures'] + 'Solution_Results.png', bbox_inches='tight')

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
            fig.savefig(paths_out['figures'] + title + '.png')

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
            fig.savefig(paths_out['figures'] + title + '.png')

        return fig
