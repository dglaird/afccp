import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import numpy as np
import copy
import pandas as pd

# Set matplotlib default font to Times New Roman
import matplotlib as mpl
mpl.rc('font', family='Times New Roman')

# afccp modules
import afccp.core.globals
import afccp.core.data.preferences


class AFSCsChart:
    def __init__(self, instance):
        """
        This is a class dedicated to creating "AFSCs Charts" which are all charts
        that include AFSCs on the x-axis. This is meant to condense the amount of code and increase
        read-ability of the various kinds of charts.
        """

        # Load attributes
        self.parameters = instance.parameters
        self.value_parameters, self.vp_name = instance.value_parameters, instance.vp_name
        self.ip = instance.mdl_p  # "instance plot parameters"
        self.solution, self.solution_name = instance.solution, instance.solution_name
        self.data_name, self.data_version = instance.data_name, instance.data_version

        # Dictionaries of instance components (sets of value parameters, solutions)
        self.vp_dict, self.solutions = instance.vp_dict, instance.solutions

        # Initialize the matplotlib figure/axes
        self.fig, self.ax = plt.subplots(figsize=self.ip['figsize'], facecolor=self.ip['facecolor'], tight_layout=True,
                                         dpi=self.ip['dpi'])

        # Label dictionary for AFSC objectives
        self.label_dict = copy.deepcopy(afccp.core.globals.obj_label_dict)

        # This is going to be a dictionary of all the various chart-specific components we need
        self.c = {"J": self.ip['J'], 'afscs': self.ip['afscs'], 'M': self.ip['M'], 'k': 0,  # Default k
                  'y_max': self.ip['y_max'], 'legend_elements': None, 'use_calculated_y_max': False}

        # If we skip AFSCs
        if self.ip["skip_afscs"]:
            self.c["tick_indices"] = np.arange(1, self.c["M"], 2).astype(int)
        else:
            self.c["tick_indices"] = np.arange(self.c["M"]).astype(int)

        # Where to save the chart
        self.paths = {"Data": instance.export_paths["Analysis & Results"] + "Data Charts/",
                      "Solution": instance.export_paths["Analysis & Results"] + self.solution_name + "/",
                      "Comparison": instance.export_paths["Analysis & Results"] + "Comparison Charts/"}

    # Main build method
    def build(self, chart_type="Data", printing=True):
        """
        Builds the specific chart based on what the user passes within the "instance plot parameters" (ip)
        """
        # Determine which chart to build
        if chart_type == "Data":
            if self.ip['data_graph'] in ['Average Merit', 'USAFA Proportion', 'Average Utility']:
                self.data_average_chart()
            elif self.ip["data_graph"] == "AFOCD Data":
                self.data_afocd_chart()
            elif self.ip["data_graph"] == "Cadet Preference Analysis":
                self.data_preference_chart()
            elif self.ip["data_graph"] == "Eligible Quota":
                self.data_quota_chart()

            # Get filename
            if self.ip["filename"] is None:
                self.ip["filename"] = \
                    self.data_name + " (" + self.data_version + ") " + self.ip["data_graph"] + " (Data).png"

        elif chart_type in ["Solution", "Comparison"]:

            # Only perform the following steps if it's for a "real" VP objective
            if self.ip['objective'] != 'Extra':

                # AFSC objective index and condense the AFSCs if this is an AFOCD objective
                self.c['k'] = np.where(self.value_parameters['objectives'] == self.ip['objective'])[0][0]
                self.condense_afscs_based_on_objective()

            # Need to know number of cadets assigned
            self.c['total_count'] = self.solution["count"][self.c['J']]

            # Determine if we sort the AFSCs by PGL or not
            if self.ip['sort_by_pgl'] and "STEM" not in self.ip['version']:
                quota = np.array([self.parameters['pgl'][j] for j in self.c['J']])

                # Sort the AFSCs by the PGL
                indices = np.argsort(quota)[::-1]
                self.c['afscs'] = self.c['afscs'][indices]
                self.c['total_count'] = self.c['total_count'][indices]
                self.c['J'] = self.c['J'][indices]

            # Sort by STEM AFSCs
            if "STEM" in self.ip['version']:

                # Sort all the AFSCs by the PGL
                sorted_indices = np.argsort(self.parameters['pgl'])[::-1]

                # Sort the AFSCs by "Not STEM", "Hybrid", "STEM" and then by PGL
                sorted_j = []
                for cat in ["Not STEM", "Hybrid", "STEM"]:
                    for j in sorted_indices:
                        if j in p['J^' + cat] and j in self.c['J']:
                            sorted_j.append(j)

                # Sort the specific elements of this chart
                indices = np.array(sorted_j)
                self.c['afscs'] = self.c['afscs'][indices]
                self.c['total_count'] = self.c['total_count'][indices]
                self.c['J'] = self.c['J'][indices]

            if self.ip['results_graph'] == 'Solution Comparison':
                self.results_solution_comparison_chart()
            else:

                if self.ip['objective'] != 'Extra':

                    # Default y-label
                    self.c['y_label'] = self.label_dict[self.ip['objective']]

                    # Shared elements
                    self.c['measure'] = self.solution['objective_measure'][self.c['J'], self.c['k']]

                    # Build the Merit Chart
                    if self.ip['objective'] == 'Merit':
                        self.results_merit_chart()

                    # Demographic Chart
                    elif self.ip['objective'] in ['USAFA Proportion', 'Male', 'Minority']:
                        self.results_demographic_chart()

                    # AFOCD Degree Tier Chart
                    elif self.ip['objective'] in ['Mandatory', 'Desired', 'Permitted', 'Tier 1', 'Tier 2',
                                                  'Tier 3', 'Tier 4']:
                        self.results_degree_tier_chart()

                    # Combined Quota Chart
                    elif self.ip['objective'] == 'Combined Quota':
                        self.results_quota_chart()

                    # Cadet/AFSC Preference Chart
                    elif self.ip['objective'] == 'Utility':
                        self.results_preference_chart()
                    elif self.ip['objective'] == 'Norm Score':
                        if self.ip['version'] == 'bar':
                            self.results_norm_score_chart()
                        else:
                            self.results_preference_chart()

                else:

                    # Demographic Charts
                    if self.ip['version'] in ['Race Chart', 'Gender Chart', 'Ethnicity Chart', 'SOC Chart',
                                              'Race Chart_proportion', 'Gender Chart_proportion',
                                              'Ethnicity Chart_proportion', 'SOC Chart_proportion']:

                        if 'race_categories' not in self.parameters:
                            return None  # We're not doing this

                        self.results_demographic_proportion_chart()

            # Get filename
            if self.ip["filename"] is None:
                self.ip["filename"] = self.data_name + " (" + self.data_version + ") " + self.solution_name + " " + \
                                      self.ip['objective'] + ' ' + self.ip["results_graph"] + " [" + \
                                      self.ip['version'] + "] (Results).png"
        else:
            raise ValueError("Error. Invalid AFSC 'main' chart type value of '" +
                             chart_type + "'. Valid inputs are 'Data' or 'Results'.")

        # Put the solution name in the title
        if self.ip["solution_in_title"]:
            self.ip['title'] = self.solution_name + ": " + self.ip['title']

        # Display title
        if self.ip['display_title']:
            self.fig.suptitle(self.ip['title'], fontsize=self.ip['title_size'])

        # Labels
        self.ax.set_ylabel(self.c["y_label"])
        self.ax.yaxis.label.set_size(self.ip['label_size'])
        self.ax.set_xlabel('AFSCs')
        self.ax.xaxis.label.set_size(self.ip['label_size'])

        if self.ip["color_afsc_text_by_grp"]:
            afsc_colors = [self.ip['bar_colors'][self.parameters['acc_grp'][j]] for j in self.c['J']]
        else:
            afsc_colors = ["black" for _ in self.c['afscs']]

        # X axis
        self.ax.tick_params(axis='x', labelsize=self.ip['afsc_tick_size'])
        self.ax.set_xticklabels(self.c["afscs"][self.c["tick_indices"]], rotation=self.ip['afsc_rotation'])
        self.ax.set_xticks(self.c["tick_indices"])
        self.ax.set(xlim=(-0.8, self.c["M"]))

        # Unique AFSC colors potentially based on accessions group
        for index, xtick in enumerate(self.ax.get_xticklabels()):
            xtick.set_color(afsc_colors[index])

        # Y axis
        self.ax.tick_params(axis='y', labelsize=self.ip['yaxis_tick_size'])
        self.ax.set(ylim=(0, self.c["y_max"]))
        if "y_ticks" in self.c:
            self.ax.set_yticklabels(self.c['y_ticks'])
            self.ax.set_yticks(self.c['y_ticks'])

        # Legend
        if self.ip["add_legend_afsc_chart"] and self.c['legend_elements'] is not None:
            self.ax.legend(handles=self.c["legend_elements"], edgecolor='black', loc=self.ip['legend_loc'],
                           fontsize=self.ip['legend_size'], ncol=self.ip['ncol'], labelspacing=1, handlelength=0.8,
                           handletextpad=0.2, borderpad=0.2, handleheight=2)

        # Save the chart
        if self.ip['save']:
            self.fig.savefig(self.paths[chart_type] + self.ip["filename"])

            if printing:
                print("Saved", self.ip["filename"], "Chart to " + self.paths[chart_type] + ".")
        else:
            if printing:
                print("Created", self.ip["filename"], "Chart.")

        # Return the chart
        return self.fig

    # Chart helper methods
    def condense_afscs_based_on_objective(self):
        """
        This method reduces the AFSCs we're looking at based on the AFSCs that have a non-zero objective weight
        for AFOCD objectives
        """

        # If it's an AFOCD objective, we only take the AFSCs that have that objective
        if self.ip['objective'] in ['Mandatory', 'Desired', 'Permitted', 'Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']:
            self.c['J'] = np.array([j for j in self.c['J'] if self.c['k'] in self.value_parameters['K^A'][j]]).astype(int)
            self.c['afscs'] = self.parameters['afscs'][self.c['J']]
            self.c['M'] = len(self.c['afscs'])

            # Make sure we're not skipping AFSCs at this point
            self.c["tick_indices"] = np.arange(self.c["M"]).astype(int)

        # Make sure at least one AFSC has this objective selected
        if self.c['M'] == 0:
            raise ValueError("Error. No AFSCs have objective '" + self.ip["objective"] + "'.")

    def determine_y_max_and_y_ticks(self):
        """
        This method calculates the correct y_max and y_ticks for this chart in place
        """
        # Y max
        self.c['y_max'] = self.ip['y_max'] * max(self.c['total_count'])
        if 100 <= self.c['y_max'] < 150:
            self.c['y_ticks'] = [50, 100, 150]
        elif 150 <= self.c['y_max'] < 200:
            self.c['y_ticks'] = [50, 100, 150, 200]
        elif 200 <= self.c['y_max'] < 250:
            self.c['y_ticks'] = [50, 100, 150, 200, 250]
        elif 250 <= self.c['y_max'] < 300:
            self.c['y_ticks'] = [50, 100, 150, 200, 250, 300]
        elif self.c['y_max'] >= 300:
            self.c['y_ticks'] = [50, 100, 150, 200, 250, 300, 350]
        else:
            self.c['y_ticks'] = [50]

    def construct_gradient_chart(self, parameter_to_use='cadet_utility'):
        """
        Constructs the gradient chart with "DIY color bar"
        """

        # Shorthand
        p = self.parameters

        for index, j in enumerate(self.c['J']):
            cadets = np.where(self.solution['j_array'] == j)[0]

            # What are we plotting
            if 'parameter_to_use' == 'cadet_utility':
                measure = p["utility"][cadets, j]
            else:
                measure = p["merit"][cadets]

            # Plot the bar
            uq = np.unique(measure)
            count_sum = 0
            for val in uq:
                count = len(np.where(measure == val)[0])

                if parameter_to_use == 'cadet_utility':
                    c = (1 - val, 0, val)  # Blue to Red
                else:
                    c = str(val)  # Grayscale
                self.ax.bar([index], count, bottom=count_sum, color=c)
                count_sum += count

            # Add the text
            self.ax.text(index, self.c['total_count'][index] + 2, int(self.c['total_count'][index]),
                         fontsize=self.ip["text_size"], horizontalalignment='center')

        # DIY Colorbar
        h = (100 / 245) * self.c['y_max']
        w1 = 0.8
        w2 = 0.74
        vals = np.arange(101) / 100
        current_height = (150 / 245) * self.c['y_max']
        self.ax.add_patch(Rectangle((self.c['M'] - 2, current_height), w1, h, edgecolor='black', facecolor='black',
                                    fill=True, lw=2))
        self.ax.text(self.c['M'] - 3.3, (245 / 245) * self.c['y_max'], '100%', fontsize=self.ip["xaxis_tick_size"])
        self.ax.text(self.c['M'] - 2.8, current_height, '0%', fontsize=self.ip["xaxis_tick_size"])
        self.ax.text((self.c['M'] - 0.95), (166 / 245) * self.c['y_max'], 'Cadet Satisfaction',
                    fontsize=self.ip["xaxis_tick_size"], rotation=270)
        for val in vals:
            if parameter_to_use == 'cadet_utility':
                c = (1 - val, 0, val)  # Blue to Red
            else:
                c = str(val)  # Grayscale
            self.ax.add_patch(Rectangle((self.c['M'] - 1.95, current_height), w2, h / 101, facecolor=c, fill=True))
            current_height += h / 101

    # "Data" Visualizations
    def data_average_chart(self):
        """
        This method builds the "Average Merit", "USAFA Proportion", and "Average Utility" data graph charts. They are
        all in a very similar format and are therefore combined
        """

        # Shorthand
        p = self.parameters

        # Get correct solution and targets
        if self.ip['data_graph'] == "Average Merit":
            metric = np.array([np.mean(p['merit'][p['I^E'][j]]) for j in self.c["J"]])
            target = 0.5
        elif self.ip['data_graph'] == 'USAFA Proportion':
            metric = np.array([len(p['I^D']['USAFA Proportion'][j]) / len(p['I^E'][j]) for j in self.c["J"]])
            target = p['usafa_proportion']
        else:
            if self.ip["eligibility"]:
                metric = np.array([np.mean(p['utility'][p['I^E'][j], j]) for j in self.c["J"]])
            else:
                metric = np.array([np.mean(p['utility'][:, j]) for j in self.c["J"]])
            target = None

        # Axis Adjustments
        self.ax.set(ylim=(0, 1))

        # Bar Chart
        self.ax.bar(self.c['afscs'], metric, color=self.ip["bar_color"], alpha=self.ip["alpha"], edgecolor='black')

        # Add a "target" line
        if target is not None:
            self.ax.axhline(y=target, color='black', linestyle='--', alpha=self.ip["alpha"])

        # Get correct label, title
        self.c['y_label'] = self.ip['data_graph']
        if self.ip['title'] is None:
            self.ip['title'] = self.ip['data_graph'] + ' Across '
            if self.ip['data_graph'] == 'Average Utility' and not self.ip['eligibility']:
                self.ip['title'] += 'All Cadets'
            else:
                self.ip['title'] += 'Eligible Cadets'
            if self.ip['eligibility_limit'] != p['N']:
                self.ip['title'] += ' for AFSCs with <= ' + \
                                    str(self.ip['eligibility_limit']) + ' Eligible Cadets'

    def data_afocd_chart(self):
        """
        This method builds the "AFOCD Data" data graph chart.
        """

        # Shorthand
        p = self.parameters

        # Legend
        self.c["legend_elements"] = [
            Patch(facecolor=self.ip["bar_colors"]["Permitted"], label='Permitted', edgecolor='black'),
            Patch(facecolor=self.ip["bar_colors"]["Desired"], label='Desired', edgecolor='black'),
            Patch(facecolor=self.ip["bar_colors"]["Mandatory"], label='Mandatory', edgecolor='black')]

        # Get solution
        mandatory_count = np.array([np.sum(p['mandatory'][:, j]) for j in self.c["J"]])
        desired_count = np.array([np.sum(p['desired'][:, j]) for j in self.c["J"]])
        permitted_count = np.array([np.sum(p['permitted'][:, j]) for j in self.c["J"]])

        # Bar Chart
        self.ax.bar(self.c["afscs"], mandatory_count, color=self.ip["bar_colors"]["Mandatory"], edgecolor='black')
        self.ax.bar(self.c["afscs"], desired_count, bottom=mandatory_count,
                    color=self.ip["bar_colors"]["Desired"], edgecolor='black')
        self.ax.bar(self.c["afscs"], permitted_count, bottom=mandatory_count + desired_count,
                    color=self.ip["bar_colors"]["Permitted"], edgecolor='black')

        # Axis Adjustments
        self.ax.set(ylim=(0, self.ip['eligibility_limit'] + self.ip['eligibility_limit'] / 100))

        # Get correct text
        self.c["y_label"] = "Number of Cadets"
        if self.ip['title'] is None:
            self.ip['title'] = 'AFOCD Degree Tier Breakdown'
            if self.ip['eligibility_limit'] != p['N']:
                self.ip['title'] += ' for AFSCs with <= ' + \
                                    str(self.ip['eligibility_limit']) + ' Eligible Cadets'

    def data_preference_chart(self):
        """
        This method generates the "Cadet Preference" charts based on the version
        specified
        """

        # Shorthand
        p = self.parameters

        # Choice Counts
        top_3_count = np.array([sum(p["Choice Count"][choice][j] for choice in [0, 1, 2]) for j in p["J"]])
        next_3_count = np.array([sum(p["Choice Count"][choice][j] for choice in [3, 4, 5]) for j in p["J"]])

        # Sets of cadets that have the AFSC in their top 3 choices and are eligible for the AFSC
        top_3_cadets = {}
        top_3_eligible_cadets = {}
        for j in p["J"]:
            top_3_cadets[j] = []
            top_3_eligible_cadets[j] = []
            for i in p["I"]:
                for choice in [0, 1, 2]:
                    if i in p["I^Choice"][choice][j]:
                        top_3_cadets[j].append(i)
                        if i in p["I^E"][j]:
                            top_3_eligible_cadets[j].append(i)
            top_3_cadets[j] = np.array(top_3_cadets[j])
            top_3_eligible_cadets[j] = np.array(top_3_eligible_cadets[j])

        # AFOCD solution
        mandatory_count = np.array([np.sum(p['mandatory'][top_3_cadets[j], j]) for j in p["J"]])
        desired_count = np.array([np.sum(p['desired'][top_3_cadets[j], j]) for j in p["J"]])
        permitted_count = np.array([np.sum(p['permitted'][top_3_cadets[j], j]) for j in p["J"]])
        ineligible_count = np.array([np.sum(p['ineligible'][top_3_cadets[j], j]) for j in p["J"]])

        # Version 1
        if self.ip["version"] == "1":
            self.c["legend_elements"] = [
                Patch(facecolor=self.ip['bar_colors']["top_choices"], label='Top 3 Choices', edgecolor='black'),
                Patch(facecolor=self.ip['bar_colors']["mid_choices"], label='Next 3 Choices', edgecolor='black')]

            # Bar Chart
            self.ax.bar(self.c["afscs"], next_3_count, color=self.ip['bar_colors']["mid_choices"], edgecolor='black')
            self.ax.bar(self.c["afscs"], top_3_count, bottom=next_3_count, color=self.ip['bar_colors']["top_choices"],
                        edgecolor='black')

            # Y max used for axis adjustments
            self.c["y_max"] = np.max(top_3_count + next_3_count)

            if self.ip['title'] is None:
                self.ip['title'] = "Cadet Preferences Placed on Each AFSC (Before Match)"

        # Version 2
        elif self.ip["version"] == "2":
            self.c["legend_elements"] = [
                Patch(facecolor=self.ip['bar_colors']["top_choices"], label='Top 3 Choices', edgecolor='black')]

            # Bar Chart
            self.ax.bar(self.c["afscs"], top_3_count, color=self.ip['bar_colors']["top_choices"], edgecolor='black')

            # Y max used for axis adjustments
            self.c["y_max"] = np.max(top_3_count)

            if self.ip['title'] is None:
                self.ip['title'] = "Cadet Top 3 Preferences Placed on Each AFSC (Before Match)"

        # Version 3
        elif self.ip["version"] == "3":
            self.c["legend_elements"] = [Patch(facecolor=self.ip['bar_colors'][objective], label=objective) for
                                         objective in ["Ineligible", "Permitted", "Desired", "Mandatory"]]

            # Bar Chart
            self.ax.bar(self.c["afscs"], mandatory_count, color=self.ip['bar_colors']["Mandatory"], edgecolor='black')
            self.ax.bar(self.c["afscs"], desired_count, bottom=mandatory_count, color=self.ip['bar_colors']["Desired"],
                   edgecolor='black')
            self.ax.bar(self.c["afscs"], permitted_count, bottom=desired_count + mandatory_count,
                        color=self.ip['bar_colors']["Permitted"], edgecolor='black')
            self.ax.bar(self.c["afscs"], ineligible_count, bottom=permitted_count + desired_count + mandatory_count,
                   color=self.ip['bar_colors']["Ineligible"], edgecolor='black')

            # Y max used for axis adjustments
            self.c["y_max"] = np.max(top_3_count)

            if self.ip['title'] is None:
                self.ip['title'] = "Cadet Degree Eligibility of Preferred Cadets on Each AFSC (Before Match)"

        # Version 4
        elif self.ip["version"] == "4":
            self.c["legend_elements"] = [Patch(facecolor=self.ip['bar_colors'][objective], label=objective) for
                                         objective in ["Permitted", "Desired", "Mandatory"]]

            # Bar Chart
            self.ax.bar(self.c["afscs"], mandatory_count, color=self.ip['bar_colors']["Mandatory"],
                        edgecolor='black')
            self.ax.bar(self.c["afscs"], desired_count, bottom=mandatory_count,
                        color=self.ip['bar_colors']["Desired"],
                        edgecolor='black')
            self.ax.bar(self.c["afscs"], permitted_count, bottom=desired_count + mandatory_count,
                        color=self.ip['bar_colors']["Permitted"], edgecolor='black')

            # Y max used for axis adjustments
            self.c["y_max"] = np.max(top_3_count)

            if self.ip['title'] is None:
                self.ip['title'] = "Cadet Degree Eligibility of Preferred Cadets on Each AFSC (Before Match)"

        # Version 5
        elif self.ip["version"] == "5":

            # Build a gradient
            for j in p["J"]:
                merit = p["merit"][top_3_eligible_cadets[j]]
                uq = np.unique(merit)
                count_sum = 0
                for val in uq:
                    count = len(np.where(merit == val)[0])
                    c = str(val)  # Grayscale
                    self.ax.bar([j], count, bottom=count_sum, color=c, zorder=2)
                    count_sum += count

                # Add the text and an outline
                self.ax.text(j, len(top_3_eligible_cadets[j]) + 4, round(np.mean(merit), 2),
                             fontsize=self.ip["text_size"], horizontalalignment='center')
                self.ax.bar([j], len(top_3_eligible_cadets[j]), color="black", zorder=1, edgecolor="black")

            # Y max used for axis adjustments
            self.c["y_max"] = np.max(top_3_count)

            # DIY Colorbar
            h = (100 / 245) * self.c["y_max"]
            w1 = 0.8
            w2 = 0.74
            current_height = (150 / 245) * self.c["y_max"]
            self.ax.add_patch(Rectangle((self.c["M"] - 2, current_height), w1, h, edgecolor='black', facecolor='black',
                                        fill=True, lw=2))
            self.ax.text(self.c["M"] - 3.3, (245 / 245) * self.c["y_max"], '100%', fontsize=self.ip["text_size"])
            self.ax.text(self.c["M"] - 2.8, current_height, '0%', fontsize=self.ip["text_size"])
            self.ax.text((self.c["M"] - 0.95), (166 / 245) * self.c["y_max"], 'Cadet Percentile',
                         fontsize=self.ip["text_size"], rotation=270)
            vals = np.arange(101) / 100
            for val in vals:
                c = str(val)  # Grayscale
                self.ax.add_patch(Rectangle((self.c["M"] - 1.95, current_height), w2, h / 101, color=c, fill=True))
                current_height += h / 101

            # Y max used for axis adjustments
            self.c["y_max"] = np.max(top_3_count)

            if self.ip['title'] is None:
                self.ip['title'] = "Merit of Preferred Cadets on Each AFSC (Before Match)"

        # Version 6
        elif self.ip["version"] == "6":
            self.c["legend_elements"] = [
                Patch(facecolor=self.ip['bar_colors']["usafa"], label='USAFA'),
                Patch(facecolor=self.ip['bar_colors']["rotc"], label='ROTC'),
                mlines.Line2D([], [], color="red", linestyle='-', linewidth=3, label="Baseline")]

            # USAFA/ROTC Numbers
            rotc_baseline = np.array([len(top_3_eligible_cadets[j]) - (
                    len(top_3_eligible_cadets[j]) * p["usafa_proportion"]) for j in p["J"]])
            usafa_count = np.array([np.sum(p['usafa'][top_3_eligible_cadets[j]]) for j in p["J"]])
            rotc_count = np.array([len(top_3_eligible_cadets[j]) - usafa_count[j] for j in p["J"]])

            # Bar Chart
            self.ax.bar(self.c["afscs"], rotc_count, color=self.ip['bar_colors']["rotc"], edgecolor='black')
            self.ax.bar(self.c["afscs"], usafa_count, bottom=rotc_count, color=self.ip['bar_colors']["usafa"],
                        edgecolor='black')

            # Add the baseline marks
            for j in p["J"]:
                self.ax.plot((j - 0.4, j + 0.4), (rotc_baseline[j], rotc_baseline[j]),
                        color="red", linestyle="-", zorder=2, linewidth=3)

            # Y max used for axis adjustments
            self.c["y_max"] = np.max(top_3_count)

            if self.ip['title'] is None:
                self.ip['title'] = "USAFA/ROTC Breakdown of Preferred Cadets on Each AFSC (Before Match)"

        # Version 7
        elif self.ip["version"] == "7":
            self.c["legend_elements"] = [
                Patch(facecolor=self.ip['bar_colors']["male"], label='Male'),
                Patch(facecolor=self.ip['bar_colors']["female"], label='Female'),
                mlines.Line2D([], [], color="red", linestyle='-', linewidth=3, label="Baseline")]

            # USAFA/ROTC Numbers
            female_baseline = np.array([len(top_3_eligible_cadets[j]) - (
                    len(top_3_eligible_cadets[j]) * p["male_proportion"]) for j in p["J"]])
            male_count = np.array([np.sum(p['male'][top_3_eligible_cadets[j]]) for j in p["J"]])
            female_count = np.array([len(top_3_eligible_cadets[j]) - male_count[j] for j in p["J"]])

            # Bar Chart
            self.ax.bar(self.c["afscs"], female_count, color=self.ip['bar_colors']["female"], edgecolor='black')
            self.ax.bar(self.c["afscs"], male_count, bottom=female_count, color=self.ip['bar_colors']["male"],
                        edgecolor='black')

            # Add the baseline marks
            for j in p["J"]:
                self.ax.plot((j - 0.4, j + 0.4), (female_baseline[j], female_baseline[j]),
                             color="red", linestyle="-", zorder=2, linewidth=3)

            # Y max used for axis adjustments
            self.c["y_max"] = np.max(top_3_count)

            if self.ip['title'] is None:
                self.ip['title'] = "Male/Female Breakdown of Preferred Cadets on Each AFSC (Before Match)"

        else:
            raise ValueError("Version '" + str(
                self.ip["version"]) + "' is not valid for the Cadet Preference Analysis data graph.")

        # Axis Adjustments
        self.ax.set(ylim=(0, self.c["y_max"] * self.ip["y_max"]))

        # Get correct y-label
        self.c["y_label"] = "Number of Cadets"

        # Get filename
        if self.ip["filename"] is None:
            self.ip["filename"] = self.data_name + " (" + self.data_version + \
                                  ") Cadet Preference Analysis Version " + self.ip["version"] + " (Data).png"

    def data_quota_chart(self):
        """
        This method builds the "Eligible Quota" data graph chart.
        """

        # Shorthand
        p = self.parameters

        # Legend
        self.c["legend_elements"] = [Patch(facecolor='blue', label='Eligible Cadets', edgecolor='black'),
                                     Patch(facecolor='black', alpha=0.5, label='AFSC Quota', edgecolor='black')]

        # Get solution
        eligible_count = p["num_eligible"][self.c["J"]]
        quota = p['pgl'][self.c["J"]]

        # Bar Chart
        self.ax.bar(self.c["afscs"], eligible_count, color='blue', edgecolor='black')
        self.ax.bar(self.c["afscs"], quota, color='black', edgecolor='black', alpha=0.5)

        # Axis Adjustments
        self.ax.set(ylim=(0, self.ip['eligibility_limit'] + self.ip['eligibility_limit'] / 100))

        # Get correct text
        self.c["y_label"] = "Number of Cadets"
        if self.ip['title'] is None:
            self.ip['title'] = 'Eligible Cadets and Quotas'
            if self.ip['eligibility_limit'] != p['N']:
                self.ip['title'] += ' for AFSCs with <= ' + \
                               str(self.ip['eligibility_limit']) + ' Eligible Cadets'

    # "Results" Visualizations
    def results_solution_comparison_chart(self):
        """
        This method plots the solution comparison chart for the chosen objective specified
        """

        # Shorthand
        p, vp = self.parameters, self.value_parameters
        k = self.c['k']

        # Make sure at least one AFSC has this objective selected
        if self.c['M'] == 0:
            raise ValueError("Error. No AFSCs have objective '" + ip["objective"] + "'.")

        # Keep track of useful variables
        x_under, x_over = [], []
        quota_percent_filled, max_quota_percent = np.zeros(self.c['M']), np.zeros(self.c['M'])
        self.c['legend_elements'] = []
        max_measure = np.zeros(self.c['M'])
        y_top = 0

        if self.ip['ncol'] != 1:
            self.ip['ncol'] = len(self.ip['solution_names'])

        # Loop through each solution
        for s, solution in enumerate(self.ip["solution_names"]):

            # Calculate objective measure
            if self.ip['version'] == 'median_preference':
                cadets = [np.where(self.solutions[solution]['j_array'] == j)[0] for j in p['J']]
                measure = np.array([np.median(p['c_pref_matrix'][cadets[j], j]) for j in self.c['J']])
                self.label_dict[self.ip['objective']] = 'Median Cadet Choice'
            elif self.ip['version'] == 'mean_preference':
                cadets = [np.where(self.solutions[solution]['j_array'] == j)[0] for j in p['J']]
                measure = np.array([np.mean(p['c_pref_matrix'][cadets[j], j]) for j in self.c['J']])
                self.label_dict[self.ip['objective']] = 'Average Cadet Choice'
            elif self.ip['version'] == 'Race Chart':
                measure = np.array([self.solutions[solution]['simpson_index'][j] for j in self.c['J']])
                self.label_dict[self.ip['objective']] = 'Simpson Diversity Index'
            else:
                measure = self.solutions[solution]["objective_measure"][self.c['J'], k]
            if self.ip["objective"] == "Combined Quota":
                self.label_dict[self.ip["objective"]] = 'Proportion of PGL Target Met'  # Adjust Label

                # Assign the right color to the AFSCs
                for idx, j in enumerate(self.c['J']):

                    # Get bounds
                    value_list = vp['objective_value_min'][j, k].split(",")
                    ub = float(value_list[1].strip())  # upper bound
                    quota = p["pgl"][j]

                    # Constraint violations
                    if quota > measure[idx]:
                        x_under.append(idx)
                    elif measure[idx] > ub:
                        x_over.append(idx)

                    quota_percent_filled[idx] = measure[idx] / quota
                    max_quota_percent[idx] = ub / quota

                # Top dot location
                y_top = max(y_top, max(quota_percent_filled))

                # Plot the points
                self.ax.scatter(self.c['afscs'], quota_percent_filled, color=self.ip["colors"][solution],
                                marker=self.ip["markers"][solution], alpha=self.ip["alpha"], edgecolor='black',
                                s=self.ip["dot_size"], zorder=self.ip["zorder"][solution])

            else:

                # Plot the points
                self.ax.scatter(self.c['afscs'], measure, color=self.ip["colors"][solution],
                                marker=self.ip["markers"][solution], alpha=self.ip["alpha"], edgecolor='black',
                                s=self.ip["dot_size"], zorder=self.ip["zorder"][solution])

            max_measure = np.array([max(max_measure[idx], measure[idx]) for idx in range(self.c['M'])])
            element = mlines.Line2D([], [], color=self.ip["colors"][solution], marker=self.ip["markers"][solution],
                                    linestyle='None', markeredgecolor='black', markersize=self.ip['legend_dot_size'],
                                    label=solution, alpha=self.ip["alpha"])
            self.c['legend_elements'].append(element)

        # Lines to the top solution's dot
        if self.ip["objective"] not in ["Combined Quota", "Mandatory", "Desired", "Permitted", 'Tier 1', 'Tier 2',
                                        'Tier 3', 'Tier 4']:
            for idx in range(self.c['M']):
                self.ax.plot((idx, idx), (0, max_measure[idx]), color='black', linestyle='--', zorder=1, alpha=0.5,
                              linewidth=2)

        # Objective Specific elements
        if self.ip["objective"] == "Merit":

            # Tick marks and extra lines
            self.c['y_ticks'] = [0, 0.35, 0.50, 0.65, 0.80, 1]
            self.ax.plot((-1, 50), (0.65, 0.65), color='blue', linestyle='-', zorder=1, alpha=0.5, linewidth=1.5)
            self.ax.plot((-1, 50), (0.5, 0.5), color='black', linestyle='--', zorder=1, alpha=0.5, linewidth=1.5)
            self.ax.plot((-1, 50), (0.35, 0.35), color='blue', linestyle='-', zorder=1, alpha=0.5, linewidth=1.5)

            # Set the max for the y-axis
            self.c['y_max'] = self.ip['y_max'] * np.max(max_measure)

        elif self.ip['version'] == 'Race Chart':
            baseline = self.parameters['baseline_simpson_index']
            self.c['y_ticks'] = [0, baseline, 1]
            self.ax.plot((-1, 50), (baseline, baseline), color='black', linestyle='--', zorder=1, alpha=0.5,
                         linewidth=1.5)

            # Set the max for the y-axis
            self.c['y_max'] = self.ip['y_max'] * np.max(max_measure)

        elif self.ip["objective"] in ["USAFA Proportion", "Male", "Minority"]:

            # Demographic Proportion elements
            prop_dict = {"USAFA Proportion": "usafa_proportion", "Male": "male_proportion",
                         "Minority": "minority_proportion"}
            up_lb = round(p[prop_dict[self.ip["objective"]]] - 0.15, 2)
            up_ub = round(p[prop_dict[self.ip["objective"]]] + 0.15, 2)
            up = round(p[prop_dict[self.ip["objective"]]], 2)
            self.c['y_ticks'] = [0, up_lb, up, up_ub, 1]

            # Add lines for the ranges
            self.ax.axhline(y=up, color='black', linestyle='--', alpha=0.5)
            self.ax.axhline(y=up_lb, color='blue', linestyle='-', alpha=0.5)
            self.ax.axhline(y=up_ub, color='blue', linestyle='-', alpha=0.5)

            # Set the max for the y-axis
            self.c['y_max'] = self.ip['y_max'] * np.max(max_measure)

        elif self.ip["objective"] in ["Mandatory", "Desired", "Permitted", "Tier 1", "Tier 2", 'Tier 3', "Tier 4"]:

            # Degree Tier elements
            self.c['y_ticks'] = [0, 0.2, 0.4, 0.6, 0.8, 1]
            minimums = np.zeros(self.c['M'])
            maximums = np.zeros(self.c['M'])

            # Assign the right color to the AFSCs
            for idx, j in enumerate(self.c['J']):
                if "Increasing" in vp["value_functions"][j, k]:
                    minimums[idx] = vp['objective_target'][j, k]
                    maximums[idx] = 1
                else:
                    minimums[idx] = 0
                    maximums[idx] = vp['objective_target'][j, k]

            # Calculate ranges
            y = [(minimums[idx], maximums[idx]) for idx in range(self.c['M'])]
            y_lines = [(0, minimums[idx]) for idx in range(self.c['M'])]

            # Plot bounds
            self.ax.scatter(range(self.c['M']), minimums, c="black", marker="_", linewidth=2, zorder=1)
            self.ax.scatter(range(self.c['M']), maximums, c="black", marker="_", linewidth=2, zorder=1)

            # Constraint Range
            self.ax.plot((range(self.c['M']), range(self.c['M'])), ([i for (i, j) in y], [j for (i, j) in y]),
                          c="black", zorder=1)

            # Line from x-axis to constraint range
            self.ax.plot((range(self.c['M']), range(self.c['M'])), ([i for (i, j) in y_lines], [j for (i, j) in y_lines]),
                          c="black", zorder=1, alpha=0.5, linestyle='--', linewidth=2)

        elif self.ip["objective"] == "Combined Quota":

            # Y axis adjustments
            self.c['y_ticks'] = [0, 0.5, 1, 1.5, 2]
            self.c['y_max'] = self.ip['y_max'] * y_top

            # Lines
            y_mins = np.repeat(1, self.c['M'])
            y_maxs = max_quota_percent
            y = [(y_mins[idx], y_maxs[idx]) for idx in range(self.c['M'])]
            y_under = [(quota_percent_filled[idx], 1) for idx in x_under]
            y_over = [(max_quota_percent[idx], quota_percent_filled[idx]) for idx in x_over]

            # Plot Bounds
            self.ax.scatter(self.c['afscs'], y_mins, c=np.repeat('black', self.c['M']), marker="_", linewidth=2, zorder=1)
            self.ax.scatter(self.c['afscs'], y_maxs, c=np.repeat('black', self.c['M']), marker="_", linewidth=2, zorder=1)

            # Plot Range Lines
            self.ax.plot((range(self.c['M']), range(self.c['M'])), ([i for (i, j) in y], [j for (i, j) in y]),
                         c='black', zorder=1)
            self.ax.plot((x_under, x_under), ([i for (i, j) in y_under], [j for (i, j) in y_under]), c='black',
                         linestyle='--', zorder=1)
            self.ax.plot((x_over, x_over), ([i for (i, j) in y_over], [j for (i, j) in y_over]), c='black',
                          linestyle='--', zorder=1)
            self.ax.plot((range(self.c['M']), range(self.c['M'])), (np.zeros(self.c['M']), np.ones(self.c['M'])),
                         c='black', linestyle='--', alpha=0.3, zorder=1)

            # Quota Line
            self.ax.axhline(y=1, color='black', linestyle='-', alpha=0.5, zorder=1)

        elif self.ip["objective"] in ["Utility", "Norm Score"] and self.ip['version'] == 'dot':

            # Fix some things for this chart (y_ticks and label)
            self.c['y_ticks'] = [0, 0.2, 0.4, 0.6, 0.8, 1]
            self.label_dict[self.ip['objective']] = afccp.core.globals.obj_label_dict[self.ip['objective']]

        # Set the max for the y-axis
        if self.ip['version'] in ['median_preference', 'mean_preference']:
            self.c['y_max'] = self.ip['y_max'] * np.max(max_measure)

        # Get y-label
        self.c['y_label'] = self.label_dict[self.ip['objective']]

        # Get names of Solutions
        solution_names = ', '.join(self.ip["solution_names"])

        # Create the title!
        if self.ip["title"] is None:
            self.ip['title'] = solution_names + ' ' + self.label_dict[self.ip["objective"]] + " Across Each AFSC"

        # Update the version of the data using the solution names
        self.ip['version'] = solution_names + ' ' + self.ip['version']

    def results_merit_chart(self):
        """
        This method constructs the different charts showing the "Balance Merit" objective
        """

        # Shorthand
        p, vp, solution = self.parameters, self.value_parameters, self.solution
        k, quota, measure = self.c['k'], p['pgl'][self.c['J']], self.c['measure']
        colors, afscs = np.array([self.ip['bar_colors']["small_afscs"] for _ in self.c['J']]), self.c['afscs']

        if self.ip["version"] == "large_only_bar":

            # Get the title
            self.ip["title"] = "Average Merit Across Each Large AFSC"

            # Y-axis
            self.c['use_calculated_y_max'] = True
            self.c['y_max'] = self.ip['y_max']  # * np.max(measure)

            # Assign the right color to the AFSCs
            for j in range(self.c['M']):
                if 0.35 <= measure[j] <= 0.65:
                    colors[j] = self.ip['bar_colors']["merit_within"]
                elif measure[j] > 0.65:
                    colors[j] = self.ip['bar_colors']["merit_above"]
                else:
                    colors[j] = self.ip['bar_colors']["merit_below"]

            # Add lines for the ranges
            self.ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)

            # Bound lines
            if self.ip['add_bound_lines']:
                self.c['y_ticks'] = [0, 0.35, 0.50, 0.65, 0.80, 1]
                self.ax.axhline(y=0.35, color='blue', linestyle='-', alpha=0.5)
                self.ax.axhline(y=0.65, color='blue', linestyle='-', alpha=0.5)
            else:
                self.c['y_ticks'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

            # Bar Chart
            self.ax.bar(afscs, measure, color=colors, edgecolor='black', alpha=self.ip["alpha"])

        elif self.ip["version"] == "bar":

            # Get the title
            self.ip["title"] = "Average Merit Across Each AFSC"

            # Set the max for the y-axis
            self.c['use_calculated_y_max'] = True
            self.c['y_max'] = self.ip['y_max']  # * np.max(measure)

            # Merit elements
            if self.ip['large_afsc_distinction']:
                self.c['legend_elements'] = [Patch(facecolor=self.ip['bar_colors']["small_afscs"], label='Small AFSC'),
                                             Patch(facecolor=self.ip['bar_colors']["large_afscs"], label='Large AFSC')]

            # Assign the right color to the AFSCs
            for j in range(self.c['M']):

                # Colors
                if self.ip['large_afsc_distinction']:
                    if quota[j] >= 40:
                        colors[j] = self.ip['bar_colors']["large_afscs"]
                    else:
                        colors[j] = self.ip['bar_colors']["small_afscs"]
                else:
                    colors[j] = self.ip['bar_color']

                # Add the text
                self.ax.text(j, measure[j] + 0.013, round(measure[j], 2),
                             fontsize=self.ip["text_size"], horizontalalignment='center')

            # Add lines for the ranges
            self.ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)

            # Bound lines
            if self.ip['add_bound_lines']:
                self.c['legend_elements'].append(mlines.Line2D([], [], color="blue", linestyle='-', label="Bound"))
                self.c['y_ticks'] = [0, 0.35, 0.50, 0.65, 0.80, 1]
                self.ax.axhline(y=0.35, color='blue', linestyle='-', alpha=0.5)
                self.ax.axhline(y=0.65, color='blue', linestyle='-', alpha=0.5)
            else:
                self.c['y_ticks'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

            # Bar Chart
            self.ax.bar(afscs, measure, color=colors, edgecolor='black', alpha=self.ip["alpha"])

        elif self.ip["version"] == "quantity_bar_gradient":

            # Get the title and label
            self.ip["title"] = "Cadet Merit Breakdown Across Each AFSC"
            self.c['y_label'] = "Number of Cadets"

            # Calculate y-axis attributes
            self.determine_y_max_and_y_ticks()

            # Build the gradient chart
            self.construct_gradient_chart(parameter_to_use='merit')

        else:  # quantity_bar_proportion  (Quartiles in this case)

            # Update the title
            self.ip["title"] = "Cadet Quartiles Across Each AFSC"
            self.c['y_label'] = "Number of Cadets"

            # Legend
            self.c['legend_elements'] = [Patch(facecolor=self.ip['bar_colors']["quartile_1"], label='1st Quartile'),
                                         Patch(facecolor=self.ip['bar_colors']["quartile_2"], label='2nd Quartile'),
                                         Patch(facecolor=self.ip['bar_colors']["quartile_3"], label='3rd Quartile'),
                                         Patch(facecolor=self.ip['bar_colors']["quartile_4"], label='4th Quartile')]

            # Calculate y-axis attributes
            self.determine_y_max_and_y_ticks()

            percentile_dict = {1: (0.75, 1), 2: (0.5, 0.75), 3: (0.25, 0.5), 4: (0, 0.25)}
            for index, j in enumerate(self.c['J']):
                cadets = np.where(self.solution['j_array'] == j)[0]
                merit = p["merit"][cadets]

                # Loop through each quartile
                count_sum = 0
                for q in [4, 3, 2, 1]:
                    lb, ub = percentile_dict[q][0], percentile_dict[q][1]
                    count = len(np.where((merit <= ub) & (merit > lb))[0])
                    self.ax.bar([index], count, bottom=count_sum,
                                color=self.ip['bar_colors']["quartile_" + str(q)], zorder=2)
                    count_sum += count

                    # Put a number on the bar
                    if count >= 10:
                        if q == 1:
                            c = "white"
                        else:
                            c = "black"
                        self.ax.text(index, (count_sum - count / 2 - 2), int(count), color=c,
                                     zorder=3, fontsize=self.ip["bar_text_size"], horizontalalignment='center')

                # Add the text and an outline
                self.ax.text(index, self.c['total_count'][index] + 2, int(self.c['total_count'][index]),
                             fontsize=self.ip["text_size"], horizontalalignment='center')
                self.ax.bar([index], self.c['total_count'][index], color="black", zorder=1, edgecolor="black")

    def results_demographic_chart(self):
        """
        Chart to visualize the demographics of the solution
        """

        # Shorthand
        p, vp, solution = self.parameters, self.value_parameters, self.solution
        k, quota, measure = self.c['k'], p['pgl'][self.c['J']], self.c['measure']
        colors, afscs = np.array([self.ip['bar_colors']["small_afscs"] for _ in self.c['J']]), self.c['afscs']

        # Demographic Proportion elements
        prop_dict = {"USAFA Proportion": "usafa_proportion", "Male": "male_proportion",
                     "Minority": "minority_proportion"}
        legend_dict = {"USAFA Proportion": "USAFA Proportion", "Male": "Male Proportion",
                       "Minority": "Minority Proportion"}
        up_lb = round(p[prop_dict[self.ip["objective"]]] - 0.15, 2)
        up_ub = round(p[prop_dict[self.ip["objective"]]] + 0.15, 2)
        up = round(p[prop_dict[self.ip["objective"]]], 2)

        if self.ip["version"] == "large_only_bar":

            # Get the title and filename
            self.ip["title"] = legend_dict[self.ip["objective"]] + " Across Large AFSCs"

            # Set the max for the y-axis
            self.c['use_calculated_y_max'] = True
            self.c['y_max'] = self.ip['y_max']  # * np.max(measure)
            self.c['y_ticks'] = [0, up_lb, up, up_ub, 1]

            # Legend elements
            self.c['legend_elements'] = [
                Patch(facecolor=self.ip['bar_colors']["large_within"], label=str(up_lb) + ' < ' + legend_dict[
                    self.ip["objective"]] + ' < ' + str(up_ub)), Patch(facecolor=self.ip['bar_colors']["large_else"],
                                                                       label='Otherwise')]

            # Assign the right color to the AFSCs
            for j in range(self.c['M']):
                if up_lb <= measure[j] <= up_ub:
                    colors[j] = self.ip['bar_colors']["large_within"]
                else:
                    colors[j] = self.ip['bar_colors']["large_else"]

            # Add lines for the ranges
            self.ax.axhline(y=up, color='black', linestyle='--', alpha=0.5)
            self.ax.axhline(y=up_lb, color='blue', linestyle='-', alpha=0.5)
            self.ax.axhline(y=up_ub, color='blue', linestyle='-', alpha=0.5)

            # Bar Chart
            self.ax.bar(afscs, measure, color=colors, edgecolor='black', alpha=self.ip["alpha"])

        elif self.ip["version"] == "bar":

            # Get the title and filename
            self.ip["title"] = legend_dict[self.ip["objective"]] + " Across Each AFSC"

            # Y-axis
            self.c['use_calculated_y_max'] = True
            self.c['y_max'] = self.ip['y_max']  # * np.max(measure)
            self.c['y_ticks'] = [0, up_lb, up, up_ub, 1]

            # Legend elements
            self.c['legend_elements'] = [Patch(facecolor=self.ip['bar_colors']["small_afscs"], label='Small AFSC'),
                                         Patch(facecolor=self.ip['bar_colors']["large_afscs"], label='Large AFSC'),
                                         mlines.Line2D([], [], color="blue", linestyle='-', label="Bound")]

            # Assign the right color to the AFSCs
            for j in range(self.c['M']):
                if quota[j] >= 40:
                    colors[j] = self.ip['bar_colors']["large_afscs"]
                else:
                    colors[j] = self.ip['bar_colors']["small_afscs"]

                # Add the text
                self.ax.text(j, measure[j] + 0.013, round(measure[j], 2),
                             fontsize=self.ip["text_size"], horizontalalignment='center')

            # Add lines for the ranges
            self.ax.axhline(y=up, color='black', linestyle='--', alpha=0.5)
            self.ax.axhline(y=up_lb, color='blue', linestyle='-', alpha=0.5)
            self.ax.axhline(y=up_ub, color='blue', linestyle='-', alpha=0.5)

            # Bar Chart
            self.ax.bar(afscs, measure, color=colors, edgecolor='black', alpha=self.ip["alpha"])

        elif self.ip["version"] == "preference_chart":
            self.c['y_label'] = "Number of Cadets"

            # Objective specific components
            if self.ip["objective"] == "USAFA Proportion":

                # Get the title
                self.ip["title"] = "USAFA/ROTC 7+ Choice Across Each AFSC"
                classes = ["USAFA", "ROTC"]

            elif self.ip["objective"] == "Male":

                # Get the title
                self.ip["title"] = "Male/Female 7+ Choice Across Each AFSC"
                classes = ["Male", "Female"]

            else:  # Minority

                # Get the title
                self.ip["title"] = "Minority/Non-Minority 7+ Choice Across Each AFSC"
                classes = ["Minority", "Non-Minority"]

            # Calculate y-axis attributes
            self.determine_y_max_and_y_ticks()

            # Categories! (Volunteers/Non-Volunteers)
            categories = ["Top 6 Choices", "7+ Choice"]

            # Get the counts for each category and class
            counts = {cls: {cat: np.zeros(self.c['M']) for cat in categories} for cls in classes}
            dem = classes[0].lower()  # reference demographic (male, usafa, minority, etc.)
            for index, afsc in enumerate(afscs):
                j = np.where(p["afscs"] == afsc)[0][0]
                cadets_assigned = np.where(self.solution['j_array'] == j)[0]
                cadets_with_demographic = np.where(p[dem] == 1)[0]
                cadets_class = {classes[0]: np.intersect1d(cadets_assigned, cadets_with_demographic),
                                classes[1]: np.array([cadet for cadet in cadets_assigned if
                                                      cadet not in cadets_with_demographic])}

                # Determine volunteer status
                if categories == ["Top 6 Choices", "7+ Choice"]:
                    cadets_with_category = np.where(p['c_pref_matrix'][:, j] < 7)[0]
                    cadets_cat = {"Top 6 Choices": np.intersect1d(cadets_assigned, cadets_with_category),
                                  "7+ Choice": np.array(
                                      [cadet for cadet in cadets_assigned if cadet not in cadets_with_category])}

                # Loop through each demographic
                for cls in classes:
                    for cat in categories:
                        counts[cls][cat][index] = len(np.intersect1d(cadets_class[cls], cadets_cat[cat]))

            # Legend
            self.c['legend_elements'] = [Patch(facecolor=self.ip['bar_colors'][cls.lower()],
                                               label=cls, edgecolor='black') for cls in classes]
            self.c['legend_elements'].append(
                Patch(facecolor=self.ip['bar_colors']["7+ Choice"], label="7+ Choice"))

            # Loop through each AFSC to plot the bars
            for index, afsc in enumerate(afscs):

                # Plot the AFOCD bars
                count_sum = 0
                for cls in classes[::-1]:
                    for cat in categories[::-1]:

                        if cat == "Top 6 Choices":
                            color = self.ip["bar_colors"][cls.lower()]
                        else:
                            color = self.ip['bar_colors'][cat]

                        # Plot the bars
                        count = counts[cls][cat][index]
                        self.ax.bar([index], count, bottom=count_sum, edgecolor="black", color=color)
                        count_sum += count

                        # Put a number on the bar
                        if count >= 10:

                            if cls in ["Male", "USAFA", "Minority"]:
                                color = "white"
                            else:
                                color = "black"
                            self.ax.text(index, (count_sum - count / 2 - 2), int(count), color=color,
                                         zorder=2, fontsize=self.ip["bar_text_size"], horizontalalignment='center')

                # Add the text
                self.ax.text(index, self.c['total_count'][index] + 2, int(self.c['total_count'][index]),
                             fontsize=self.ip["text_size"], horizontalalignment='center')

        else:  # Sorted Sized Bar Chart
            self.c['y_label'] = "Number of Cadets"

            # Objective specific components
            if self.ip["objective"] == "USAFA Proportion":
                self.ip["title"] = "Source of Commissioning Breakdown Across Each AFSC"
                class_1_color = self.ip['bar_colors']["usafa"]
                class_2_color = self.ip['bar_colors']["rotc"]

                # Legend
                self.c['legend_elements'] = [Patch(facecolor=class_1_color, label='USAFA'),
                                             Patch(facecolor=class_2_color, label='ROTC')]

            elif self.ip["objective"] == "Minority":
                self.ip["title"] = "Minority/Non-Minority Breakdown Across Each AFSC"
                class_1_color = self.ip['bar_colors']["minority"]
                class_2_color = self.ip['bar_colors']["non-minority"]

                # Legend
                self.c['legend_elements'] = [Patch(facecolor=class_1_color, label='Minority'),
                                             Patch(facecolor=class_2_color, label='Non-Minority')]

            else:
                self.ip["title"] = "Gender Breakdown Across Each AFSC"
                class_1_color = self.ip['bar_colors']["male"]
                class_2_color = self.ip['bar_colors']["female"]

                # Legend
                self.c['legend_elements'] = [Patch(facecolor=class_1_color, label='Male'),
                                             Patch(facecolor=class_2_color, label='Female')]

            # Calculate y-axis attributes
            self.determine_y_max_and_y_ticks()

            # Get "class" objective measures (number of cadets with demographic)
            class_1 = measure * self.c['total_count']
            class_2 = (1 - measure) * self.c['total_count']
            self.ax.bar(afscs, class_2, color=class_2_color, zorder=2, edgecolor="black")
            self.ax.bar(afscs, class_1, bottom=class_2, color=class_1_color, zorder=2, edgecolor="black")
            for j, afsc in enumerate(afscs):
                if class_2[j] >= 10:
                    self.ax.text(j, class_2[j] / 2, int(class_2[j]), color="black",
                                 zorder=3, fontsize=self.ip["bar_text_size"], horizontalalignment='center')
                if class_1[j] >= 10:
                    self.ax.text(j, class_2[j] + class_1[j] / 2, int(class_1[j]), color="white", zorder=3,
                                 fontsize=self.ip["bar_text_size"], horizontalalignment='center')

                # Add the text and an outline
                self.ax.text(j, self.c['total_count'][j] + 2, round(measure[j], 2), fontsize=self.ip["text_size"],
                             horizontalalignment='center')

    def results_demographic_proportion_chart(self):
        """
        Chart to visualize the demographics of the solution across each AFSC
        """

        # Shorthand
        p, vp, solution = self.parameters, self.value_parameters, self.solution

        # Category Dictionary
        category_dict = {"Race Chart": p['race_categories'], 'Ethnicity Chart': p['ethnicity_categories'],
                         'Gender Chart': ['Male', 'Female'], 'SOC Chart': ['USAFA', 'ROTC']}
        title_dict = {"Race Chart": 'Racial Demographics Across Each AFSC',
                      'Ethnicity Chart': 'Ethnicity Demographics Across Each AFSC',
                      'Gender Chart': 'Gender Breakdown Across Each AFSC',
                      'SOC Chart': 'Source of Commissioning Breakdown Across Each AFSC'}
        afsc_num_dict = {"Race Chart": "simpson_index", "Ethnicity Chart": "simpson_index_eth",
                         "Gender Chart": "male_proportion_afscs", "SOC Chart": "usafa_proportion_afscs"}
        baseline_dict = {"Gender Chart": "male_proportion", "SOC Chart": "usafa_proportion",
                         "Race Chart": "baseline_simpson_index", "Ethnicity Chart": "baseline_simpson_index_eth"}

        # Proportion Chart
        if '_proportion' in self.ip['version']:
            proportion_chart = True
            version = self.ip['version'][:-11]
        else:
            proportion_chart = False
            version = self.ip['version']

        # Extract the specific information for this chart from the dictionaries above
        self.ip['title'] = title_dict[version]
        categories = category_dict[version]
        afsc_num_dict_s = afsc_num_dict[version]
        baseline_p = baseline_dict[version]

        # Legend elements
        self.c['legend_elements'] = []
        for cat in categories[::-1]:  # Flip the legend
            color = self.ip['bar_colors'][cat]
            self.c['legend_elements'].append(Patch(facecolor=color, label=cat, edgecolor='black'))

        # Calculate y-axis attributes
        if proportion_chart:

            # Get y axis characteristics
            self.c['use_calculated_y_max'] = True
            self.c['y_ticks'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            self.c['y_label'] = "Proportion of Cadets"
            self.c['y_max'] *= 1.03  # expand it a little
            self.ip['legend_size'] = self.ip['proportion_legend_size']  # Change the size of the legend
            self.ip['ncol'] = len(p['race_categories'])  # Adjust number of columns for legend
            self.ip['text_bar_threshold'] = self.ip['proportion_text_bar_threshold']  # Adjust this threshold

            # Baseline
            if len(categories) == 2:
                self.c['legend_elements'].append(mlines.Line2D([], [], color="black", linestyle='--', label="Baseline"))
                self.ax.axhline(y=p[baseline_p], color='black', linestyle='--', alpha=1, zorder=4)
                self.c['y_ticks'] = [0, round(p[baseline_p], 2), 1]
        else:
            self.determine_y_max_and_y_ticks()
            self.c['y_label'] = "Number of Cadets"

        # Loop through each category and AFSC pair
        quantities, proportions = {}, {}
        for cat in categories:
            quantities[cat], proportions[cat] = np.zeros(self.c['M']), np.zeros(self.c['M'])
            for idx, j in enumerate(self.c['J']):
                cadets = np.intersect1d(p['I^' + cat], solution['cadets_assigned'][j])

                # Load metrics
                quantities[cat][idx] = int(len(cadets))
                proportions[cat][idx] = len(cadets) / len(solution['cadets_assigned'][j])

        # Loop through each AFSC to add the text above the bars
        for idx, j in enumerate(self.c['J']):

            # Get the text for the top of the bar
            txt = str(solution[afsc_num_dict_s][j])
            if txt[0] == '0' and txt != '0.0':
                txt = txt[1:]
            elif txt == 'nan':
                txt = ''

            # Add the text
            if proportion_chart:
                self.ax.text(idx, 1.005, txt, verticalalignment='bottom', fontsize=self.ip["text_size"],
                             horizontalalignment='center')
            else:
                self.ax.text(idx, solution['count'][j] + 2, txt, verticalalignment='bottom',
                             fontsize=self.ip["text_size"], horizontalalignment='center')

        # Plot the data
        totals = np.zeros(self.c['M'])
        for cat in quantities:
            color = self.ip['bar_colors'][cat]

            if proportion_chart:
                self.ax.bar(range(self.c['M']), proportions[cat], bottom=totals, color=color, zorder=2,
                            edgecolor='black')
                totals += proportions[cat]
            else:
                self.ax.bar(range(self.c['M']), quantities[cat], bottom=totals, color=color, zorder=2,
                            edgecolor='black')
                totals += quantities[cat]

            # If it's a dark color, change text color to white
            if 'Black' in cat or "Male" in cat:
                text_color = 'white'
            else:
                text_color = 'black'

            # Put a number on the bar
            for idx, j in enumerate(self.c['J']):
                if proportion_chart:
                    if proportions[cat][idx] >= self.ip['text_bar_threshold'] / max(solution['count']):

                        # Rotate triple digit text
                        if quantities[cat][idx] >= 100:
                            rotation = 90
                        else:
                            rotation = 0

                        # Place the text
                        self.ax.text(idx, (totals[idx] - proportions[cat][idx] / 2), int(quantities[cat][idx]),
                                     color=text_color, zorder=2, fontsize=self.ip["text_size"],
                                     horizontalalignment='center', verticalalignment='center', rotation=rotation)
                else:
                    if proportions[cat][idx] >= self.ip['text_bar_threshold'] / max(solution['count']):

                        # Place the text
                        self.ax.text(idx, (totals[idx] - quantities[cat][idx] / 2), int(quantities[cat][idx]),
                                     color=text_color, zorder=2, fontsize=self.ip["text_size"],
                                     horizontalalignment='center', verticalalignment='center')

    def results_degree_tier_chart(self):
        """
        Builds the degree tier results chart
        """

        # Shorthand
        p, vp, solution = self.parameters, self.value_parameters, self.solution
        k, quota, measure = self.c['k'], p['pgl'][self.c['J']], self.c['measure']
        colors, afscs = np.array([self.ip['bar_colors']["small_afscs"] for _ in self.c['J']]), self.c['afscs']

        # Get the title
        self.ip["title"] = self.ip["objective"] + " Proportion Across Each AFSC"

        # Set the max for the y-axis
        self.c['use_calculated_y_max'] = True
        self.c['y_max'] = self.ip['y_max']  # * np.max(measure)

        minimums = np.zeros(self.c['M'])
        maximums = np.zeros(self.c['M'])
        x_under = []
        x_over = []
        x_within = []

        # Assign the right color to the AFSCs
        for index, j in enumerate(self.c['J']):
            if "Increasing" in vp["value_functions"][j, k]:
                minimums[index] = vp['objective_target'][j, k]
                maximums[index] = 1
            else:
                minimums[index] = 0
                maximums[index] = vp['objective_target'][j, k]

            if minimums[index] <= measure[index] <= maximums[index]:
                colors[index] = "blue"
                x_within.append(index)
            else:
                colors[index] = "red"
                if measure[index] < minimums[index]:
                    x_under.append(index)
                else:
                    x_over.append(index)

        # Plot points
        self.ax.scatter(afscs, measure, c=colors, linewidths=4, s=self.ip["dot_size"], zorder=3)

        # Calculate ranges
        y_within = [(minimums[j], maximums[j]) for j in x_within]
        y_under_ranges = [(minimums[j], maximums[j]) for j in x_under]
        y_over_ranges = [(minimums[j], maximums[j]) for j in x_over]
        y_under = [(measure[j], minimums[j]) for j in x_under]
        y_over = [(maximums[j], measure[j]) for j in x_over]

        # Plot bounds
        self.ax.scatter(afscs, minimums, c="black", marker="_", linewidth=2)
        self.ax.scatter(afscs, maximums, c="black", marker="_", linewidth=2)

        # Plot Ranges
        self.ax.plot((x_within, x_within), ([i for (i, j) in y_within], [j for (i, j) in y_within]),
                     c="black")
        self.ax.plot((x_under, x_under), ([i for (i, j) in y_under_ranges], [j for (i, j) in y_under_ranges]),
                     c="black")
        self.ax.plot((x_over, x_over),([i for (i, j) in y_over_ranges], [j for (i, j) in y_over_ranges]), c="black")

        # How far off
        self.ax.plot((x_under, x_under), ([i for (i, j) in y_under], [j for (i, j) in y_under]), c='red', linestyle='--')
        self.ax.plot((x_over, x_over), ([i for (i, j) in y_over], [j for (i, j) in y_over]), c='red', linestyle='--')

    def results_quota_chart(self):
        """
        This method produces the "Combined Quota" chart for each AFSC
        """

        # Shorthand
        p, vp, solution = self.parameters, self.value_parameters, self.solution
        k, quota, measure = self.c['k'], p['pgl'][self.c['J']], self.c['measure']
        colors, afscs = np.array([self.ip['bar_colors']["small_afscs"] for _ in self.c['J']]), self.c['afscs']

        if self.ip["version"] == "dot":

            # Get the title and label
            self.ip["title"] = "Percent of PGL Target Met Across Each AFSC"
            self.c['y_label'] = "Percent of PGL Target Met"  # Manual change from objective label dictionary

            # Set the max for the y-axis
            self.c['use_calculated_y_max'] = True
            self.c['y_ticks'] = [0, 0.5, 1, 1.5, 2]

            # Degree Tier elements
            x_under = []
            x_over = []
            x_within = []
            quota_percent_filled = np.zeros(self.c['M'])
            max_quota_percent = np.zeros(self.c['M'])

            # Assign the right color to the AFSCs
            for index, j in enumerate(self.c['J']):

                # Get bounds
                value_list = vp['objective_value_min'][j, k].split(",")
                max_measure = float(value_list[1].strip())
                if quota[index] > measure[index]:
                    colors[index] = "red"
                    x_under.append(index)
                elif quota[index] <= measure[index] <= max_measure:
                    colors[index] = "blue"
                    x_within.append(index)
                else:
                    colors[index] = "orange"
                    x_over.append(index)

                quota_percent_filled[index] = measure[index] / quota[index]
                max_quota_percent[index] = max_measure / quota[index]

            # Plot points
            self.ax.scatter(afscs, quota_percent_filled, c=colors, linewidths=4, s=self.ip["dot_size"], zorder=3)

            # Set the max for the y-axis
            self.c['y_max'] = self.ip['y_max'] * np.max(quota_percent_filled)

            # Lines
            y_mins = np.repeat(1, self.c['M'])
            y_maxs = max_quota_percent
            y = [(y_mins[j], y_maxs[j]) for j in range(self.c['M'])]
            y_under = [(quota_percent_filled[j], 1) for j in x_under]
            y_over = [(max_quota_percent[j], quota_percent_filled[j]) for j in x_over]

            # Plot Bounds
            self.ax.scatter(afscs, y_mins, c=np.repeat('blue', self.c['M']), marker="_", linewidth=2)
            self.ax.scatter(afscs, y_maxs, c=np.repeat('blue', self.c['M']), marker="_", linewidth=2)

            # Plot Range Lines
            self.ax.plot((np.arange(self.c['M']), np.arange(self.c['M'])), ([i for (i, j) in y], [j for (i, j) in y]),
                         c='blue')
            self.ax.plot((x_under, x_under), ([i for (i, j) in y_under], [j for (i, j) in y_under]), c='red',
                    linestyle='--')
            self.ax.plot((x_over, x_over), ([i for (i, j) in y_over], [j for (i, j) in y_over]), c='orange',
                    linestyle='--')
            self.ax.plot((np.arange(self.c['M']), np.arange(self.c['M'])), (np.zeros(self.c['M']), np.ones(self.c['M'])),
                         c='black', linestyle='--', alpha=0.3)

            # Quota Line
            self.ax.axhline(y=1, color='black', linestyle='-', alpha=0.5)

            # Put quota text
            y_top = round(max(quota_percent_filled))
            y_spacing = (y_top / 80)
            for j in range(self.c['M']):
                if int(measure[j]) >= 100:
                    self.ax.text(j, quota_percent_filled[j] + 1.4 * y_spacing, int(measure[j]),
                                 fontsize=self.ip["xaxis_tick_size"], multialignment='right',
                                 horizontalalignment='center')
                elif int(measure[j]) >= 10:
                    self.ax.text(j, quota_percent_filled[j] + y_spacing, int(measure[j]),
                                 fontsize=self.ip["xaxis_tick_size"], multialignment='right',
                                 horizontalalignment='center')
                else:
                    self.ax.text(j, quota_percent_filled[j] + y_spacing, int(measure[j]),
                                 fontsize=self.ip["xaxis_tick_size"], multialignment='right',
                                 horizontalalignment='center')

        else:  # quantity_bar

            # Get the title and label
            self.ip["title"] = "Number of Cadets Assigned to Each AFSC against PGL"
            self.c['y_label'] = self.label_dict[self.ip["objective"]]

            # Calculate y-axis attributes
            self.determine_y_max_and_y_ticks()

            # Legend
            self.c['legend_elements'] = [Patch(facecolor=self.ip['bar_colors']["pgl"], label='PGL Target',
                                               edgecolor="black"),
                                         Patch(facecolor=self.ip['bar_colors']["surplus"],
                                               label='Cadets Exceeding PGL Target', edgecolor="black"),
                                         Patch(facecolor=self.ip['bar_colors']["failed_pgl"], label='PGL Target Not Met',
                                               edgecolor="black")]

            # Add the text and quota lines
            for j in range(self.c['M']):

                # Add the text
                self.ax.text(j, measure[j] + 2, int(measure[j]),
                             fontsize=self.ip["text_size"], horizontalalignment='center')

                # Determine which category the AFSC falls into
                line_color = "black"
                if measure[j] > quota[j]:
                    self.ax.bar([j], quota[j], color=self.ip['bar_colors']["pgl"], edgecolor="black")
                    self.ax.bar([j], measure[j] - quota[j], bottom=quota[j],
                                color=self.ip['bar_colors']["surplus"], edgecolor="black")
                elif measure[j] < quota[j]:
                    self.ax.bar([j], measure[j], color=self.ip['bar_colors']["failed_pgl"], edgecolor="black")
                    self.ax.plot((j - 0.4, j - 0.4), (quota[j], measure[j]),
                            color=self.ip['bar_colors']["failed_pgl"], linestyle="--", zorder=2)
                    self.ax.plot((j + 0.4, j + 0.4), (quota[j], measure[j]),
                            color=self.ip['bar_colors']["failed_pgl"], linestyle="--", zorder=2)
                    line_color = self.ip['bar_colors']["failed_pgl"]
                else:
                    self.ax.bar([j], measure[j], color=self.ip['bar_colors']["pgl"], edgecolor="black")

                # Add the PGL lines
                self.ax.plot((j - 0.4, j + 0.4), (quota[j], quota[j]), color=line_color, linestyle="-", zorder=2)

    def results_preference_chart(self):
        """
        This method builds the charts for cadet/AFSC preferences
        """

        # Shorthand
        p, vp, solution = self.parameters, self.value_parameters, self.solution
        k, quota, measure = self.c['k'], p['pgl'][self.c['J']], self.c['measure']
        colors, afscs = np.array([self.ip['bar_colors']["small_afscs"] for _ in self.c['J']]), self.c['afscs']

        if self.ip["version"] in ["quantity_bar_proportion", "quantity_bar_choice"]:

            # Counts
            counts = {"bottom_choices": np.zeros(self.c['M']), "mid_choices": np.zeros(self.c['M']),
                      "top_choices": np.zeros(self.c['M'])}

            # Colors
            colors = self.ip['bar_colors']

            if self.ip["objective"] == "Utility":

                # Get the title
                self.ip["title"] = "Cadet Preference Breakdown Across Each AFSC"

                if self.ip["version"] == 'quantity_bar_choice':

                    # Legend (and colors)
                    self.c['legend_elements'] = []
                    self.ip['legend_size'] = int(self.ip['legend_size'] * 0.7)
                    colors = {}
                    for choice in range(p['P'])[:10]:
                        colors[str(choice + 1)] = self.ip['choice_colors'][choice + 1]
                        self.c['legend_elements'].append(Patch(facecolor=colors[str(choice + 1)],
                                                               label=str(choice + 1), edgecolor='black'))
                    colors['All Others'] = self.ip['all_other_choice_colors']
                    self.c['legend_elements'].append(Patch(facecolor=colors['All Others'],
                                                           label='All Others', edgecolor='black'))

                    # Counts
                    counts = {"All Others": np.zeros(self.c['M'])}
                    for choice in range(p['P'])[:10][::-1]:
                        counts[str(choice + 1)] = np.zeros(self.c['M'])
                    for index, j in enumerate(self.c['J']):
                        afsc = p['afscs'][j]
                        total = 0
                        for choice in range(p['P'])[:10]:
                            counts[str(choice + 1)][index] = solution["choice_counts"]["TOTAL"][afsc][choice]
                            total += counts[str(choice + 1)][index]
                        counts['All Others'][index] = self.c['total_count'][index] - total

                else:

                    # Legend
                    self.c['legend_elements'] = [Patch(facecolor=self.ip['bar_colors']["top_choices"],
                                                       label='Top 3 Choices', edgecolor='black'),
                                                 Patch(facecolor=self.ip['bar_colors']["mid_choices"],
                                                       label='Next 3 Choices', edgecolor='black'),
                                                 Patch(facecolor=self.ip['bar_colors']["bottom_choices"],
                                                       label='All Others', edgecolor='black')]

                    # Cadet Choice Counts
                    for index, j in enumerate(self.c['J']):
                        counts['top_choices'][index] = solution['choice_counts']['TOTAL']['Top 3'][j]
                        counts['mid_choices'][index] = solution['choice_counts']['TOTAL']['Next 3'][j]
                        counts['bottom_choices'][index] = solution['choice_counts']['TOTAL']['All Others'][j]



            else:

                # Get the title
                self.ip["title"] = "AFSC Preference Breakdown"

                # Legend
                self.c['legend_elements'] = [Patch(facecolor=self.ip['bar_colors']["top_choices"], label='Top Third',
                                                   edgecolor='black'),
                                             Patch(facecolor=self.ip['bar_colors']["mid_choices"], label='Middle Third',
                                                   edgecolor='black'),
                                             Patch(facecolor=self.ip['bar_colors']["bottom_choices"],
                                                   label='Bottom Third', edgecolor='black')]

                # AFSC Choice Counts
                for i, j in enumerate(self.solution['j_array']):
                    if j in self.c['J']:
                        index = np.where(self.c['J'] == j)[0][0]
                        if p["afsc_utility"][i, j] < 1 / 3:
                            counts["bottom_choices"][index] += 1
                        elif p["afsc_utility"][i, j] < 2 / 3:
                            counts["mid_choices"][index] += 1
                        else:
                            counts["top_choices"][index] += 1

            # Set label
            self.c['y_label'] = "Number of Cadets"

            # Calculate y-axis attributes
            self.determine_y_max_and_y_ticks()

            # Loop through each AFSC to plot the bars
            for index, j in enumerate(self.c["J"]):

                count_sum = 0
                for cat in counts:
                    text_color = "black"

                    # Plot the bars
                    count = counts[cat][index]
                    self.ax.bar([index], count, bottom=count_sum, edgecolor="black", color=colors[cat])
                    count_sum += count

                    # Put a number on the bar
                    if count >= self.ip['text_bar_threshold']:
                        self.ax.text(index, (count_sum - count / 2 - 2), int(count), color=text_color, zorder=2,
                                     fontsize=self.ip["bar_text_size"], horizontalalignment='center')

                # Add the text
                self.ax.text(index, self.c['total_count'][index] + 2, int(self.c['total_count'][index]),
                             fontsize=self.ip["text_size"], horizontalalignment='center')

        elif self.ip["version"] in ["dot", "bar"]:

            # Get the title
            self.ip["title"] = self.label_dict[self.ip["objective"]] + " Across Each AFSC"

            # Average Utility Chart (simple)
            self.c['y_ticks'] = [0, 0.2, 0.4, 0.6, 0.8, 1]
            self.c['use_calculated_y_max'] = True
            self.ax.bar(afscs, measure, color="black", edgecolor='black', alpha=self.ip["alpha"])

            for j in range(self.c['M']):

                # Add the text
                self.ax.text(j, measure[j] + 0.013, round(measure[j], 2), fontsize=self.ip["text_size"],
                             horizontalalignment='center')

        elif self.ip["version"] == "quantity_bar_gradient":

            # Get the title and label
            self.ip["title"] = "Cadet Satisfaction Breakdown Across Each AFSC"
            self.c['y_label'] = 'Number of Cadets'

            # Calculate y-axis attributes
            self.determine_y_max_and_y_ticks()

            # Build the gradient chart
            self.construct_gradient_chart(parameter_to_use='cadet_utility')

    def results_norm_score_chart(self):
        """
        This method constructs the different charts showing the "Norm Score" objective
        """

        # Shorthand
        p, vp, solution = self.parameters, self.value_parameters, self.solution
        k, quota, measure = self.c['k'], p['pgl'][self.c['J']], self.c['measure']
        colors, afscs = np.array([self.ip["bar_color"] for _ in self.c['J']]), self.c['afscs']

        if self.ip["version"] == "bar":

            # Get the title and label
            self.ip["title"] = "Normalized Score Across Each AFSC"

            # Y-axis
            self.c['use_calculated_y_max'] = True
            self.c['y_max'] = self.ip['y_max']
            self.c['y_ticks'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

            # Add the text
            for j in range(self.c['M']):

                # Add the text
                self.ax.text(j, measure[j] + 0.013, round(measure[j], 2),
                             fontsize=self.ip["text_size"], horizontalalignment='center')

            # Bar Chart
            self.ax.bar(afscs, measure, color=colors, edgecolor='black', alpha=self.ip["alpha"])


class AccessionsGroupChart:
    def __init__(self, instance):
        """
        This is a class dedicated to creating "Accessions Group Charts" which are all charts
        that include the "accessions groups" on the x-axis alongside the basline. This is meant to condense the amount
        of code and increase read-ability of the various kinds of charts.
        """

        # Load attributes
        self.parameters = instance.parameters
        self.value_parameters, self.vp_name = instance.value_parameters, instance.vp_name
        self.ip = instance.mdl_p  # "instance plot parameters"
        self.solution, self.solution_name = instance.solution, instance.solution_name
        self.data_name, self.data_version = instance.data_name, instance.data_version

        # Dictionaries of instance components (sets of value parameters, solutions)
        self.vp_dict, self.solutions = instance.vp_dict, instance.solutions

        # Initialize the matplotlib figure/axes
        self.fig, self.ax = plt.subplots(figsize=self.ip['figsize'], facecolor=self.ip['facecolor'], tight_layout=True,
                                         dpi=self.ip['dpi'])

        # Label dictionary for AFSC objectives
        self.label_dict = copy.deepcopy(afccp.core.globals.obj_label_dict)

        # Where to save the chart
        self.paths = {"Data": instance.export_paths["Analysis & Results"] + "Data Charts/",
                      "Solution": instance.export_paths["Analysis & Results"] + self.solution_name + "/",
                      "Comparison": instance.export_paths["Analysis & Results"] + "Comparison Charts/"}

        # Initialize "c" dictionary (specific chart parameters)
        self.c = {"x_labels": ["All Cadets"]}
        for acc_grp in self.parameters['afscs_acc_grp']:
            self.c['x_labels'].append(acc_grp)
        self.c['g'] = len(self.c['x_labels'])

    def build(self, chart_type="Data", printing=True):
        """
        Builds the specific chart based on what the user passes within the "instance plot parameters" (ip)
        """

        # Determine what kind of chart we're showing
        if chart_type == "Data":  # "Before Solution" chart
            pass

        elif chart_type == "Solution":  # Solution chart

            if 'race_categories' not in self.parameters:
                return None  # We're not doing this

            self.results_demographic_chart()

        # Put the solution name in the title if specified
        if self.ip["solution_in_title"]:
            self.ip['title'] = self.solution['name'] + ": " + self.ip['title']

        # Display title
        if self.ip['display_title']:
            self.fig.suptitle(self.ip['title'], fontsize=self.ip['title_size'])

        # Labels
        self.ax.set_ylabel(self.c["y_label"])
        self.ax.yaxis.label.set_size(self.ip['label_size_acc'])
        self.ax.set_xlabel('Accessions Group')
        self.ax.xaxis.label.set_size(self.ip['label_size_acc'])

        # Color the x-axis labels
        if self.ip["color_afsc_text_by_grp"]:
            label_colors = [self.ip['bar_colors'][grp] for grp in self.c['x_labels']]
        else:
            label_colors = ["black" for _ in self.c['x_labels']]

        # X axis
        self.ax.tick_params(axis='x', labelsize=self.ip['xaxis_tick_size'])
        self.ax.set_xticklabels(self.c['x_labels'])

        # Unique label colors potentially based on accessions group
        for index, xtick in enumerate(self.ax.get_xticklabels()):
            xtick.set_color(label_colors[index])

        # Y axis
        self.ax.tick_params(axis='y', labelsize=self.ip['yaxis_tick_size'])
        self.ax.set(ylim=(0, self.ip["y_max"] * 1.03))

        # Legend
        if self.c['legend_elements'] is not None:
            self.ax.legend(handles=self.c["legend_elements"], edgecolor='black', loc=self.ip['legend_loc'],
                           fontsize=self.ip['acc_legend_size'], ncol=self.ip['ncol'], labelspacing=1, handlelength=0.8,
                           handletextpad=0.2, borderpad=0.2, handleheight=2)

        # Get the filename
        if self.ip["filename"] is None:
            self.ip["filename"] = self.data_name + " (" + self.data_version + ") " + self.solution['name'] + \
                                  " Accessions Group [" + self.ip['version'] + "] (Results).png"

        # Save the chart
        if self.ip['save']:
            self.fig.savefig(self.paths[chart_type] + self.ip["filename"])

            if printing:
                print("Saved", self.ip["filename"], "Chart to " + self.paths[chart_type] + ".")
        else:
            if printing:
                print("Created", self.ip["filename"], "Chart.")

        # Return the chart
        return self.fig

    def results_demographic_chart(self):
        """
        Displays a demographic breakdown across accessions groups
        """

        # Shorthand
        p, vp, solution = self.parameters, self.value_parameters, self.solution

        # Update certain things that apply to all versions of this kind of chart
        self.c['y_label'] = 'Proportion of Cadets'

        # Category Dictionary
        category_dict = {"Race Chart": p['race_categories'], 'Ethnicity Chart': p['ethnicity_categories'],
                         'Gender Chart': ['Male', 'Female'], 'SOC Chart': ['USAFA', 'ROTC']}
        title_dict = {"Race Chart": 'Racial Demographics Across Each Accessions Group',
                      'Ethnicity Chart': 'Ethnicity Demographics Across Each Accessions Group',
                      'Gender Chart': 'Gender Breakdown Across Each Accessions Group',
                      'SOC Chart': 'Source of Commissioning Breakdown Across Each Accessions Group'}
        baseline_num_dict = {"Race Chart": "baseline_simpson_index", "Ethnicity Chart": "baseline_simpson_index_eth",
                             "Gender Chart": "male_proportion", "SOC Chart": "usafa_proportion"}
        acc_grp_num_dict = {"Race Chart": "simpson_index_", "Ethnicity Chart": "simpson_index_eth_",
                            "Gender Chart": "male_proportion_", "SOC Chart": "usafa_proportion_"}

        # Extract the specific information for this chart from the dictionaries above
        self.ip['title'] = title_dict[self.ip['version']]
        categories = category_dict[self.ip['version']]
        baseline_p = baseline_num_dict[self.ip['version']]
        acc_grp_num_dict_s = acc_grp_num_dict[self.ip['version']]

        # Legend elements
        self.c['legend_elements'] = []
        for cat in categories[::-1]:  # Flip the legend
            color = self.ip['bar_colors'][cat]
            self.c['legend_elements'].append(Patch(facecolor=color, label=cat, edgecolor='black'))

        # Baseline
        if len(categories) == 2:
            self.c['legend_elements'].append(mlines.Line2D([], [], color="black", linestyle='--', label="Baseline"))
            self.ax.axhline(y=p[baseline_p], color='black', linestyle='--', zorder=4)
            self.c['y_ticks'] = [0, round(p[baseline_p], 2), 1]

        self.ip['ncol'] = len(self.c['legend_elements'])  # Adjust number of columns for legend

        # Loop through each category and accessions group pair
        quantities, proportions = {}, {}
        for cat in categories:
            quantities[cat], proportions[cat] = np.zeros(self.c['g']), np.zeros(self.c['g'])
            for idx, grp in enumerate(self.c['x_labels']):

                # Get metrics from this category group in this accessions group
                if grp == "All Cadets":
                    cadets = p['I^' + cat]

                    # Load metrics
                    quantities[cat][idx] = int(len(cadets))
                    proportions[cat][idx] = len(cadets) / p['N']
                else:
                    cadets = np.intersect1d(p['I^' + cat], solution['I^' + grp])

                    # Load metrics
                    quantities[cat][idx] = int(len(cadets))
                    proportions[cat][idx] = len(cadets) / len(solution['I^' + grp])

        # Loop through each accession group to add the text above the bars
        for idx, grp in enumerate(self.c['x_labels']):
            if grp == "All Cadets":
                txt = str(round(p[baseline_p], 2))
            else:
                txt = str(solution[acc_grp_num_dict_s + grp])
            self.ax.text(idx, 1.005, txt, verticalalignment='bottom', fontsize=self.ip["acc_text_size"],
                         horizontalalignment='center')

        # Plot the data
        totals = np.zeros(self.c['g'])
        for cat in quantities:
            color = self.ip['bar_colors'][cat]
            self.ax.bar(self.c['x_labels'], proportions[cat], bottom=totals, color=color, zorder=2, edgecolor='black')
            totals += proportions[cat]

            # Put a number on the bar
            for idx in range(self.c['g']):
                if proportions[cat][idx] >= self.ip['acc_text_bar_threshold'] / max(solution['count']):

                    # If it's a dark color, change text color to white
                    if 'Black' in cat or "Male" in cat:
                        text_color = 'white'
                    else:
                        text_color = 'black'

                    # Place the text
                    self.ax.text(idx, (totals[idx] - proportions[cat][idx] / 2), int(quantities[cat][idx]),
                                 color=text_color, zorder=2, fontsize=self.ip["acc_bar_text_size"],
                                 horizontalalignment='center', verticalalignment='center')


class ValueFunctionChart:
    def __init__(self, x=[0, 0.5, 1], y=[0, 1, 0], mdl_p={'x_pt': None, 'y_pt': None, 'title': None,
                                                          'display_title': True, 'figsize': (10, 10),
                                                          'facecolor': 'white', 'save': True, 'breakpoints': None,
                                                          'x_ticks': None, 'crit_point': None, 'label_size': 25,
                                                          'yaxis_tick_size': 25, 'xaxis_tick_size': 25, 'x_label': None,
                                                          'filepath': None, 'graph_color': 'black',
                                                          'breakpoint_color': 'black'}):

        # Initialize chart
        self.mdl_p = mdl_p
        self.fig, self.ax = plt.subplots(figsize=self.mdl_p['figsize'], facecolor=self.mdl_p['facecolor'],
                                         tight_layout=True)
        self.ax.set_facecolor(self.mdl_p['facecolor'])

        # Axes
        for spine in ['bottom', 'top', 'left', 'right']:
            self.ax.spines[spine].set_color(self.mdl_p['graph_color'])

        # Title
        if self.mdl_p['title'] is None:
            self.mdl_p['title'] = "Example Value Function Graph"
        if self.mdl_p['display_title']:
            self.fig.suptitle(self.mdl_p['title'], fontsize=self.mdl_p['label_size'], color=self.mdl_p['graph_color'])

        # Plot function
        self.ax.plot(x, y, color=self.mdl_p['graph_color'], linewidth=3)

        # Breakpoints
        if self.mdl_p['breakpoints'] is not None:
            if self.mdl_p['breakpoints'] is True:  # Assign breakpoints to the x and y coordinates
                self.ax.scatter(x, y, color=self.mdl_p['breakpoint_color'], s=100)
            elif self.mdl_p['breakpoints'] is not False:  # Additional breakpoints provided
                self.ax.scatter(self.mdl_p['breakpoints'][0], self.mdl_p['breakpoints'][1],
                                color=self.mdl_p['breakpoint_color'], s=100)

        # Critical Point
        if self.mdl_p['crit_point'] is not None:
            if type(self.mdl_p['crit_point']) == list:
                for point in self.mdl_p['crit_point']:
                    self.ax.plot((point, point), (0, 1), c=self.mdl_p['breakpoint_color'], linestyle="--", linewidth=3)
            else:
                self.ax.plot((self.mdl_p['crit_point'], self.mdl_p['crit_point']), (0, 1),
                             c=self.mdl_p['breakpoint_color'], linestyle="--", linewidth=3)

        # Specified points to highlight
        if self.mdl_p['x_pt'] is not None:
            self.ax.scatter(self.mdl_p['x_pt'], self.mdl_p['y_pt'], color="blue", s=50)
            self.ax.plot((self.mdl_p['x_pt'],
                          self.mdl_p['x_pt']), (0, self.mdl_p['y_pt']), c="blue", linestyle="--", linewidth=3)
            self.ax.plot((0, self.mdl_p['x_pt']),
                         (self.mdl_p['y_pt'], self.mdl_p['y_pt']), c="blue", linestyle="--", linewidth=3)
            self.ax.text(x=self.mdl_p['x_pt'], y=self.mdl_p['y_pt'],
                         s=str(round(self.mdl_p['x_pt'], 2)) + ", " + str(round(self.mdl_p['y_pt'], 2)))

        # Set ticks and labels
        self.ax.set_yticks([1])  # 0 ... 1 (just want the "1" on the y axis!)
        if self.mdl_p['x_ticks'] is not None:
            ax.set_xticks(self.mdl_p['x_ticks'])

        # Adjust axes and ticks
        self.ax.tick_params(axis='x', labelsize=self.mdl_p['xaxis_tick_size'], colors=self.mdl_p['graph_color'])
        self.ax.tick_params(axis='y', labelsize=self.mdl_p['yaxis_tick_size'], colors=self.mdl_p['graph_color'])
        self.ax.set_facecolor(self.mdl_p['facecolor'])
        self.ax.yaxis.label.set_size(self.mdl_p['label_size'])
        self.ax.yaxis.label.set_color(self.mdl_p['graph_color'])
        self.ax.set_ylabel('Value')
        self.ax.xaxis.label.set_size(self.mdl_p['label_size'])
        self.ax.xaxis.label.set_color(self.mdl_p['graph_color'])

        # Label and window margins
        if self.mdl_p['x_label'] is None:
            self.mdl_p['x_label'] = 'Measure'
        self.ax.set_xlabel(self.mdl_p['x_label'])
        self.ax.margins(x=0)
        self.ax.margins(y=0)
        plt.ylim(0, 1.05)

        # Save the chart if necessary
        if self.mdl_p['filepath'] is None:
            self.mdl_p['filepath'] = 'Example_Value_Function.png'  # Just put the example in the working directory
        if self.mdl_p['save']:
            self.fig.savefig(self.mdl_p['filepath'], bbox_inches='tight')


def individual_weight_graph(instance):
    """
    This function creates the chart for either the individual weight function for cadets or the actual
    individual weights on the AFSCs
    :return: chart
    """

    # Shorthand
    p, vp, ip = instance.parameters, instance.value_parameters, instance.mdl_p

    # Initialize figure and title
    if ip["title"] is None:
        if ip["cadets_graph"]:
            title = 'Individual Weight on Cadets'
        else:
            title = 'Individual Weight on AFSCs'
    else:
        title = ip["title"]

    # Build figure
    if ip["cadets_graph"]:

        # Cadets Graph
        fig, ax = plt.subplots(figsize=ip["square_figsize"], facecolor=ip["facecolor"], dpi=ip["dpi"],
                               tight_layout=True)
        # ax.set_aspect('equal', adjustable='box')

        # Get x and y coordinates
        if 'merit_all' in p:
            x = p['merit_all']
        else:
            x = p['merit']
        y = vp['cadet_weight'] / np.max(vp['cadet_weight'])

        # Plot
        ax.scatter(x, y, color=ip["bar_color"], alpha=ip["alpha"], linewidth=3)

        # Labels
        ax.set_ylabel('Cadet Weight', fontname='Times New Roman')
        ax.yaxis.label.set_size(ip["label_size"])
        ax.set_xlabel('Percentile', fontname='Times New Roman')
        ax.xaxis.label.set_size(ip["label_size"])

        # Ticks
        # x_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        # ax.set_xticklabels(x_ticks, fontname='Times New Roman')
        ax.tick_params(axis='x', labelsize=ip["xaxis_tick_size"])
        ax.tick_params(axis='y', labelsize=ip["yaxis_tick_size"])

        # Margins
        ax.set(xlim=(-0.02, 1.02))
        # ax.margins(x=0)
        # ax.margins(y=0)

    else:  # AFSC Chart

        # AFSCs Graph
        fig, ax = plt.subplots(figsize=ip["figsize"], facecolor=ip["facecolor"], dpi=ip["dpi"], tight_layout=True)

        # Labels
        ax.set_ylabel('AFSC Weight')
        ax.yaxis.label.set_size(ip["label_size"])
        ax.set_xlabel('AFSCs')
        ax.xaxis.label.set_size(ip["label_size"])

        # We can skip AFSCs
        if ip["skip_afscs"]:
            tick_indices = np.arange(1, p["M"], 2).astype(int)
        else:
            tick_indices = np.arange(p["M"])

        # Plot
        afscs = p['afscs'][:p["M"]]
        ax.bar(afscs, vp['afsc_weight'], color=ip["bar_color"], alpha=ip["alpha"])

        # Ticks
        ax.set(xlim=(-0.8, p["M"]))
        ax.tick_params(axis='x', labelsize=ip["afsc_tick_size"])
        ax.set_yticks([])
        ax.set_xticklabels(afscs[tick_indices], rotation=ip["afsc_rotation"])
        ax.set_xticks(tick_indices)

    if ip["display_title"]:
        ax.set_title(title, fontsize=ip["label_size"])

    if ip["save"]:
        fig.savefig(instance.export_paths['Analysis & Results'] + 'Value Parameters/' + title + '.png',
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
    used_indices = np.array([np.where(p["afscs"] == afsc)[0][0] for afsc in afscs])
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
    preferences, utilities_array = afccp.core.data.preferences.get_utility_preferences(p)

    # AFOCD
    afocd_objectives = ["Mandatory", "Desired", "Permitted"]
    afocd_k = {objective: np.where(vp["objectives"] == objective)[0][0] for objective in afocd_objectives}
    afocd_count = {objective: full_count * instance.metrics["objective_measure"][:, afocd_k[objective]]
                   for objective in afocd_objectives}
    afocd_count = {objective: afocd_count[objective][indices] for objective in afocd_objectives}  # Re-sort

    # Counts
    top_3_count = np.zeros(p["M"])
    next_3_count = np.zeros(p["M"])
    non_vol_count = np.zeros(p["M"])
    for i, j in enumerate(instance.solution):

        # Preference Counts
        afsc = p["afscs"][j]
        if afsc in preferences[i, 0:3]:
            top_3_count[j] += 1
        elif afsc in preferences[i, 3:6]:
            next_3_count[j] += 1
        else:
            non_vol_count[j] += 1

    # Re-sort preferences
    top_3_count, next_3_count = top_3_count[indices], next_3_count[indices]
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
                       edgecolor='black', color=ip['bar_colors']["bottom_choices"])
                ax.bar(label_locations[index] + bar_width * c, next_3_count[index], bar_width,
                       bottom=non_vol_count[index], edgecolor='black',
                       color=ip['bar_colors']["mid_choices"])
                ax.bar(label_locations[index] + bar_width * c, top_3_count[index], bar_width,
                       bottom=non_vol_count[index] + next_3_count[index], edgecolor='black',
                       color=ip['bar_colors']["top_choices"])

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
        fig.savefig(afccp.core.globals.paths['figures'] + instance.data_name + "/results/" + ip['filename'] + '.png',
                    bbox_inches='tight')

    return fig


def cadet_utility_histogram(instance, filepath=None):
    """
    Builds the Cadet Utility histogram
    """

    # Shorthand
    ip = instance.mdl_p

    # Shared elements
    fig, ax = plt.subplots(figsize=ip["figsize"], facecolor=ip["facecolor"], dpi=ip["dpi"], tight_layout=True)
    bins = np.arange(21) / 20

    if ip["solution_names"] is not None:  # Comparing two or more solutions
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
        legend_elements = []
        for solution_name in ip["solution_names"]:
            value = instance.solutions[solution_name]['cadet_utility_achieved']
            ax.hist(value, bins=bins, edgecolor='black', color=ip["colors"][solution_name], alpha=0.5)
            legend_elements.append(Patch(facecolor=ip["colors"][solution_name], label=solution_name,
                                         alpha=0.5, edgecolor='black'))

        ax.legend(handles=legend_elements, edgecolor='black', fontsize=ip["legend_size"], loc='upper left',
                  ncol=ip["num_solutions"], columnspacing=0.8, handletextpad=0.25, borderaxespad=0.5, borderpad=0.4)
    else:

        # Get the title and filename
        ip["title"] = "Cadet Utility Results Histogram"
        ip["filename"] = instance.solution_name + " Cadet_Utility_Histogram"
        if ip["solution_in_title"]:
            ip['title'] = instance.solution_name + ": " + ip['title']

        value = instance.solution["cadet_utility_achieved"]
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
        ip["filename"] = ip["title"] + '.png'

    # Save the figure
    if ip["save"]:
        if filepath is None:
            filepath = instance.export_paths['Analysis & Results'] + "Results Charts/"
        fig.savefig(filepath + ip["filename"], bbox_inches='tight')

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
        fig.savefig(afccp.core.globals.paths['figures'] + instance.data_name + "/results/" + ip["filename"] + '.png',
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
        fig.savefig(afccp.core.globals.paths['figures'] + instance.data_name + "/results/" + title + '.png',
                    bbox_inches='tight')

    return fig


# Sensitivity Analysis
def pareto_graph(instance, pareto_df, solution_names=None, dimensions=None, save=True, title=None, figsize=(10, 8),
                 facecolor='white', display_title=False, l_word='Value', filepath=None):
    """
    Builds the Pareto Frontier Chart for adjusting the overall weight on cadets
    :param filepath: path to the folder to save this chart in
    :param solution_names: other solutions to plot on the chart
    :param instance: problem instance
    :param l_word: "Label word" for whether we're referring to these as "values" or "utilities"
    :param display_title: if we should display a title or not
    :param save: If we should save the figure
    :param title: If we should include a title or not
    :param pareto_df: data frame of pareto analysis
    :param dimensions: N and M
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :return: figure
    """

    # Shorthand
    ip = instance.mdl_p

    # Colors and Axis
    cm = plt.cm.get_cmap('RdYlBu')
    label_size = 20
    xaxis_tick_size = 20
    yaxis_tick_size = 20

    # Chart
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)
    ax.set_aspect('equal', adjustable='box')

    sc = ax.scatter(pareto_df[l_word + ' on AFSCs'], pareto_df[l_word + ' on Cadets'], c=pareto_df['Weight on Cadets'],
                    s=100, cmap=cm, edgecolor='black', zorder=1)
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

    # Plot solution point(s)
    if solution_names is not None:
        for solution_name in solution_names:
            solution = instance.solutions[solution_name]
            ax.scatter(solution['afsc_utility_overall'], solution['cadet_utility_overall'],
                       c=ip["colors"][solution_name],
                       s=100, edgecolor='black', zorder=2, marker=ip["markers"][solution_name])
            plt.text(solution['afsc_utility_overall'],
                     solution['cadet_utility_overall'] + 0.003, solution_name, fontsize=15,
                     horizontalalignment='center')

    # Labels
    ax.set_ylabel(l_word + ' on Cadets')
    ax.yaxis.label.set_size(label_size)
    ax.set_xlabel(l_word + ' on AFSCs')
    ax.xaxis.label.set_size(label_size)

    # Axis
    ax.tick_params(axis='y', labelsize=yaxis_tick_size)
    ax.tick_params(axis='x', labelsize=xaxis_tick_size)

    # Save the figure
    if save:
        if filepath is None:
            filepath = instance.export_paths['Analysis & Results'] + "Results Charts/"
        if solution_names is None:
            filename = instance.data_name + " " + title + '.png'
        else:
            string_names = ', '.join(solution_names)
            filename = instance.data_name + " " + title + '(' + string_names + ').png'
        fig.savefig(filepath + filename, bbox_inches='tight')

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
    j = np.where(parameters['afscs'] == afsc)[0][0]
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
        fig.savefig(afccp.core.globals.paths['figures'] + instance.data_name + "/value parameters/" + title + '.png',
                    bbox_inches='tight')

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
        fig.savefig(afccp.core.globals.paths['figures'] + instance.data_name + "/value parameters/" + title + '.png',
                    bbox_inches='tight')

    return fig


def solution_results_graph(parameters, value_parameters, solutions, vp_name, k, save=False, colors=None,
                           figsize=(19, 7), facecolor='white'):
    """
    Builds the Graph to show how well we meet each of the objectives
    :param colors: colors of the solutions
    :param vp_name: value parameter name (to access from solutions)
    :param k: objective index
    :param facecolor: color of the background of the graph
    :param figsize: size of the figure
    :param save: Whether we should save the graph
    :param parameters: fixed cadet/AFSC data
    :param value_parameters: value parameters
    :param solutions: solution metrics dictionary
    :return: figure
    """

    # Load the data
    indices = value_parameters['J^E'][k]
    afscs = parameters['afscs'][indices]
    minimums = np.zeros(len(indices))
    maximums = np.zeros(len(indices))
    num_solutions = len(solutions.keys())
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
    for s_num, solution_name in enumerate(solutions.keys()):
        if k == 2:
            measures = solutions[solution_name][vp_name]['objective_measure'][indices, k] / \
                       parameters['quota'][indices]
        elif k == 3:
            measures = solutions[solution_name][vp_name]['objective_measure'][indices, k] / \
                       parameters['usafa_quota'][indices]
        elif k == 4:
            measures = solutions[solution_name][vp_name]['objective_measure'][indices, k] / \
                       (parameters['quota'][indices] - parameters['usafa_quota'][indices])
        else:
            measures = solutions[solution_name][vp_name]['objective_measure'][indices, k]

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
        fig.savefig(afccp.core.globals.paths['figures'] + instance.data_name + "/results/Solution Results Graph.png",
                    bbox_inches='tight')

    return fig


def solution_similarity_graph(instance, coords, solution_names, filepath=None):
    """
    This is the chart that compares the approximate and exact models (with genetic algorithm) in solve time and
    objective value
    """

    # Load in plot parameters
    ip = instance.mdl_p
    ip["figsize"] = (10, 10)

    if ip["title"] is None:
        ip["title"] = instance.data_name + " Solution Similarity"

    # Create figure
    fig, ax = plt.subplots(figsize=ip["figsize"], facecolor=ip["facecolor"], tight_layout=True)
    ax.set_aspect('equal', adjustable='box')

    # Plot the solution dot
    legend_elements = []
    special_solutions = []
    for i, solution_name in enumerate(solution_names):
        x, y = coords[i, 0], coords[i, 1]

        # "Special" Solutions to show
        if solution_name in ip['solution_names']:
            special_solutions.append(solution_name)
            ax.scatter(x, y, c=ip["colors"][solution_name], marker=ip["markers"][solution_name], edgecolor="black",
                       s=ip["sim_dot_size"], zorder=2)

            # Add legend element
            legend_elements.append(mlines.Line2D([], [], color=ip["colors"][solution_name],
                                                 marker=ip["markers"][solution_name], linestyle='None',
                                                 markeredgecolor='black', markersize=20, label=solution_name))

        # "Basic" solutions
        else:

            ax.scatter(x, y, c=ip['default_sim_color'], marker=ip["default_sim_marker"], edgecolor="black",
                       s=ip["sim_dot_size"], zorder=2)

    ax.legend(handles=legend_elements, edgecolor='black', fontsize=ip["legend_size"], loc='upper right',
              ncol=len(legend_elements), columnspacing=0.4, handletextpad=0.1, borderaxespad=0.5, borderpad=0.2)

    # Remove tick marks
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Save the figure
    if ip["save"]:
        if filepath is None:
            filepath = instance.export_paths['Analysis & Results'] + "Results Charts/"
        if len(special_solutions) > 0:
            string_names = ', '.join(special_solutions)
            filename = ip['title'] + '(' + string_names + ').png'
        else:
            filename = ip['title'] + '.png'
        fig.savefig(filepath + filename, bbox_inches='tight')

    return fig
