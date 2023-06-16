# Import libraries
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.text import Annotation
from matplotlib.transforms import Transform, Bbox
import numpy as np
import os
import copy
import pandas as pd
import afccp.core.globals
import afccp.core.solutions.handling

# Import pyomo models if library is installed
if afccp.core.globals.use_pyomo:
    import afccp.core.solutions.pyomo_models

# Set matplotlib default font to Times New Roman
import matplotlib as mpl
mpl.rc('font', family='Times New Roman')

class CadetBoardFigure:
    def __init__(self, instance, printing=None):
        """
        This is the object to construct the "AFSC/Cadet Board" graph and animation to show where cadets get placed in
        a solution and how they move around through various algorithms. The problem instance is the only parameter
        passed. We extract various attributes of the CadetCareerProblem instance for use in this 'CadetBoardFigure'
        instance. One attribute we need in the instance is 'solution_iterations' which is a dictionary of a particular
        set of solutions used in the figure. The 'b' dictionary contains the necessary animation/plot hyperparameters
        as defined in afccp.core.data.ccp_helping_functions.py.
        """

        # Initialize attributes that we take directly from the CadetCareerProblem instance
        self.p, self.vp = instance.parameters, instance.value_parameters
        self.b, self.data_name, self.data_version = instance.mdl_p, instance.data_name, instance.data_version
        self.solution_iterations, self.mdl_p = instance.solution_iterations, instance.mdl_p
        self.paths = instance.export_paths
        self.printing = printing

        # Load in hex values/colors
        filepath = afccp.core.globals.paths['support'] + 'data/value_hex_translation.xlsx'
        if self.mdl_p['use_rainbow_hex']:
            hex_df = afccp.core.globals.import_data(filepath, sheet_name='Rainbow')
        else:
            hex_df = afccp.core.globals.import_data(filepath)
        self.v_hex_dict = {hex_df.loc[i, 'Value']: hex_df.loc[i, 'Hex'] for i in range(len(hex_df))}

        # Figure Height
        self.b['fh'] = self.b['fw'] * self.b['fh_ratio']

        # Border Widths
        for i in ['t', 'l', 'r', 'b', 'u']:
            self.b['bw^' + i] = self.b['fw'] * self.b['bw^' + i + '_ratio']

        # AFSC border/buffer widths
        self.b['abw^lr'] = self.b['fw'] * self.b['abw^lr_ratio']
        self.b['abw^ud'] = self.b['fw'] * self.b['abw^ud_ratio']

        # Legend width/height
        if self.b['add_legend']:
            self.b['lw'] = self.b['fw'] * self.b['lw_ratio']
            self.b['lh'] = self.b['fw'] * self.b['lh_ratio']
        else:
            self.b['lw'], self.b['lh'] = 0, 0

        # Basic information about this sequence for the animation
        self.b['afscs'], self.b['J'] = self.mdl_p['afscs'], self.mdl_p['J']
        self.b['M'] = len(self.b['afscs'])

        # Loop through all the key value pairs of solution iterations and add them to "b"
        for key in self.solution_iterations:
            self.b[key] = self.solution_iterations[key]

        # Cadets in the "denominator" basically
        if self.b['cadets_solved_for'] == 'ROTC Rated':
            self.b['cadets'] = self.p['Rated Cadets']['rotc']
        else:
            self.b['cadets'] = self.p['I']
        self.b['N'] = len(self.b['cadets'])

        # These are attributes to use in the title of each iteration
        self.num_unmatched = self.b['N']
        self.average_afsc_choice = None
        self.average_cadet_choice = None

        # Initialize Figure
        self.fig, self.ax = plt.subplots(figsize=self.b['b_figsize'], tight_layout=True, dpi=self.b['dpi'],
                                         facecolor=self.b['figure_color'])
        self.ax.set_facecolor(self.b['figure_color'])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set(xlim=(0, self.b['fw']))
        self.ax.set(ylim=(0, self.b['fh']))

    # Main functions
    def main(self):
        """
        Main method to call all other methods based on what parameters the user provides
        """

        # Run through some initial preprocessing (calculating 'n' for example)
        self.preprocessing()
        x_y_initialized = self.import_board_parameters()  # Potentially import x and y

        # If we weren't able to initialize x and y coordinates for the board, determine that here
        if not x_y_initialized:

            # Determine x and y coordinates (and potentially 's')
            if self.b['use_pyomo_model']:
                self.calculate_afsc_x_y_s_through_pyomo()
            else:
                self.calculate_afsc_x_y_through_algorithm()

        # Create the rest of the main figure
        self.calculate_cadet_box_x_y()
        self.initialize_board()

        # Create all the iteration frames
        if self.b['save_iteration_frames']:
            self.export_board_parameters()  # Save the board parameters

            # Make the "focus" directory if needed
            folder_path = self.paths['Analysis & Results'] + 'Cadet Board/' + self.b['sequence']
            if self.b['focus'] not in os.listdir(folder_path):
                os.mkdir(folder_path + '/' + self.b['focus'])

            # Simple solution kinds of graphs
            if self.b['type'] in ['Solutions', 'ROTC Rated Board']:

                if self.printing:
                    print("Creating " + str(len(self.b['solutions'])) + " animation images...")

                # Loop through each solution
                for s in self.b['solutions']:
                    self.solution_iteration_frame(s, cadets_to_show='cadets_matched')

            # Matching Algorithm Proposals & Rejections
            elif self.b['type'] in ['MA1']:

                if self.printing:  # "Plus 2" to account for orientation and final solution frames
                    print("Creating " + str(len(self.b['solutions']) + 2) + " animation images...")

                # Save the orientation slide
                filepath = folder_path + '/' + self.b['focus'] + '/0 (Orientation).png'
                self.fig.savefig(filepath)

                # Loop through each iteration
                for s in self.b['solutions']:
                    self.solution_iteration_frame(s, cadets_to_show='cadets_proposing', kind='Proposals')
                    self.rejections_iteration_frame(s, kind='Rejections')

                # Final Solution
                self.solution_iteration_frame(s, cadets_to_show='cadets_matched', kind='Final Solution')
                if self.printing:
                    print('Done.')

    def preprocessing(self):
        """
        This method preprocesses the different specs for this particular figure instance
        """

        # Maximum number of cadets assigned to each AFSC across solutions
        self.b['max_assigned'] = {j: 0 for j in self.b["J"]}

        # Subset of cadets assigned to the AFSC in each solution
        self.b['cadets_matched'], self.b['counts'] = {}, {}

        if 'proposals' in self.b:
            self.b['cadets_proposing'] = {}

        # Loop through each solution (iteration)
        for s in self.b['solutions']:
            self.b['cadets_matched'][s], self.b['counts'][s] = {}, {}

            if 'proposals' in self.b:
                self.b['cadets_proposing'][s] = {}

            # Loop through each AFSC
            for j in self.b['J']:
                self.b['cadets_matched'][s][j] = np.where(self.b['solutions'][s] == j)[0]  # cadets assigned to this AFSC
                self.b['counts'][s][j] = len(self.b['cadets_matched'][s][j])  # number of cadets assigned to this AFSC

                # cadets proposing to this AFSC
                max_count = self.b['counts'][s][j]
                if 'proposals' in self.b:
                    self.b['cadets_proposing'][s][j] = np.where(self.b['proposals'][s] == j)[0]
                    proposal_counts = len(self.b['cadets_proposing'][s][j])  # number of cadets assigned to this AFSC
                    max_count = max(self.b['counts'][s][j], proposal_counts)

                # Update maximum number of cadets assigned if necessary
                if max_count > self.b['max_assigned'][j]:
                    self.b['max_assigned'][j] = max_count

            # Get number of unassigned cadets at the end of the iterations
            if s == self.b['last_s']:
                self.b['unassigned_cadets'] = np.where(self.b['solutions'][s] == self.p['M'])[0]  # cadets left unmatched
                self.b['N^u'] = len(self.b['unassigned_cadets'])  # number of cadets left unmatched

        # Determine number of cadet boxes for AFSCs based on nearest square
        squares_required = [max(self.b['max_assigned'][j], self.p['quota_max'][j]) for j in self.b['J']]
        n = np.ceil(np.sqrt(squares_required)).astype(int)
        n2 = (np.ceil(np.sqrt(squares_required)) ** 2).astype(int)
        self.b['n'] = {j: n[idx] for idx, j in enumerate(self.b['J'])}
        self.b['n^2'] = {j: n2[idx] for idx, j in enumerate(self.b['J'])}

        # Number of boxes in row of unmatched box
        self.b['n^u'] = int((self.b['fw'] - self.b['bw^r'] - self.b['bw^l']) / self.b['s'])

        # Number of rows in unmatched box
        self.b['n^urow'] = int(self.b['N^u'] / self.b['n^u'])

        # Sort the AFSCs by 'n'
        n = np.array([self.b['n'][j] for j in self.b['J']])  # Convert dictionary to numpy array
        indices = np.argsort(n)[::-1]  # Get list of indices that would sort n
        sorted_J = self.b['J'][indices]  # J Array sorted by n
        sorted_n = n[indices]  # n Array sorted by n
        self.b['J^sorted'] = {index: sorted_J[index] for index in range(self.b['M'])}  # Translate 'new j' to 'real j'
        self.b['n^sorted'] = {index: sorted_n[index] for index in range(self.b['M'])}  # Translate 'new n' to 'real n'
        self.b['J^translated'] = {sorted_J[index]: index for index in range(self.b['M'])}  # Translate 'real j' to 'new j'

    # Determine layout of board
    def calculate_afsc_x_y_through_algorithm(self):
        """
        This method calculates the x and y locations of the AFSC boxes using a very simple algorithm.
        """

        # Determine x and y coordinates of bottom left corner of AFSC squares algorithmically
        self.b['x'], self.b['y'] = {j: 0 for j in self.b['J']}, {j: 0 for j in self.b['J']}
        n = np.array([self.b['n'][j] for j in self.b['J']])  # Convert dictionary to numpy array

        # Start at top left corner of main container (This is the algorithm)
        x, y = self.b['bw^l'], self.b['fh'] - self.b['bw^t']
        current_max_n = np.max(n)
        for j in self.b['J']:
            check_x = x + self.b['s'] * self.b['n'][j]

            if check_x > self.b['fw'] - self.b['bw^r'] - self.b['lw']:
                x = self.b['bw^l']  # move back to left-most column
                y = y - self.b['s'] * current_max_n - self.b['abw^ud']  # drop to next row
                current_max_n = self.b['n'][j]

            self.b['x'][j], self.b['y'][j] = x, y - self.b['s'] * self.b['n'][j]  # bottom left corner of box
            x += self.b['s'] * self.b['n'][j] + self.b['abw^lr']  # move over to next column

        if self.printing:
            print("Board parameters 'x' and 'y' determined through simple algorithm.")

    def calculate_afsc_x_y_s_through_pyomo(self):
        """
        This method calculates the x and y locations of the AFSC boxes, as well as the size (s) of the cadet boxes,
        using the pyomo optimization model to determine the optimal placement of all these objects
        """

        if not afccp.core.globals.use_pyomo:
            raise ValueError("Pyomo not installed.")

        # Build the model
        model = afccp.core.solutions.pyomo_models.cadet_board_preprocess_model(self)

        # Get coordinates and size of boxes by solving the model
        self.b['s'], self.b['x'], self.b['y'] = afccp.core.solutions.pyomo_models.solve_pyomo_model(
            self, model, "CadetBoard", q=None, printing=self.printing)

        if self.printing:
            print("Board parameters 'x' and 'y' determined through pyomo model.")

    def calculate_cadet_box_x_y(self):
        """
        This method uses the x and y coordinates of the AFSC boxes, along with the size of the cadet boxes, to calculate
        the x and y coordinates of all the individual cadet boxes.
        """

        # Get coordinates of all cadet boxes
        self.b['cb_coords'] = {}
        for j in self.b['J']:
            self.b['cb_coords'][j] = {}

            # Bottom left corner of top left cadet square
            x, y = self.b['x'][j], self.b['y'][j] + self.b['s'] * (self.b['n'][j] - 1)

            # Loop through all cadet boxes to get individual coordinates of bottom left corner of each cadet box
            i = 0
            for r in range(self.b['n'][j]):
                for c in range(self.b['n'][j]):
                    x_i = x + c * self.b['s']
                    y_i = y - r * self.b['s']
                    self.b['cb_coords'][j][i] = (x_i, y_i)
                    i += 1

    def initialize_board(self):
        """
        This method takes all the necessary board parameters and constructs the board to then be manipulated in other
        algorithms based on what the user wants to do.
        """

        # Loop through each AFSC to add certain elements
        self.b['afsc_name_text'] = {}
        self.b['afsc_fontsize'] = {}
        self.b['c_boxes'] = {}
        self.b['c_circles'] = {}
        for j in self.b['J']:

            # AFSC names
            if self.b['afsc_names_sized_box']:

                # Calculate fontsize and put AFSC name in middle of box
                x = self.b['x'][j] + (self.b['n'][j] / 2) * self.b['s']
                y = self.b['y'][j] + (self.b['n'][j] / 2) * self.b['s']
                w, h = self.b['n'][j] * self.b['s'], self.b['n'][j] * self.b['s']
                self.b['afsc_fontsize'][j] = get_fontsize_for_text_in_box(self.ax, self.p['afscs'][j], (x, y), w, h,
                                                                       va='center')
                va = 'center'
            else:

                # AFSC fontsize is given and put AFSC name above box
                x = self.b['x'][j] + (self.b['n'][j] / 2) * self.b['s']
                y = self.b['y'][j] + self.b['n'][j] * self.b['s'] + 0.02
                self.b['afsc_fontsize'][j] = self.b['afsc_title_size']
                va = 'bottom'

            self.b['afsc_name_text'][j] = self.ax.text(x, y, self.p['afscs'][j], fontsize=self.b['afsc_fontsize'][j],
                                                       horizontalalignment='center', verticalalignment=va,
                                                       color=self.b['text_color'])

            # Loop through each cadet to add the cadet boxes and circles
            self.b['c_boxes'][j] = {}
            self.b['c_circles'][j] = {}
            for i in range(self.b['n^2'][j]):  # All cadet boxes

                # If we are under the maximum number of cadets allowed
                if i + 1 <= self.p['quota_max'][j]:

                    # If we are under the PGL
                    if i + 1 <= self.p['pgl'][j]:
                        linestyle = self.b['pgl_linestyle']
                        color = self.b['pgl_color']
                        alpha = self.b['pgl_alpha']

                    # 'Surplus' Range
                    else:
                        linestyle = self.b['surplus_linestyle']
                        color = self.b['surplus_color']
                        alpha = self.b['surplus_alpha']

                    # Make the rectangle patch (cadet box)
                    self.b['c_boxes'][j][i] = patches.Rectangle(self.b['cb_coords'][j][i], self.b['s'], self.b['s'],
                                                                linestyle=linestyle, linewidth=1, facecolor=color,
                                                                alpha=alpha, edgecolor=self.b['cb_edgecolor'])

                    # Add the patch to the figure
                    self.ax.add_patch(self.b['c_boxes'][j][i])

                # If we are under the maximum number of cadets assigned to this AFSC across the solutions
                if i + 1 <= self.b['max_assigned'][j]:

                    # Make the circle patch (cadet)
                    x, y = self.b['cb_coords'][j][i][0] + (self.b['s'] / 2), \
                           self.b['cb_coords'][j][i][1] + (self.b['s'] / 2)
                    self.b['c_circles'][j][i] = patches.Circle((x, y), radius = (self.b['s'] / 2) * 0.8,
                                                                linestyle='-', linewidth=1, facecolor='black',
                                                                alpha=1, edgecolor='black')

                    # Add the patch to the figure
                    self.ax.add_patch(self.b['c_circles'][j][i])

                    # Hide the circle
                    self.b['c_circles'][j][i].set_visible(False)

        if self.b['draw_containers']:

            # Build box around main container
            self.b['container'] = patches.Rectangle((self.b['bw^l'], self.b['bw^b']), self.b['fw'] - self.b['bw^r'] -
                                                    self.b['bw^l'], self.b['fh'] - self.b['bw^t'] - self.b['bw^b'],
                                          linestyle='-', linewidth=1, edgecolor='black', facecolor='none')
            self.ax.add_patch(self.b['container'])

            # Build box around legend
            self.b['legend_box'] = patches.Rectangle((self.b['fw'] - self.b['bw^r'] - self.b['lw'], self.b['fh'] -
                                                      self.b['bw^t'] - self.b['lh']), self.b['lw'], self.b['lh'],
                                                     linestyle='-', linewidth=1, edgecolor='black', facecolor='none')
            self.ax.add_patch(self.b['legend_box'])

        # Build unmatched cadets box if necessary
        if self.b['N^u'] > 0:
            self.b['unmatched_box'] = patches.Rectangle((self.b['bw^l'], self.b['bw^b']), self.b['fw'] - self.b['bw^r'] -
                                                        self.b['bw^l'], self.b['bw^b'] + self.b['n^urow'] * self.b['s'],
                                                        linestyle='-', linewidth=1, edgecolor='black', facecolor='none')
            self.ax.add_patch(self.b['unmatched_box'])

        # Remove tick marks
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Add the title
        if self.b['b_title'] is None:
            title = "Round 0 (Orientation)"
        self.fig.suptitle(title, fontsize=self.b['b_title_size'], color=self.b['text_color'])

        # Save the figure
        if self.b['save_board_default']:
            folder_path = self.paths['Analysis & Results'] + 'Cadet Board/'
            if self.b['sequence'] not in os.listdir(folder_path):
                os.mkdir(folder_path + self.b['sequence'])

            # Get the filepath and save the "default" graph
            filepath = folder_path + self.b['sequence'] + '/Default Board'
            if type(self.mdl_p['afscs_to_show']) == str:
                filepath += ' (' + self.mdl_p['afscs_to_show'] + ' Cadets).png'
            else:
                filepath += ' (M = ' + str(self.b['M']) + ').png'
            self.fig.savefig(filepath)

    # Iteration functions
    def solution_iteration_frame(self, s, cadets_to_show='cadets_matched', kind=None):
        """
        This method reconstructs the figure to reflect the cadet/afsc state in this iteration
        """

        # AFSC Normalized scores
        self.b['scores'] = {j: 0 for j in self.b['J']}

        # Loop through each AFSC
        for j in self.b['J']:

            # Sort the cadets based on whatever method we choose
            unsorted_cadets = self.b[cadets_to_show][s][j]
            cadets = self.sort_cadets(j, unsorted_cadets)

            # Make sure we have cadets assigned to this AFSC in this frame
            if len(cadets) > 0:

                # Change the colors of the circles based on the desired method
                self.change_circle_features(j, cadets)

                # Hide the circles that aren't in the solution
                for i in range(len(cadets), self.b['max_assigned'][j]):

                    # Hide the circle
                    self.b['c_circles'][j][i].set_visible(False)

                # Update the text above the AFSC square
                self.update_afsc_text(s, j)

            else:  # There aren't any assigned cadets yet!
                self.b['afsc_name_text'][j].set_color('white')
                self.b['afsc_name_text'][j].set_text(self.p['afscs'][j] + ": 0")

        # Update the title of the figure
        self.update_title_text(s, kind=kind)

        # Save the figure
        if self.b['save_iteration_frames']:
            self.save_iteration_frame(s, kind=kind)

    def rejections_iteration_frame(self, s, kind='Rejections'):
        """
        This method reconstructs the figure to reflect the cadet/afsc state in this iteration
        """

        # Rejection 'Xs' lines
        line_1, line_2 = {}, {}

        # Loop through each AFSC
        for j in self.b['J']:

            # Sort the cadets based on whatever method we choose
            unsorted_cadets = self.b['cadets_proposing'][s][j]
            cadets_proposing = self.sort_cadets(j, unsorted_cadets)

            # Rejection lines
            line_1[j], line_2[j] = {}, {}
            for i, cadet in enumerate(cadets_proposing):
                if cadet not in self.b['cadets_matched'][s][j]:

                    # Get line coordinates
                    x_values_1 = [self.b['cb_coords'][j][i][0], self.b['cb_coords'][j][i][0] + self.b['s']]
                    y_values_1 = [self.b['cb_coords'][j][i][1], self.b['cb_coords'][j][i][1] + self.b['s']]
                    x_values_2 = [self.b['cb_coords'][j][i][0], self.b['cb_coords'][j][i][0] + self.b['s']]
                    y_values_2 = [self.b['cb_coords'][j][i][1] + self.b['s'], self.b['cb_coords'][j][i][1]]

                    # Plot the 'Big Red X' lines
                    line_1[j][i] = self.ax.plot(x_values_1, y_values_1, linestyle='-', c='red')
                    line_2[j][i] = self.ax.plot(x_values_2, y_values_2, linestyle='-', c='red')

        # Update the title of the figure
        self.update_title_text(s, kind=kind)

        # Save the figure
        if self.b['save_iteration_frames']:
            self.save_iteration_frame(s, kind)

        # Remove the "Big Red X" lines
        for j in self.b['J']:
            for i in line_1[j]:
                line = line_1[j][i].pop(0)
                line.remove()
                line = line_2[j][i].pop(0)
                line.remove()

    # Iteration helping functions
    def sort_cadets(self, j, cadets_unsorted):
        """
        This method sorts the cadets in this frame through some means
        """

        # Sort the cadets by AFSC preferences
        if self.mdl_p['sort_cadets_by'] == 'AFSC Preferences':
            indices = np.argsort(self.p['a_pref_matrix'][cadets_unsorted, j])
            cadets_sorted = cadets_unsorted[indices]

        # Return the sorted cadets
        return cadets_sorted

    def change_circle_features(self, j, cadets):
        """
        This method determines the color and edgecolor of the circles to show
        """

        # Colors based on cadet utility
        if self.b['focus'] == 'Cadet Utility':
            utility = self.p['utility'][cadets, j]

            # Change the cadet circles to reflect the appropriate colors
            for i, cadet in enumerate(cadets):

                # Change circle color
                color = self.v_hex_dict[round(utility[i], 2)]
                self.b['c_circles'][j][i].set_facecolor(color)

                # Show the circle
                self.b['c_circles'][j][i].set_visible(True)

        elif self.b['focus'] == 'Cadet Choice':
            choice = self.p['c_pref_matrix'][cadets, j]

            # Change the cadet circles to reflect the appropriate colors
            for i, cadet in enumerate(cadets):

                # Change circle color
                if choice[i] in self.mdl_p['choice_colors']:
                    color = self.mdl_p['choice_colors'][choice[i]]
                else:
                    color = self.mdl_p['all_other_choice_colors']
                self.b['c_circles'][j][i].set_facecolor(color)

                # Show the circle
                self.b['c_circles'][j][i].set_visible(True)

        elif self.b['focus'] == 'AFSC Choice':
            choice = self.p['a_pref_matrix'][cadets, j]

            # Change the cadet circles to reflect the appropriate colors
            for i, cadet in enumerate(cadets):

                value = round(1 - (choice[i] / self.p['num_eligible'][j]), 2)

                # Change circle color
                color = self.v_hex_dict[value]
                self.b['c_circles'][j][i].set_facecolor(color)

                # Show the circle
                self.b['c_circles'][j][i].set_visible(True)

        elif self.b['focus'] == 'ROTC Rated Interest':
            afsc_index = np.where(self.b['J'] == j)[0][0]

            # Change the cadet circles to reflect the appropriate colors
            for i, cadet in enumerate(cadets):
                idx = self.p['Rated Cadet Index Dict']['rotc'][cadet]
                interest = self.p['rr_interest_matrix'][idx, afsc_index]

                # Change circle color
                color = self.mdl_p['interest_colors'][interest]
                self.b['c_circles'][j][i].set_facecolor(color)

                # Show the circle
                self.b['c_circles'][j][i].set_visible(True)

    def update_afsc_text(self, s, j):
        """
        This method updates the text above the AFSC squares
        """

        # Set of ranks for all the cadets "considered" in this solution for this AFSC
        cadets_considered = np.intersect1d(self.b['cadets'], self.p['I^E'][j])
        ranks = self.p['a_pref_matrix'][cadets_considered, j]
        achieved_ranks = self.p['a_pref_matrix'][self.b['cadets_matched'][s][j], j]

        # Calculate AFSC Norm Score and use it in the new text
        self.b['scores'][j] = round(afccp.core.solutions.handling.calculate_afsc_norm_score_general(
            ranks, achieved_ranks), 2)

        # Change the text for the AFSCs
        if self.b['afsc_text_to_show'] == 'Norm Score':
            color = self.v_hex_dict[self.b['scores'][j]]  # New AFSC color
            afsc_text = self.p['afscs'][j] + ": " + str(self.b['scores'][j])
            self.b['afsc_name_text'][j].set_color(color)

        # Determine average cadet choice and use it in the new text
        elif self.b['afsc_text_to_show'] == 'Cadet Choice':
            average_choice = round(np.mean(self.p['c_pref_matrix'][self.b['cadets_matched'][s][j], j]), 2)
            color = 'white'
            afsc_text = self.p['afscs'][j] + ": " + str(average_choice)
            self.b['afsc_name_text'][j].set_color(color)

        # Text shows number of cadets matched/proposing
        else:
            afsc_text = self.p['afscs'][j] + ": " + str(len(cadets))

        # Update the text
        self.b['afsc_name_text'][j].set_text(afsc_text)

    def update_title_text(self, s, kind=None):
        """
        This method purely updates the text in the title of the figure
        """

        # Change the text and color of the title
        if kind == 'Proposals':
            title_text = 'Round ' + str(s + 1) + ' (Proposals)'

            # Get the color of the title
            if s + 1 in self.b['choice_colors']:
                title_color = self.b['choice_colors'][s + 1]
            else:
                title_color = self.b['all_other_choice_colors']
        else:
            title_color = self.b['text_color']

            # Update the title text in a specific way
            if kind == 'Final Solution':
                title_text = 'Final Solution'
            elif kind == 'Rejections':
                title_text = 'Round ' + str(s + 1) + ' (Rejections)'

                # Get the color of the title
                if s + 1 in self.b['choice_colors']:
                    title_color = self.b['choice_colors'][s + 1]
                else:
                    title_color = self.b['all_other_choice_colors']
            else:
                title_text = self.b['iteration_names'][s]

        # All unmatched cadets in the solution (even the ones we're not considering)
        unmatched_cadets_all = np.where(self.b['solutions'][s] == self.p['M'])[0]

        # Unmatched cadets that we're concerned about in this solution (This really just applies to Rated)
        unmatched_cadets = np.intersect1d(unmatched_cadets_all, self.b['cadets'])
        self.num_unmatched = len(unmatched_cadets)
        matched_cadets = np.array([i for i in self.b['cadets'] if i not in unmatched_cadets])

        # Calculate average cadet choice based on matched cadets
        choices = np.zeros(len(matched_cadets))
        for idx, i in enumerate(matched_cadets):
            j = self.b['solutions'][s][i]
            choices[idx] = self.p['c_pref_matrix'][i, j]
        self.average_cadet_choice = round(np.mean(choices), 2)
        title_text += ' Averages: Cadets (' + str(self.average_cadet_choice)

        # Calculate AFSC weighted average score (and add number of unmatched cadets)
        counts = np.array([len(np.where(self.b['solutions'][s] == j)[0]) for j in self.b['J']])
        weights = counts / np.sum(counts)
        scores = np.array([self.b['scores'][j] for j in self.b['J']])
        self.average_afsc_choice = round(np.dot(weights, scores), 2)
        title_text += ') AFSCs (' + str(self.average_afsc_choice) + '), ' + str(self.num_unmatched) + ' Unmatched.'

        # Update the title
        self.fig.suptitle(title_text, fontsize=self.b['b_title_size'], color=title_color)

    # Export/Save functions
    def save_iteration_frame(self, s, kind=None):
        """
        Saves the iteration frame to the appropriate folder
        """

        # Save the figure
        if self.b['save_iteration_frames']:

            # 'Sequence' Folder
            folder_path = self.paths['Analysis & Results'] + 'Cadet Board/'
            if self.b['sequence'] not in os.listdir(folder_path):
                os.mkdir(folder_path + self.b['sequence'])

            # 'Sequence Focus' Sub-folder
            sub_folder_name = self.b['focus']
            if sub_folder_name not in os.listdir(folder_path + self.b['sequence'] + '/'):
                os.mkdir(folder_path + self.b['sequence'] + '/' + sub_folder_name)
            sub_folder_path = folder_path + self.b['sequence']  + '/' + sub_folder_name + '/'
            if kind is None:
                filepath = sub_folder_path + str(s + 1) + '.png'
            elif kind == "Final Solution":
                filepath = sub_folder_path + str(s + 2) + ' (' + kind + ').png'
            else:
                filepath = sub_folder_path + str(s + 1) + ' (' + kind + ').png'

            # Save frame
            self.fig.savefig(filepath)

    def export_board_parameters(self):
        """
        This function exports the board parameters back to excel
        """
        # 'Sequence' Folder
        folder_path = self.paths['Analysis & Results'] + 'Cadet Board/'
        if self.b['sequence'] not in os.listdir(folder_path):
            os.mkdir(folder_path + self.b['sequence'])

        # Create dataframe
        df = pd.DataFrame({'J': [j for j in self.b['J']],
                           'AFSC': [self.p['afscs'][j] for j in self.b['J']],
                           'x': [self.b['x'][j] for j in self.b['J']],
                           'y': [self.b['y'][j] for j in self.b['J']],
                           'n': [self.b['n'][j] for j in self.b['J']],
                           's': [self.b['s'] for _ in self.b['J']]})

        # Export file
        filepath = folder_path + self.b['sequence'] + '/Board Parameters.csv'
        df.to_csv(filepath, index=False)

        if self.printing:
            print("Sequence parameters (J, x, y, n, s) exported to", filepath)

    def import_board_parameters(self):
        """
        This method imports the board parameters from excel if applicable
        """

        # 'Sequence' Folder
        folder_path = self.paths['Analysis & Results'] + 'Cadet Board/'
        if self.b['sequence'] in os.listdir(folder_path):

            # Import the file if we have it
            if 'Board Parameters.csv' in os.listdir(folder_path + self.b['sequence']):
                filepath = folder_path + self.b['sequence'] + '/Board Parameters.csv'
                df = afccp.core.globals.import_csv_data(filepath)

                # Load parameters
                self.b['J'] = np.array(df['J'])
                self.b['afscs'] = np.array(df['AFSC'])
                self.b['s'] = float(df.loc[0, 's'])
                for key in ['x', 'y', 'n']:
                    self.b[key] = {j: df.loc[idx, key] for idx, j in enumerate(self.b['J'])}

                if self.printing:
                    print("Sequence parameters (J, x, y, n, s) imported from", filepath)
                return True

            else:

                if self.printing:
                    print("Sequence folder '" + self.b['sequence'] + "' in 'Cadet Board' analysis sub-folder, but no "
                                                                     "board parameter file found within sequence folder.")
                return False

        else:
            if self.printing:
                print("No sequence folder '" + self.b['sequence'] + "' in 'Cadet Board' analysis sub-folder.")
            return False


# Function to determine font size of object constrained to specific box
def get_fontsize_for_text_in_box(self, txt, xy, width, height, *, transform=None,
                                 ha='center', va='center', **kwargs):
    """
    Determines fontsize of the text that needs to be inside a specific box
    """

    # Transformation
    if transform is None:
        if isinstance(self, plt.Axes):
            transform = self.transData
        if isinstance(self, plt.Figure):
            transform = self.transFigure

    # Align the x and y
    x_data = {'center': (xy[0] - width / 2, xy[0] + width / 2),
              'left': (xy[0], xy[0] + width),
              'right': (xy[0] - width, xy[0])}
    y_data = {'center': (xy[1] - height / 2, xy[1] + height / 2),
              'bottom': (xy[1], xy[1] + height),
              'top': (xy[1] - height, xy[1])}

    (x0, y0) = transform.transform((x_data[ha][0], y_data[va][0]))
    (x1, y1) = transform.transform((x_data[ha][1], y_data[va][1]))

    # Rectangle region size to constrain the text
    rect_width = x1 - x0
    rect_height = y1 - y0

    # Doing stuff
    fig = self.get_figure() if isinstance(self, plt.Axes) else self
    dpi = fig.dpi
    rect_height_inch = rect_height / dpi
    fontsize = rect_height_inch * 72

    # Put on the text
    if isinstance(self, plt.Axes):
        text = self.annotate(txt, xy, ha=ha, va=va, xycoords=transform,
                             **kwargs)

    # Adjust the fontsize according to the box size.
    text.set_fontsize(fontsize)
    bbox: Bbox = text.get_window_extent(fig.canvas.get_renderer())
    adjusted_size = fontsize * rect_width / bbox.width
    text.set_fontsize(adjusted_size)

    # Remove the text but return the font size
    text.remove()
    return text.get_fontsize()