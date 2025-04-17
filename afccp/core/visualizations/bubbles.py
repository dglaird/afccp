import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from matplotlib.transforms import Transform, Bbox
import numpy as np
import os
import copy
import pandas as pd

# Set matplotlib default font to Times New Roman
import matplotlib as mpl
mpl.rc('font', family='Times New Roman')

# afccp modules
import afccp.core.globals
import afccp.core.solutions.handling

# Import pyomo models if library is installed
if afccp.core.globals.use_pyomo:
    import afccp.core.solutions.optimization

class BubbleChart:
    def __init__(self, instance, printing=None):
        """
        Initialize an "AFSC/Cadet Bubble" chart and animation object.

        This class is designed to construct a graphical representation of the "AFSC/Cadet Bubble," showing the placement
        of cadets in a solution and their movement through various algorithms. The problem instance is the only required
        parameter.

        Args:
            instance: A CadetCareerProblem instance, containing various attributes and parameters necessary for constructing
                the bubble chart.
            printing (bool, None): A flag to control whether to print information during chart creation and animation. If set
                to True, the class will print progress and debugging information. If set to False, it will suppress printing.
                If None, the class will use the default printing setting from the instance.

        Notes:
        - This constructor extracts various attributes from the `CadetCareerProblem` instance provided as `instance`.
        - The `solution_iterations` attribute of the problem instance is expected to be a dictionary of a particular set of
          solutions used in the figure.
        - The 'b' dictionary contains hyperparameters for the animation/plot, as defined in `afccp.core.data.ccp_helping_functions.py`.

        Attributes:
        - p: A dictionary containing parameters extracted from the `instance`.
        - vp: A dictionary containing value parameters extracted from the `instance`.
        - b: A dictionary containing hyperparameters for the bubble chart, populated from the `instance`.
        - data_name: The name of the data used for the chart.
        - data_version: The version of the data used for the chart.
        - solution: The solution data extracted from the `instance`.
        - mdl_p: Model parameters extracted from the `instance`.
        - paths: Export paths from the `instance`.
        - printing: A boolean flag for controlling printing behavior during chart creation and animation.
        - v_hex_dict: A dictionary mapping value parameter values to their corresponding hexadecimal colors.
        - ...
        """

        # Initialize attributes that we take directly from the CadetCareerProblem instance
        self.p, self.vp = instance.parameters, instance.value_parameters
        self.b, self.data_name, self.data_version = instance.mdl_p, instance.data_name, instance.data_version
        self.solution, self.mdl_p = instance.solution, instance.mdl_p
        self.paths = instance.export_paths
        self.printing = printing

        # Load in hex values/colors
        filepath = afccp.core.globals.paths['files'] + 'value_hex_translation.xlsx'
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

        # Set up "solutions" properly
        if 'iterations' in self.solution:
            self.b['solutions'] = copy.deepcopy(self.solution['iterations']['matches'])
            self.b['last_s'] = self.solution['iterations']['last_s']
        else:
            self.b['solutions'] = {0: self.solution['j_array']}
            self.b['last_s'] = 0

        # Basic information about this sequence for the animation
        self.b['afscs'] = self.mdl_p['afscs']

        # Determine which cadets were solved for in this solution
        if self.b['cadets_solved_for'] is None:
            self.b['cadets_solved_for'] = self.solution['cadets_solved_for']

        # Cadets in the "denominator" basically
        if self.b['cadets_solved_for'] == 'ROTC Rated':
            self.b['cadets'] = self.p['Rated Cadets']['rotc']
            self.b['max_afsc'] = self.p['rotc_quota']
            self.b['min_afsc'] = self.p['rotc_quota']
            self.b['afscs'] = np.array([afsc for afsc in self.b['afscs'] if "_U" not in afsc])
            self.soc = "rotc"
        elif self.b['cadets_solved_for'] == 'USAFA Rated':
            self.b['cadets'] = self.p['Rated Cadets']['usafa']
            self.b['max_afsc'] = self.p['usafa_quota']
            self.b['min_afsc'] = self.p['usafa_quota']
            self.b['afscs'] = np.array([afsc for afsc in self.b['afscs'] if "_R" not in afsc])
            self.soc = 'usafa'
        else:
            self.b['cadets'] = self.p['I']
            self.b['max_afsc'] = self.p['quota_max']
            self.b['min_afsc'] = self.p['pgl']
            self.soc = 'both'

        # Correct cadet parameters
        self.b['N'] = len(self.b['cadets'])

        # Correct AFSC parameters
        self.b['M'] = len(self.b['afscs'])
        self.b['J'] = np.array([np.where(afsc == self.p['afscs'])[0][0] for afsc in self.b['afscs']])

        # These are attributes to use in the title of each iteration
        self.num_unmatched = self.b['N']
        self.average_afsc_choice = None
        self.average_cadet_choice = None

        # Initialize Figure
        self.fig, self.ax = plt.subplots(figsize=self.b['b_figsize'], dpi=self.b['dpi'],
                                         facecolor=self.b['figure_color'], tight_layout=True)
        self.ax.set_facecolor(self.b['figure_color'])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set(xlim=(-self.b['x_ext_left'], self.b['fw'] + self.b['x_ext_right']))
        self.ax.set(ylim=(-self.b['y_ext_left'], self.b['fh'] + self.b['y_ext_right']))

        # Remove tick marks
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

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

        # Redistribute the AFSCs along each row by spacing out the x coordinates
        if self.b['redistribute_x']:
            self.redistribute_x_along_row()

        # Only saving one image for a single solution
        if 'iterations' not in self.solution:
            self.b['save_iteration_frames'] = False
            self.b['build_orientation_slides'] = False
            self.b['save_board_default'] = False

        # Create the rest of the main figure
        self.calculate_cadet_box_x_y()

        # Save the board parameters
        self.export_board_parameters()

        # Build out the orientation slides
        if self.b['build_orientation_slides']:

            # Orientation slides first
            self.orientation_slides()
        else:

            # Initialize the board!
            self.initialize_board()

        # Just making one picture
        if 'iterations' not in self.solution:
            self.solution_iteration_frame(0, cadets_to_show='cadets_matched', kind='Final Solution')

            # Save frame to solution sub-folder with solution name
            filepath = self.paths['Analysis & Results'] + self.solution['name'] + '/' + self.solution['name'] + ' ' +\
                       self.b['chart_filename'] + '.png'
            self.fig.savefig(filepath)

            if self.printing:
                print('Done.')

        # Create all the iteration frames
        if self.b['save_iteration_frames']:

            # Make the "focus" directory if needed
            folder_path = self.paths['Analysis & Results'] + 'Cadet Board/' + self.solution['iterations']['sequence']
            if self.b['focus'] not in os.listdir(folder_path):
                os.mkdir(folder_path + '/' + self.b['focus'])

            # ROTC Rated Board
            if self.solution['iterations']['type'] in ['ROTC Rated Board']:

                if self.printing:
                    print("Creating " + str(len(self.solution['iterations']['matches'])) + " animation images...")

                # Loop through each solution
                for s in self.b['solutions']:
                    self.solution_iteration_frame(s, cadets_to_show='cadets_matched')

            # Matching Algorithm Proposals & Rejections
            elif self.solution['iterations']['type'] in ['HR', 'Rated SOC HR']:

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

        # Default AFSC fontsize and whether they're on two lines or not (likely overwritten later)
        self.b['afsc_fontsize'] = {j: self.b['afsc_title_size'] for j in self.b['J']}
        self.b['afsc_title_two_lines'] = {j: False for j in self.b['J']}

        # Maximum number of cadets assigned to each AFSC across solutions
        self.b['max_assigned'] = {j: 0 for j in self.b["J"]}

        # Subset of cadets assigned to the AFSC in each solution
        self.b['cadets_matched'], self.b['counts'] = {}, {}

        # Proposal iterations
        if 'iterations' in self.solution:
            if 'proposals' in self.solution['iterations']:
                self.b['cadets_proposing'] = {}

        # Loop through each solution (iteration)
        for s in self.b['solutions']:
            self.b['cadets_matched'][s], self.b['counts'][s] = {}, {}

            # Proposal iterations
            if 'iterations' in self.solution:
                if 'proposals' in self.solution['iterations']:
                    self.b['cadets_proposing'][s] = {}

            # Loop through each AFSC
            for j in self.b['J']:
                self.b['cadets_matched'][s][j] = np.where(self.b['solutions'][s] == j)[0]  # cadets assigned to this AFSC
                self.b['counts'][s][j] = len(self.b['cadets_matched'][s][j])  # number of cadets assigned to this AFSC
                max_count = self.b['counts'][s][j]

                # Proposal iterations
                if 'iterations' in self.solution:
                    if 'proposals' in self.solution['iterations']:
                        self.b['cadets_proposing'][s][j] = np.where(self.solution['iterations']['proposals'][s] == j)[0]
                        proposal_counts = len(self.b['cadets_proposing'][s][j])  # number of proposing cadets
                        max_count = max(self.b['counts'][s][j], proposal_counts)

                # Update maximum number of cadets assigned if necessary
                if max_count > self.b['max_assigned'][j]:
                    self.b['max_assigned'][j] = max_count

            # Get number of unassigned cadets at the end of the iterations
            if s == self.b['last_s']:
                self.b['unassigned_cadets'] = np.where(self.b['solutions'][s] == self.p['M'])[0]  # cadets left unmatched
                self.b['N^u'] = len(self.b['unassigned_cadets'])  # number of cadets left unmatched

        # Determine number of cadet boxes for AFSCs based on nearest square
        squares_required = [max(self.b['max_assigned'][j], self.b['max_afsc'][j]) for j in self.b['J']]
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

    def orientation_slides(self):
        """
        Build out the orientation slides for a particular sequence (intended to be used on ONE AFSC)
        """

        # Make the "orientation" directory if needed
        folder_path = self.paths['Analysis & Results'] + 'Cadet Board/' + self.solution['iterations']['sequence']
        if 'Orientation' not in os.listdir(folder_path):
            os.mkdir(folder_path + '/Orientation')

        # Save the "zero" slide (just black screen)
        filepath = folder_path + '/Orientation/0.png'
        self.fig.savefig(filepath)

        # Create first frame
        self.initialize_board(include_surplus=False)

        # Save the real first frame
        filepath = folder_path + '/Orientation/1.png'
        self.fig.savefig(filepath)

        # Reset Figure
        self.fig, self.ax = plt.subplots(figsize=self.b['b_figsize'], dpi=self.b['dpi'],
                                         facecolor=self.b['figure_color'], tight_layout=True)
        self.ax.set_facecolor(self.b['figure_color'])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set(xlim=(-self.b['x_ext_left'], self.b['fw'] + self.b['x_ext_right']))
        self.ax.set(ylim=(-self.b['y_ext_left'], self.b['fh'] + self.b['y_ext_right']))

        # Create second frame
        self.initialize_board(include_surplus=True)

        # Save the second frame
        filepath = folder_path + '/Orientation/2.png'
        self.fig.savefig(filepath)

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
        model = afccp.core.solutions.optimization.cadet_board_preprocess_model(self.b)

        # Get coordinates and size of boxes by solving the model
        self.b['s'], self.b['x'], self.b['y'] = afccp.core.solutions.optimization.solve_pyomo_model(
            self, model, "CadetBoard", q=None, printing=self.printing)

        if self.printing:
            print("Board parameters 'x' and 'y' determined through pyomo model.")

    def redistribute_x_along_row(self):
        """
        This method re-calculates the x coordinates by spacing out the AFSCs along each row
        """

        # Unique y coordinates
        y_unique = np.unique(np.array([round(self.b['y'][j], 4) for j in self.b['J']]))[::-1]

        # Need to get ordered list of AFSCs in each row
        sorted_J = np.array([j for j in self.b['J^translated']])
        rows = {row: [] for row in range(len(y_unique))}
        for j in sorted_J:
            y = round(self.b['y'][j], 4)
            row = np.where(y_unique == y)[0][0]
            rows[row].append(j)

        # Loop through each row to determine optimal spacing
        for row in rows:

            # Only adjust spacing for rows with more than one AFSC
            if len(rows[row]) > 1:

                # Calculate total spacing to play around with
                total_spacing = self.b['fw'] - self.b['bw^l'] - self.b['bw^r']
                for j in rows[row]:
                    total_spacing -= (self.b['s'] * self.b['n'][j])

                # Spacing used to fill in the gaps
                new_spacing = total_spacing / (len(rows[row]) - 1)

                # Loop through each AFSC in this row to calculate the new x position
                for num, j in enumerate(rows[row]):

                    # Calculate the appropriate x coordinate
                    if num == 0:
                        x = self.b['x'][j] + (self.b['n'][j] * self.b['s']) + new_spacing
                    else:
                        self.b['x'][j] = x
                        x += (self.b['n'][j] * self.b['s']) + new_spacing

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

    def initialize_board(self, include_surplus=True):
        """
        This method takes all the necessary board parameters and constructs the board to then be manipulated in other
        algorithms based on what the user wants to do.
        """

        # Loop through each AFSC to add certain elements
        self.b['afsc_name_text'] = {}
        self.b['c_boxes'] = {}
        self.b['c_circles'] = {}
        self.b['c_rank_text'] = {}
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
                ha = 'center'
            else:

                # AFSC fontsize is given and put AFSC name above box
                x = self.b['x'][j] + (self.b['n'][j] / 2) * self.b['s']
                y = self.b['y'][j] + self.b['n'][j] * self.b['s'] + 0.02
                va = 'bottom'
                ha = 'center'

                self.b['x'] = {key: round(val, 4) for key, val in self.b['x'].items()}
                self.b['y'] = {key: round(val, 4) for key, val in self.b['y'].items()}

                # Are we on a bottom edge?
                row = np.array([j_p for j_p, val in self.b['y'].items() if val == self.b['y'][j]])
                x_coords = np.array([self.b['x'][j_p] for j_p in row])
                if self.b['x'][j] == np.max(x_coords) and self.b['y'][j] <= 0.03:  # We're at the right edge
                    x = self.b['x'][j] + (self.b['n'][j]) * self.b['s']
                    ha = 'right'
                elif self.b['x'][j] == np.min(x_coords) and self.b['y'][j] <= 0.03:  # We're at the left edge
                    x = self.b['x'][j]
                    ha = 'left'

            # AFSC text
            self.b['afsc_name_text'][j] = self.ax.text(x, y, self.p['afscs'][j], fontsize=self.b['afsc_fontsize'][j],
                                                       horizontalalignment=ha, verticalalignment=va,
                                                       color=self.b['text_color'])

            # Cadet box text size
            cb_s = get_fontsize_for_text_in_box(self.ax, "0", (0, 0), self.b['s'], self.b['s'], va='center')

            # Loop through each cadet to add the cadet boxes and circles
            self.b['c_boxes'][j] = {}
            self.b['c_circles'][j] = {}
            self.b['c_rank_text'][j] = {}
            for i in range(self.b['n^2'][j]):  # All cadet boxes

                # If we are under the maximum number of cadets allowed
                if i + 1 <= self.b['max_afsc'][j]:

                    # Boxes based on SOC PGL Breakouts
                    if 'SOC PGL' in self.b['focus']:

                        # If we are under the USAFA PGL
                        if i + 1 <= self.p['usafa_quota'][j]:
                            linestyle = self.b['pgl_linestyle']
                            color = self.b['usafa_pgl_color']
                            alpha = self.b['pgl_alpha']

                        # We're in the ROTC range
                        elif i + 1 <= self.p['usafa_quota'][j] + self.p['rotc_quota'][j]:
                            linestyle = self.b['pgl_linestyle']
                            color = self.b['rotc_pgl_color']
                            alpha = self.b['pgl_alpha']

                        # 'Surplus' Range
                        else:
                            linestyle = self.b['surplus_linestyle']
                            color = self.b['surplus_color']
                            alpha = self.b['surplus_alpha']

                    else:

                        # If we are under the PGL
                        if i + 1 <= self.b['min_afsc'][j]:
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
                    if include_surplus or linestyle == self.b['pgl_linestyle']:
                        self.ax.add_patch(self.b['c_boxes'][j][i])

                # If we are under the maximum number of cadets assigned to this AFSC across the solutions
                if i + 1 <= self.b['max_assigned'][j]:

                    # Make the circle patch (cadet)
                    x, y = self.b['cb_coords'][j][i][0] + (self.b['s'] / 2), \
                           self.b['cb_coords'][j][i][1] + (self.b['s'] / 2)
                    self.b['c_circles'][j][i] = patches.Circle(
                        (x, y), radius = (self.b['s'] / 2) * self.b['circle_radius_percent'], linestyle='-', linewidth=1,
                        facecolor='black', alpha=1, edgecolor='black')

                    # Add the patch to the figure
                    self.ax.add_patch(self.b['c_circles'][j][i])

                    # Hide the circle
                    self.b['c_circles'][j][i].set_visible(False)

                    # We may want to include rank text on the cadets
                    if self.b['show_rank_text']:
                        self.b['c_rank_text'][j][i] = self.ax.text(x, y, '0', fontsize=cb_s, horizontalalignment='center',
                                                                   verticalalignment='center',
                                                                   color=self.b['rank_text_color'])
                        self.b['c_rank_text'][j][i].set_visible(False)


        # Remove tick marks
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Add the title
        if self.b['b_title'] is None:
            title = "Round 0 (Orientation)"
        else:
            title = self.b['b_title']
        self.fig.suptitle(title, fontsize=self.b['b_title_size'], color=self.b['text_color'])

        # Add the legend if necessary
        if self.b['b_legend']:
            self.create_legend()

        # Save the figure
        if self.b['save_board_default']:
            folder_path = self.paths['Analysis & Results'] + 'Cadet Board/'
            if self.solution['iterations']['sequence'] not in os.listdir(folder_path):
                os.mkdir(folder_path + self.b['sequence'])

            # Get the filepath and save the "default" graph
            filepath = folder_path + self.solution['iterations']['sequence'] + '/Default Board'
            if type(self.mdl_p['afscs_to_show']) == str:
                filepath += ' (' + self.mdl_p['afscs_to_show'] + ' Cadets).png'
            else:
                filepath += ' (M = ' + str(self.b['M']) + ').png'
            self.fig.savefig(filepath)

    def create_legend(self):

        # Initialize legend elements
        legend_elements = []
        legend_title = self.b['focus']
        self.b['chart_filename'] = self.b['focus']  # Filename for chart
        if self.b['focus'] in ['Cadet Choice', 'Specific Choice', 'Tier 1']:

            # Add legend elements
            for c in np.arange(1, 11):
                legend_elements.append(
                    Line2D([0], [0], marker='o', label=str(c), markerfacecolor=self.mdl_p['choice_colors'][c],
                           markersize=self.mdl_p['b_legend_marker_size'], color='black', markeredgecolor='black'))
            legend_elements.append(
                Line2D([0], [0], marker='o', label='11+', markerfacecolor=self.mdl_p['all_other_choice_colors'],
                       markersize=self.mdl_p['b_legend_marker_size'], color='black', markeredgecolor='black'))

            if self.b['focus'] in ['Specific Choice', 'Tier 1']:

                # Update filename and legend title
                self.b['chart_filename'] = self.mdl_p['afsc'] + " " + self.b['focus']
                if "Specific Choice" in self.b['focus']:
                    legend_title = "Cadet Choice For " + self.mdl_p['afsc']
                else:
                    legend_title = "Cadet Choice For Their Assigned AFSC"

                # Get the AFSC we're highlighting
                j_focus = np.where(self.p['afscs'] == self.mdl_p['afsc'])[0][0]

                # Add "Unqualified" legend element if we have enough people ineligible for this AFSC
                if len(self.p['I^E'][j_focus]) <= (self.p['N'] - 20):  # Might be handful of cadets on PRP or something
                    legend_elements.append(
                        Line2D([0], [0], marker='o', label=self.mdl_p['afsc'] + "\nUnqualified",
                               markerfacecolor=self.mdl_p['unfocused_color'],
                               markersize=self.mdl_p['b_legend_marker_size'], color='black', markeredgecolor='black'))

        elif self.b['focus'] == 'Cadet Choice Categories':

            # Add legend elements
            for c in np.arange(1, 7):
                legend_elements.append(
                    Line2D([0], [0], marker='o', label=str(c), markerfacecolor=self.mdl_p['choice_colors'][c],
                           markersize=self.mdl_p['b_legend_marker_size'], color='black', markeredgecolor='black'))
            legend_elements.append(
                Line2D([0], [0], marker='o', label='Selected',
                       markerfacecolor=self.mdl_p['choice_colors'][8],
                       markersize=self.mdl_p['b_legend_marker_size'], color='black', markeredgecolor='black'))
            legend_elements.append(
                Line2D([0], [0], marker='o', label='Not Selected',
                       markerfacecolor=self.mdl_p['choice_colors'][9],
                       markersize=self.mdl_p['b_legend_marker_size'], color='black', markeredgecolor='black'))
            legend_elements.append(
                Line2D([0], [0], marker='o', label='Last 3', markerfacecolor=self.mdl_p['all_other_choice_colors'],
                       markersize=self.mdl_p['b_legend_marker_size'], color='black', markeredgecolor='black'))

        elif self.b['focus'] == 'ROTC Rated Interest':

            # Add legend elements
            for level in self.mdl_p['interest_colors']:
                legend_elements.append(
                    Line2D([0], [0], marker='o', label=level, markerfacecolor=self.mdl_p['interest_colors'][level],
                           markersize=self.mdl_p['b_legend_marker_size'], color='black', markeredgecolor='black'))

        elif self.b['focus'] == 'Rated Choice':

            # Add legend elements
            for c in np.arange(1, len(self.b['J']) + 1):
                legend_elements.append(
                    Line2D([0], [0], marker='o', label=str(c), markerfacecolor=self.mdl_p['choice_colors'][c],
                           markersize=self.mdl_p['b_legend_marker_size'], color='black', markeredgecolor='black'))

        elif self.b['focus'] == 'Reserves':

            # Add legend elements
            legend_elements = [
                Line2D([0], [0], marker='o', label="Matched", markerfacecolor=self.mdl_p['matched_slot_color'],
                       markersize=self.mdl_p['b_legend_marker_size'], color='black', markeredgecolor='black'),
                Line2D([0], [0], marker='o', label="Reserved", markerfacecolor=self.mdl_p['reserved_slot_color'],
                       markersize=self.mdl_p['b_legend_marker_size'], color='black', markeredgecolor='black')]

        elif self.b['focus'] == 'SOC PGL':

            # Add legend elements
            legend_elements = [
                Line2D([0], [0], marker='o', label="USAFA", markerfacecolor=self.b['usafa_bubble'],
                       markersize=self.mdl_p['b_legend_marker_size'], color='black', markeredgecolor='black'),
                Line2D([0], [0], marker='o', label="ROTC", markerfacecolor=self.b['rotc_bubble'],
                       markersize=self.mdl_p['b_legend_marker_size'], color='black', markeredgecolor='black')]

        # Create legend
        leg = self.ax.legend(handles=legend_elements, edgecolor='white', loc=self.b['b_legend_loc'], facecolor='black',
                             fontsize=self.mdl_p['b_legend_size'], ncol=len(legend_elements), labelspacing=1,
                             handlelength=0.8, handletextpad=0.2, borderpad=0.2, handleheight=2,
                             title=legend_title)

        # Change title color in legend
        title = leg.get_title()
        title.set_color('white'), title.set_fontsize(self.mdl_p['b_legend_title_size'])

        # Change label colors in legend
        for text in leg.get_texts():
            text.set_color("white")

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
                self.change_circle_features(s, j, cadets)

                # Hide the circles/text that aren't in the solution
                for i in range(len(cadets), self.b['max_assigned'][j]):

                    # Hide the circle
                    self.b['c_circles'][j][i].set_visible(False)

                    # If rank text is included
                    if self.b['show_rank_text']:
                        self.b['c_rank_text'][j][i].set_visible(False)

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

        # Sort the cadets by SOC
        if self.b['focus'] == 'SOC PGL':
            indices = np.argsort(self.p['usafa'][cadets_unsorted])[::-1]

        # Sort the cadets by AFSC preferences
        elif self.mdl_p['sort_cadets_by'] == 'AFSC Preferences':
            indices = np.argsort(self.p['a_pref_matrix'][cadets_unsorted, j])

        # Return the sorted cadets
        return cadets_unsorted[indices]

    def change_circle_features(self, s, j, cadets):
        """
        This method determines the color and edgecolor of the circles to show
        """

        # Colors based on cadet utility
        if self.b['focus'] == 'Cadet Utility':
            utility = self.p['cadet_utility'][cadets, j]

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

        elif self.b['focus'] == 'Cadet Choice Categories':
            choice = self.p['c_pref_matrix'][cadets, j]

            # Change the cadet circles to reflect the appropriate colors
            for i, cadet in enumerate(cadets):

                # If the AFSC was a top 6 choice, we use that color
                if choice[i] in [1, 2, 3, 4, 5, 6]:
                    color = self.mdl_p['choice_colors'][choice[i]]

                # If the AFSC was at least selected, we make it that color
                elif j in self.p['J^Selected'][cadet]:

                    # Use the color for the 8th choice
                    color = self.mdl_p['choice_colors'][8]

                # If the AFSC was not in the bottom 3 choices, we make it that color
                elif j not in self.p['J^Bottom 2 Choices'][cadet] and j != self.p['J^Last Choice'][cadet]:

                    # Use the color for the 9th choice
                    color = self.mdl_p['choice_colors'][9]

                # Otherwise, it's a bottom 3 choice
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

        elif self.b['focus'] == 'Reserves':

            # Change the cadet circles to reflect the appropriate colors
            for i, cadet in enumerate(cadets):
                if cadet in self.solution['iterations']['matched'][s]:
                    color = self.b['matched_slot_color']
                elif cadet in self.solution['iterations']['reserves'][s]:
                    color = self.b['reserved_slot_color']
                else:
                    color = self.b['unmatched_color']

                # Change circle color
                self.b['c_circles'][j][i].set_facecolor(color)

                # Show the circle
                self.b['c_circles'][j][i].set_visible(True)

        elif self.b['focus'] == 'SOC PGL':

            # Change the cadet circles to reflect the appropriate colors
            for i, cadet in enumerate(cadets):
                if cadet in self.p['usafa_cadets']:
                    color = self.b['usafa_bubble']
                else:
                    color = self.b['rotc_bubble']

                # Change circle color
                self.b['c_circles'][j][i].set_facecolor(color)

                # Show the circle
                self.b['c_circles'][j][i].set_visible(True)

        elif self.b['focus'] == 'Rated Choice':

            # Change the cadet circles to reflect the appropriate colors
            for i, cadet in enumerate(cadets):
                rated_choices = self.p['Rated Choices'][self.soc][cadet]

                # Get color of this choice
                if j in rated_choices:
                    choice = np.where(rated_choices == j)[0][0] + 1
                else:
                    choice = 100  # Arbitrary big number

                # Change circle color
                if choice in self.mdl_p['choice_colors']:
                    color = self.mdl_p['choice_colors'][choice]
                else:
                    color = self.mdl_p['all_other_choice_colors']
                self.b['c_circles'][j][i].set_facecolor(color)

                # Show the circle
                self.b['c_circles'][j][i].set_visible(True)

        elif 'Specific Choice' in self.b['focus']:

            # Get the AFSC we're highlighting
            j_focus = np.where(self.p['afscs'] == self.mdl_p['afsc'])[0][0]
            choice = self.p['c_pref_matrix'][cadets, j_focus]

            # Change the cadet circles to reflect the appropriate colors
            for i, cadet in enumerate(cadets):

                # Change circle color
                if choice[i] in self.mdl_p['choice_colors']:
                    color = self.mdl_p['choice_colors'][choice[i]]
                elif choice[i] == 0:  # Ineligible
                    color = self.mdl_p['unfocused_color']
                else:  # All other choices
                    color = self.mdl_p['all_other_choice_colors']
                self.b['c_circles'][j][i].set_facecolor(color)

                # Show the circle
                self.b['c_circles'][j][i].set_visible(True)

        elif 'Tier 1' in self.b['focus']:
            choice = self.p['c_pref_matrix'][cadets, j]

            # Get the AFSC we're highlighting
            j_focus = np.where(self.p['afscs'] == self.mdl_p['afsc'])[0][0]

            # Change the cadet circles to reflect the appropriate colors
            for i, cadet in enumerate(cadets):

                # Change circle color
                if '1' in self.p['qual'][cadet, j_focus]:
                    if choice[i] in self.mdl_p['choice_colors']:
                        color = self.mdl_p['choice_colors'][choice[i]]
                    else:
                        color = self.mdl_p['all_other_choice_colors']
                else:
                    color = self.mdl_p['unfocused_color']
                self.b['c_circles'][j][i].set_facecolor(color)

                # # Edgecolor (Don't worry about exception anymore)
                # if 'E' in self.p['qual'][cadet, j_focus]:# and j == j_focus:
                #     self.b['c_circles'][j][i].set_edgecolor(self.mdl_p['exception_edge'])
                # else:
                #     self.b['c_circles'][j][i].set_edgecolor(self.mdl_p['base_edge'])

                # Show the circle
                self.b['c_circles'][j][i].set_visible(True)

        # Cadet rank text
        if self.b['show_rank_text']:
            choice = self.p['a_pref_matrix'][cadets, j]
            for i, cadet in enumerate(cadets):
                txt = str(choice[i])
                x, y = self.b['cb_coords'][j][i][0] + (self.b['s'] / 2), \
                       self.b['cb_coords'][j][i][1] + (self.b['s'] / 2)
                w, h = self.b['s'] * self.b['circle_radius_percent'], self.b['s'] * self.b['circle_radius_percent']
                fontsize = get_fontsize_for_text_in_box(self.ax, txt, (x, y), w, h, va='center')

                # Adjust fontsize for single digit ranks
                if int(txt) < 10:
                    fontsize = int(fontsize * self.b['fontsize_single_digit_adj'])
                self.b['c_rank_text'][j][i].set_text(txt)
                self.b['c_rank_text'][j][i].set_fontsize(fontsize)
                self.b['c_rank_text'][j][i].set_visible(True)

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

        # If we want to put this AFSC title on two lines or not
        if self.b['afsc_title_two_lines'][j]:
            afsc_text = self.p['afscs'][j] + ":\n"
        else:
            afsc_text = self.p['afscs'][j] + ": "

        # Change the text for the AFSCs
        if self.b['focus'] == 'SOC PGL':
            more = 'neither'
            for soc, other_soc in {'usafa': 'rotc', 'rotc': 'usafa'}.items():
                soc_cadets = len(np.intersect1d(self.b['cadets_matched'][s][j], self.p[soc + '_cadets']))
                soc_pgl = self.p[soc + '_quota'][j]
                diff = soc_cadets - soc_pgl
                if diff > 0:
                    more = soc
                    afsc_text += '+' + str(diff)

            if more == 'neither':
                color = 'white'
                afsc_text += '+0'
            else:
                color = self.b[more + '_bubble']
            self.b['afsc_name_text'][j].set_color(color)


        elif self.b['afsc_text_to_show'] == 'Norm Score':
            color = self.v_hex_dict[self.b['scores'][j]]  # New AFSC color
            afsc_text += str(self.b['scores'][j])
            self.b['afsc_name_text'][j].set_color(color)

        # Determine average cadet choice and use it in the new text
        elif self.b['afsc_text_to_show'] == 'Cadet Choice':
            average_choice = round(np.mean(self.p['c_pref_matrix'][self.b['cadets_matched'][s][j], j]), 2)
            color = 'white'
            afsc_text += str(average_choice)
            self.b['afsc_name_text'][j].set_color(color)

        # Text shows number of cadets matched/proposing
        else:
            afsc_text += str(len(self.b['cadets_matched'][s][j]))

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
                title_text = 'Solution'
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

        # Calculate AFSC weighted average score (and add number of unmatched cadets)
        counts = np.array([len(np.where(self.b['solutions'][s] == j)[0]) for j in self.b['J']])
        weights = counts / np.sum(counts)
        scores = np.array([self.b['scores'][j] for j in self.b['J']])
        self.average_afsc_choice = round(np.dot(weights, scores), 2)

        # Add title text
        if self.b['focus'] in ['Specific Choice', 'Tier 1']:
            title_text += ' Highlighting Results for ' + self.mdl_p['afsc']
        else:
            percent_text = str(np.around(self.solution['top_3_choice_percent'] * 100, 3)) + "%"
            title_text += ' Results: Cadet Top3: ' + percent_text
            title_text += ', AFSC Score: ' + str(np.around(self.average_afsc_choice, 2))

        # Update the title
        if self.b['b_title'] is not None:  # We specified a title directly
            title_text = self.b['b_title']
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
            if self.solution['iterations']['sequence'] not in os.listdir(folder_path):
                os.mkdir(folder_path + self.solution['iterations']['sequence'])

            # 'Sequence Focus' Sub-folder
            sub_folder_name = self.b['focus']
            if sub_folder_name not in os.listdir(folder_path + self.solution['iterations']['sequence'] + '/'):
                os.mkdir(folder_path + self.solution['iterations']['sequence'] + '/' + sub_folder_name)
            sub_folder_path = folder_path + self.solution['iterations']['sequence']  + '/' + sub_folder_name + '/'
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

        if 'iterations' not in self.solution:

            # Solutions Folder
            filepath = self.paths['Analysis & Results'] + self.solution['name'] + '/Board Parameters.csv'
            if self.solution['name'] not in os.listdir(self.paths['Analysis & Results']):
                os.mkdir(self.paths['Analysis & Results'] + self.solution['name'] + '/')
        else:

            # 'Sequence' Folder
            folder_path = self.paths['Analysis & Results'] + 'Cadet Board/'
            filepath = folder_path + self.solution['iterations']['sequence'] + '/Board Parameters.csv'
            if self.solution['iterations']['sequence'] not in os.listdir(folder_path):
                os.mkdir(folder_path + self.solution['iterations']['sequence'])

        # Create dataframe
        df = pd.DataFrame({'J': [j for j in self.b['J']],
                           'AFSC': [self.p['afscs'][j] for j in self.b['J']],
                           'x': [self.b['x'][j] for j in self.b['J']],
                           'y': [self.b['y'][j] for j in self.b['J']],
                           'n': [self.b['n'][j] for j in self.b['J']],
                           's': [self.b['s'] for _ in self.b['J']],
                           'afsc_fontsize': [self.b['afsc_fontsize'][j] for j in self.b['J']],
                           'afsc_title_two_lines': [self.b['afsc_title_two_lines'][j] for j in self.b['J']]})

        # Export file
        df.to_csv(filepath, index=False)

        if self.printing:
            print("Sequence parameters (J, x, y, n, s) exported to", filepath)

    def import_board_parameters(self):
        """
        This method imports the board parameters from excel if applicable
        """

        # 'Solutions' Folder
        if 'iterations' not in self.solution:
            folder_path = self.paths['Analysis & Results'] + self.solution['name']

            # Import the file if we have it
            if 'Board Parameters.csv' in os.listdir(folder_path):
                filepath = folder_path + '/Board Parameters.csv'
                df = afccp.core.globals.import_csv_data(filepath)

                # Load parameters
                self.b['J'] = np.array(df['J'])
                self.b['afscs'] = np.array(df['AFSC'])
                self.b['s'] = float(df.loc[0, 's'])
                for key in ['x', 'y', 'n', 'afsc_fontsize', 'afsc_title_two_lines']:
                    self.b[key] = {j: df.loc[idx, key] for idx, j in enumerate(self.b['J'])}

                if self.printing:
                    print("Sequence parameters (J, x, y, n, s) imported from", filepath)
                return True

            else:

                if self.printing:
                    print("No Sequence parameters found in solution analysis sub-folder '" +
                          self.solution['name'] + "'.")
                return False


        # 'Sequence' Folder
        folder_path = self.paths['Analysis & Results'] + 'Cadet Board/'
        if self.solution['iterations']['sequence'] in os.listdir(folder_path):

            # Import the file if we have it
            if 'Board Parameters.csv' in os.listdir(folder_path + self.solution['iterations']['sequence']):
                filepath = folder_path + self.solution['iterations']['sequence'] + '/Board Parameters.csv'
                df = afccp.core.globals.import_csv_data(filepath)

                # Load parameters
                self.b['J'] = np.array(df['J'])
                self.b['afscs'] = np.array(df['AFSC'])
                self.b['s'] = float(df.loc[0, 's'])
                for key in ['x', 'y', 'n', 'afsc_fontsize', 'afsc_title_two_lines']:
                    self.b[key] = {j: df.loc[idx, key] for idx, j in enumerate(self.b['J'])}

                if self.printing:
                    print("Sequence parameters (J, x, y, n, s) imported from", filepath)
                return True

            else:

                if self.printing:
                    print("Sequence folder '" + self.solution['iterations']['sequence'] + "' in 'Cadet Board' analysis sub-folder, but no "
                                                                     "board parameter file found within sequence folder.")
                return False

        else:
            if self.printing:
                print("No sequence folder '" + self.solution['iterations']['sequence'] + "' in 'Cadet Board' analysis sub-folder.")
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