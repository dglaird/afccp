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

# Set matplotlib default font to Times New Roman
import matplotlib as mpl
mpl.rc('font', family='Times New Roman')

class CadetBoardFigure:
    def __init__(self, instance):
        """
        This is the object to construct the "AFSC/Cadet Board" graph and animation to show where cadets get placed in
        a solution and how they move around through various algorithms. The problem instance is the only parameter
        passed. We extract various attributes of the CadetCareerProblem instance for use in this 'CadetBoardFigure'
        instance. One attribute we need in the instance is 'solution_iterations' which is a dictionary of a particular
        set of solutions used in the figure. The 'b' dictionary contains the necessary animation/plot hyperparameters
        as defined in afccp.core.data.ccp_helping_functions.py.
        """

        # Load in hex values/colors
        filepath = afccp.core.globals.paths['support'] + 'data/value_hex_translation.xlsx'
        hex_df = afccp.core.globals.import_data(filepath)
        self.v_hex_dict = {hex_df.loc[i, 'Value']: hex_df.loc[i, 'Hex'] for i in range(len(hex_df))}

        # Initialize attributes that we take directly from the CadetCareerProblem instance
        self.p, self.vp = instance.parameters, instance.value_parameters
        self.b, self.data_name, self.data_version = instance.b, instance.data_name, instance.data_version
        self.solution_iterations = instance.solution_iterations
        self.paths = instance.export_paths

        # Figure Height
        self.b['fh'] = self.b['fw'] * self.b['fh_ratio']

        # Border Widths
        for i in ['t', 'l', 'r', 'b', 'u']:
            self.b['bw^' + i] = self.b['fw'] * self.b['bw^' + i + '_ratio']

        # AFSC border/buffer widths
        self.b['abw^lr'] = self.b['fw'] * self.b['abw^lr_ratio']
        self.b['abw^ud'] = self.b['fw'] * self.b['abw^ud_ratio']

        # Legend width/height
        self.b['lw'] = self.b['fw'] * self.b['lw_ratio']
        self.b['lh'] = self.b['fw'] * self.b['lh_ratio']

        # Information from the solution iterations dictionary
        self.b['afscs'], self.b['J'] = self.solution_iterations['afscs'], self.solution_iterations['J']
        self.b['N'], self.b['M'] = self.p['N'], len(self.b['afscs'])
        self.b['solutions'] = self.solution_iterations['solutions']
        self.b['iteration_names'] = self.solution_iterations['iteration_names']
        self.b['last_s'] = self.solution_iterations['last_s']
        self.b['sequence'] = self.solution_iterations['sequence']

        # Initialize Figure
        self.fig, self.ax = plt.subplots(figsize=self.b['b_figsize'], tight_layout=True, dpi=self.b['dpi'])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set(xlim=(0, self.b['fw']))
        self.ax.set(ylim=(0, self.b['fh']))

    def main(self):
        """
        Main method to call all other methods based on what parameters the user provides
        """
        self.preprocessing()
        self.calculate_afsc_x_y_through_algorithm()
        self.calculate_cadet_box_x_y()
        self.initialize_board()

        # Create all the iteration frames
        if self.b['save_iteration_frames']:
            for s in self.b['solutions']:
                self.iteration_frame(s, self.b['focus'])

    def preprocessing(self):
        """
        This method preprocesses the different specs for this particular figure instance
        """

        # Maximum number of cadets assigned to each AFSC across solutions
        self.b['max_assigned'] = {j: 0 for j in self.b["J"]}

        # Subset of cadets assigned to the AFSC in each solution
        self.b['cadets'], self.b['counts'] = {}, {}

        # Loop through each solution (iteration)
        for s in self.b['solutions']:
            self.b['cadets'][s], self.b['counts'][s] = {}, {}

            # Loop through each AFSC
            for j in self.b['J']:
                self.b['cadets'][s][j] = np.where(self.b['solutions'][s] == j)[0]  # cadets assigned to this AFSC
                self.b['counts'][s][j] = len(self.b['cadets'][s][j])  # number of cadets assigned to this AFSC

                # Update maximum number of cadets assigned if necessary
                if self.b['counts'][s][j] > self.b['max_assigned'][j]:
                    self.b['max_assigned'][j] = self.b['counts'][s][j]

            # Get number of unassigned cadets at the end of the iterations
            if s == self.b['last_s']:
                self.b['unassigned_cadets'] = np.where(self.b['solutions'][s] == "*")[0]  # cadets left unmatched
                self.b['N^u'] = len(self.b['unassigned_cadets'])  # number of cadets left unmatched

        # Determine number of cadet boxes for AFSCs based on nearest square
        squares_required = [max(self.b['max_assigned'][j], self.p['quota_max'][j]) for j in self.b['J']]
        n = np.ceil(np.sqrt(squares_required)).astype(int)
        n2 = (np.ceil(np.sqrt(squares_required)) ** 2).astype(int)
        self.b['n'] = {j: n[idx] for idx, j in enumerate(self.b['J'])}
        self.b['n^2'] = {j: n2[idx] for idx, j in enumerate(self.b['J'])}

    def calculate_afsc_x_y_through_algorithm(self):
        """
        This method calculates the x and y locations of the AFSC boxes using a very simple algorithm.
        """

        # Number of boxes in row of unmatched box
        self.b['n^u'] = int((self.b['fw'] - self.b['bw^r'] - self.b['bw^l']) / self.b['s'])

        # Number of rows in unmatched box
        self.b['n^urow'] = int(self.b['N^u'] / self.b['n^u'])

        # Sort the AFSCs by 'n'
        n = np.array([self.b['n'][j] for j in self.b['J']])
        indices = np.argsort(n)[::-1]
        self.b['afscs'] = self.b['afscs'][indices]
        self.b['J'] = self.b['J'][indices]

        # Determine x and y coordinates of bottom left corner of AFSC squares algorithmically
        self.b['x'], self.b['y'] = {j: 0 for j in self.b['J']}, {j: 0 for j in self.b['J']}

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

    def calculate_afsc_x_y_s_through_pyomo(self):
        """
        This method calculates the x and y locations of the AFSC boxes, as well as the size (s) of the cadet boxes,
        using the pyomo optimization model to determine the optimal placement of all these objects
        """
        pass

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



            self.b['afsc_name_text'] = self.ax.text(x, y, self.p['afscs'][j], fontsize=self.b['afsc_fontsize'][j],
                                                    horizontalalignment='center', verticalalignment=va)

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

        # Save the figure
        if self.b['save_board_default']:
            folder_path = self.paths['Analysis & Results'] + 'Cadet Board/'
            if self.b['sequence'] not in os.listdir(folder_path):
                os.mkdir(folder_path + self.b['sequence'])
            filepath = folder_path + self.b['sequence'] + '/Default Board.png'
            self.fig.savefig(filepath)

    def iteration_frame(self, s, focus='Cadet Utility'):
        """
        This method reconstructs the figure to reflect the cadet/afsc state in this iteration
        """

        # Loop through each AFSC
        for j in self.b['J']:

            # List of cadets matched to this AFSC in this iteration
            cadets = self.b['cadets'][s][j]

            if focus == 'Cadet Utility':
                utility = self.p['utility'][cadets, j]
                indices = np.argsort(utility)[::-1]
                utility = utility[indices]
                cadets = cadets[indices]

                # Change the cadet circles that are in the solution
                for i, cadet in enumerate(cadets):

                    # Change circle color
                    color = self.v_hex_dict[round(utility[i], 2)]
                    self.b['c_circles'][j][i].set_facecolor(color)

                    # Show the circle
                    self.b['c_circles'][j][i].set_visible(True)

                # Hide the circles that aren't in the solution
                for i in range(len(cadets), self.b['max_assigned'][j]):

                    # Hide the circle
                    self.b['c_circles'][j][i].set_visible(False)

        # Save the figure
        if self.b['save_iteration_frames']:

            # 'Sequence' Folder
            folder_path = self.paths['Analysis & Results'] + 'Cadet Board/'
            if self.b['sequence'] not in os.listdir(folder_path):
                os.mkdir(folder_path + self.b['sequence'])

            # 'Sequence Focus' Sub-folder
            sub_folder_name = focus
            if sub_folder_name not in os.listdir(folder_path + self.b['sequence'] + '/'):
                os.mkdir(folder_path + self.b['sequence'] + '/' + sub_folder_name)
            filepath = folder_path + self.b['sequence']  + '/' + sub_folder_name + '/' + str(s + 1) + '.png'

            # Save frame
            self.fig.savefig(filepath)

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