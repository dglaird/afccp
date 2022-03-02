# Import Libraries
import tkinter as tk
from tkinter import font as tkFont
import tkinter
from tkinter import messagebox
from tkinter import *
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename, asksaveasfilename
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from problem_class import *

# The GUI global object
global root


# Useful functions for placing widgets
def xy(x, y, window_width=None, window_height=None):
    if window_width is None:
        window_width = root.window_width
        window_height = root.window_height
    x = (x / 1440) * window_width
    y = (1 - (y / 750)) * window_height
    return x, y


def _font(font_size, window_height=None):
    if window_height is None:
        window_height = root.window_height
    return int(font_size * (window_height / 750))


def _x(w):
    return w * (root.window_width / 1440)


def _y(h):
    return h * (root.window_height / 750)


def create_background_image(frame, image):
    screen = Image.open(image)

    # resize the image to be the size of the screen
    resize_image = screen.resize((root.window_width, root.window_height))
    background = ImageTk.PhotoImage(resize_image)

    # create canvas to add background image to
    canvas = tk.Canvas(frame, width=root.window_width, height=root.window_height)
    canvas.pack()

    # Keep a reference in case this code is put in a function.
    canvas.background = background
    canvas.create_image(0, 0, anchor=tk.NW, image=background)

    return canvas


# Custom widget classes
class HoverButton(tk.Button):
    def __init__(self, master, text, width, height, action, font_size, initial_state=NORMAL, **kwargs):
        tk.Button.__init__(self, master=master, text=text, width=width, height=height,
                           image=tk.PhotoImage(width=1, height=1), font=(root.button_font, font_size),
                           fg=root.default_fg, bg=root.default_bg, cursor="hand2", compound='c', relief=GROOVE,
                           state=initial_state, **kwargs)
        self.default_bg = root.default_bg
        self.changed_bg = root.active_bg
        self.default_fg = root.default_fg
        self.changed_fg = root.active_fg
        self.action = action
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)

    def on_enter(self, e):
        self.config(bg=self.changed_bg, fg=self.changed_fg, image=tk.PhotoImage(width=1, height=1), compound='c')

    def on_leave(self, e):
        self.config(bg=self.default_bg, fg=self.default_fg, image=tk.PhotoImage(width=1, height=1), compound='c')

    def on_click(self, e):
        global root

        if self.action == 'Data':
            root.show_page(Data)
        elif self.action == 'Main':
            root.show_page(Main)
        elif self.action == 'Title':
            root.show_page(Title)
        elif self.action == 'EditWeights':
            root.show_page(EditWeights)
        elif self.action == 'Objectives':
            root.show_page(Objectives)
        elif self.action == 'Results':
            root.show_page(Results)
        elif self.action == 'ResultsChart':
            root.show_page(ResultsChart)
        elif self.action == 'DataCharts':
            root.show_page(DataChart)
        elif self.action == 'Constraints':
            root.show_page(Constraints)
        elif self.action == 'Sensitivity':
            root.show_page(Sensitivity)
        elif self.action == 'Import Data':
            root.pages[Data].import_data()
        elif self.action == 'Simulate Random':
            root.pages[Data].Simulate_Random_Data()
        elif self.action == 'Simulate Realistic':
            root.pages[Data].Simulate_Realistic_Data()
        elif self.action == 'Simulate Perfect':
            root.pages[Data].Simulate_Perfect_Data()
        elif self.action == 'Reset':
            root.pages[Main].Reset_Problem()
        elif self.action == 'Import Weights':
            root.pages[Main].import_value_parameters()
        elif self.action == 'Default Weights':
            root.pages[Main].Default_Value_Parameters()
        elif self.action == 'Export':
            root.pages[Main].Export_To_Excel()
        elif self.action == 'EditWeights Update':
            root.pages[EditWeights].Update_Weights()
        elif self.action == 'Update Data Chart':
            root.pages[DataChart].Update_Chart()
        elif self.action == 'Objectives Update':
            root.pages[Objectives].Update_Objectives()
        elif self.action == 'Export Defaults':
            filename = asksaveasfilename()
            model_value_parameters_to_defaults(root.instance.parameters, root.instance.value_parameters,
                                               filepath=filename)
        elif self.action == 'Solve':
            root.pages[Main].Solve_Button_Click()
        elif self.action == 'Update Constraints':
            root.pages[Constraints].Update_AFSC_Constraints()
        elif self.action == 'ChangeChart_L':
            root.pages[DataChart].Change_Chart('Left')
        elif self.action == 'ChangeChart_R':
            root.pages[DataChart].Change_Chart('Right')
        elif self.action == 'ChangeResultsChart_L':
            root.pages[ResultsChart].Change_Chart('Left')
        elif self.action == 'ChangeResultsChart_R':
            root.pages[ResultsChart].Change_Chart('Right')
        elif self.action == "Compare Solutions":
            if root.sensitivity_chart_page == 'Measures':
                root.show_page(CompareMeasures)
            elif root.sensitivity_chart_page == 'Objective Values':
                root.show_page(CompareObjectiveValues)
            elif root.sensitivity_chart_page == 'AFSC Values':
                root.show_page(CompareAFSCValues)
        elif self.action == "Compare Values":
            if root.values_chart_page == 'Objectives':
                root.sensitivity_chart_page = 'Objective Values'
                root.show_page(CompareObjectiveValues)
            else:
                root.sensitivity_chart_page = 'AFSC Values'
                root.show_page(CompareAFSCValues)
        elif self.action == 'By Objective':
            root.sensitivity_chart_page = 'Objective Values'
            root.values_chart_page = 'Objectives'
            root.show_page(CompareObjectiveValues)
        elif self.action == 'By AFSC':
            root.sensitivity_chart_page = 'AFSC Values'
            root.values_chart_page = 'AFSCs'
            root.show_page(CompareAFSCValues)
        elif self.action == "Compare Measures":
            root.sensitivity_chart_page = 'Measures'
            root.show_page(CompareMeasures)
        elif self.action == "Compare Weights":
            root.show_page(CompareWeights)
        elif self.action == "Change Solution":
            root.pages[Sensitivity].Change_Solution()
        elif self.action == "Change Values":
            root.pages[Sensitivity].Change_Weights()
        elif self.action == "LSP":
            root.pages[Sensitivity].LSP_clicked()
        elif self.action == "Pareto":
            root.pages[Sensitivity].Pareto_clicked()
        elif self.action == "Con_All_None":
            root.pages[Sensitivity].Constraint_Analysis('all_to_none')
        elif self.action == "Con_None_All":
            root.pages[Sensitivity].Constraint_Analysis('none_to_all')
        elif self.action == "Export Report":
            root.pages[Sensitivity].Export_Report()
        elif self.action == 'ChangeMeasuresChart_L':
            root.pages[CompareMeasures].Change_Chart('Left')
        elif self.action == 'ChangeMeasuresChart_R':
            root.pages[CompareMeasures].Change_Chart('Right')

    def update_state(self, state):
        self.config(state=state)


class HoverEntry(tk.Entry):
    def __init__(self, master, font_size, width, justify=LEFT, **kwargs):
        # font_size = int(font_size * (root.window_width / 1440))
        tk.Entry.__init__(self, master=master, bg=root.default_bg, fg=root.default_fg,
                          font=(root.entry_font, font_size), width=width, justify=justify, **kwargs)
        self.default_bg = root.default_bg
        self.changed_bg = root.active_bg
        self.default_fg = root.default_fg
        self.changed_fg = root.active_fg
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.config(bg=self.changed_bg, fg=self.changed_fg)

    def on_leave(self, e):
        self.config(bg=self.default_bg, fg=self.default_fg)


# Class for the entire GUI itself (main window)
class GUI(tk.Tk):

    # __init__ function for the GUI class
    def __init__(self, *args, **kwargs):

        # __init__ function for the Tk class
        tk.Tk.__init__(self, *args, **kwargs)

        # Create the container- all page frames are contained here
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand="True")

        # Screen specifications
        scaled_dimensions = 30  # standard scaled-size is 30!
        self.window_width = 48 * scaled_dimensions
        self.window_height = 25 * scaled_dimensions
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.center_x = int(self.screen_width / 2 - self.window_width / 2)
        self.center_y = int(self.screen_height / 2 - self.window_height / 2) - 50
        # self.resizable(False, False)  # Disable window resizing

        # set the position of the window in the center of the screen
        self.geometry(f'{self.window_width}x{self.window_height}+{self.center_x}+{self.center_y}')

        # Set title
        self.title("USAF Career Field Assignment Model")  # title of window

        # Specify container stuff
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Initialize pages
        self.pages = {}

        # Initialize attributes that are constant
        self.title_font = "Bahnschrift"
        self.label_font = "Copperplate"
        self.button_font = "Impact"
        self.bg_color = '#97c2e9'
        self.entry_font = "Calibri"
        # self.graph_color = "#103A83"
        self.graph_color = self.bg_color
        self.active_bg = "#6fa7d9"
        self.active_fg = 'black'
        self.default_bg = 'black'
        self.default_fg = 'white'
        self.chart_title_size = _font(20, self.window_height)
        self.chart_label_size = _font(20, self.window_height)
        self.chart_legend_size = _font(20, self.window_height)
        self.chart_xtick_size = _font(15, self.window_height)
        self.chart_ytick_size = _font(15, self.window_height)
        self.chart_afsc_size = _font(15, self.window_height)
        self.chart_afsc_rotation = None
        self.chart_figsize = (_font(14, self.window_height), _font(6, self.window_height))
        # self.chart_color = '#8a9fb5'
        self.chart_color = self.bg_color
        self.chart_coords = xy(20, 75, self.window_width, self.window_height)

        # Initialize attributes that must be reset
        self.perfect = False
        self.current_solution_name = ""
        self.current_vp_name = ""
        self.vp_dict = {}
        self.z_dict = {}
        self.metrics_dict = {}
        self.instance = None
        self.chart_num = 1
        self.current_selected_solutions_dict = {}
        self.current_selected_vp_dict = {}
        self.solution_dict = {}
        self.sensitivity_chart_page = 'Measures'
        self.values_chart_page = 'Objectives'

    def Init_Title_Data(self):

        # Reset attributes
        self.perfect = False
        self.current_solution_name = ""
        self.current_vp_name = ""
        self.vp_dict = {}
        self.z_dict = {}
        self.metrics_dict = {}
        self.solution_dict = {}
        self.instance = None
        self.chart_num = 1
        self.current_selected_solutions_dict = {}
        self.current_selected_vp_dict = {}
        self.sensitivity_chart_page = 'Measures'
        self.values_chart_page = 'Objectives'

        # Loop through data page layouts
        for F in (Title, Data):
            # initialize the classes for each page, attach them to container
            page = F(self.container)
            self.pages[F] = page  # add the page to a dictionary
            page.grid(row=0, column=0, sticky="nsew")  # expand the page to fill the container

        self.show_page(Title)  # set Title screen to be the initial screen

    def Init_Main_Frame(self):
        """
        Initializes the "Main" page after data has been imported. This is the main page the user will be navigating
        :return: None.
        """

        # Loop through all data parameter pages
        for F2 in (Main, DataChart):  # These pages depend on the fixed parameters

            # initialize the classes for each page, attach them to container
            page = F2(self.container)
            self.pages[F2] = page  # add the page to a dictionary
            page.grid(row=0, column=0, sticky="nsew")  # expand the page to fill the container

        self.show_page(Main)  # set Main screen to be the initial screen

        # Change continue button
        self.pages[Title].continue_button = HoverButton(self.pages[Title].canvas, text='Continue Matching',
                                                        width=260, height=_y(50), action='Main', font_size=_font(24))

        self.pages[Title].canvas.create_window(xy(1420, 20), anchor=tk.SE, window=self.pages[Title].continue_button)

    def Init_Parameter_Frames(self):
        """
        Initializes the "Edit Weights" and "Objectives" pages after the initial value parameters have been specified.
        :param container: the container that the pages are attached to
        :return: None.
        """
        # Loop through all parameter pages
        for F2 in (EditWeights, Objectives, Constraints):  # These pages depend on the value parameters

            # initialize the classes for each page, attach them to container
            page = F2(self.container)
            self.pages[F2] = page  # add the page to a dictionary
            page.grid(row=0, column=0, sticky="nsew")  # expand the page to fill the container

        self.show_page(Main)  # set Main screen to be the initial screen

    def show_page(self, cont):
        """
        Shows the page
        :param cont: Page name
        :return: None.
        """
        # takes the page name as argument and displays that page
        page = self.pages[cont]
        page.tkraise()

    def Init_Results_Frame(self):
        """
        Initializes the "Results" page after model has been solved.
        :return: None.
        """

        # Loop through results page layouts
        for F in (Results, ResultsChart, Sensitivity, CompareMeasures, CompareObjectiveValues, CompareAFSCValues,
                  CompareWeights):

            if F in (Sensitivity, CompareMeasures, CompareWeights, CompareObjectiveValues, CompareAFSCValues):
                if len(self.vp_dict.keys()) == 1 and len(self.z_dict.keys()) == 1:

                    # initialize the classes for each page, attach them to container
                    page = F(self.container)
                    self.pages[F] = page  # adds the page to a dictionary
                    page.grid(row=0, column=0, sticky="nsew")  # expands the page to fill the container
                else:
                    if F == Sensitivity:
                        self.pages[F].Update_Screen()

            else:
                # initialize the classes for each page, attach them to container
                page = F(self.container)
                self.pages[F] = page  # adds the page to a dictionary
                page.grid(row=0, column=0, sticky="nsew")  # expands the page to fill the container

        self.show_page(Main)  # set Main screen to be the initial screen

    def New_Results(self):

        num_parameter_sets = len(self.vp_dict.keys())
        if num_parameter_sets == 0:
            vp_name = "P" + str(num_parameter_sets + 1)
            self.vp_dict[vp_name] = copy.deepcopy(self.instance.value_parameters)
            solution_vp_name = vp_name
            self.current_vp_name = vp_name
        else:
            new = True  # We initially assume this is a new set of value parameters

            # Loop through each distinct set of value parameters
            for vp_name in self.vp_dict.keys():

                # We initially assume this set of value parameters is the same as the new set of value parameters
                same = True

                # Loop through each value parameter
                for key in self.instance.value_parameters.keys():

                    # We don't care about changes in the constraints since those don't affect how we measure a solution
                    if key.find('min') < 0 and key.find('constraint') < 0 and key not in ['I_C', 'J_C', 'L', 'r',
                                                                                          'K_A', 'F_v']:

                        # If we are dealing with a dictionary as the parameter (sets of objectives/AFSCs)
                        if type(self.instance.value_parameters[key]) is dict:

                            # Loop through each index to see if the contents are the same
                            sub_same = True
                            for sub_key in self.instance.value_parameters[key].keys():
                                if np.all(np.ravel(self.vp_dict[vp_name][key][sub_key]) !=
                                          np.ravel(self.instance.value_parameters[key][sub_key])):
                                    sub_same = False
                                    break

                            if not sub_same:
                                same = False
                                break
                        elif key == 'F_bp':

                            # If we're dealing with a breakpoint thing, we gotta check each individually
                            sub_same = True
                            for j in self.instance.parameters['J']:
                                for k in self.instance.value_parameters['K_A'][j]:
                                    for l in self.instance.value_parameters['L'][j][k]:

                                        try:
                                            if self.instance.value_parameters[key][j][k][l] != \
                                                    self.vp_dict[vp_name][key][j][k][l]:
                                                sub_same = False
                                                break

                                        # index out of range error means the set of breakpoints is different
                                        except:
                                            sub_same = False
                                            break

                                    if not sub_same:
                                        break
                                if not sub_same:
                                    break
                            if not sub_same:
                                same = False
                                break

                        else:

                            # If this value parameter for this set is not the same as that of the new vp set...
                            check_key = np.ravel(self.vp_dict[vp_name][key])
                            current_key = np.ravel(self.instance.value_parameters[key])
                            if np.all(check_key != current_key):
                                same = False
                                break

                # If we managed to get through all the checks and the two sets are in fact the same, this new set
                # is not actually new
                if same:
                    new = False
                    solution_vp_name = vp_name
                    self.current_vp_name = vp_name
                    break

            # If this set of value parameters does not match any other set...
            if new:
                new_vp_name = "P" + str(num_parameter_sets + 1)
                self.vp_dict[new_vp_name] = copy.deepcopy(self.instance.value_parameters)
                solution_vp_name = new_vp_name
                self.current_vp_name = new_vp_name

        # Check if the solution is new
        if len(self.solution_dict.keys()) == 0:
            self.current_solution_name += "_1"
            self.solution_dict[self.current_solution_name] = self.instance.solution
            self.metrics_dict[self.current_solution_name] = {}
            self.metrics_dict[self.current_solution_name][solution_vp_name] = copy.deepcopy(self.instance.metrics)
            self.z_dict[self.current_solution_name] = {}
            self.z_dict[self.current_solution_name][solution_vp_name] = self.instance.metrics['z']
        else:
            same = False
            for solution_name in self.z_dict.keys():

                p_i = compare_solutions(self.instance.solution, self.solution_dict[solution_name])
                if p_i == 1:
                    self.current_solution_name = solution_name
                    same = True
                    break

            if not same:
                self.current_solution_name += "_" + str(len(self.z_dict.keys()) + 1)
                self.solution_dict[self.current_solution_name] = self.instance.solution
                self.metrics_dict[self.current_solution_name] = {}
                self.metrics_dict[self.current_solution_name][solution_vp_name] = copy.deepcopy(self.instance.metrics)
                self.z_dict[self.current_solution_name] = {}
                self.z_dict[self.current_solution_name][solution_vp_name] = self.instance.metrics['z']

            for solution_name in self.z_dict.keys():
                for vp, vp_name in enumerate(list(self.vp_dict.keys())):
                    if vp_name not in self.metrics_dict[solution_name].keys():
                        self.metrics_dict[solution_name][vp_name] = copy.deepcopy(
                            measure_solution_quality(self.solution_dict[solution_name],
                                                     self.instance.parameters, self.vp_dict[vp_name]))

                        self.z_dict[solution_name][vp_name] = self.metrics_dict[solution_name][vp_name]['z']

        self.Init_Results_Frame()


# Title Page
class Title(tk.Frame):

    # Initialize title screen class
    def __init__(self, parent):

        # Initialize tk frame class
        tk.Frame.__init__(self, parent)
        self.canvas = create_background_image(self, "GUI_Background_Title.jpg")

        # initialize button
        self.begin_button = HoverButton(self.canvas, text='Begin Matching', width=_x(220), height=_y(50),
                                        action='Data', font_size=_font(24))
        self.canvas.create_window(xy(1420, 20), anchor=tk.SE, window=self.begin_button)


# Data Generation Page
class Data(tk.Frame):

    # Initialize data screen class
    def __init__(self, parent):

        # Initialize tk frame class
        tk.Frame.__init__(self, parent)
        self.canvas = create_background_image(self, "GUI_Background_Data.png")

        # Place buttons. This is more complicated than it needs to be, but I thought it would be cleaner... I was wrong
        b_names = ["Back", "Import", "Simulate Random", "Simulate Realistic", "Simulate Perfect"]
        params_dict = {'action': {"Back": "Title", "Import": "Import Data"},
                       'coords': {"Back": (20, 20), "Import": (360, 512.5), "Simulate Random": (1050, 462.5),
                                  "Simulate Realistic": (1110, 462.5), "Simulate Perfect": (1080, 300)},
                       'anchor': {"Back": tk.SW, "Import": tk.N, "Simulate Random": tk.NE, "Simulate Realistic": tk.NW,
                                  "Simulate Perfect": tk.CENTER}, 'height': {'Back': 40}, 'font': {'Back': 20},
                       'width': {'Back': 70, 'Import': 100}}
        param_defaults = {'width': 270, 'height': 50, 'font': 24}
        b_dict = {}
        for b, b_name in enumerate(b_names):
            button_params = {}
            for param in params_dict:
                if b_name in params_dict[param]:
                    button_params[param] = params_dict[param][b_name]
                else:
                    if param == 'action':
                        button_params[param] = b_name
                    else:
                        button_params[param] = param_defaults[param]
            b_dict[b_name] = HoverButton(self.canvas, text=b_name, width=_x(button_params['width']),
                                         height=_y(button_params['height']), action=button_params['action'],
                                         font_size=_font(button_params['font']))
            self.canvas.create_window(xy(button_params['coords'][0], button_params['coords'][1]),
                                      anchor=button_params['anchor'], window=b_dict[b_name])

        # Title
        self.canvas.create_text(xy(720, 700), text="New Problem Instance",
                                font=root.title_font + " " + str(_font(50)))

        # Import Data
        self.canvas.create_text(xy(360, 562.5), text="Import Data",
                                font=root.label_font + " " + str(_font(36)))
        # Simulate Data
        self.canvas.create_text(xy(1080, 562.5), text="Simulate Data",
                                font=root.label_font + " " + str(_font(36)))

        # N
        self.canvas.create_text(xy(880, 502.5), text="N", font=root.entry_font + " " + str(_font(24)))
        self.N_entry = HoverEntry(master=self.canvas, font_size=_font(24), width=4, justify=LEFT)
        self.N_entry.insert(tk.END, 1600)
        self.canvas.create_window(xy(900, 502.5), anchor=tk.W, window=self.N_entry)

        # M
        self.canvas.create_text(xy(1070, 502.5), text="M", anchor=tk.E,
                                font=root.entry_font + " " + str(_font(24)))
        self.M_entry = HoverEntry(master=self.canvas, font_size=_font(24), width=2, justify=LEFT)
        self.M_entry.insert(tk.END, 32)
        self.canvas.create_window(xy(1080, 502.5), anchor=tk.W, window=self.M_entry)

        # P
        self.canvas.create_text(xy(1210, 502.5), text="P", font=root.entry_font + " " + str(_font(24)))
        self.P_entry = HoverEntry(master=self.canvas, font_size=_font(24), width=2, justify=LEFT)
        self.P_entry.insert(tk.END, 6)
        self.canvas.create_window(xy(1230, 502.5), anchor=tk.W, window=self.P_entry)

        # Warning text
        self.warning_text = self.canvas.create_text(xy(1080, 385), fill='red', text=" ", anchor=tk.CENTER,
                                                    font=root.label_font + " " + str(_font(24)))

    @staticmethod
    def Import_Data():
        """
        Imports the data
        :return: None.
        """
        global root
        filepath = askopenfilename()  # asks user for file name

        try:
            root.instance = CadetCareerProblem(filepath=filepath)
            root.Init_Main_Frame()
        except:
            tkinter.messagebox.showerror(message="Please Choose a valid Cadet/AFSC dataset.")

    def Simulate_Random_Data(self):
        """
        Simulate random date using specified parameters
        :return: None
        """
        global root

        try:
            if int(self.N_entry.get()) > 0 and int(self.M_entry.get()) > 0 and int(self.P_entry.get()) > 0 and \
                    (int(self.M_entry.get()) > int(self.P_entry.get())):

                root.instance = CadetCareerProblem('Random', N=int(self.N_entry.get()), M=int(self.M_entry.get()),
                                                   P=int(self.P_entry.get()))
                root.Init_Main_Frame()
            else:
                self.canvas.itemconfig(self.warning_text, text="Select valid 'random' simulation parameters: \n"
                                                           "                  N > 0, M > P > 0")
        except:
            self.canvas.itemconfig(self.warning_text, text="Input integers for the simulation parameters")

    def Simulate_Realistic_Data(self):
        """
        Simulate realistic data using specified number of cadets
        :return: None
        """
        global root
        try:
            if int(self.N_entry.get()) > 100 and int(self.M_entry.get()) == 32 and int(self.P_entry.get()) == 6:

                root.instance = CadetCareerProblem("Realistic", N=int(self.N_entry.get()))
                root.Init_Main_Frame()
            else:
                self.canvas.itemconfig(self.warning_text, text="Select valid 'realistic' simulation parameters: \n"
                                                               "                  N > 100, M = 32, P = 6")
        except:
            self.canvas.itemconfig(self.warning_text, text="Input integers for the simulation parameters")

    def Simulate_Perfect_Data(self):
        """
        Simulate "perfect" data using specified number of cadets
        :return: None.
        """

        global root

        try:
            if (int(self.N_entry.get()) % 4 == 0) and (int(self.N_entry.get()) % int(self.M_entry.get()) == 0) and \
                    int(self.M_entry.get()) >= 2 and (int(self.P_entry.get()) <= int(self.M_entry.get())) and \
                    int(self.N_entry.get()) >= 8:
                root.perfect = True
                root.instance = CadetCareerProblem('Perfect', N=int(self.N_entry.get()), M=int(self.M_entry.get()),
                                                   P=int(self.P_entry.get()))
                root.Init_Main_Frame()
            else:
                self.canvas.itemconfig(self.warning_text, text="Select valid 'perfect' simulation parameters: \n"
                                                           "N is divisible by 4 and M, M >= 2, N >= 8, M >= P")
        except:
            self.canvas.itemconfig(self.warning_text, text="Input integers for the simulation parameters")


# Main Page
class Main(tk.Frame):

    # Initialize main screen class
    def __init__(self, parent):

        # Initialize tk frame class
        tk.Frame.__init__(self, parent)

        self.canvas = create_background_image(self, "GUI_Background_Main.jpg")

        # Back Button
        back_button = HoverButton(self.canvas, text='Back', width=_x(70), height=_y(40), action='Title',
                                  font_size=_font(20))
        self.canvas.create_window(xy(20, 20), anchor=tk.SW, window=back_button)

        # More Button
        more_button = HoverButton(self.canvas, text='More', width=_x(70), height=_y(40), action='DataCharts',
                                  font_size=_font(20))
        self.canvas.create_window(xy(1420, 20), anchor=tk.SE, window=more_button)

        # Title
        self.canvas.create_text(xy(720, 700), text="New Problem Instance",
                                font=root.title_font + " " + str(_font(40)))

        # Reset Problem
        self.canvas.create_text(xy(130, 615), text="Reset Problem", font=root.label_font + " " + str(_font(24)))
        self.reset_button = HoverButton(self.canvas, text='Reset', width=_x(95), height=_y(40),
                                        action='Reset', font_size=_font(20))
        self.canvas.create_window(xy(130, 560), anchor=tk.CENTER, window=self.reset_button)

        # Specify Value Parameters
        self.value_parameter_text = self.canvas.create_text(xy(460, 615), text="Specify Value Parameters",
                                                            font=root.label_font + " " + str(_font(24)))

        # Import Button
        self.import_button = HoverButton(self.canvas, text='Import', width=_x(95), height=_y(40),
                                         action='Import Weights', font_size=_font(20))
        self.canvas.create_window(xy(283, 560), anchor=tk.CENTER, window=self.import_button)

        # Default Button
        self.default_button = HoverButton(self.canvas, text='Default', width=_x(95), height=_y(40),
                                          action='Default Weights', font_size=_font(20))
        self.canvas.create_window(xy(403, 560), anchor=tk.CENTER, window=self.default_button)

        # Export Button
        self.export_text = self.canvas.create_text(xy(1310, 615), text="Export to Excel",
                                                   font=root.label_font + " " + str(_font(24)))
        self.export_button = HoverButton(self.canvas, text='Export', width=_x(95), height=_y(40),
                                         action='Export', font_size=_font(20))
        self.canvas.create_window(xy(1310, 560), anchor=tk.CENTER, window=self.export_button)

        # Warning text
        self.user_text1 = self.canvas.create_text(xy(460, 530), fill='red', text=" ", anchor=tk.CENTER,
                                                  font=root.label_font + " " + str(_font(20)))

        chart = root.instance.display_data_graph(graph='Average Utility', facecolor='black', alpha=1,
                                                 title='Average Utility Across All AFSCs',
                                                 figsize=(_font(20), _font(4)), dpi=_font(60), label_size=_font(18),
                                                 afsc_tick_size=_font(15), yaxis_tick_size=_font(15))

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.graph_canvas = FigureCanvasTkAgg(chart, master=self.canvas)
        self.graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(xy(720, 20), anchor=tk.S, window=self.graph_canvas.get_tk_widget())

        # Data Statistics
        statistic_titles = ['Number of Cadets:', 'Number of AFSCs:', 'Number of Preferences:', 'USAFA Proportion:',
                            '2nd Choice Average Utility:', '3rd Choice Average Utility:', '4th Choice Average Utility:',
                            '5th Choice Average Utility:', '6th Choice Average Utility:']

        preferences, utilities_array = get_utility_preferences(root.instance.parameters)

        choice_utils = np.zeros(5)
        for p in [1, 2, 3, 4, 5]:
            if p < root.instance.parameters['P']:
                choice_utils[p - 1] = np.mean(utilities_array[:, p])

        statistics = {'Number of Cadets:': root.instance.parameters['N'],
                      'Number of AFSCs:': root.instance.parameters['M'],
                      'Number of Preferences:': root.instance.parameters['P'],
                      'USAFA Proportion:': round(sum(root.instance.parameters['usafa']) /
                                                 root.instance.parameters['N'], 2)}
        for p, stat in enumerate(statistic_titles[4:9]):
            if choice_utils[p] != 0:
                statistics[stat] = round(choice_utils[p], 2)
            else:
                statistics[stat] = "N/A"

        start_x = 110
        start_y = 450
        pivot_x = start_x
        pivot_y = start_y
        for i, statistic in enumerate(statistic_titles):
            self.canvas.create_text(xy(pivot_x, pivot_y), text=statistic, anchor=tk.W,
                                    font=root.label_font + " " + str(_font(18)))

            self.canvas.create_text(xy(pivot_x + 300, pivot_y), text=statistics[statistic], anchor=tk.W,
                                    font=root.label_font + " " + str(_font(18)))

            pivot_y -= 80

            if (i + 1) % 3 == 0:
                pivot_y = start_y
                pivot_x += 420

    def Reset_Problem(self):
        root.Init_Title_Data()

    def Import_Value_Parameters(self):
        global root
        filepath = askopenfilename()  # asks user for file name

        try:
            root.instance.import_value_parameters(filepath)

            self.Init_Value_Parameters()
        except:
            tkinter.messagebox.showerror(message="Please Choose a valid parameter file.")

    def Default_Value_Parameters(self):
        global root

        root.instance.import_default_value_parameters()
        self.Init_Value_Parameters()

    def Export_To_Excel(self):
        filepath = asksaveasfilename()
        try:
            root.instance.export_to_excel(filepath)
        except:
            tkinter.messagebox.showerror(message="Something went wrong.")

    def Init_Value_Parameters(self):

        # Initialize value parameter pages
        root.Init_Parameter_Frames()

        # Alter User Text on Main Screen
        self.canvas.itemconfig(self.value_parameter_text, text="Edit Value Parameters")
        self.edit_button = HoverButton(self.canvas, text='Edit', width=_x(95), height=_y(40), action='EditWeights',
                                       font_size=_font(20))
        self.canvas.create_window(xy(523, 560), anchor=tk.CENTER, window=self.edit_button)
        self.constraints_button = HoverButton(self.canvas, text='Constraints', width=_x(95), height=_y(40),
                                              action='Constraints', font_size=_font(16))
        self.canvas.create_window(xy(643, 560), anchor=tk.CENTER, window=self.constraints_button)
        self.solve_model_text = self.canvas.create_text(xy(980, 615), text="Solve the Model",
                                                        font=root.label_font + " " + str(_font(24)))

        # Solver variables
        self.solver_var = tk.StringVar()
        self.solver_options = ['Import', 'VFT', 'Stable', 'Greedy', 'Random', 'Original', 'Genetic']
        self.solver_var.set(self.solver_options[0])
        self.solver_drop = tk.OptionMenu(self, self.solver_var, *self.solver_options)
        self.solver_drop.config(font=tkFont.Font(family=root.button_font, size=_font(20)), bg='black',
                                fg='white', activebackground=root.active_bg, activeforeground='black', width=6,
                                relief=GROOVE, cursor='hand2')
        self.canvas.create_window(xy(797, 560), anchor=tk.CENTER, window=self.solver_drop)

        self.solve_button = HoverButton(self.canvas, text='Solve', width=_x(95), height=_y(40), action='Solve',
                                        font_size=_font(20))
        self.canvas.create_window(xy(917, 560), anchor=tk.CENTER, window=self.solve_button)

    def Solve_Button_Click(self):
        global root

        if self.solver_var.get() == 'Stable':
            root.instance.stable_matching()
        elif self.solver_var.get() == 'Greedy':
            root.instance.greedy_method()
        elif self.solver_var.get() == 'Random':
            root.instance.generate_random_solution()
        elif self.solver_var.get() == 'VFT':
            root.instance.full_vft_model_solve(printing=True)
        elif self.solver_var.get() == 'Original':
            root.instance.solve_original_pyomo_model()
        elif self.solver_var.get() == 'Genetic':
            root.instance.genetic_algorithm(initialize=True, printing=True)
        elif self.solver_var.get() == 'Import':
            filename = askopenfilename()
            root.instance.import_solution(filename)
        else:
            root.instance.generate_random_solution()

        # Change Text
        self.canvas.itemconfig(self.solve_model_text, text="View Solver Results")

        # Add buttons
        self.results_button = HoverButton(self.canvas, text='Results', width=_x(95), height=_y(40), action='Results',
                                          font_size=_font(20))
        self.canvas.create_window(xy(1037, 560), anchor=tk.CENTER, window=self.results_button)
        self.sensitivity_button = HoverButton(self.canvas, text='Sensitivity', width=_x(95), height=_y(40),
                                              action='Sensitivity', font_size=_font(16))
        self.canvas.create_window(xy(1157, 560), anchor=tk.CENTER, window=self.sensitivity_button)

        root.current_solution_name = self.solver_var.get()
        root.New_Results()


# Data Chart Page
class DataChart(tk.Frame):

    # Initialize data chart screen class
    def __init__(self, parent):

        # Initialize tk frame class
        tk.Frame.__init__(self, parent)
        self.canvas = create_background_image(self, "GUI_Background_Main.jpg")

        # Back Button
        back_button = HoverButton(self.canvas, text='Back', width=_x(70), height=_y(40), action='Main',
                                  font_size=_font(20))
        self.canvas.create_window(xy(20, 20), anchor=tk.SW, window=back_button)

        # Title
        self.canvas.create_text(xy(720, 708), text="Dataset Charts", font=root.title_font + " " + str(_font(40)))

        # Arrow Buttons
        root.chart_num = 1  # this variable indicates which chart to show based on what arrow was clicked

        self.left_arrow = HoverButton(self.canvas, text='<-', width=_x(36), height=_y(40), action='ChangeChart_L',
                                      font_size=_font(20))
        self.canvas.create_window(xy(20, 690), anchor=tk.SW, window=self.left_arrow)

        self.right_arrow = HoverButton(self.canvas, text='->', width=_x(36), height=_y(40), action='ChangeChart_R',
                                       font_size=_font(20))
        self.canvas.create_window(xy(1420, 690), anchor=tk.SE, window=self.right_arrow)

        # Data chart
        chart = root.instance.display_data_graph(
            figsize=root.chart_figsize, facecolor=root.chart_color, graph='AFOCD Data', alpha=1,
            afsc_tick_size=root.chart_afsc_size, yaxis_tick_size=root.chart_ytick_size,
            legend_size=root.chart_legend_size, title_size=root.chart_title_size,
            label_size=root.chart_label_size, eligibility=True)

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.graph_canvas = FigureCanvasTkAgg(chart, master=self.canvas)
        self.graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(root.chart_coords, anchor=tk.SW, window=self.graph_canvas.get_tk_widget())

        # Chart Parameters
        self.num_cadet_entry = HoverEntry(master=self.canvas, font_size=_font(20), width=4, justify=CENTER)
        self.num_cadet_entry.insert(tk.END, root.instance.parameters['N'])
        self.canvas.create_window(xy(735, 20), anchor=tk.SW, window=self.num_cadet_entry)

        # Num cadet text
        self.canvas.create_text(xy(120, 20), text="Max Number of Eligible Cadets for AFSCs", anchor=tk.SW,
                                font=root.label_font + " " + str(_font(24)))

        # Update Chart button
        self.update_button = HoverButton(self.canvas, text='Update Chart', width=_x(200), height=_y(40),
                                         action='Update Data Chart', font_size=_font(20))
        self.canvas.create_window(xy(1080, 20), anchor=tk.S, window=self.update_button)

    def Change_Chart(self, direction):
        """
        Picks the chart to the right or left
        :param direction: right or left
        :return: None
        """

        global root

        if direction == 'Left':
            root.chart_num -= 1
        else:
            root.chart_num += 1

        if root.chart_num < 1:
            root.chart_num = 5
        elif root.chart_num > 5:
            root.chart_num = 1

        self.Update_Chart()

    def Update_Chart(self):
        """
        Updates the data chart to be shown
        :return: None
        """
        num = int(self.num_cadet_entry.get())
        charts = ['AFOCD Data', 'Average Merit', 'USAFA Proportion', 'Eligible Quota', 'Average Utility']
        chart = root.instance.display_data_graph(
            figsize=root.chart_figsize, facecolor=root.chart_color, graph=charts[root.chart_num - 1],
            afsc_tick_size=root.chart_afsc_size, num=num, eligibility=True, yaxis_tick_size=root.chart_ytick_size,
            legend_size=root.chart_legend_size, title_size=root.chart_title_size, label_size=root.chart_label_size)

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.graph_canvas = FigureCanvasTkAgg(chart, master=self.canvas)
        self.graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(root.chart_coords, anchor=tk.SW,
                                  window=self.graph_canvas.get_tk_widget())


# Edit Weights Page
class EditWeights(tk.Frame):

    # Initialize title screen class
    def __init__(self, parent):
        global root

        # Initialize tk frame class
        tk.Frame.__init__(self, parent)
        self.bg_color = root.bg_color
        self.canvas = create_background_image(self, "GUI_Background_Weights.png")

        # Back Button
        back_button = HoverButton(self.canvas, text='Back', width=_x(70), height=_y(40), action='Main',
                                  font_size=_font(20))
        self.canvas.create_window(xy(20, 20), anchor=tk.SW, window=back_button)

        # Title
        self.canvas.create_text(xy(720, 700), text="Edit Parameters", font=root.title_font + " " + str(_font(40)))

        # Cadet/AFSC Labels
        self.canvas.create_text(xy(240, 700), text="Cadets", font=root.label_font + " " + str(_font(30)))
        self.canvas.create_text(xy(1200, 700), text="AFSCs", font=root.label_font + " " + str(_font(30)))

        # Adjust Overall Weights
        self.canvas.create_text(xy(720, 630), text="Adjust Weights", font=root.label_font + " " + str(_font(24)))

        # Cadets
        self.cadet_weight_entry = HoverEntry(master=self.canvas, font_size=_font(24), width=4, justify=CENTER)
        self.cadet_weight_entry.insert(tk.END, root.instance.value_parameters['cadets_overall_weight'])
        self.canvas.create_window(xy(690, 580), anchor=tk.E, window=self.cadet_weight_entry)

        # AFSCs
        self.afsc_weight_entry = HoverEntry(master=self.canvas, font_size=_font(24), width=4, justify=CENTER)
        self.afsc_weight_entry.insert(tk.END, root.instance.value_parameters['afscs_overall_weight'])
        self.canvas.create_window(xy(750, 580), anchor=tk.W, window=self.afsc_weight_entry)

        # Adjust Weight Functions
        self.canvas.create_text(xy(720, 530), text="Adjust Functions", font=root.label_font + " " + str(_font(24)))

        # Cadet Weight Function
        self.c_function_var = tk.StringVar()
        self.c_function_options = ['Linear', 'Equal']
        self.c_function_var.set(self.c_function_options[0])
        self.c_function_drop = tk.OptionMenu(self, self.c_function_var, *self.c_function_options)
        self.c_function_drop.config(font=tkFont.Font(family=root.button_font, size=_font(18)), bg='black',
                                    fg='white', activebackground=root.active_bg, activeforeground='black', width=5,
                                    relief=GROOVE, cursor='hand2')
        self.canvas.create_window(xy(690, 480), anchor=tk.E, window=self.c_function_drop)

        # AFSC Weight Function
        self.afsc_function_var = tk.StringVar()
        self.afsc_function_options = ['Custom', 'Size', 'Piece', 'Equal']
        self.afsc_function_var.set(self.afsc_function_options[0])
        self.afsc_function_drop = tk.OptionMenu(self, self.afsc_function_var, *self.afsc_function_options)
        self.afsc_function_drop.config(font=tkFont.Font(family=root.button_font, size=_font(18)),
                                       bg='black', fg='white', activebackground=root.active_bg,
                                       activeforeground='black', width=5, relief=GROOVE, cursor='hand2')
        self.canvas.create_window(xy(750, 480), anchor=tk.W, window=self.afsc_function_drop)

        # Update Button
        self.update_button = HoverButton(self.canvas, text='Update Weights', width=_x(200), height=_y(40),
                                         action='EditWeights Update', font_size=_font(20))
        self.canvas.create_window(xy(720, 380), anchor=tk.S, window=self.update_button)

        # Cadet Weight Graph
        self.cadet_weight_graph = root.instance.display_weight_function(display_title=False,
                                                                        cadets=True, figsize=(_font(8), _font(5.5)),
                                                                        dpi=_font(55), facecolor=self.bg_color,
                                                                        gui_chart=True,
                                                                        label_size=_font(20))

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.cadet_weight_graph_canvas = FigureCanvasTkAgg(self.cadet_weight_graph, master=self.canvas)
        self.cadet_weight_graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(xy(262, 680), anchor=tk.N, window=self.cadet_weight_graph_canvas.get_tk_widget())

        # AFSC Weight Graph
        self.afsc_weight_graph = root.instance.display_weight_function(display_title=False,
                                                                       cadets=False, figsize=(_font(8), _font(5.5)),
                                                                       dpi=_font(55), facecolor=self.bg_color,
                                                                       gui_chart=True,
                                                                       label_size=_font(20), afsc_tick_size=_font(15))

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.afsc_weight_graph_canvas = FigureCanvasTkAgg(self.afsc_weight_graph, master=self.canvas)
        self.afsc_weight_graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(xy(1178, 680), anchor=tk.N, window=self.afsc_weight_graph_canvas.get_tk_widget())

        # arrays of tkinter objects
        self.afsc_weight_text = {}
        self.afsc_swing_weight_entries = {}

        # the top weighted AFSC is the "reference" and has a swing weight of 100
        reference = np.max(root.instance.value_parameters['afsc_weight'])

        # the swing weights are whatever percentage of the reference value they are
        root.afsc_swing_weight = np.round((root.instance.value_parameters['afsc_weight'] / reference) * 100, 1)

        start_y = 300
        start_x = 30

        # Loop through all AFSCs to place their weight widgets
        for j, afsc in enumerate(root.instance.parameters['afsc_vector']):
            if 0 <= j <= 9:
                pivot_x = start_x + 138 * j
                pivot_y = start_y
            elif 10 <= j <= 19:
                pivot_x = start_x + 138 * (j - 10)
                pivot_y = start_y - 75
            elif 20 <= j <= 29:
                pivot_x = start_x + 138 * (j - 20)
                pivot_y = start_y - 75 * 2
            else:
                pivot_x = start_x + 138 * (j - 29)
                pivot_y = start_y - 75 * 3

            # AFSC Name
            self.canvas.create_text(xy(pivot_x + 32, pivot_y), text=afsc, font=root.label_font + " " + str(_font(18)))

            # Local Weight Text
            self.afsc_weight_text[j] = self.canvas.create_text(
                xy(pivot_x + 96, pivot_y + 20), text=round(root.instance.value_parameters['afsc_weight'][j], ndigits=3),
                font=root.label_font + " " + str(_font(16)))

            # Swing Weight Entry
            self.afsc_swing_weight_entries[j] = HoverEntry(master=self.canvas, font_size=_font(16), width=5,
                                                           justify=CENTER)
            if root.afsc_swing_weight[j] == 100.0:
                root.afsc_swing_weight[j] = int(100)
            self.afsc_swing_weight_entries[j].insert(tk.END, root.afsc_swing_weight[j])
            self.canvas.create_window(xy(pivot_x + 96, pivot_y - 20), anchor=tk.CENTER,
                                      window=self.afsc_swing_weight_entries[j])

        # Adjust Objectives Button
        self.adjust_objectives_button = HoverButton(self.canvas, text='Adjust Objectives', width=_x(212), height=_y(40),
                                                    action='Objectives', font_size=_font(20))
        self.canvas.create_window(xy(1420, 20), anchor=tk.SE, window=self.adjust_objectives_button)

        # Export as Defaults Button
        self.default_button = HoverButton(self.canvas, text='Export as Defaults', width=_x(262), height=_y(40),
                                          action='Export Defaults', font_size=_font(20))
        self.canvas.create_window(xy(720, 1344), anchor=tk.CENTER, window=self.default_button)

    def Update_Weights(self):
        global root

        # Update AFSC Weight Chart
        if self.afsc_function_var.get() == 'Size':

            # Update Local Weights
            root.instance.value_parameters['afsc_weight'] = root.instance.parameters['quota'] / \
                                                            sum(root.instance.parameters['quota'])
            root.instance.value_parameters['afsc_weight_function'] = 'Linear'

            # Update Swing Weights
            reference = np.max(root.instance.value_parameters['afsc_weight'])
            root.afsc_swing_weight = np.round((root.instance.value_parameters['afsc_weight'] / reference) * 100, 1)

            # Update Swing Weight Entries and Local Weight Text
            for j in range(root.instance.parameters['M']):
                self.afsc_swing_weight_entries[j].insert(tk.END, '1')
                self.afsc_swing_weight_entries[j].delete(first=0, last=tk.END)
                self.afsc_swing_weight_entries[j].insert(tk.END, root.afsc_swing_weight[j])
                self.canvas.itemconfig(self.afsc_weight_text[j], text=round(
                    root.instance.value_parameters['afsc_weight'][j], ndigits=3))

        elif self.afsc_function_var.get() == 'Equal':

            # Update Local Weights
            root.instance.value_parameters['afsc_weight'] = np.repeat(1 / root.instance.parameters['M'],
                                                                      root.instance.parameters['M'])

            # Update Swing Weights
            reference = np.max(root.instance.value_parameters['afsc_weight'])
            root.afsc_swing_weight = np.round((root.instance.value_parameters['afsc_weight'] / reference) * 100, 1)
            root.instance.value_parameters['afsc_weight_function'] = 'Equal'

            # Update Swing Weight Entries and Local Weight Text
            for j in range(root.instance.parameters['M']):
                self.afsc_swing_weight_entries[j].insert(tk.END, '1')
                self.afsc_swing_weight_entries[j].delete(first=0, last=tk.END)
                self.afsc_swing_weight_entries[j].insert(tk.END, root.afsc_swing_weight[j])
                self.canvas.itemconfig(self.afsc_weight_text[j], text=round(
                    root.instance.value_parameters['afsc_weight'][j], ndigits=3))

        elif self.afsc_function_var.get() == 'Piece':

            # Generate AFSC weights
            swing_weights = np.zeros(root.instance.parameters['M'])
            for j, quota in enumerate(root.instance.parameters['quota']):
                if quota >= 200:
                    swing_weights[j] = 1
                elif 150 <= quota < 200:
                    swing_weights[j] = 0.9
                elif 100 <= quota < 150:
                    swing_weights[j] = 0.8
                elif 50 <= quota < 100:
                    swing_weights[j] = 0.7
                elif 25 <= quota < 50:
                    swing_weights[j] = 0.6
                else:
                    swing_weights[j] = 0.5

            # Update Local Weights
            root.instance.value_parameters['afsc_weight'] = np.around(swing_weights / sum(swing_weights), 4)

            # Update Swing Weights
            reference = np.max(root.instance.value_parameters['afsc_weight'])
            root.afsc_swing_weight = np.round((root.instance.value_parameters['afsc_weight'] / reference) * 100, 1)
            root.instance.value_parameters['afsc_weight_function'] = 'Equal'

            # Update Swing Weight Entries and Local Weight Text
            for j in range(root.instance.parameters['M']):
                self.afsc_swing_weight_entries[j].insert(tk.END, '1')
                self.afsc_swing_weight_entries[j].delete(first=0, last=tk.END)
                self.afsc_swing_weight_entries[j].insert(tk.END, root.afsc_swing_weight[j])
                self.canvas.itemconfig(self.afsc_weight_text[j], text=round(
                    root.instance.value_parameters['afsc_weight'][j], ndigits=3))

        else:

            # Update Swing Weights
            for j in range(root.instance.parameters['M']):
                root.afsc_swing_weight[j] = self.afsc_swing_weight_entries[j].get()

            # Update Local Weights
            root.instance.value_parameters['afsc_weight'] = root.afsc_swing_weight / \
                                                            sum(root.afsc_swing_weight)
            root.instance.value_parameters['afsc_weight_function'] = 'Custom'

            # Update Local Weight Text
            for j in range(root.instance.parameters['M']):
                self.canvas.itemconfig(self.afsc_weight_text[j], text=round(
                    root.instance.value_parameters['afsc_weight'][j], ndigits=3))

        # AFSC Weight Graph
        self.afsc_weight_graph = root.instance.display_weight_function(
            cadets=False, figsize=(_font(8), _font(5.5)), dpi=_font(55), facecolor=self.bg_color,
            gui_chart=True, display_title=False,
            label_size=_font(20), afsc_tick_size=_font(15))
        self.afsc_weight_graph_canvas = FigureCanvasTkAgg(self.afsc_weight_graph, master=self.canvas)
        self.afsc_weight_graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(xy(1178, 680), anchor=tk.N, window=self.afsc_weight_graph_canvas.get_tk_widget())

        # Update Cadet Weight Graph
        if self.c_function_var.get() == 'Linear':
            root.instance.value_parameters['cadet_weight'] = root.instance.parameters['merit'] / \
                                                             root.instance.parameters['sum_merit']
            self.cadet_weight_y = np.arange(start=0, stop=1, step=1 / 10)
            root.instance.value_parameters['cadet_weight_function'] = 'Linear'
        else:
            root.instance.value_parameters['cadet_weight'] = np.repeat(1 / root.instance.parameters['N'],
                                                                       root.instance.parameters['N'])
            root.instance.value_parameters['cadet_weight_function'] = 'Equal'
            self.cadet_weight_y = np.repeat(1, 20)

        # Cadet Weight Graph
        self.cadet_weight_graph = root.instance.display_weight_function(
            cadets=True, figsize=(_font(8), _font(5.5)), dpi=_font(55), facecolor=self.bg_color, gui_chart=True,
            label_size=_font(20), display_title=False)

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.cadet_weight_graph_canvas = FigureCanvasTkAgg(self.cadet_weight_graph, master=self.canvas)
        self.cadet_weight_graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(xy(262, 680), anchor=tk.N,
                                  window=self.cadet_weight_graph_canvas.get_tk_widget())

        # Get the overall weights from the entries
        root.instance.value_parameters['cadets_overall_weight'] = self.cadet_weight_entry.get()
        root.instance.value_parameters['afscs_overall_weight'] = self.afsc_weight_entry.get()

        # Sum them
        global_weight_sums = float(float(root.instance.value_parameters['cadets_overall_weight']) +
                                   float(root.instance.value_parameters['afscs_overall_weight']))

        # Force them to sum to one
        root.instance.value_parameters['cadets_overall_weight'] = round(float(
            root.instance.value_parameters['cadets_overall_weight']) / float(global_weight_sums), 2)
        root.instance.value_parameters['afscs_overall_weight'] = round(float(
            root.instance.value_parameters['afscs_overall_weight']) / float(global_weight_sums), 2)

        # Place them back in the entries
        self.cadet_weight_entry.insert(tk.END, '1')
        self.cadet_weight_entry.delete(first=0, last=tk.END)
        self.cadet_weight_entry.insert(tk.END, root.instance.value_parameters['cadets_overall_weight'])
        self.afsc_weight_entry.insert(tk.END, '1')
        self.afsc_weight_entry.delete(first=0, last=tk.END)
        self.afsc_weight_entry.insert(tk.END, root.instance.value_parameters['afscs_overall_weight'])


# AFSC individual objectives page
class Objectives(tk.Frame):

    # Initialize objectives screen class
    def __init__(self, parent):
        global root

        root.afsc_objective_swing_weight = np.zeros([root.instance.parameters['M'],
                                                     root.instance.value_parameters['O']])

        self.objectives = {}
        for k, objective in enumerate(root.instance.value_parameters['objectives']):
            if k in root.instance.value_parameters['K_A'][0]:
                self.objectives[objective] = k

        self.color = root.bg_color

        # Initialize tk frame class
        tk.Frame.__init__(self, parent, bg=self.color)
        self.canvas = create_background_image(self, "GUI_Background_Weights.png")

        # Back Button
        back_button = HoverButton(self.canvas, text='Back', width=_x(70), height=_y(40), action='EditWeights',
                                  font_size=_font(20))
        self.canvas.create_window(xy(20, 20), anchor=tk.SW, window=back_button)

        # AFSC dropdown
        self.afsc_font = tkFont.Font(family=root.title_font, size=_font(28))
        self.afsc = tk.StringVar()
        self.afscs = root.instance.parameters['afsc_vector']
        self.afsc.set(self.afscs[0])
        self.afsc_drop = tk.OptionMenu(self, self.afsc, *self.afscs,
                                       command=lambda x: self.Update_Objectives(initial=True))
        self.afsc_drop.config(font=self.afsc_font, bg=root.default_bg, fg=root.default_fg,
                              activebackground=root.active_bg, activeforeground=root.active_fg, width=5, relief=GROOVE,
                              cursor='hand2')
        self.canvas.create_window(xy(720, 700), anchor=tk.CENTER, window=self.afsc_drop)

        # Update Parameters Button
        self.update_button = HoverButton(self.canvas, text='Update Parameters', width=_x(240), height=_y(40),
                                         action='Objectives Update', font_size=_font(20))
        self.canvas.create_window(xy(1080, 700), anchor=tk.CENTER, window=self.update_button)

        # Parameter Text
        self.canvas.create_text(xy(80, 600), text="Weights", anchor=tk.S, font=root.label_font + " " + str(_font(28)))
        self.canvas.create_text(xy(300, 600), text="Objectives", anchor=tk.S,
                                font=root.label_font + " " + str(_font(28)))

        # tkinter objective arrays
        self.objective_swing_weight_entries = {}
        self.objective_local_weight_text = {}
        self.objective_button = {}
        self.obj_var = tk.IntVar()  # variable containing which objective is selected
        self.obj_var.set(root.instance.value_parameters['K_A'][0][0])  # initially the first objective is selected

        # Set the buttons and labels for the objectives
        ceiling = 575
        spacing = 6
        height = 42

        # Init First AFSC Swing Weights
        reference = max(root.instance.value_parameters['objective_weight'][0, :])
        root.afsc_objective_swing_weight[0, :] = (root.instance.value_parameters['objective_weight'][0, :] /
                                                  reference) * 100

        for loc, objective in enumerate(list(self.objectives.keys())):
            pivot_y = ceiling - loc * (height + spacing) - spacing
            k = self.objectives[objective]

            # Swing Weight Entries
            self.objective_swing_weight_entries[k] = HoverEntry(master=self.canvas, font_size=_font(20),
                                                                width=5, justify=CENTER)
            self.objective_swing_weight_entries[k].insert(tk.END, round(root.afsc_objective_swing_weight[0, k],
                                                                        ndigits=2))
            self.canvas.create_window(xy(80, pivot_y), anchor=tk.E, window=self.objective_swing_weight_entries[k])

            # Local Weight Text
            self.objective_local_weight_text[k] = self.canvas.create_text(
                xy(95, pivot_y), text=round(root.instance.value_parameters['objective_weight'][0, k], ndigits=2),
                font=root.label_font + " " + str(_font(20)), anchor=tk.W)

            # Objective selector button
            self.objective_button[k] = tk.Radiobutton(self, variable=self.obj_var, value=k, bg=self.color,
                                                      fg='black', text=objective, cursor='hand2',
                                                      font=(root.label_font, _font(20)),
                                                      activebackground=self.color, activeforeground='black',
                                                      command=lambda: self.Update_Objectives())

            self.canvas.create_window(xy(200, pivot_y), anchor=tk.W, window=self.objective_button[k])

        k = root.instance.value_parameters['K_A'][0][0]
        chart = root.instance.show_value_function(self.afsc.get(), root.instance.value_parameters['objectives'][k],
                                                  label_size=_font(20),
                                                  yaxis_tick_size=_font(15), xaxis_tick_size=_font(15),
                                                  facecolor=self.color, figsize=(_font(5), _font(5)))

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.graph_canvas = FigureCanvasTkAgg(chart, master=self.canvas)
        self.graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(_x(1340), _y(100), anchor=tk.NE, window=self.graph_canvas.get_tk_widget())

        # Export as Defaults Button
        self.default_button = HoverButton(self.canvas, text='Export as Defaults', width=_x(262), height=_y(40),
                                          action='Export Defaults', font_size=_font(20))
        self.canvas.create_window(_x(720), _y(700), anchor=tk.CENTER, window=self.default_button)

    def Update_Objectives(self, initial=False):
        global root
        afsc_name = self.afsc.get()  # get name of afsc we're dealing with
        j = np.where(root.instance.parameters['afsc_vector'] == afsc_name)[0][0]  # index of afsc

        if initial:

            # Update Swing Weights from Local Weights
            reference = max(root.instance.value_parameters['objective_weight'][j, :])
            root.afsc_objective_swing_weight[j, :] = (root.instance.value_parameters['objective_weight'][j, :] /
                                                      reference) * 100

            # Remove old objective stuff
            for loc, objective in enumerate(list(self.objectives.keys())):
                k = self.objectives[objective]
                self.objective_swing_weight_entries[k].destroy()
                self.objective_button[k].destroy()
                self.canvas.itemconfig(self.objective_local_weight_text[k], text="")

            # Get new objectives
            self.objectives = {}
            for k, objective in enumerate(root.instance.value_parameters['objectives']):
                if k in root.instance.value_parameters['K_A'][j]:
                    self.objectives[objective] = k

            # tkinter objective arrays
            self.objective_swing_weight_entries = {}
            self.objective_local_weight_text = {}
            self.objective_button = {}
            self.obj_var = tk.IntVar()  # variable containing which objective is selected
            self.obj_var.set(root.instance.value_parameters['K_A'][j][0])  # initially the first objective is selected

            # Set the buttons and labels for the objectives
            ceiling = 575
            spacing = 6
            height = 42

            for loc, objective in enumerate(list(self.objectives.keys())):
                pivot_y = ceiling - loc * (height + spacing) - spacing
                k = self.objectives[objective]

                # Swing Weight Entries
                self.objective_swing_weight_entries[k] = HoverEntry(master=self.canvas, font_size=_font(20),
                                                                    width=5, justify=CENTER)
                self.objective_swing_weight_entries[k].insert(tk.END,
                                                              round(root.afsc_objective_swing_weight[j, k], ndigits=2))
                self.canvas.create_window(xy(80, pivot_y), anchor=tk.E,
                                          window=self.objective_swing_weight_entries[k])

                # Local Weight Text
                self.objective_local_weight_text[k] = self.canvas.create_text(
                    xy(95, pivot_y), text=round(root.instance.value_parameters['objective_weight'][j, k], ndigits=2),
                    font=root.label_font + " " + str(_font(20)), anchor=tk.W)

                # Objective selector button
                self.objective_button[k] = tk.Radiobutton(self, variable=self.obj_var, value=k, bg=self.color,
                                                          fg='black', text=objective, cursor='hand2',
                                                          font=(root.label_font, _font(20)),
                                                          activebackground=self.color, activeforeground='black',
                                                          command=lambda: self.Update_Objectives())

                self.canvas.create_window(xy(200, pivot_y), anchor=tk.W, window=self.objective_button[k])

        else:

            for loc, objective in enumerate(list(self.objectives.keys())):
                k = self.objectives[objective]

                # Update Swing Weights from Swing Weight Entries
                root.afsc_objective_swing_weight[j, k] = self.objective_swing_weight_entries[k].get()

            # Update Local Weights from Swing Weights
            root.instance.value_parameters['objective_weight'][j, :] = root.afsc_objective_swing_weight[j, :] / \
                                                                       sum(root.afsc_objective_swing_weight[j, :])

            for loc, objective in enumerate(list(self.objectives.keys())):
                k = self.objectives[objective]

                # Update Local Weight Text from Local Weights
                self.canvas.itemconfig(self.objective_local_weight_text[k], text=round(
                    root.instance.value_parameters['objective_weight'][j, k], ndigits=2))

        # Display Graph
        k = self.obj_var.get()
        objective = root.instance.value_parameters['objectives'][k]

        chart = root.instance.show_value_function(self.afsc.get(), objective, label_size=_font(20),
                                                  yaxis_tick_size=_font(15), xaxis_tick_size=_font(15),
                                                  facecolor=self.color, figsize=(_font(5), _font(5)))

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.graph_canvas = FigureCanvasTkAgg(chart, master=self.canvas)
        self.graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(_x(1340), _y(100), anchor=tk.NE,
                                  window=self.graph_canvas.get_tk_widget())

    def Set_Individual_Across_AFSCs(self):
        """
        Takes this AFSCs value parameters and generalizes them across the AFSCs
        :return: None.
        """
        global root

        # Update parameters again just in case we didn't before
        self.Update_Objectives(initial=False)

        # # Update all weights
        # for k in range(value_parameters['O']):
        #     root.afsc_objective_swing_weight[:, k]


# Edit Constraints Page
class Constraints(tk.Frame):

    # Initialize title screen class
    def __init__(self, parent):
        global root

        # Initialize tk frame class
        tk.Frame.__init__(self, parent)

        self.bg_color = root.bg_color
        self.canvas = create_background_image(self, "GUI_Background_Main.jpg")

        # Back Button
        back_button = HoverButton(self.canvas, text='Back', width=_x(70), height=_y(40), action='Main',
                                  font_size=_font(20))
        self.canvas.create_window(xy(20, 20), anchor=tk.SW,
                                  window=back_button)

        # Update Button
        self.update_button = HoverButton(self.canvas, text='Update Constraints', width=_x(240), height=_y(40),
                                         action='Update Constraints', font_size=_font(20))
        self.canvas.create_window(_x(1080), _y(50), anchor=tk.CENTER,
                                  window=self.update_button)

        # Title
        self.canvas.create_text(_x(720), _y(50), text="Edit Constraints", font=root.title_font + " " + str(_font(40)))

        # AFSC Objective selector
        self.afsc_objective_options = ["Overall"]
        for objective in root.instance.value_parameters['objectives']:
            self.afsc_objective_options.append(objective)
        self.afsc_objective_var = tk.StringVar()
        self.afsc_objective_var.set(self.afsc_objective_options[0])
        self.afsc_objective_drop = tk.OptionMenu(self, self.afsc_objective_var, *self.afsc_objective_options,
                                                 command=lambda x: self.Change_AFSC_Objective())
        self.afsc_objective_drop.config(font=tkFont.Font(family=root.button_font, size=_font(20)),
                                        bg='black', fg='white', activebackground=root.active_bg,
                                        activeforeground='black', width=15, relief=GROOVE, cursor='hand2')
        self.canvas.create_window(_x(720), _y(656), anchor=tk.N, window=self.afsc_objective_drop)

        # arrays of tkinter objects
        self.afsc_convex_constraint_entries = {}
        self.afsc_min_value_entries = {}
        self.afsc_min_value_text = {}
        start_y = 550
        start_x = 30
        for j, afsc in enumerate(root.instance.parameters['afsc_vector']):
            delta_x = 175
            delta_y = 120

            if 0 <= j <= 7:
                pivot_x = start_x + delta_x * j
                pivot_y = start_y
            elif 8 <= j <= 15:
                pivot_x = start_x + delta_x * (j - 8)
                pivot_y = start_y - delta_y
            elif 16 <= j <= 23:
                pivot_x = start_x + delta_x * (j - 16)
                pivot_y = start_y - delta_y * 2
            elif 24 <= j <= 31:
                pivot_x = start_x + delta_x * (j - 24)
                pivot_y = start_y - delta_y * 3
            else:
                pivot_x = start_x + delta_x * (j - 32)
                pivot_y = start_y - delta_y * 4

            # AFSC Name
            name_x = 80
            self.canvas.create_text(xy(pivot_x + name_x, pivot_y), text=afsc, anchor=tk.E,
                                    font=root.label_font + " " + str(_font(22)))

            entry_x = 120
            entry_y = 5

            # Min Value Entry
            self.afsc_min_value_entries[j] = HoverEntry(master=self.canvas, font_size=_font(18), width=5,
                                                        justify=CENTER)

            self.canvas.create_window(xy(pivot_x + entry_x, pivot_y + entry_y), anchor=tk.S,
                                      window=self.afsc_min_value_entries[j])
            self.afsc_min_value_text[j] = self.canvas.create_text(
                xy(pivot_x + entry_x, pivot_y + entry_y + 40), text='Min Value', anchor=tk.S,
                font=root.label_font + " " + str(_font(9)))

            # Convex Constraint Entry
            self.afsc_convex_constraint_entries[j] = HoverEntry(master=self.canvas, font_size=_font(18), width=5,
                                                                justify=CENTER)

            self.canvas.create_window(xy(pivot_x + entry_x, pivot_y - entry_y), anchor=tk.N,
                                      window=self.afsc_convex_constraint_entries[j])
            self.canvas.create_text(xy(pivot_x + entry_x, pivot_y - 5), anchor=tk.N, text='Type',
                                    font=root.label_font + " " + str(_font(9)))

            # Update Min Value Entries from min values
            self.afsc_min_value_entries[j].insert(tk.END, '1')
            self.afsc_min_value_entries[j].delete(first=0, last=tk.END)
            self.afsc_min_value_entries[j].insert(tk.END, str(root.instance.value_parameters['afsc_value_min'][j]))
            self.canvas.itemconfig(self.afsc_min_value_text[j], text='Min Value')
            self.afsc_min_value_entries[j].config(state=tk.NORMAL)

            # Update Convex Constraint Entries
            self.afsc_convex_constraint_entries[j].insert(tk.END, '1')
            self.afsc_convex_constraint_entries[j].delete(first=0, last=tk.END)
            self.afsc_convex_constraint_entries[j].insert(tk.END, '0')
            self.afsc_convex_constraint_entries[j].config(state=tk.DISABLED)

    def Change_AFSC_Objective(self):
        """
        Changes the constraints shown on the screen
        :return:
        """
        for k_check in range(root.instance.value_parameters['O'] + 1):
            if self.afsc_objective_var.get() == self.afsc_objective_options[k_check]:
                k = k_check
                break

        if k == 0:
            for j, afsc in enumerate(root.instance.parameters['afsc_vector']):
                # Update Min Value Entries from min values
                self.afsc_min_value_entries[j].insert(tk.END, '1')
                self.afsc_min_value_entries[j].delete(first=0, last=tk.END)
                self.afsc_min_value_entries[j].insert(tk.END, str(root.instance.value_parameters['afsc_value_min'][j]))
                self.canvas.itemconfig(self.afsc_min_value_text[j], text='Min Value')
                self.afsc_min_value_entries[j].config(state=tk.NORMAL)

                # Update Convex Constraint Entries
                self.afsc_convex_constraint_entries[j].insert(tk.END, '1')
                self.afsc_convex_constraint_entries[j].delete(first=0, last=tk.END)
                self.afsc_convex_constraint_entries[j].insert(tk.END, '0')
                self.afsc_convex_constraint_entries[j].config(state=tk.DISABLED)
        else:
            k -= 1
            for j, afsc in enumerate(root.instance.parameters['afsc_vector']):

                # Update Min Value Entries from min values
                self.afsc_min_value_entries[j].config(state=tk.NORMAL)
                self.afsc_min_value_entries[j].insert(tk.END, '1')
                self.afsc_min_value_entries[j].delete(first=0, last=tk.END)
                self.afsc_min_value_entries[j].insert(tk.END, str(
                    root.instance.value_parameters['objective_value_min'][j, k]))

                # Update Min Value Text
                if root.instance.value_parameters['constraint_type'][j, k] == 0:
                    self.canvas.itemconfig(self.afsc_min_value_text[j], text='N/A')
                    self.afsc_min_value_entries[j].config(state=tk.DISABLED)
                elif root.instance.value_parameters['constraint_type'][j, k] == 1:
                    self.canvas.itemconfig(self.afsc_min_value_text[j], text='Min')
                    self.afsc_min_value_entries[j].config(state=tk.NORMAL)
                elif root.instance.value_parameters['constraint_type'][j, k] == 2:
                    self.canvas.itemconfig(self.afsc_min_value_text[j], text='Min')
                    self.afsc_min_value_entries[j].config(state=tk.NORMAL)
                elif root.instance.value_parameters['constraint_type'][j, k] == 3:
                    self.canvas.itemconfig(self.afsc_min_value_text[j], text='Min, Max')
                    self.afsc_min_value_entries[j].config(state=tk.NORMAL)
                elif root.instance.value_parameters['constraint_type'][j, k] == 4:
                    self.canvas.itemconfig(self.afsc_min_value_text[j], text='Min, Max')
                    self.afsc_min_value_entries[j].config(state=tk.NORMAL)

                # Update Convex Constraint Entries
                self.afsc_convex_constraint_entries[j].config(state=tk.NORMAL)
                self.afsc_convex_constraint_entries[j].insert(tk.END, '1')
                self.afsc_convex_constraint_entries[j].delete(first=0, last=tk.END)
                self.afsc_convex_constraint_entries[j].insert(
                    tk.END, str(root.instance.value_parameters['constraint_type'][j, k]))

    def Update_AFSC_Constraints(self):
        """
        Updates the value parameters based on the constraints on the page
        :return: None
        """
        global root

        for k_check in range(root.instance.value_parameters['O'] + 1):
            if self.afsc_objective_var.get() == self.afsc_objective_options[k_check]:
                k = k_check
                break

        if k == 0:
            for j, afsc in enumerate(root.instance.parameters['afsc_vector']):
                self.afsc_min_value_entries[j].config(state=tk.NORMAL)
                root.instance.value_parameters['afsc_value_min'][j] = self.afsc_min_value_entries[j].get()
        else:
            k -= 1
            for j, afsc in enumerate(root.instance.parameters['afsc_vector']):
                root.instance.value_parameters['objective_value_min'][j, k] = self.afsc_min_value_entries[j].get()
                root.instance.value_parameters['constraint_type'][j, k] = \
                    self.afsc_convex_constraint_entries[j].get()

                # Update Min Value Text
                if root.instance.value_parameters['constraint_type'][j, k] == 0:
                    self.canvas.itemconfig(self.afsc_min_value_text[j], text='N/A')
                    self.afsc_min_value_entries[j].config(state=tk.DISABLED)
                elif root.instance.value_parameters['constraint_type'][j, k] == 1:
                    self.canvas.itemconfig(self.afsc_min_value_text[j], text='Min')
                    self.afsc_min_value_entries[j].config(state=tk.NORMAL)
                elif root.instance.value_parameters['constraint_type'][j, k] == 2:
                    self.canvas.itemconfig(self.afsc_min_value_text[j], text='Min')
                    self.afsc_min_value_entries[j].config(state=tk.NORMAL)
                elif root.instance.value_parameters['constraint_type'][j, k] == 3:
                    self.canvas.itemconfig(self.afsc_min_value_text[j], text='Min, Max')
                    self.afsc_min_value_entries[j].config(state=tk.NORMAL)
                elif root.instance.value_parameters['constraint_type'][j, k] == 4:
                    self.canvas.itemconfig(self.afsc_min_value_text[j], text='Min, Max')
                    self.afsc_min_value_entries[j].config(state=tk.NORMAL)


# Results Page
class Results(tk.Frame):

    # Initialize results screen class
    def __init__(self, parent):

        # Initialize tk frame class
        tk.Frame.__init__(self, parent)
        self.canvas = create_background_image(self, "GUI_Background_Main.jpg")

        # Back Button
        back_button = HoverButton(self.canvas, text='Back', width=_x(70),
                                  height=_y(40), action='Main', font_size=_font(20))
        self.canvas.create_window(xy(20, 20), anchor=tk.SW,
                                  window=back_button)

        # More Button
        more_button = HoverButton(self.canvas, text='More', width=_x(70),
                                  height=_y(40), action='ResultsChart', font_size=_font(20))
        self.canvas.create_window(xy(1420, 20), anchor=tk.SE, window=more_button)

        # Title
        self.canvas.create_text(_x(720), _y(50), text="Solution Results", font=root.title_font + " " + str(_font(40)))

        # Solution Results Text
        self.canvas.create_text(xy(720, 610), text=round(root.instance.metrics['z'], 3),
                                font=root.label_font + " " + str(_font(24)))
        self.canvas.create_text(xy(720, 645), text='Z', font=root.label_font + " " + str(_font(24)))
        self.canvas.create_text(xy(600, 610), text=round(root.instance.metrics['cadets_overall_value'], 3),
                                font=root.label_font + " " + str(_font(24)))
        self.canvas.create_text(xy(600, 645), text='Cadets', font=root.label_font + " " + str(_font(24)))
        self.canvas.create_text(xy(840, 610), text=round(root.instance.metrics['afscs_overall_value'], 3),
                                font=root.label_font + " " + str(_font(24)))
        self.canvas.create_text(xy(840, 645), text='AFSCs', font=root.label_font + " " + str(_font(24)))

        # Cadet Solution Frame
        self.cadet_frame = tk.Frame(self.canvas, bg='black', height=_y(150), width=_x(240))
        self.canvas.create_window(xy(600, 595), anchor=tk.N, window=self.cadet_frame)
        self.cadet_scrollbar = tk.Scrollbar(self.cadet_frame)
        self.cadet_scrollbar.pack(side=RIGHT, fill=Y)

        self.cadet_solution_list = tk.Listbox(self.cadet_frame, yscrollcommand=self.cadet_scrollbar.set, bg='black',
                                              width=_font(25), height=_font(11), fg='white')
        self.cadet_solution_list.insert(tk.END, 'Cadet' + ' ' * 4 + 'Matched' + ' ' * 4 + 'Value')

        for i in range(root.instance.parameters['N']):
            list_str = ""
            encrypt_i = str(root.instance.parameters['SS_encrypt'][i])
            afsc_i = str(root.instance.metrics['afsc_solution'][i])
            value_i = str(round(root.instance.metrics['cadet_value'][i], 2))
            list_str += " " * 2 + encrypt_i + " " * 2 + " " * 12
            if len(afsc_i) == 2:
                list_str += afsc_i + " " * 8
            else:
                list_str += afsc_i + " " * 7
            list_str += " " * 2 + value_i
            # if len(encrypt_i) == 1:
            #     list_str += " " * 2 + encrypt_i + " " * 2
            # elif len(encrypt_i) == 2:
            #     list_str += " " * 2 + encrypt_i + " " * 2
            # elif len(encrypt_i) == 3:
            #     list_str += " " * 2 + encrypt_i + " " * 2
            # else:
            #     list_str += " " * 2 + encrypt_i
            # list_str += " " * 8
            # if len(afsc_i) == 3:
            #     list_str += " " * 4 + afsc_i + " " * 4
            # else:
            #     list_str += " " * 2 + afsc_i + " " * 2
            # list_str += " " * 8
            # if len(value_i) == 3:
            #     list_str += " " * 2 + value_i + " " * 2
            # else:
            #     list_str += " " * 2 + value_i
            self.cadet_solution_list.insert(tk.END, list_str)

        self.cadet_solution_list.pack()
        self.cadet_scrollbar.config(command=self.cadet_solution_list.yview, activebackground=root.active_bg,
                                    bg=root.default_bg, cursor='hand2')

        self.cadet_utility_hist = root.instance.display_results_graph(
            graph='Cadet Utility', figsize=(_font(10), _font(7)), dpi=_font(45), facecolor='black',
            label_size=_font(25), xaxis_tick_size=_font(20), yaxis_tick_size=_font(20), gui_chart=True)

        # Average Cadet Utility
        self.canvas.create_text(xy(20, 375), anchor=tk.W, font=root.label_font + " " + str(_font(15)),
                                text='Unweighted Average Cadet Utility:   ' + str(round(np.mean(
                                    root.instance.metrics['cadet_value']), 3)))

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.cadet_utility_hist_canvas = FigureCanvasTkAgg(self.cadet_utility_hist, master=self.canvas)
        self.cadet_utility_hist_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(xy(20, 720), anchor=tk.NW, window=self.cadet_utility_hist_canvas.get_tk_widget())

        # AFSC objective values by AFSC chart
        self.objective_values_chart = root.instance.display_results_graph(alpha=1, bar_color='black',
                                                                          graph='Objective Values', figsize=(_font(10), _font(7)), dpi=_font(45), facecolor='black',
                                                                          label_size=_font(25), xaxis_tick_size=_font(20), yaxis_tick_size=_font(20), gui_chart=True,
                                                                          afsc=root.instance.parameters['afsc_vector'][0])

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.objective_values_chart_canvas = FigureCanvasTkAgg(self.objective_values_chart, master=self.canvas)
        self.objective_values_chart_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(xy(1420, 720), anchor=tk.NE,
                                  window=self.objective_values_chart_canvas.get_tk_widget())

        # AFSC selector
        self.afsc_var = tk.StringVar()
        self.afsc_var.set(root.instance.parameters['afsc_vector'][0])
        self.afsc_var_drop = tk.OptionMenu(self, self.afsc_var, *root.instance.parameters['afsc_vector'],
                                                 command=lambda x: self.Change_AFSC())
        self.afsc_var_drop.config(font=tkFont.Font(family=root.button_font, size=_font(20)),
                                  bg='black', fg='white', activebackground=root.active_bg,
                                  activeforeground='black', width=6, relief=GROOVE, cursor='hand2')
        self.canvas.create_window(xy(1225, 375), anchor=tk.CENTER, window=self.afsc_var_drop)

        # AFSC Solution Frame
        self.afsc_frame = tk.Frame(self.canvas, bg='black', height=_y(150), width=_x(240))
        self.canvas.create_window(xy(840, 595), anchor=tk.N, window=self.afsc_frame)
        self.afsc_scrollbar = tk.Scrollbar(self.afsc_frame)
        self.afsc_scrollbar.pack(side=RIGHT, fill=Y)

        self.afsc_solution_list = tk.Listbox(self.afsc_frame, yscrollcommand=self.afsc_scrollbar.set, bg='black',
                                             width=_font(12), height=_font(11), fg='white')
        self.afsc_solution_list.insert(tk.END, 'AFSC' + ' ' * 3 + 'Value')

        for j in range(root.instance.parameters['M']):
            self.afsc_solution_list.insert(tk.END, str(root.instance.parameters['afsc_vector'][j]) + ' ' * 5 +
                                           str(round(root.instance.metrics['afsc_value'][j], 2)))

        self.afsc_solution_list.pack()
        self.afsc_scrollbar.config(command=self.afsc_solution_list.yview, activebackground=root.active_bg,
                                   bg=root.default_bg, cursor='hand2')

        # Build Average Objective Values Chart
        self.afsc_objective_options = ["Overall"]
        for objective in root.instance.value_parameters['objectives']:
            self.afsc_objective_options.append(objective)
        self.afsc_objective_measure_titles = ['AFSC Overall Values', 'Average Merit', 'USAFA Proportion',
                                              'Number of Cadets Assigned', 'Number of USAFA Cadets Assigned',
                                              'Number of ROTC Cadets Assigned', 'Mandatory Degree Proportion',
                                              'Desired Degree Proportion', 'Permitted Degree Proportion',
                                              'Average Cadet Utility', 'Male Proportion', 'Minority Proportion']
        self.objective_colors = ['black', '#cc241f', '#cc7e1f', '#d4ca1c', '#78d41c', '#1cd43a', '#1cd4ce', '#1c75d4',
                                 '#341cd4', '#901cd4', '#d41cbe', '#d41c65']

        self.average_afsc_objective_values_graph = root.instance.display_results_graph(gui_chart=True, alpha=1,
                                                                                       graph='AFSC Value',
                                                                                       figsize=(_font(19), _font(4)),
                                                                                       dpi=_font(58), facecolor='black',
                                                                                       label_size=_font(18),
                                                                                       afsc_tick_size=_font(15),
                                                                                       yaxis_tick_size=_font(15),
                                                                                       value_type="Overall",
                                                                                       bar_color='black')

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.average_afsc_objective_values_graph_canvas = FigureCanvasTkAgg(self.average_afsc_objective_values_graph,
                                                                            master=self.canvas)
        self.average_afsc_objective_values_graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(xy(720, 20), anchor=tk.S,
                                  window=self.average_afsc_objective_values_graph_canvas.get_tk_widget())

        # AFSC Objective selector
        self.afsc_objective_var = tk.StringVar()
        self.afsc_objective_var.set(self.afsc_objective_measure_titles[0])
        self.afsc_objective_drop = tk.OptionMenu(self, self.afsc_objective_var, *self.afsc_objective_measure_titles,
                                                 command=lambda x: self.Change_AFSC_Objective())
        self.afsc_objective_drop.config(font=tkFont.Font(family=root.button_font, size=_font(20)),
                                        bg=self.objective_colors[0], fg='white', activebackground=root.active_bg,
                                        activeforeground='black',
                                        width=30, relief=GROOVE, cursor='hand2')
        self.canvas.create_window(xy(720, 375), anchor=tk.CENTER, window=self.afsc_objective_drop)

        start_y = 335
        start_x = 30
        self.objective_measures = {}
        for j, afsc in enumerate(root.instance.parameters['afsc_vector']):
            if 0 <= j <= 9:
                pivot_x = start_x + 140 * j
                pivot_y = start_y
            elif 10 <= j <= 19:
                pivot_x = start_x + 140 * (j - 10)
                pivot_y = start_y - 30
            elif 20 <= j <= 29:
                pivot_x = start_x + 140 * (j - 20)
                pivot_y = start_y - 30 * 2
            elif j == 30:
                pivot_x = 30
                pivot_y = start_y - 30 * 3
            elif j == 31:
                pivot_x = 140 * 9 + 30
                pivot_y = start_y - 30 * 3
            elif j == 32:
                pivot_x = 30
                pivot_y = start_y - 30 * 4
            elif j == 33:
                pivot_x = 140 * 9 + 30
                pivot_y = start_y - 30 * 4
            else:
                pivot_x = 2000
                pivot_y = start_y - 30 * 4

            # AFSC Values
            self.canvas.create_text(xy(pivot_x, pivot_y), anchor=tk.W, text=afsc + ':',
                                    font=root.label_font + " " + str(_font(14)))
            self.objective_measures[j] = self.canvas.create_text(xy(pivot_x + 70, pivot_y), anchor=tk.W,
                                                                 text=str(
                                                                     round(root.instance.metrics['afsc_value'][j], 2)),
                                                                 font=root.label_font + " " + str(_font(14)))

    def Change_AFSC_Objective(self):
        """
        Changes the graph and objective measures on the screen
        :return:
        """

        potential_objectives = np.array(['Overall', 'Merit', 'USAFA Proportion', 'Combined Quota', 'USAFA Quota',
                                         'ROTC Quota', 'Mandatory', 'Desired', 'Permitted', 'Utility', 'Male',
                                         'Minority'])
        lookup_dict = {}
        for k, obj_name in enumerate(self.afsc_objective_measure_titles):
            lookup_dict[obj_name] = potential_objectives[k]

        value_type = lookup_dict[self.afsc_objective_var.get()]
        k = np.where(potential_objectives == value_type)[0][0]
        self.average_afsc_objective_values_graph = root.instance.display_results_graph(gui_chart=True,
                                                                                       graph='AFSC Value',
                                                                                       figsize=(_font(19), _font(4)),
                                                                                       dpi=_font(58), facecolor='black',
                                                                                       label_size=_font(18),
                                                                                       afsc_tick_size=_font(15),
                                                                                       yaxis_tick_size=_font(15),
                                                                                       value_type=value_type,
                                                                                       alpha=1,
                                                                                       bar_color='black')

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.average_afsc_objective_values_graph_canvas = FigureCanvasTkAgg(self.average_afsc_objective_values_graph,
                                                                            master=self.canvas)
        self.average_afsc_objective_values_graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(xy(720, 20), anchor=tk.S,
                                  window=self.average_afsc_objective_values_graph_canvas.get_tk_widget())

        if k == 0:
            measures = root.instance.metrics['afsc_value']
        else:
            measures = root.instance.metrics['objective_measure'][:, k - 1]
        for j, afsc in enumerate(root.instance.parameters['afsc_vector']):
            self.canvas.itemconfig(self.objective_measures[j], text=round(measures[j], 2))

    def Change_AFSC(self):
        """
        Changes the objective values on the screen
        :return:
        """

        afsc = self.afsc_var.get()
        self.objective_values_chart = root.instance.display_results_graph(afsc=afsc, alpha=1, bar_color='black',
                                                                          graph='Objective Values', figsize=(_font(10), _font(7)), dpi=_font(45), facecolor='black',
                                                                          label_size=_font(25), xaxis_tick_size=_font(20), yaxis_tick_size=_font(20), gui_chart=True)

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.objective_values_chart_canvas = FigureCanvasTkAgg(self.objective_values_chart, master=self.canvas)
        self.objective_values_chart_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(xy(1420, 720), anchor=tk.NE,
                                  window=self.objective_values_chart_canvas.get_tk_widget())


# Results Chart Page
class ResultsChart(tk.Frame):

    # Initialize data chart screen class
    def __init__(self, parent):
        # Initialize tk frame class
        tk.Frame.__init__(self, parent)

        self.canvas = create_background_image(self, "GUI_Background_Main.jpg")

        # Back Button
        back_button = HoverButton(self.canvas, text='Back', width=_x(70), height=_y(40), action='Results',
                                  font_size=_font(20))
        self.canvas.create_window(xy(20, 20), anchor=tk.SW, window=back_button)

        # Title
        self.canvas.create_text(_x(720), _y(42), text="Results Charts", font=root.title_font + " " + str(_font(40)))

        # Arrow Buttons
        self.results_chart_num = 1  # this variable indicates which chart to show based on what arrow was clicked

        self.left_arrow = HoverButton(self.canvas, text='<-', width=_x(36), height=_y(40),
                                      action='ChangeResultsChart_L', font_size=_font(20))
        self.canvas.create_window(_x(20), _y(60), anchor=tk.SW, window=self.left_arrow)

        self.right_arrow = HoverButton(self.canvas, text='->', width=_x(36),
                                       height=_y(40), action='ChangeResultsChart_R', font_size=_font(20))
        self.canvas.create_window(_x(1420), _y(60), anchor=tk.SE, window=self.right_arrow)

        chart = root.instance.display_results_graph(
            figsize=root.chart_figsize, facecolor=root.chart_color, graph='Average Merit',
            afsc_tick_size=root.chart_afsc_size, afsc_rotation=root.chart_afsc_rotation,
            yaxis_tick_size=root.chart_ytick_size, legend_size=root.chart_legend_size, title_size=root.chart_title_size,
            label_size=root.chart_label_size, alpha=1, bar_color='black')

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.graph_canvas = FigureCanvasTkAgg(chart, master=self.canvas)
        self.graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(root.chart_coords, anchor=tk.SW, window=self.graph_canvas.get_tk_widget())

    def Change_Chart(self, direction):
        """
        Picks the chart to the right or left
        :param direction: right or left
        :return: None
        """

        global root
        if direction == 'Left':
            self.results_chart_num -= 1
        else:
            self.results_chart_num += 1

        if self.results_chart_num < 1:
            self.results_chart_num = 7
        elif self.results_chart_num > 7:
            self.results_chart_num = 1

        self.Update_Chart()

    def Update_Chart(self):
        """
        Updates the results chart to be shown
        :return: None
        """
        chart_dict = {1: 'Average Merit', 2: 'Combined Quota', 3: 'USAFA Proportion', 4: 'Mandatory', 5: 'Desired',
                      6: 'Permitted', 7: 'Average Utility'}
        if self.results_chart_num in [4, 5, 6]:
            chart = root.instance.display_results_graph(
                figsize=root.chart_figsize, facecolor=root.chart_color, graph='AFOCD Proportion',
                afsc_tick_size=root.chart_afsc_size, afsc_rotation=root.chart_afsc_rotation,
                yaxis_tick_size=root.chart_ytick_size, legend_size=root.chart_legend_size,
                title_size=root.chart_title_size, degree=chart_dict[self.results_chart_num],
                label_size=root.chart_label_size, alpha=1, bar_color='black')
        else:
            chart = root.instance.display_results_graph(
                figsize=root.chart_figsize, facecolor=root.chart_color, graph=chart_dict[self.results_chart_num],
                afsc_tick_size=root.chart_afsc_size, afsc_rotation=root.chart_afsc_rotation,
                yaxis_tick_size=root.chart_ytick_size, legend_size=root.chart_legend_size,
                title_size=root.chart_title_size, alpha=1, bar_color='black',
                label_size=root.chart_label_size)

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.graph_canvas = FigureCanvasTkAgg(chart, master=self.canvas)
        self.graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(root.chart_coords, anchor=tk.SW, window=self.graph_canvas.get_tk_widget())


# Sensitivity Analysis Page
class Sensitivity(tk.Frame):

    # Initialize results screen class
    def __init__(self, parent):

        # Initialize tk frame class
        tk.Frame.__init__(self, parent)
        self.canvas = create_background_image(self, "GUI_Background_Main.jpg")

        # Back Button
        back_button = HoverButton(self.canvas, text='Back', width=_x(70), height=_y(40), action='Main',
                                  font_size=_font(20))
        self.canvas.create_window(xy(20, 20), anchor=tk.SW, window=back_button)

        # Export Button
        export_button = HoverButton(self.canvas, text='Export', width=_x(100), height=_y(40), action='Export Report',
                                  font_size=_font(20))
        self.canvas.create_window(xy(1420, 20), anchor=tk.SE, window=export_button)

        # Title
        self.title = self.canvas.create_text(xy(720, 700), text="Sensitivity Analysis",
                                             font=root.title_font + " " + str(_font(40)))

        # Labels
        self.canvas.create_text(xy(40, 650), text="Solutions", anchor=tk.SW,
                                font=root.label_font + " " + str(_font(30)))
        self.canvas.create_text(xy(1400, 650), text="    Value\nParameters",
                                anchor=tk.SE, font=root.label_font + " " + str(_font(30)))

        # Buttons
        self.comp_solutions_button = HoverButton(self.canvas, text='Compare Solutions', width=_x(220),
                                                 height=_y(40), action='Compare Solutions', font_size=_font(20))
        self.canvas.create_window(xy(720 - 60, 630), anchor=tk.NE, window=self.comp_solutions_button)
        self.comp_values_button = HoverButton(self.canvas, text='Compare Weights', width=_x(220),
                                                 height=_y(40), action='Compare Weights', font_size=_font(20))
        self.canvas.create_window(xy(720 + 60, 630), anchor=tk.NW, window=self.comp_values_button)
        self.change_solution_button = HoverButton(self.canvas, text='Change Instance', width=_x(210),
                                                  height=_y(40), action='Change Solution', font_size=_font(20))
        self.canvas.create_window(xy(20, 150), anchor=tk.NW, window=self.change_solution_button)
        self.change_weights_button = HoverButton(self.canvas, text='Change Instance', width=_x(210),
                                                 height=_y(40), action='Change Values', font_size=_font(20))
        self.canvas.create_window(xy(1420, 150), anchor=tk.NE, window=self.change_weights_button)
        self.pareto_button = HoverButton(self.canvas, text='Pareto', width=_x(220),
                                         height=_y(40), action='Pareto', font_size=_font(20))
        self.canvas.create_window(xy(720 - 60, 550), anchor=tk.NE, window=self.pareto_button)
        self.lsp_button = HoverButton(self.canvas, text='LSP', width=_x(220),
                                      height=_y(40), action='LSP', font_size=_font(20))
        self.canvas.create_window(xy(720 + 60, 550), anchor=tk.NW, window=self.lsp_button)
        self.con_button1 = HoverButton(self.canvas, text='All -> None', width=_x(220),
                                       height=_y(40), action='Con_All_None', font_size=_font(20))
        self.canvas.create_window(xy(720 + 60, 470), anchor=tk.NW, window=self.con_button1)
        self.con_button2 = HoverButton(self.canvas, text='None -> All', width=_x(220),
                                       height=_y(40), action='Con_None_All', font_size=_font(20))
        self.canvas.create_window(xy(720 - 60, 470), anchor=tk.NE, window=self.con_button2)

        # Solution/VP Grid Parameters
        self.row_y = 640
        self.name_x = 80
        self.box_x = 40
        self.delta_y = 30

        # Place solution texts
        self.solution_names = {}
        self.solution_boxes = {}
        self.solution_box_vars = {}
        y = self.row_y
        for i, solution_name in enumerate(root.z_dict.keys()):
            self.solution_names[solution_name] = self.canvas.create_text(
                xy(self.name_x, y), text=solution_name + " *", anchor=tk.NW,
                font=root.label_font + " " + str(_font(12)))
            self.solution_box_vars[solution_name] = tk.IntVar(value=1)
            self.solution_boxes[solution_name] = tk.Checkbutton(
                self.canvas, onvalue=1, offvalue=0, variable=self.solution_box_vars[solution_name], bg=root.default_bg,
                fg=root.default_bg, cursor='hand2', command=lambda: self.Update_Screen())
            self.canvas.create_window(xy(self.box_x, y), anchor=tk.NW, window=self.solution_boxes[solution_name])
            root.current_selected_solutions_dict[solution_name] = copy.deepcopy(root.z_dict[solution_name])
            y -= self.delta_y

        # Place parameter texts
        self.parameter_names = {}
        self.parameter_boxes = {}
        self.parameter_box_vars = {}
        y = self.row_y
        for i, parameter_name in enumerate(root.vp_dict.keys()):
            self.parameter_names[parameter_name] = self.canvas.create_text(
                xy(1440-self.name_x, y), text="* " + parameter_name, anchor=tk.NE,
                font=root.label_font + " " + str(_font(12)))
            self.parameter_box_vars[parameter_name] = tk.IntVar(value=1)
            self.parameter_boxes[parameter_name] = tk.Checkbutton(
                self.canvas, onvalue=1, offvalue=0, variable=self.parameter_box_vars[parameter_name],
                bg=root.default_bg, fg=root.default_bg, cursor='hand2', command=lambda: self.Update_Screen())
            self.canvas.create_window(xy(1440-self.box_x, y), anchor=tk.NE, window=self.parameter_boxes[parameter_name])
            root.current_selected_vp_dict[parameter_name] = copy.deepcopy(root.vp_dict[parameter_name])
            y -= self.delta_y

        # Objective value matrix
        self.canvas.create_rectangle(xy(720 - 420, 375), xy(720 + 420, 20))
        self.canvas.create_text(xy(720 - 420 - 15, 375 * (3/4)), text='Value Parameters', anchor=tk.E, angle=90,
                                font=root.label_font + " " + str(_font(20)))
        self.canvas.create_text(xy(720, 375 + 10), text='Solutions', anchor=tk.S,
                                font=root.label_font + " " + str(_font(20)))

        # Matrix Parameters
        self.header_font = 20
        self.element_font = 15
        self.matrix_x_space = 60
        self.matrix_y_space = 30
        self.top_left_x = 720 - 420 + 80
        self.top_left_y = 375 - 33
        self.buffer_x = 40
        self.buffer_y = 30

        self.m_value_font = tkFont.Font(family=root.label_font, size=_font(28))
        self.m_value = tk.StringVar()
        self.m_values = ['Z', 'AFSC', 'Cadet']
        self.m_value.set(self.m_values[0])
        self.m_value_drop = tk.OptionMenu(self, self.m_value, *self.m_values,
                                          command=lambda x: self.Update_Screen())
        self.m_value_drop.config(font=self.m_value_font, bg=root.default_bg, fg=root.default_fg,
                                 activebackground=root.active_bg,
                                 activeforeground=root.active_fg, width=5, relief=GROOVE, cursor='hand2')
        self.canvas.create_window(xy(self.top_left_x, self.top_left_y), anchor=tk.CENTER, window=self.m_value_drop)

        # Initially all boxes are checked
        self.current_selected_z_dict = {}
        self.current_checked_solutions = []
        self.current_checked_parameters = []
        num_solutions = 0
        for solution_name in root.z_dict:
            self.current_checked_solutions.append(solution_name)
            self.current_selected_z_dict[solution_name] = {}
            num_vps = 0
            for vp_name in root.vp_dict:
                num_vps += 1
                self.current_selected_z_dict[solution_name][vp_name] = root.z_dict[solution_name][vp_name]
                self.current_checked_parameters.append(vp_name)
            num_solutions += 1

        # Place row labels
        self.row_indices = []
        for i in range(num_vps):
            self.row_indices.append(self.canvas.create_text(
                xy(self.top_left_x, self.top_left_y - self.buffer_y - (i + 1) * self.matrix_y_space), text=str(i + 1),
                anchor=tk.CENTER, font=root.label_font + " " + str(_font(self.header_font))))

        # Place column labels
        self.col_indices = []
        for j in range(num_solutions):
            self.col_indices.append(self.canvas.create_text(
                xy(self.top_left_x + self.buffer_x + (j + 1) * self.matrix_x_space, self.top_left_y), text=str(j + 1),
                anchor=tk.CENTER, font=root.label_font + " " + str(_font(self.header_font))))

        # Add Z values to matrix
        self.matrix_value_text = []
        for j, solution_name in enumerate(self.current_selected_z_dict.keys()):
            self.matrix_value_text.append([])
            for i, vp_name in enumerate(self.current_selected_z_dict[solution_name].keys()):
                z = str(round(self.current_selected_z_dict[solution_name][vp_name], 3))
                self.matrix_value_text[j].append(self.canvas.create_text(
                    xy(self.top_left_x + self.buffer_x + (j + 1) * self.matrix_x_space,
                       self.top_left_y - self.buffer_y - (i + 1) * self.matrix_y_space),
                    text=z, anchor=tk.CENTER, font=root.label_font + " " + str(_font(self.element_font))))

    def Update_Screen(self):
        """
        Updates the screen
        :return: None
        """
        global root

        # Place solution texts
        y = self.row_y
        for i, solution_name in enumerate(root.z_dict.keys()):
            if solution_name not in self.solution_names.keys():
                self.solution_names[solution_name] = self.canvas.create_text(
                    xy(self.name_x, y), text=solution_name + " *", anchor=tk.NW,
                    font=root.label_font + " " + str(_font(12)))
                self.solution_box_vars[solution_name] = tk.IntVar(value=1)
                self.solution_boxes[solution_name] = tk.Checkbutton(
                    self.canvas, onvalue=1, offvalue=0, variable=self.solution_box_vars[solution_name],
                    bg=root.default_bg, fg=root.default_bg, cursor='hand2', command=lambda: self.Update_Screen())
                self.canvas.create_window(xy(self.box_x, y), anchor=tk.NW, window=self.solution_boxes[solution_name])
            else:
                if solution_name == root.current_solution_name:
                    self.canvas.itemconfig(self.solution_names[solution_name], text=solution_name + " *")
                else:
                    self.canvas.itemconfig(self.solution_names[solution_name], text=solution_name)

            y -= self.delta_y

        # Place parameter texts
        y = self.row_y
        for i, parameter_name in enumerate(root.vp_dict.keys()):
            if parameter_name not in self.parameter_names.keys():
                self.parameter_names[parameter_name] = self.canvas.create_text(
                    xy(1440-self.name_x, y), text="* " + parameter_name, anchor=tk.NE,
                    font=root.label_font + " " + str(_font(12)))
                self.parameter_box_vars[parameter_name] = tk.IntVar(value=1)
                self.parameter_boxes[parameter_name] = tk.Checkbutton(
                    self.canvas, onvalue=1, offvalue=0, variable=self.parameter_box_vars[parameter_name],
                    bg=root.default_bg, fg=root.default_bg, cursor='hand2', command=lambda: self.Update_Screen())
                self.canvas.create_window(xy(1440-self.box_x, y), anchor=tk.NE,
                                          window=self.parameter_boxes[parameter_name])
            else:
                if parameter_name == root.current_vp_name:
                    self.canvas.itemconfig(self.parameter_names[parameter_name], text="* " + parameter_name)
                else:
                    self.canvas.itemconfig(self.parameter_names[parameter_name], text=parameter_name)

            y -= self.delta_y

        # Get list of solutions that are checked
        self.current_checked_solutions = []
        checked_solution_nums = []
        for i, solution_name in enumerate(self.solution_names.keys()):
            if self.solution_box_vars[solution_name].get() == 1:
                self.current_checked_solutions.append(solution_name)
                checked_solution_nums.append(i + 1)

        # Get list of value parameter sets that are checked
        self.current_checked_parameters = []
        checked_vp_nums = []
        for i, parameter_name in enumerate(self.parameter_names.keys()):
            if self.parameter_box_vars[parameter_name].get() == 1:
                self.current_checked_parameters.append(parameter_name)
                checked_vp_nums.append(i + 1)

        # Clear value matrix
        for j, solution_name in enumerate(self.current_selected_z_dict.keys()):
            self.canvas.itemconfig(self.col_indices[j], text="")
            for i, vp_name in enumerate(self.current_selected_z_dict[solution_name].keys()):
                self.canvas.itemconfig(self.matrix_value_text[j][i], text="")
                self.canvas.itemconfig(self.row_indices[i], text="")

        # Load checked solutions dictionary
        root.current_selected_solutions_dict = {}
        for solution_name in self.current_checked_solutions:
            root.current_selected_solutions_dict[solution_name] = root.solution_dict[solution_name]

        # Load checked value parameters dictionary
        root.current_selected_vp_dict = {}
        for value_parameter_name in self.current_checked_parameters:
            root.current_selected_vp_dict[value_parameter_name] = copy.deepcopy(root.vp_dict[value_parameter_name])

        # Place row labels
        self.row_indices = []
        for i, num in enumerate(checked_vp_nums):
            self.row_indices.append(self.canvas.create_text(
                xy(self.top_left_x, self.top_left_y - self.buffer_y - (i + 1) * self.matrix_y_space), text=str(num),
                anchor=tk.CENTER, font=root.label_font + " " + str(_font(self.header_font))))

        # Place column labels
        self.col_indices = []
        for j, num in enumerate(checked_solution_nums):
            self.col_indices.append(self.canvas.create_text(
                xy(self.top_left_x + self.buffer_x + (j + 1) * self.matrix_x_space, self.top_left_y), text=str(num),
                anchor=tk.CENTER, font=root.label_font + " " + str(_font(self.header_font))))

        # Get correct value
        m_dict = {'Z': 'z', 'AFSC': 'afscs_overall_value', 'Cadet': 'cadets_overall_value'}
        _v = m_dict[self.m_value.get()]

        # Add values to matrix
        self.matrix_value_text = []
        self.current_selected_z_dict = {}
        for j, solution_name in enumerate(root.current_selected_solutions_dict.keys()):
            self.current_selected_z_dict[solution_name] = {}
            self.matrix_value_text.append([])
            for i, vp_name in enumerate(root.current_selected_vp_dict.keys()):
                self.current_selected_z_dict[solution_name][vp_name] = root.z_dict[solution_name][vp_name]
                val = str(round(root.metrics_dict[solution_name][vp_name][_v], 3))
                self.matrix_value_text[j].append(self.canvas.create_text(
                    xy(self.top_left_x + self.buffer_x + (j + 1) * self.matrix_x_space,
                       self.top_left_y - self.buffer_y - (i + 1) * self.matrix_y_space),
                    text=val, anchor=tk.CENTER, font=root.label_font + " " + str(_font(self.element_font))))

        root.pages[CompareMeasures].Update_Chart()
        root.pages[CompareWeights].Update_Screen()

    def LSP_clicked(self):
        print("This button will perform the LSP sensitivity analysis")

    def Pareto_clicked(self):
        print("This button will perform the Pareto Chart sensitivity analysis")

    def Constraint_Analysis(self, kind):
        print('This button will perform the ' + kind + ' constraint analysis')

    def Export_Report(self):
        print('This button will export a sensitivity report')

    def Change_Solution(self):
        if len(self.current_checked_solutions) != 0:
            solution_name = self.current_checked_solutions[0]
            root.current_solution_name = solution_name
            root.instance.solution = root.solution_dict[solution_name]
            root.instance.measure_solution()
            root.New_Results()
            root.show_page(Sensitivity)
            self.Update_Screen()

    def Change_Weights(self):
        if len(self.current_checked_parameters) != 0:
            vp_name = self.current_checked_parameters[0]
            root.current_vp_name = vp_name
            root.instance.value_parameters = root.vp_dict[vp_name]
            root.instance.measure_solution()
            root.Init_Parameter_Frames()
            root.New_Results()
            root.show_page(Sensitivity)
            self.Update_Screen()


# Compare Objective Measures Page
class CompareMeasures(tk.Frame):

    # Initialize data chart screen class
    def __init__(self, parent):

        # Initialize tk frame class
        tk.Frame.__init__(self, parent)
        self.canvas = create_background_image(self, "GUI_Background_Main.jpg")

        # Back Button
        back_button = HoverButton(self.canvas, text='Back', width=_x(70),
                                  height=_y(40), action='Sensitivity', font_size=_font(20))
        self.canvas.create_window(xy(20, 20), anchor=tk.SW, window=back_button)

        # Values Button
        values_button = HoverButton(self.canvas, text='Values', width=_x(90),
                                    height=_y(40), action='Compare Values', font_size=_font(20))
        self.canvas.create_window(xy(1420, 20), anchor=tk.SE, window=values_button)

        # Solution Similarity Button
        similarity_button = HoverButton(self.canvas, text='Solution Similarity', width=_x(220),
                                        height=_y(40), action='Similarity', font_size=_font(20))
        self.canvas.create_window(xy(720 - 200, 20), anchor=tk.S, window=similarity_button)

        # Save Figure Button
        save_fig_button = HoverButton(self.canvas, text='Save Figure', width=_x(220),
                                      height=_y(40), action='Save Figure_Measures', font_size=_font(20))
        self.canvas.create_window(xy(720 + 200, 20), anchor=tk.S, window=save_fig_button)

        # Title
        self.canvas.create_text(xy(720, 708), text="Compare AFSC Measures",
                                font=root.title_font + " " + str(_font(40)))

        # Arrow Buttons
        self.chart_num = 1  # this variable indicates which chart to show based on what arrow was clicked

        self.left_arrow = HoverButton(self.canvas, text='<-', width=_x(36),
                                      height=_y(40), action='ChangeMeasuresChart_L', font_size=_font(20))
        self.canvas.create_window(_x(20), _y(60), anchor=tk.SW, window=self.left_arrow)

        self.right_arrow = HoverButton(self.canvas, text='->', width=_x(36),
                                       height=_y(40), action='ChangeMeasuresChart_R', font_size=_font(20))
        self.canvas.create_window(_x(1420), _y(60), anchor=tk.SE, window=self.right_arrow)

        checked_vp_name = list(root.current_selected_vp_dict.keys())[0]
        checked_solutions = root.current_selected_solutions_dict.keys()
        checked_metrics = {}
        for solution_name in checked_solutions:
            checked_metrics[solution_name] = root.metrics_dict[solution_name][checked_vp_name]

        chart = root.instance.display_results_graph(
            figsize=root.chart_figsize, facecolor=root.chart_color, graph='Average Merit',
            afsc_tick_size=root.chart_afsc_size, afsc_rotation=root.chart_afsc_rotation, metrics_dict=checked_metrics,
            yaxis_tick_size=root.chart_ytick_size, legend_size=root.chart_legend_size, title_size=root.chart_title_size,
            label_size=root.chart_label_size, xaxis_tick_size=root.chart_xtick_size)

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.graph_canvas = FigureCanvasTkAgg(chart, master=self.canvas)
        self.graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(root.chart_coords, anchor=tk.SW, window=self.graph_canvas.get_tk_widget())

    def Change_Chart(self, direction):
        """
        Picks the chart to the right or left
        :param direction: right or left
        :return: None
        """

        if direction == 'Left':
            self.chart_num -= 1
        else:
            self.chart_num += 1

        if self.chart_num < 1:
            self.chart_num = 8
        elif self.chart_num > 8:
            self.chart_num = 1

        self.Update_Chart()

    def Update_Chart(self):
        """
        Updates the results chart to be shown
        :return: None
        """

        chart_dict = {1: 'Average Merit', 2: 'USAFA Proportion', 3: 'Combined Quota', 4: 'AFOCD Proportion|Mandatory',
                      5: 'AFOCD Proportion|Desired', 6: 'AFOCD Proportion|Permitted', 7: 'Average Utility',
                      8: 'Cadet Utility'}

        if self.chart_num in [4, 5, 6]:
            graph = chart_dict[self.chart_num].split('|')[0]
            degree = chart_dict[self.chart_num].split('|')[1]
        else:
            graph = chart_dict[self.chart_num]
            degree = 'Mandatory'

        checked_vp_name = list(root.current_selected_vp_dict.keys())[0]
        checked_solutions = root.current_selected_solutions_dict.keys()
        checked_metrics = {}
        for solution_name in checked_solutions:
            checked_metrics[solution_name] = root.metrics_dict[solution_name][checked_vp_name]

        chart = root.instance.display_results_graph(
            figsize=root.chart_figsize, facecolor=root.chart_color, graph=graph, degree=degree,
            afsc_tick_size=root.chart_afsc_size, afsc_rotation=root.chart_afsc_rotation, metrics_dict=checked_metrics,
            yaxis_tick_size=root.chart_ytick_size, legend_size=root.chart_legend_size, title_size=root.chart_title_size,
            label_size=root.chart_label_size, xaxis_tick_size=root.chart_xtick_size)

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.graph_canvas = FigureCanvasTkAgg(chart, master=self.canvas)
        self.graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(root.chart_coords, anchor=tk.SW, window=self.graph_canvas.get_tk_widget())


# Compare Objective Values Page
class CompareObjectiveValues(tk.Frame):

    # Initialize data chart screen class
    def __init__(self, parent):

        # Initialize tk frame class
        tk.Frame.__init__(self, parent)
        self.canvas = create_background_image(self, "GUI_Background_Main.jpg")

        # Back Button
        back_button = HoverButton(self.canvas, text='Back', width=_x(70),
                                  height=_y(40), action='Sensitivity', font_size=_font(20))
        self.canvas.create_window(xy(20, 20), anchor=tk.SW, window=back_button)

        # Measures Button
        measures_button = HoverButton(self.canvas, text='Measures', width=_x(120),
                                      height=_y(40), action='Compare Measures', font_size=_font(20))
        self.canvas.create_window(xy(1420, 20), anchor=tk.SE, window=measures_button)

        # By AFSC Button
        by_afsc_button = HoverButton(self.canvas, text='Compare AFSCs', width=_x(240),
                                      height=_y(40), action='By AFSC', font_size=_font(20))
        self.canvas.create_window(xy(720 - 200, 20), anchor=tk.S, window=by_afsc_button)

        # Save Figure Button
        save_fig_button = HoverButton(self.canvas, text='Save Figure', width=_x(220),
                                      height=_y(40), action='Save Figure_ObjectiveValues', font_size=_font(20))
        self.canvas.create_window(xy(720 + 200, 20), anchor=tk.S, window=save_fig_button)

        # Title
        self.canvas.create_text(xy(200, 708), text="Compare AFSC Values for ", anchor=tk.W,
                                font=root.title_font + " " + str(_font(35)))

        # Objective dropdown
        self.objective_font = tkFont.Font(family=root.title_font, size=_font(35))
        self.objective = tk.StringVar()
        self.objectives = np.hstack((['Overall'], root.instance.value_parameters['objectives']))
        self.objective.set(self.objectives[0])
        self.objective_drop = tk.OptionMenu(self, self.objective, *self.objectives,
                                            command=lambda x: self.Update_Chart())
        self.objective_drop.config(font=self.objective_font, bg=root.default_bg, fg=root.default_fg,
                                   activebackground=root.active_bg,
                                   activeforeground=root.active_fg, width=15, relief=GROOVE, cursor='hand2')
        self.canvas.create_window(xy(1050, 708), anchor=tk.CENTER, window=self.objective_drop)

        checked_vp_name = list(root.current_selected_vp_dict.keys())[0]
        checked_solutions = root.current_selected_solutions_dict.keys()
        checked_metrics = {}
        for solution_name in checked_solutions:
            checked_metrics[solution_name] = root.metrics_dict[solution_name][checked_vp_name]

        chart = root.instance.display_results_graph(
            figsize=root.chart_figsize, facecolor=root.chart_color, graph='AFSC Value',
            afsc_tick_size=root.chart_afsc_size, afsc_rotation=root.chart_afsc_rotation, metrics_dict=checked_metrics,
            yaxis_tick_size=root.chart_ytick_size, legend_size=root.chart_legend_size, title_size=root.chart_title_size,
            label_size=root.chart_label_size, xaxis_tick_size=root.chart_xtick_size)

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.graph_canvas = FigureCanvasTkAgg(chart, master=self.canvas)
        self.graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(root.chart_coords, anchor=tk.SW, window=self.graph_canvas.get_tk_widget())

    def Update_Chart(self):
        """
        Updates the results chart to be shown
        :return: None
        """

        checked_vp_name = list(root.current_selected_vp_dict.keys())[0]
        checked_solutions = root.current_selected_solutions_dict.keys()
        checked_metrics = {}
        for solution_name in checked_solutions:
            checked_metrics[solution_name] = root.metrics_dict[solution_name][checked_vp_name]

        value_type = self.objective.get()
        chart = root.instance.display_results_graph(
            figsize=root.chart_figsize, facecolor=root.chart_color, graph='AFSC Value',
            afsc_tick_size=root.chart_afsc_size, afsc_rotation=root.chart_afsc_rotation, metrics_dict=checked_metrics,
            yaxis_tick_size=root.chart_ytick_size, legend_size=root.chart_legend_size, title_size=root.chart_title_size,
            label_size=root.chart_label_size, value_type=value_type, xaxis_tick_size=root.chart_xtick_size)

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.graph_canvas = FigureCanvasTkAgg(chart, master=self.canvas)
        self.graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(root.chart_coords, anchor=tk.SW, window=self.graph_canvas.get_tk_widget())


# Compare AFSC Values Page
class CompareAFSCValues(tk.Frame):

    # Initialize data chart screen class
    def __init__(self, parent):

        # Initialize tk frame class
        tk.Frame.__init__(self, parent)
        self.canvas = create_background_image(self, "GUI_Background_Main.jpg")

        # Back Button
        back_button = HoverButton(self.canvas, text='Back', width=_x(70),
                                  height=_y(40), action='Sensitivity', font_size=_font(20))
        self.canvas.create_window(xy(20, 20), anchor=tk.SW, window=back_button)

        # Measures Button
        measures_button = HoverButton(self.canvas, text='Measures', width=_x(120),
                                      height=_y(40), action='Compare Measures', font_size=_font(20))
        self.canvas.create_window(xy(1420, 20), anchor=tk.SE, window=measures_button)

        # By Objective Button
        by_objective_button = HoverButton(self.canvas, text='Compare Objectives', width=_x(240),
                                          height=_y(40), action='By Objective', font_size=_font(20))
        self.canvas.create_window(xy(720 - 200, 20), anchor=tk.S, window=by_objective_button)

        # Save Figure Button
        save_fig_button = HoverButton(self.canvas, text='Save Figure', width=_x(220),
                                      height=_y(40), action='Save Figure_AFSCValues', font_size=_font(20))
        self.canvas.create_window(xy(720 + 200, 20), anchor=tk.S, window=save_fig_button)

        # Title
        self.canvas.create_text(xy(200, 708), text="Compare Objective Values for ", anchor=tk.W,
                                font=root.title_font + " " + str(_font(35)))

        # AFSC dropdown
        self.afsc_font = tkFont.Font(family=root.title_font, size=_font(35))
        self.afsc = tk.StringVar()
        self.afscs = root.instance.parameters['afsc_vector']
        self.afsc.set(self.afscs[0])
        self.afsc_drop = tk.OptionMenu(self, self.afsc, *self.afscs,
                                            command=lambda x: self.Update_Chart())
        self.afsc_drop.config(font=self.afsc_font, bg=root.default_bg, fg=root.default_fg,
                              activebackground=root.active_bg,
                              activeforeground=root.active_fg, width=5, relief=GROOVE, cursor='hand2')
        self.canvas.create_window(xy(950, 708), anchor=tk.CENTER, window=self.afsc_drop)

        checked_vp_name = list(root.current_selected_vp_dict.keys())[0]
        checked_solutions = root.current_selected_solutions_dict.keys()
        checked_metrics = {}
        for solution_name in checked_solutions:
            checked_metrics[solution_name] = root.metrics_dict[solution_name][checked_vp_name]

        chart = root.instance.display_results_graph(
            figsize=root.chart_figsize, facecolor=root.chart_color, graph='Objective Values', afsc=self.afsc.get(),
            afsc_tick_size=root.chart_afsc_size, afsc_rotation=root.chart_afsc_rotation, metrics_dict=checked_metrics,
            yaxis_tick_size=root.chart_ytick_size, legend_size=root.chart_legend_size, title_size=root.chart_title_size,
            label_size=root.chart_label_size, xaxis_tick_size=root.chart_xtick_size)

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.graph_canvas = FigureCanvasTkAgg(chart, master=self.canvas)
        self.graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(root.chart_coords, anchor=tk.SW, window=self.graph_canvas.get_tk_widget())

    def Update_Chart(self):
        """
        Updates the results chart to be shown
        :return: None
        """

        checked_vp_name = list(root.current_selected_vp_dict.keys())[0]
        checked_solutions = root.current_selected_solutions_dict.keys()
        checked_metrics = {}
        for solution_name in checked_solutions:
            checked_metrics[solution_name] = root.metrics_dict[solution_name][checked_vp_name]

        chart = root.instance.display_results_graph(
            figsize=root.chart_figsize, facecolor=root.chart_color, graph='Objective Values', afsc=self.afsc.get(),
            afsc_tick_size=root.chart_afsc_size, afsc_rotation=root.chart_afsc_rotation, metrics_dict=checked_metrics,
            yaxis_tick_size=root.chart_ytick_size, legend_size=root.chart_legend_size, title_size=root.chart_title_size,
            label_size=root.chart_label_size, xaxis_tick_size=root.chart_xtick_size)

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.graph_canvas = FigureCanvasTkAgg(chart, master=self.canvas)
        self.graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(root.chart_coords, anchor=tk.SW, window=self.graph_canvas.get_tk_widget())


# Weight Comparison Page
class CompareWeights(tk.Frame):

    # Initialize data chart screen class
    def __init__(self, parent):
        # Initialize tk frame class
        tk.Frame.__init__(self, parent)
        self.canvas = create_background_image(self, "GUI_Background_Main.jpg")

        # Back Button
        back_button = HoverButton(self.canvas, text='Back', width=_x(70),
                                  height=_y(40), action='Sensitivity', font_size=_font(20))
        self.canvas.create_window(xy(20, 20), anchor=tk.SW,
                                  window=back_button)

        # AFSC dropdown
        self.afsc_font = tkFont.Font(family=root.title_font, size=_font(28))
        self.afsc = tk.StringVar()
        self.afscs = root.instance.parameters['afsc_vector']
        self.afsc.set(self.afscs[0])
        self.afsc_drop = tk.OptionMenu(self, self.afsc, *self.afscs,
                                       command=lambda x: self.Update_Screen())
        self.afsc_drop.config(font=self.afsc_font, bg=root.default_bg, fg=root.default_fg,
                              activebackground=root.active_bg,
                              activeforeground=root.active_fg, width=5, relief=GROOVE, cursor='hand2')
        self.canvas.create_window(_x(720), _y(10), anchor=tk.N, window=self.afsc_drop)

        chart = afsc_objective_weights_graph(root.instance.parameters, root.vp_dict, self.afsc.get(),
                                             figsize=root.chart_figsize, label_size=root.chart_label_size,
                                             facecolor=root.chart_color, title_size=root.chart_title_size,
                                             legend_size=root.chart_legend_size, xaxis_tick_size=root.chart_xtick_size,
                                             yaxis_tick_size=root.chart_ytick_size)

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.graph_canvas = FigureCanvasTkAgg(chart, master=self.canvas)
        self.graph_canvas.draw()

        self.chart_x = _x(20)
        self.chart_y = _y(675)

        # place the canvas on the page
        self.canvas.create_window(self.chart_x, self.chart_y, anchor=tk.SW, window=self.graph_canvas.get_tk_widget())

    def Update_Screen(self):
        """
        Updates the results chart to be shown
        :return: None
        """
        chart = afsc_objective_weights_graph(root.instance.parameters, root.current_selected_vp_dict, self.afsc.get(),
                                             figsize=root.chart_figsize, label_size=root.chart_label_size,
                                             facecolor=root.chart_color, title_size=root.chart_title_size,
                                             legend_size=root.chart_legend_size, xaxis_tick_size=root.chart_xtick_size,
                                             yaxis_tick_size=root.chart_ytick_size)

        # create the Tkinter canvas containing the Matplotlib figure and attach it to the page canvas
        self.graph_canvas = FigureCanvasTkAgg(chart, master=self.canvas)
        self.graph_canvas.draw()

        # place the canvas on the page
        self.canvas.create_window(self.chart_x, self.chart_y, anchor=tk.SW, window=self.graph_canvas.get_tk_widget())


# Run the GUI
if __name__ == "__main__":
    global root

    # Initialize the GUI structure
    root = GUI()

    # Initialize pages
    root.Init_Title_Data()

    root.mainloop()  # main loop
