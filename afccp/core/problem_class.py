# Import libraries
import copy

from afccp.core.comprehensive_functions import *
import datetime
import glob
import os


class CadetCareerProblem:
    def __init__(self, data_name=None, filepath=None, N=1600, M=32, P=6, printing=False):
        """
        This is the initialization function for the AFSC/Cadet problem. We can import data directly by specifying
        a filepath (or the filepath can be determined by the data_name). We can also generate data by providing a
        data_name that contains "Random", "Realistic", or "Perfect" in the name.
        :param data_name: name of the data set. If we want to import data using the data_name, this needs to be
        the name of the excel file without the .xlsx extension
        :param filepath: we can specify the filepath directly if need be
        :param N: Number of cadets to generate
        :param M: Number of AFSCs to generate
        :param P: Number of AFSC preferences to generate for each cadet
        :param printing: Whether we should print status updates or not
        """

        # Get a list of full instance files in the directory
        dir_folder = "instances"
        directory = paths[dir_folder]
        self.data_files = {'Generated': {'Random': [], 'Perfect': [], 'Realistic': []},
                           'Real': {'Scrubbed': [], 'Year': []}}
        for file_name in glob.iglob(directory + '*.xlsx', recursive=True):
            start_index = file_name.find(dir_folder) + len(dir_folder) + 1
            end_index = len(file_name) - 5
            full_name = file_name[start_index:end_index]
            sections = full_name.split(' ')
            if len(sections) != 2:
                if 'Generated' == sections[0]:
                    for data_variant in self.data_files['Generated']:
                        if data_variant in sections[1]:
                            self.data_files['Generated'][data_variant].append(full_name)
                elif "Real" == sections[0]:
                    if len(sections[1]) == 1:
                        self.data_files['Real']['Scrubbed'].append(full_name)
                    else:
                        self.data_files['Real']['Year'].append(full_name)
        if data_name is None:

            # If we don't specify a data name or a filepath, we generate random data
            if filepath is None:
                self.data_name = "Random_" + str(len(self.data_files['Generated']["Random"]) + 1)
                self.data_type = "Generated"
                self.data_variant = "Random"
                generate = True
            else:
                full_name = filepath.split('/')
                full_name = full_name[len(full_name) - 1]
                self.data_name = full_name.split(' ')[1]
                self.data_type = full_name.split(' ')[0]
                generate = False
        else:
            self.data_name = data_name
            self.data_type = "Real"
            generate = False
            for data_variant in self.data_files['Generated']:
                if data_variant == self.data_name:
                    self.data_name = data_variant + "_" + str(len(self.data_files['Generated'][data_variant]) + 1)
                    self.data_type = "Generated"
                    generate = True
                    break
                elif data_variant in self.data_name and '_' in self.data_name:
                    self.data_type = "Generated"
                    generate = False
                    break

        # Get data variant
        if self.data_type == "Real":
            if len(self.data_name) == 1:
                self.data_variant = "Scrubbed"
            else:
                self.data_variant = "Year"
        else:
            for data_variant in self.data_files['Generated']:
                if data_variant in self.data_name:
                    self.data_variant = data_variant

        # Create the "full name" which we piece together all the current available information
        self.full_name = self.data_type + ' ' + self.data_name

        # List of instance variants for this particular problem instance (same data, different solutions/vps)
        self.instance_files = np.array([full_name for full_name in self.data_files[
            self.data_type][self.data_variant] if self.full_name in full_name])

        if filepath is None:

            # Since all fixed data are the same here, just import the cadet/AFSC parameters for one of the versions
            if len(self.instance_files) > 0:
                self.filepath = paths['instances'] + self.instance_files[0] + '.xlsx'
            else:
                self.filepath = paths['instances'] + self.full_name + '.xlsx'
        else:
            self.filepath = filepath

        # initialize more instance attributes
        self.printing = printing
        self.default_value_parameters = None
        self.value_parameters = None
        self.metrics = None
        self.X = None
        self.measure = None
        self.value = None
        self.pyomo_z = None
        self.year = None
        self.solution = None
        self.gp_parameters = None
        self.solution_dict = None
        self.metrics_dict = None
        self.solution_name = None
        self.vp_dict = None
        self.vp_name = None

        # Check if we're generating data, and the data variant
        if self.data_variant == 'Random' and generate:

            if printing:
                print('Generating ' + self.data_name + ' problem instance...')
            parameters = simulate_model_fixed_parameters(N=N, P=P, M=M)
            self.parameters = model_fixed_parameters_set_additions(parameters)

        elif self.data_variant == 'Perfect' and generate:

            if printing:
                print('Generating ' + self.data_name + ' problem instance...')
            parameters, self.solution = perfect_example_generator(N=N, P=P, M=M)
            self.parameters = model_fixed_parameters_set_additions(parameters)

        elif self.data_variant == 'Realistic' and generate:

            if printing:
                print('Generating ' + self.data_name + ' problem instance...')

            if use_sdv:
                data = simulate_realistic_fixed_data(N=N)
                cadets_fixed, afscs_fixed = convert_realistic_data_parameters(data)
                parameters = model_fixed_parameters_from_data_frame(cadets_fixed, afscs_fixed)
            else:
                parameters = simulate_model_fixed_parameters(N=N, P=P, M=M)

            self.parameters = model_fixed_parameters_set_additions(parameters)
        else:

            if printing:
                print('Importing ' + self.data_name + ' problem instance...')
            cadets_fixed, afscs_fixed = import_fixed_cadet_afsc_data_from_excel(self.filepath)
            parameters = model_fixed_parameters_from_data_frame(cadets_fixed, afscs_fixed)
            self.parameters = model_fixed_parameters_set_additions(parameters)

        if printing:
            if generate:
                print('Generated.')
            else:
                print('Imported.')

    # Observe "Fixed" Data
    def display_data_graph(self, graph='Average Merit', save=False, printing=None, facecolor='white', title=None,
                           figsize=(16, 10), display_title=True, num=None, eligibility=False, thesis_chart=False,
                           label_size=35, afsc_tick_size=30, yaxis_tick_size=30, afsc_rotation=None, alpha=0.5,
                           dpi=100, title_size=None, legend_size=None, bar_color='black', skip_afscs=None):
        """
        This method plots different aspects of the fixed parameters of the problem instance.
        :param graph: which data element to visualize (Average Merit, USAFA Proportion, Eligible Quota, AFOCD Data,
        Average Utility)
        :param save: if we should save the graph
        :param printing: Whether we should print status updates or not
        :param facecolor: color of the figure
        :param title: title of the figure
        :param figsize: size of the figure
        :param display_title: Whether we should put a title on the figure or not
        :param num: Maximum number of eligible cadets for each AFSC to show
        :param eligibility: This applies to the Average Utility chart. This determines if we should calculate the
        average utility across the eligible cadets (True) or all cadets (False)
        :param thesis_chart: If this chart is used in the thesis document (determines some of these parameters)
        :param label_size: size of the labels
        :param afsc_tick_size: size of the x axis AFSC tick labels
        :param yaxis_tick_size: size of the y axis ticks
        :param afsc_rotation: how much the AFSC tick labels should be rotated
        :param alpha: alpha used on the bar plots
        :param dpi: dots per inch for figure
        :param title_size: size of the title
        :param legend_size: size of the legend
        :param bar_color: color of the bars
        :param skip_afscs: if we should have every other AFSC shown (useful for letter class years)
        :return: chart
        """

        # Find number of AFSCs to show
        if num is not None:
            num_afscs = sum([1 if len(self.parameters['I^E'][j]) < num else 0 for j in self.parameters['J']])
        else:
            num_afscs = self.parameters['M']

        if afsc_rotation is None:
            if self.data_variant == "Real":
                if num_afscs > 18:
                    afsc_rotation = 45
                else:
                    afsc_rotation = 0
            else:
                if num_afscs < 25:
                    afsc_rotation = 0
                else:
                    afsc_rotation = 45

        if skip_afscs is None:
            if self.data_variant == "Real":
                skip_afscs = False
            else:
                if num_afscs < self.parameters['M']:
                    skip_afscs = True
                else:
                    skip_afscs = False

        if thesis_chart:
            figsize = (16, 10)
            display_title = False
            save = True
            label_size = 35
            afsc_tick_size = 30
            yaxis_tick_size = 30

        chart = data_graph(self.parameters, save=save, figsize=figsize, facecolor=facecolor, eligibility=eligibility,
                           title=title, display_title=display_title, num=num, label_size=label_size,
                           afsc_tick_size=afsc_tick_size, graph=graph, yaxis_tick_size=yaxis_tick_size,
                           afsc_rotation=afsc_rotation, dpi=dpi, bar_color=bar_color, alpha=alpha,
                           title_size=title_size, legend_size=legend_size, skip_afscs=skip_afscs)

        if printing is None:
            printing = self.printing

        if printing:
            chart.show()

        return chart

    def find_ineligible_cadets(self, solution=None):
        """
        This procedure takes in a solution (presumably AFPC solution) and then finds the cadets that are
        assigned to AFSCs that they are ineligible for. This is used in cases where I didn't have the qual
        matrix and I had to generate it myself using the 2021 AFOCD. For some of the older years where
        qualifications were different, this would mean my qual matrix may say that a cadet is ineligible
        for an AFSC they got. I used this function to tell me how to change the qualifications based on
        the cadets' CIPs. If no cadets are printed out, then none were ineligible.
        :param solution: solution vector
        """

        if solution is None:
            solution = self.solution

        find_original_solution_ineligibility(self.parameters, solution)

    # Adjust Data
    def adjust_qualification_matrix(self, printing=None):
        """
        This procedure simply re-runs the CIP to Qual function in case I change qualifications or I
        identify errors since some cadets in the AFPC solution may receive AFSCs for which they
        are ineligible for.
        :param printing: Whether we should print status updates or not
        """
        if printing is None:
            printing = self.printing

        if printing:
            print('Adjusting qualification matrix...')

        parameters = copy.deepcopy(self.parameters)
        qual_matrix = cip_to_qual(parameters['afsc_vector'], parameters['cip1'].astype(str),
                                  cip2=parameters['cip2'].astype(str))
        parameters['qual'] = qual_matrix
        parameters['ineligible'] = (qual_matrix == 'I') * 1
        parameters['eligible'] = (parameters['ineligible'] == 0) * 1
        parameters['mandatory'] = (qual_matrix == 'M') * 1
        parameters['desired'] = (qual_matrix == 'D') * 1
        parameters['permitted'] = (qual_matrix == 'P') * 1
        parameters = model_fixed_parameters_set_additions(parameters)
        self.parameters = copy.deepcopy(parameters)

    # Specify Value Parameters
    def import_value_parameters(self, filepath=None, vp_name=None, num_breakpoints=None, set_value_parameters=True,
                                printing=None):
        """
        Imports weight and value parameters from a problem instance excel file
        :param vp_name: name of value parameters
        :param set_value_parameters: if we want to set these value parameters as our object attribute
        :param filepath: filepath to import from (if none specified, will use filepath attribute)
        :param num_breakpoints: Number of breakpoints to use for the value functions
        :param printing: :param printing: Whether we should print status updates or not
        """
        if vp_name is not None:
            full_name = self.data_type + " " + self.data_name + " " + vp_name
            if filepath is None:
                full_name = [file_name for file_name in self.instance_files if full_name in file_name][0]
                filepath = paths['instances'] + full_name + '.xlsx'
        else:
            if filepath is None:
                filepath = paths['instances'] + self.instance_files[0] + '.xlsx'
            vp_name = self.instance_files[0].split(' ')[2]

        # Set correct names
        self.vp_name = vp_name
        self.full_name = self.data_type + " " + self.data_name + " " + self.vp_name

        if printing is None:
            printing = self.printing

        # Import value parameters
        value_parameters = model_value_parameters_from_excel(self.parameters, filepath, printing=printing,
                                                             num_breakpoints=num_breakpoints)
        value_parameters = model_value_parameters_set_additions(value_parameters)
        value_parameters = condense_value_functions(self.parameters, value_parameters)
        value_parameters = model_value_parameters_set_additions(value_parameters, printing)

        if set_value_parameters:
            self.value_parameters = value_parameters
            self.vp_name = vp_name

        if printing:
            print('Imported.')

        return value_parameters

    def import_default_value_parameters(self, filepath=None, no_constraints=False, num_breakpoints=24,
                                        generate_afsc_weights=True, printing=None):
        """
        Import default value parameter setting and generate value parameters for this instance from those
        ones.
        :param filepath: filepath to import from (if none specified, will use filepath attribute)
        :param num_breakpoints: Number of breakpoints to use for the value functions
        :param printing: Whether we should print status updates or not
        :param no_constraints: If we don't want to use the predefined constraints
        :param generate_afsc_weights: If we generate the AFSC weights from the weight function, or just
        import the weights directly from the "AFSC Weight" column
        """

        if printing is None:
            printing = self.printing

        if filepath is None:
            filepath = paths['Data Processing Support'] + 'Value_Parameters_Defaults_' + self.data_type + '.xlsx'

        if self.perfect:
            filepath = paths['Data Processing Support'] + 'Value_Parameters_Defaults_Perfect.xlsx'

        self.default_value_parameters = default_value_parameters_from_excel(filepath, num_breakpoints=num_breakpoints,
                                                                            printing=printing)
        value_parameters = generate_value_parameters_from_defaults(
            self.parameters, generate_afsc_weights=generate_afsc_weights,
            default_value_parameters=self.default_value_parameters)

        if no_constraints:
            value_parameters['constraint_type'] = np.zeros([self.parameters['M'], value_parameters['O']])
        value_parameters = model_value_parameters_set_additions(value_parameters)
        value_parameters = condense_value_functions(self.parameters, value_parameters)
        self.value_parameters = model_value_parameters_set_additions(value_parameters)

        if self.printing:
            print('Imported.')

    def generate_realistic_value_parameters(self, default_value_parameters=None, constraints_df=None,
                                            deterministic=True, printing=None, constrain_merit=False):
        """
        This procedure only works on actual class years. We have data on what the quotas and constraints were
        supposed to be, as well as what was actually implemented with the original solution for each class year.
        This is how we can generate more accurate sets of value parameters based on what we know about the
        original solution.
        :param default_value_parameters: optional set of default value parameters (we would just import it
        otherwise)
        :param constraints_df: dataframe of constraints used for the real class years
        :param deterministic: if we're generating sets of value parameters deterministically or not
        :param printing: Whether we should print status updates or not
        :param constrain_merit: If we want to constrain merit
        """

        if printing is None:
            printing = self.printing

        if printing:
            if deterministic:
                print("Generating deterministic set of value parameters...")
            else:
                print("Generating non-deterministic set of value parameters...")

        if default_value_parameters is None:
            filepath = paths['Data Processing Support'] + 'Value_Parameters_Defaults_' + self.data_type + '.xlsx'
            if self.data_variant == 'Perfect':
                filepath = paths['Data Processing Support'] + 'Value_Parameters_Defaults_Perfect.xlsx'
            default_value_parameters = default_value_parameters_from_excel(filepath)
            self.default_value_parameters = default_value_parameters

        if self.data_type == "Real":  # Not scrubbed AFSC data
            data_type = self.data_name  # Pull in the class year used
        else:
            data_type = self.data_type

        if constraints_df is None:
            if self.data_type == "Real":
                constraints_df = import_data(
                    paths['Data Processing Support'] + 'Value_Parameter_Sets_Options_Real.xlsx',
                    sheet_name=data_type + ' Constraint Options')
            else:
                constraints_df = import_data(
                    paths['Data Processing Support'] + 'Value_Parameter_Sets_Options_Scrubbed.xlsx',
                    sheet_name=data_type + ' Constraint Options')

        value_parameters = value_parameter_realistic_generator(self.parameters, default_value_parameters,
                                                               constraints_df, deterministic=deterministic,
                                                               constrain_merit=constrain_merit,
                                                               data_type=data_type)
        value_parameters = model_value_parameters_set_additions(value_parameters)
        value_parameters = condense_value_functions(self.parameters, value_parameters)
        self.value_parameters = model_value_parameters_set_additions(value_parameters)

        if printing:
            print('Generated.')

    def change_weight_function(self, cadets=True, function=None):
        """
        Changes the weight function on either cadets or AFSCs
        :param cadets: if this is for cadets (True) or AFSCs (False)
        :param function: new weight function to use
        """

        if cadets:
            if function is None:
                function = 'Linear'

            # Update weight function
            self.value_parameters['cadet_weight_function'] = function

            # Update Cadet Weights
            if function == 'Linear':
                self.value_parameters['cadet_weight'] = self.parameters['merit'] / np.sum(self.parameters['merit'])
            else:
                self.value_parameters['cadet_weight'] = np.repeat(1 / self.parameters['N'], self.parameters['N'])
        else:
            if function is None:
                function = "Piece"

            # Update weight function
            self.value_parameters['afsc_weight_function'] = function

            # Update AFSC Weights
            if function == 'Size':
                self.value_parameters['afsc_weight'] = self.parameters['quota'] / np.sum(self.parameters['quota'])

            elif function == 'Equal':
                self.value_parameters['afsc_weight'] = np.repeat(1 / self.parameters['M'], self.parameters['M'])

            else:

                # Update Swing Weights
                swing_weights = np.zeros(self.parameters['M'])
                for j, quota in enumerate(self.parameters['quota']):
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
                self.value_parameters['afsc_weight'] = swing_weights / sum(swing_weights)

    def update_value_parameter_dictionary(self, value_parameters, vp_name=None):
        pass

    # Translate Parameters
    def vft_to_gp_parameters(self, gp_df_dict=None, printing=None):
        """
        Converts the instance parameters and value parameters to parameters used by Rebecca's model
        :param gp_df_dict: dictionary of dataframes used by Rebecca's model
        :param printing: Whether we should print status updates or not
        """

        if printing is None:
            printing = self.printing

        self.gp_parameters = translate_vft_to_gp_parameters(self.parameters, self.value_parameters, gp_df_dict,
                                                            printing=printing)

    # Observe Value Parameters
    def show_value_function(self, afsc='13N', objective='Combined Quota', printing=None, label_size=25,
                            yaxis_tick_size=25, xaxis_tick_size=25, figsize=(12, 10), facecolor='white',
                            thesis_chart=False, title=None, save=False, display_title=True):
        """
        This method plots a specific AFSC objective value function
        :param afsc: AFSC to show the function for
        :param objective: objective for that AFSC to show the function for
        :param printing: Whether we should print status updates or not
        :param label_size: size of the labels
        :param yaxis_tick_size: size of the y axis ticks
        :param xaxis_tick_size: size of the x axis ticks
        :param figsize: size of the figure
        :param facecolor: color of the figure
        :param thesis_chart: If this chart is used in the thesis document (determines some of these parameters)
        :param title: title of the figure
        :param save: if we should save the graph
        :param display_title: Whether we should put a title on the figure or not
        :return: figure
        """

        if printing is None:
            printing = self.printing

        if afsc is None:
            afsc = self.parameters['afsc_vector'][0]
        if objective is None:
            objective = self.value_parameters['objectives'][0]

        if thesis_chart:
            figsize = (12, 10)
            display_title = False
            save = True
            label_size = 35
            xaxis_tick_size = 30
            yaxis_tick_size = 30
            title = 'Value_Function_' + afsc + '_' + objective

        value_function_chart = plot_value_function(afsc, objective, self.parameters, self.value_parameters,
                                                   printing=printing, label_size=label_size,
                                                   yaxis_tick_size=yaxis_tick_size, facecolor=facecolor,
                                                   xaxis_tick_size=xaxis_tick_size, figsize=figsize, save=save,
                                                   display_title=display_title, title=title)

        if printing:
            value_function_chart.show()

        return value_function_chart

    def display_weight_function(self, cadets=True, save=False, figsize=(19, 7), facecolor='white', gui_chart=False,
                                display_title=True, thesis_chart=False, title=None, label_size=35, afsc_tick_size=25,
                                yaxis_tick_size=30, afsc_rotation=None, xaxis_tick_size=30, skip_afscs=None, dpi=100,
                                printing=None):
        """
        This method plots the weight function used on either cadets or AFSCs
        :param cadets: if the weight function is for cadets or not (AFSCs)
        :param gui_chart: if this method is used for the GUI
        :param afsc_tick_size: size of the x axis AFSC tick labels
        :param afsc_rotation: how much the AFSC tick labels should be rotated
        :param skip_afscs: if we should have every other AFSC shown (useful for letter class years)
        :param dpi: dots per inch for figure
        :param printing: Whether we should print status updates or not
        :param label_size: size of the labels
        :param yaxis_tick_size: size of the y axis ticks
        :param xaxis_tick_size: size of the x axis ticks
        :param figsize: size of the figure
        :param facecolor: color of the figure
        :param thesis_chart: If this chart is used in the thesis document (determines some of these parameters)
        :param title: title of the figure
        :param save: if we should save the graph
        :param display_title: Whether we should put a title on the figure or not
        :return: figure
        """

        if printing is None:
            printing = self.printing

        if afsc_rotation is None:
            if self.data_type == "Real":
                afsc_rotation = 45
            else:
                afsc_rotation = 0

        if skip_afscs is None:
            if self.data_type == "Real":
                skip_afscs = False
            else:
                skip_afscs = True

        if thesis_chart:
            figsize = (16, 10)
            display_title = False
            save = True
            label_size = 35
            afsc_tick_size = 30
            yaxis_tick_size = 30
            xaxis_tick_size = 30
            scrub_afscs = True
            if cadets:
                title = 'Weight_Chart_Cadets'
            else:
                title = 'Weight_Chart_AFSCs'
            chart = individual_weight_graph(self.parameters, self.value_parameters, cadets, save, figsize, facecolor,
                                            display_title, title, label_size, afsc_tick_size, gui_chart, dpi,
                                            yaxis_tick_size, afsc_rotation, xaxis_tick_size, skip_afscs=skip_afscs)
        else:
            chart = individual_weight_graph(self.parameters, self.value_parameters, cadets, save, figsize, facecolor,
                                            display_title, title, label_size, afsc_tick_size, gui_chart, dpi,
                                            yaxis_tick_size, afsc_rotation, xaxis_tick_size, skip_afscs=skip_afscs)

        if printing:
            chart.show()

        return chart

    # Import/Convert Solutions
    def import_solution(self, data_name=None, filepath=None, standard=True, solution_name=None, set_solution=True,
                        printing=None):
        """
        Imports a solution to the problem instance object using either the data_name or filepath
        :param set_solution: if we want to set this solution as the object's solution attribute (self.solution)
        :param data_name: name of problem instance file
        :param solution_name: Name of the solution
        :param filepath: filepath to import the solution from (will take attribute filepath by default)
        :param standard: if we should import solutions the standard way or not (for the real class years I initially
        had excel files that had the 'Cadets Fixed" sheet, "AFSCs Fixed" sheet and an additional sheet called "Original
        Solution" that had the original solution with no other value parameters or metrics in the workbook
        :param printing: Whether we should print status updates or not
        :return solution
        """

        if printing is None:
            printing = self.printing

        # Determine filepath and data_name
        if filepath is None:
            if data_name is None:
                filepath, data_name = self.filepath, self.data_name
            else:
                filepath = paths['instances'] + data_name + '.xlsx'
        else:
            data_name = filepath.split('/')
            data_name = data_name[len(data_name) - 1]
            data_name = data_name[:-5]

        # Import solution
        solution = import_solution_from_excel(filepath=filepath, standard=standard, printing=printing)

        # Add solution to solution dictionary
        self.add_solution_to_dictionary(solution, data_name, solution_name=solution_name, solution_method="Import")

        # Set the solution attribute
        if set_solution:
            self.solution = solution
            self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                    printing=printing)
        if printing:
            print('Imported.')

        return solution

    def add_solution_to_dictionary(self, solution, data_name=None, solution_name=None, solution_method="Import"):

        # Determine name of solution
        if solution_name is None:

            # Initially assume the name is Method_i
            if self.solution_dict is not None:
                count = 1
                for s_name in self.solution_dict:
                    if solution_method in s_name:
                        count += 1
                solution_name = solution_method + '_' + str(count)

            # Check to see if we can rename solution to something based on how it was solved
            if solution_method == "Import":
                for option in ['Original', 'VFT', 'GP', 'Stable', 'Greedy', 'Genetic']:
                    if option in data_name:
                        index = data_name.index(option)
                        solution_name = data_name[index:]
                        break

        # Add solution to dictionary if it is a new solution
        if self.solution_dict is None:
            self.solution_dict = {solution_name: solution}
        else:

            # Check if this solution is a new solution
            new = True
            for s_name in self.solution_dict:
                p_i = compare_solutions(self.solution_dict[s_name], solution)
                if p_i == 1:
                    new = False
                    break

            # If it is new, we add it to the dictionary
            if new:
                self.solution_dict[solution_name] = solution

    def scrub_afsc_solution(self, filepath, year, printing=None):
        """
        Takes in a real class year solution and converts the real AFSCs into the scrubbed letter verisons
        :param filepath: filepath of the solution (not optional because we know this instance has scrubbed AFSCs
        and so the solution has the real ones labeled)
        :param year: year of the solution
        :param printing: Whether we should print status updates or not
        :return: solution
        """

        if printing is None:
            printing = self.printing

        real_solution = import_solution_from_excel(filepath=filepath)
        year_afsc_table = import_data(filepath=paths['Data Processing Support'] + "Year_AFSCs_Table.xlsx",
                                      sheet_name=str(year))
        real_afscs = np.array(year_afsc_table['AFSC'])
        old_afscs = np.sort(real_afscs)
        scrubbed_solution = np.zeros(self.parameters['N']).astype(int)
        for j, afsc in enumerate(real_afscs):
            _name = self.parameters['afsc_vector'][j]
            old_j = np.where(old_afscs == afsc)[0][0]
            indices = np.where(real_solution == old_j)[0]
            scrubbed_solution[indices] = j
        self.solution = scrubbed_solution
        self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                printing=printing)
        return self.solution

    def afsc_solution_to_solution(self, afsc_solution, printing=None):
        """
        Simply converts a vector of labeled AFSCs to a vector of AFSC indices
        :param afsc_solution: solution vector of AFSC names (strings)
        :param printing: Whether or not we should print status updates
        :return: solution vector of AFSC indices
        """
        if printing is None:
            printing = self.printing

        solution = np.zeros(len(afsc_solution))
        for i in range(self.parameters['N']):
            solution[i] = np.where(self.parameters['afsc_vector'] == afsc_solution[i])[0]

        self.solution = solution
        self.metrics = measure_solution_quality(solution, self.parameters, self.value_parameters,
                                                printing=printing)
        return solution

    def x_from_excel(self, filepath=None):
        """
        This method loads in an X matrix from excel
        :param filepath: filepath of the X matrix
        """

        # Load X dataframe From Excel
        if filepath is None:
            if '.xlsx' in self.filepath:
                data_name = self.filepath[:-len('.xlsx')]
            else:
                data_name = self.filepath
            filepath = data_name + '_X.xlsx'
        X = import_data(filepath)

        # Convert to X matrix
        X.drop(labels='Encrypt_PII', axis=1, inplace=True)
        self.X = np.array(X)

    def init_exact_solution_from_x(self, printing=None):
        """
        This method creates a dictionary of initial variables used in the exact optimization model if we want to
        start it with a solution
        :param printing: Whether we should print statues updates or not
        :return: dictionary of exact pyomo model initial variables
        """
        if printing is None:
            printing = self.printing

        if printing:
            print('Initializing VFT solution from X matrix...')

        if self.X is None:
            self.x_from_excel()

        # Evaluate Solution
        metrics = measure_solution_quality(self.X, self.parameters, self.value_parameters)
        values = metrics['objective_value']
        measures = metrics['objective_measure']

        lam, y = x_to_solution_initialization(self.parameters, self.value_parameters, measures, values)
        initialization = {'X': self.X, 'lam': lam, 'y': y, 'F_X': values}
        return initialization

    # Solve Models
    def generate_random_solution(self, printing=None):
        """
        Generate random solution by assigning cadets to AFSCs that they're eligible for
        :param printing: Whether or not to print status updates
        """
        if printing is None:
            printing = self.printing

        if printing:
            print('Generating random solution...')

        J_E = self.parameters['J_E']
        I = self.parameters['I']
        self.solution = np.array([np.random.choice(J_E[i]) for i in I])
        self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                printing=printing)

    def stable_matching(self, printing=None, set_solution=True):
        """
        This method solves the stable marriage heuristic for an initial solution
        :param printing: Whether or not to print status updates
        :param set_solution: If we set the solution as the current object's solution
        :return: solution vector
        """
        if printing is None:
            printing = self.printing

        solution = stable_marriage_model_solve(self.parameters, self.value_parameters, printing=printing)

        if set_solution:
            self.solution = solution
            self.measure_solution(solution, printing=printing)

        return solution

    def greedy_method(self, printing=None, set_solution=True):
        """
        This method solves the greedy heuristic for an initial solution
        :param printing: Whether or not to print status updates
        :param set_solution: If we set the solution as the current object's solution
        :return: solution vector
        """
        if printing is None:
            printing = self.printing

        solution = greedy_model_solve(self.parameters, self.value_parameters, printing=printing)

        if set_solution:
            self.solution = solution
            self.measure_solution(solution, printing=printing)

        return solution

    def genetic_algorithm(self, initialize=True, pop_size=10, stopping_time=60, num_crossover_points=3,
                          initial_solutions=None, mutation_rate=0.05, num_time_points=100, constraints="None",
                          penalty_scale=10, time_eval=False, num_mutations=None, percent_step=10,
                          con_tolerance=0.95, printing=None, con_fail_dict=None):
        """
        This is the genetic algorithm. The hyper-parameters to the algorithm can be tuned, and this is meant to be
        solved in conjunction with the pyomo model solution. Use that as the initial solution, and then we evolve
        from there
        :param initialize: if we want to initialize the algorithm with solutions
        :param con_fail_dict: dictionary of failed constraints in the approximate model that we adhere to
        :param con_tolerance: constraint fail tolerance (we can meet X % of the constraint or above and be ok)
        :param penalty_scale: how to scale the penalties to further decrease the fitness value
        :param constraints: how we handle fitness in terms of constraints (penalty, fail, or other)
        :param pop_size: population size
        :param stopping_time: how long to run the GA for
        :param num_crossover_points: k for multi-point crossover
        :param initial_solutions: solutions to initialize the population with
        :param mutation_rate: how likely a gene is to mutate
        :param num_time_points: how many observations to collect for the time evaluation df
        :param time_eval: if we should get a time evaluation df
        :param num_mutations: how many genes are up for mutation
        :param percent_step: what percent checkpoints we should display updates
        :param printing: if we should display updates
        """

        if printing is None:
            printing = self.printing

        if time_eval:
            start_time = time.perf_counter()

        if initialize:
            if initial_solutions is None:
                if self.solution is not None:
                    solution1 = self.stable_matching(printing, set_solution=False)
                    solution2 = self.greedy_method(printing, set_solution=False)
                    initial_solutions = np.array([self.solution, solution1, solution2])
                    con_fail_dict = self.get_constraint_fail_dictionary()
                else:
                    solution1 = self.stable_matching(printing, set_solution=False)
                    solution2 = self.greedy_method(printing, set_solution=False)
                    initial_solutions = np.array([solution1, solution2])
            else:
                if con_fail_dict is None and self.solution is not None:
                    con_fail_dict = self.get_constraint_fail_dictionary()

        result = genetic_algorithm(self.parameters, self.value_parameters, pop_size=pop_size,
                                   stopping_time=stopping_time,
                                   num_crossover_points=num_crossover_points, initial_solutions=initial_solutions,
                                   mutation_rate=mutation_rate, num_time_points=num_time_points,
                                   constraints=constraints, con_fail_dict=con_fail_dict,
                                   penalty_scale=penalty_scale, time_eval=time_eval, num_mutations=num_mutations,
                                   percent_step=percent_step, con_tolerance=con_tolerance, printing=printing)
        if time_eval:
            solve_time = time.perf_counter() - start_time
            self.solution = result[0]
            self.measure_solution(printing=printing)
            time_eval_df = result[1]
            return time_eval_df, solve_time
        else:
            self.solution = result
            self.measure_solution(printing=printing)
            return result

    def solve_vft_pyomo_model(self, solver_name="cbc", approximate=True, max_time=None, report=False,
                              timing=False, add_breakpoints=True, initial=None, init_from_X=False, printing=None):
        """
        Solve the VFT model using pyomo
        :param init_from_X: if we have an X matrix to initialize the solution with
        :param initial: if this model has a warm start or not
        :param add_breakpoints: if we should add breakpoints to adjust the approximate model
        :param timing: If we want to time the model
        :param max_time: max time in seconds the solver is allowed to solve
        :param approximate: if the model is convex or not
        :param report: if we want to grab all the information to sanity check the solution
        :param solver_name: name of solver
        :param printing: if we should print something
        :return: solution
        """
        if printing is None:
            printing = self.printing

        if not approximate:
            if solver_name == 'cbc':
                solver_name = 'ipopt'

        if initial is None:
            if init_from_X:
                initial = self.init_exact_solution_from_x()

        if use_pyomo:
            model = vft_model_build(self.parameters, self.value_parameters, convex=approximate,
                                    add_breakpoints=add_breakpoints, initial=initial, printing=printing)
            if report:
                if timing:
                    self.solution, self.X, self.measure, self.value, self.pyomo_z, solve_time = vft_model_solve(
                        model, self.parameters, self.value_parameters, solve_name=solver_name, approximate=approximate,
                        max_time=max_time, report=True, timing=True, printing=printing)
                else:
                    self.solution, self.X, self.measure, self.value, self.pyomo_z = vft_model_solve(
                        model, self.parameters, self.value_parameters, solve_name=solver_name, approximate=approximate,
                        max_time=max_time, report=True, printing=printing)
            else:
                if timing:
                    self.solution, solve_time = vft_model_solve(model, self.parameters, self.value_parameters,
                                                                solve_name=solver_name, approximate=approximate,
                                                                max_time=max_time, timing=True, printing=printing)
                else:
                    self.solution = vft_model_solve(model, self.parameters, self.value_parameters,
                                                    solve_name=solver_name, approximate=approximate, max_time=max_time,
                                                    printing=printing)
        else:
            if printing:
                print('Pyomo not available')
            self.generate_random_solution()

        self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                printing=printing)
        self.solution = self.solution.astype(int)

        if timing:
            return solve_time
        else:
            return self.solution

    def solve_original_pyomo_model(self, printing=None):
        """
        Solve the original AFPC model using pyomo
        :param printing: Whether the procedure should print something
        """

        if printing is None:
            printing = self.printing

        if use_pyomo:
            model = original_pyomo_model_build(printing)
            data = convert_parameters_to_original_model_inputs(self.parameters, self.value_parameters, printing)
            self.solution = solve_original_pyomo_model(data, model, printing=printing)
        else:
            if printing:
                print('Pyomo not available')
            self.generate_random_solution()
        self.measure_solution(self.solution, self.parameters, self.value_parameters)

    def solve_gp_pyomo_model(self, gp_df_dict=None, max_time=None, solve_name='cbc', printing=None):
        """
        Solve the Goal Programming Model (Created by Lt. Reynolds)
        :param gp_df_dict: dictionary of dataframes used for parameters
        :param max_time: max solve time for the model
        :param solve_name: name of the solver
        :param printing: Whether the procedure should print something
        :return: solution
        """

        if printing is None:
            printing = self.printing

        if self.gp_parameters is None:
            self.vft_to_gp_parameters(gp_df_dict=gp_df_dict)

        r_model = gp_model_build(self.gp_parameters, printing=printing)
        self.solution, self.X = gp_model_solve(r_model, self.gp_parameters, max_time=max_time, solve_name=solve_name,
                                               printing=printing)
        self.measure_solution()

        return self.solution

    def full_vft_model_solve(self, ga_max_time=60 * 10, pyomo_max_time=10, printing=None, ga_printing=False):
        """
        This is the main method to solve the problem instance. We first solve the pyomo Approximate model, and then
        evolve it using the GA
        :param ga_max_time: the genetic algorithm's time to solve
        :param pyomo_max_time: max time to solve the pyomo model
        :param printing: Whether the procedure should print something
        :param ga_printing: If we want to print status updates during the genetic algorithm
        :return: solution z
        """
        if printing is None:
            printing = self.printing

        if printing:
            now = datetime.datetime.now()
            print('Solving VFT Model for ' + str(pyomo_max_time) + ' seconds at ' + now.strftime('%H:%M:%S') + '...')
        self.solve_vft_pyomo_model(max_time=pyomo_max_time, printing=False)

        if printing:
            now = datetime.datetime.now()
            print('Solution value of ' + str(round(self.metrics['z'], 4)) + ' obtained.')
            print('Solving Genetic Algorithm for ' + str(ga_max_time) + ' seconds at ' +
                  now.strftime('%H:%M:%S') + '...')
        self.genetic_algorithm(initialize=True, stopping_time=ga_max_time, printing=ga_printing, constraints='Fail')
        if printing:
            print('Solution value of ' + str(round(self.metrics['z'], 4)) + ' obtained.')

        return round(self.metrics['z'], 4)

    # Measure Solutions
    def get_constraint_fail_dictionary(self, metrics=None):
        """
        Get a dictionary of failed constraints. This is used since the Approximate model initial solution must be
        rounded, and we may miss a few constraints by 1 cadet each. This allows the GA to not reject the solution
        initially
        :param metrics: solution metrics
        :return: constraint fail dictionary
        """
        if metrics is None:
            metrics = self.metrics

        con_fail_dict = {}
        J = self.parameters['J']
        K_C = self.value_parameters['K_C']
        quota_k = np.where(self.value_parameters['objectives'] == 'Combined Quota')[0][0]
        for j in J:

            for k, objective in enumerate(self.value_parameters['objectives']):

                if k in K_C[j]:

                    measure = metrics['objective_measure'][j, k]
                    count = metrics['objective_measure'][j, quota_k]
                    target_quota = self.parameters['quota'][j]

                    # Constrained Approximate Measure
                    if self.value_parameters['constraint_type'][j, k] == 3:
                        value_list = self.value_parameters['objective_value_min'][j, k].split(",")
                        min_measure = float(value_list[0].strip())
                        max_measure = float(value_list[1].strip())

                        # Get correct measure constraint
                        if objective not in ['Combined Quota', 'USAFA Quota', 'ROTC Quota']:
                            if (measure * count) / target_quota < min_measure:
                                new_min = floor(1000 * (measure * count) / target_quota) / 1000
                                con_fail_dict[(j, k)] = '> ' + str(new_min)
                            elif (measure * count) / target_quota > max_measure:
                                new_max = ceil(1000 * (measure * count) / target_quota) / 1000
                                con_fail_dict[(j, k)] = '< ' + str(new_max)
                        else:
                            if measure < min_measure:
                                con_fail_dict[(j, k)] = '> ' + str(measure)
                            elif measure > max_measure:
                                con_fail_dict[(j, k)] = '< ' + str(measure)

                    # Constrained Exact Measure
                    elif self.value_parameters['constraint_type'][j, k] == 4:

                        value_list = self.value_parameters['objective_value_min'][j, k].split(",")
                        min_measure = float(value_list[0].strip())
                        max_measure = float(value_list[1].strip())

                        if measure < min_measure:
                            con_fail_dict[(j, k)] = '> ' + str(measure)
                        elif measure > max_measure:
                            con_fail_dict[(j, k)] = '< ' + str(measure)

        return con_fail_dict

    def measure_solution(self, solution=None, value_parameters=None, approximate=False, matrix=False,
                         printing=None, set_solution=True, return_z=True):
        """
        Measure a solution
        :param return_z: if we want to return z or the full metrics
        :param set_solution: if we we want to set the solution metrics as the object's attribute of metrics
        :param matrix: if we want to measure a solution matrix (x) or the vector of AFSC indices (solution)
        :param approximate: whether we measure the approximate value or not (using target quota instead of count)
        :param printing: Whether or not the procedure should print the matrix
        :param solution: either vector or matrix of matched cadets to AFSCs
        :param value_parameters: weight/value parameters
        :return: solution metrics
        """
        if solution is None:

            if matrix:
                solution = self.X
            else:
                solution = self.solution
        if value_parameters is None:
            value_parameters = self.value_parameters

        if printing is None:
            printing = self.printing

        metrics = measure_solution_quality(solution, self.parameters, value_parameters,
                                           approximate=approximate, printing=printing)

        if set_solution:
            self.metrics = metrics
            self.solution = solution

        if return_z:
            return round(metrics['z'], 4)
        else:
            return metrics

    def measure_fitness(self, solution=None, constraints='Fail', penalty_scale=1, printing=None, con_fail_dict=None,
                        first=True):
        """
        This is the fitness function method (could be slightly different depending on how the constraints are handled)
        :param first: if this is a solution in the initial population
        :param con_fail_dict: dictionary used for constraints
        :param penalty_scale: how much to penalize failed constraints
        :param constraints: how we handle failed constraints
        :param printing: whether the procedure should print something
        :param solution: solution vector
        :return: fitness score
        """
        if solution is None:
            solution = self.solution

        if printing is None:
            printing = self.printing

        if constraints == 'Fail':
            if con_fail_dict is None:
                con_fail_dict = self.get_constraint_fail_dictionary()

        metrics = ga_fitness_function(solution, self.parameters, self.value_parameters, constraints=constraints,
                                      penalty_scale=penalty_scale, con_fail_dict=con_fail_dict, first=first,
                                      printing=printing)

        return metrics

    # Observe Results
    def display_results_graph(self, graph='Average Merit', title=None, save=None, printing=None, facecolor='white',
                              figsize=(19, 7), display_title=True, metrics_dict=None, thesis_chart=False,
                              degree='Mandatory', label_size=35, afsc_tick_size=30, yaxis_tick_size=30,
                              afsc_rotation=None, xaxis_tick_size=30, dpi=100, gui_chart=False, value_type="Overall",
                              afsc='14N', title_size=None, legend_size=None, y_max=1.1, bar_color='black',
                              alpha=0.5, dot_size=100, skip_afscs=None, colors=None):
        """
        Builds the AFSC Results graphs
        :param graph: kind of results graph to show
        :param printing: whether or not to print status updates
        :param thesis_chart: if this is meant to go in the thesis document
        :param degree: which degree to show if necessary
        :param xaxis_tick_size: x axis tick sizes
        :param afsc: which AFSC to show if necessary
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
        :param skip_afscs:  Whether we should label every other AFSC
        :param dpi: dots per inch for figure
        :param bar_color: color of bars for figure (for certain kinds of graphs)
        :param alpha: alpha parameter for the bars of the figure
        :param title_size: font size of the title
        :param legend_size: font size of the legend
        :return: figure
        """

        if metrics_dict is None:
            metrics = self.metrics
        else:
            metrics = None

        if afsc_rotation is None:
            if self.data_type == "Real":
                afsc_rotation = 45
            else:
                afsc_rotation = 0

        if skip_afscs is None:
            if self.data_type == "Real":
                skip_afscs = False
            else:
                skip_afscs = True

        if thesis_chart:
            if figsize == (19, 7):
                figsize = (16, 10)
            display_title = False
            save = True
            label_size = 35
            afsc_tick_size = 30
            yaxis_tick_size = 30
            afsc_rotation = 0
            xaxis_tick_size = 30

        if graph == 'Average Merit':
            if thesis_chart and title is None:
                title = 'Results_Merit'
            chart = average_merit_results_graph(self.parameters, self.value_parameters, metrics=metrics,
                                                metrics_dict=metrics_dict, save=save, figsize=figsize,
                                                facecolor=facecolor, title=title, display_title=display_title,
                                                label_size=label_size, afsc_tick_size=afsc_tick_size,
                                                yaxis_tick_size=yaxis_tick_size, y_max=y_max, title_size=title_size,
                                                legend_size=legend_size, afsc_rotation=afsc_rotation, alpha=alpha,
                                                skip_afscs=skip_afscs, dot_size=dot_size)

        elif graph == 'AFSC Value':
            if thesis_chart and title is None:
                title = 'Results_AFSC'
            chart = afsc_value_results_graph(self.parameters, self.value_parameters, metrics=metrics,
                                             metrics_dict=metrics_dict, save=save, figsize=figsize, dpi=dpi,
                                             facecolor=facecolor, title=title, display_title=display_title,
                                             label_size=label_size, afsc_tick_size=afsc_tick_size, y_max=y_max,
                                             yaxis_tick_size=yaxis_tick_size, value_type=value_type,
                                             afsc_rotation=afsc_rotation, gui_chart=gui_chart, bar_color=bar_color,
                                             alpha=alpha, dot_size=dot_size, title_size=title_size,
                                             legend_size=legend_size, skip_afscs=skip_afscs, colors=colors)
        elif graph == 'Combined Quota':
            if thesis_chart and title is None:
                title = 'Results_Quota'
            dot_size = int(4 * dot_size / 5)
            chart = quota_fill_results_graph(self.parameters, self.value_parameters, metrics=metrics,
                                             metrics_dict=metrics_dict, save=save, figsize=figsize,
                                             facecolor=facecolor, title=title, display_title=display_title,
                                             label_size=label_size, afsc_tick_size=afsc_tick_size,
                                             yaxis_tick_size=yaxis_tick_size, y_max=y_max, skip_afscs=skip_afscs,
                                             afsc_rotation=afsc_rotation, dot_size=dot_size, title_size=title_size,
                                             legend_size=legend_size)
        elif graph == 'USAFA Proportion':
            if thesis_chart and title is None:
                title = 'Results_USAFA'
            chart = usafa_proportion_results_graph(self.parameters, self.value_parameters, metrics=metrics,
                                                   metrics_dict=metrics_dict, save=save, figsize=figsize,
                                                   facecolor=facecolor, title=title, display_title=display_title,
                                                   label_size=label_size, afsc_tick_size=afsc_tick_size,
                                                   yaxis_tick_size=yaxis_tick_size, y_max=y_max, skip_afscs=skip_afscs,
                                                   afsc_rotation=afsc_rotation, dot_size=dot_size,
                                                   title_size=title_size, legend_size=legend_size)
        elif graph == 'AFOCD Proportion':
            if thesis_chart and title is None:
                title = 'Results_' + degree
            chart = afocd_degree_proportions_results_graph(self.parameters, self.value_parameters, metrics=metrics,
                                                           metrics_dict=metrics_dict, save=save, figsize=figsize,
                                                           facecolor=facecolor, title=title,
                                                           display_title=display_title, degree=degree,
                                                           label_size=label_size, afsc_tick_size=afsc_tick_size,
                                                           yaxis_tick_size=yaxis_tick_size, y_max=y_max,
                                                           afsc_rotation=afsc_rotation, dot_size=dot_size,
                                                           title_size=title_size, legend_size=legend_size)
        elif graph == 'Average Utility':
            if thesis_chart and title is None:
                title = 'Results_Utility'
            chart = average_utility_results_graph(self.parameters, self.value_parameters, metrics=metrics,
                                                  metrics_dict=metrics_dict, save=save, figsize=figsize,
                                                  facecolor=facecolor, title=title, display_title=display_title,
                                                  label_size=label_size, afsc_tick_size=afsc_tick_size,
                                                  yaxis_tick_size=yaxis_tick_size, y_max=y_max,
                                                  afsc_rotation=afsc_rotation, bar_color=bar_color, alpha=alpha,
                                                  dot_size=dot_size, skip_afscs=skip_afscs, title_size=title_size,
                                                  legend_size=legend_size)
        elif graph == 'Cadet Utility':
            if thesis_chart and title is None:
                title = 'Results_Cadet_Utility'
            chart = cadet_utility_histogram(metrics=metrics, metrics_dict=metrics_dict, save=save, figsize=figsize,
                                            facecolor=facecolor, title=title, display_title=display_title,
                                            label_size=label_size, xaxis_tick_size=xaxis_tick_size,
                                            yaxis_tick_size=yaxis_tick_size, dpi=dpi, gui_chart=gui_chart,
                                            title_size=title_size, legend_size=legend_size)
        elif graph == 'Objective Values':
            if thesis_chart and title is None:
                title = 'Results_Objective_Values_' + afsc
            chart = afsc_objective_values_graph(self.parameters, self.value_parameters, metrics=metrics, save=save,
                                                figsize=figsize, facecolor=facecolor, title=title,
                                                display_title=display_title, label_size=label_size, y_max=y_max,
                                                xaxis_tick_size=xaxis_tick_size, afsc=afsc, metrics_dict=metrics_dict,
                                                yaxis_tick_size=yaxis_tick_size, dpi=dpi, gui_chart=gui_chart,
                                                bar_color=bar_color, alpha=alpha, dot_size=dot_size,
                                                title_size=title_size, legend_size=legend_size)
        if printing is None:
            printing = self.printing

        if printing:
            chart.show()

        return chart

    # Sensitivity Analysis
    def overall_weights_pareto_analysis(self, step=10, full_solve=True, ga_max_time=60 * 2,
                                        printing=None, filepath=None, import_df=False, thesis_chart=False, save=False,
                                        figsize=(16, 10), facecolor='white', display_title=True):

        if filepath is None:
            filepath = paths['Analysis & Results'] + self.data_name + '_Pareto_Analysis.xlsx'

        if printing is None:
            printing = self.printing

        if printing:
            print("Conducting pareto analysis on problem...")

        if import_df:
            pareto_df = import_data(filepath, sheet_name='Pareto Chart')
            chart = pareto_graph(pareto_df, dimensions=(self.parameters['N'], self.parameters['M']), save=save,
                                 figsize=figsize, facecolor=facecolor, thesis_chart=thesis_chart,
                                 display_title=display_title)
            if printing:
                chart.show()

        elif use_pyomo:

            # Initialize arrays
            num_points = int(100 / step + 1)
            cadet_overall_values = np.zeros(num_points)
            afsc_overall_values = np.zeros(num_points)
            cadet_overall_weights = np.arange(1, 0, -(step / 100))
            cadet_overall_weights = np.append(cadet_overall_weights, 0)

            # Iterate over the number of points needed for the Pareto Chart
            for point in range(num_points):
                self.value_parameters['cadets_overall_weight'] = cadet_overall_weights[point]
                self.value_parameters['afscs_overall_weight'] = 1 - cadet_overall_weights[point]

                if printing:
                    print("Calculating point " + str(point + 1) + " out of " + str(num_points) + "...")

                if full_solve:
                    self.full_vft_model_solve(ga_max_time=ga_max_time)
                else:
                    self.solve_vft_pyomo_model(max_time=10)

                cadet_overall_values[point] = self.metrics['cadets_overall_value']
                afsc_overall_values[point] = self.metrics['afscs_overall_value']

                if printing:
                    print('For an overall weight on cadets of ' + str(cadet_overall_weights[point]) +
                          ', calculated value on cadets: ' + str(round(cadet_overall_values[point], 2)) +
                          ', value on afscs: ' + str(round(afsc_overall_values[point], 2)) +
                          ', and a Z of ' + str(round(self.metrics['z'], 2)) + '.')

            pareto_df = pd.DataFrame(
                {'Weight on Cadets': cadet_overall_weights, 'Value on Cadets': cadet_overall_values,
                 'Value on AFSCs': afsc_overall_values})

            with pd.ExcelWriter(filepath) as writer:  # Export to excel
                pareto_df.to_excel(writer, sheet_name="Pareto Chart", index=False)

            chart = pareto_graph(pareto_df, dimensions=(self.parameters['N'], self.parameters['M']), save=save,
                                 figsize=figsize, facecolor=facecolor, thesis_chart=thesis_chart,
                                 display_title=display_title)

            if printing:
                chart.show()

        else:
            if printing:
                print("Pyomo not available")

    def least_squares_procedure(self, t_solution, delta=0, printing=None, show_graph=True, names=None, afsc=None,
                                colors=None, save=False, figsize=(19, 7), facecolor="white", title=None,
                                display_title=True, thesis_chart=False):

        if printing is None:
            printing = self.printing

        if use_pyomo:
            value_parameters = least_squares_procedure(self.parameters, self.value_parameters, self.solution,
                                                       t_solution, delta=delta, printing=printing)
        else:
            value_parameters = self.value_parameters

            if printing:
                print("Pyomo not available")

        if show_graph:
            if names is None:
                names = ['Set 1', 'Set 2']
            vp_dict = {names[0]: self.value_parameters, names[1]: value_parameters}
            if afsc is None:
                afsc = self.parameters['afsc_vector'][0]
            chart = self.display_afsc_objective_weights_chart(vp_dict, afsc, printing=printing, colors=colors,
                                                              save=save, figsize=figsize, facecolor=facecolor,
                                                              title=title, display_title=display_title,
                                                              thesis_chart=thesis_chart)

            if printing:
                print('t', self.measure_solution(t_solution, value_parameters=value_parameters, set_solution=False))
                print('original', self.measure_solution(self.solution, value_parameters=value_parameters,
                                                        set_solution=False))
            return chart
        else:
            return value_parameters

    def display_afsc_objective_weights_chart(self, value_parameters_dict=None, afsc=None, printing=None, colors=None,
                                             save=False, figsize=(14, 6), facecolor="white", title=None,
                                             display_title=True, thesis_chart=False, title_size=None, bar_color=None,
                                             legend_size=None, label_size=20, xaxis_tick_size=15, yaxis_tick_size=15):

        if printing is None:
            printing = self.printing

        if value_parameters_dict is None:
            value_parameters_dict = {'P1': self.value_parameters}

        if title_size is None:
            title_size = label_size
            legend_size = label_size

        if afsc is None:
            afsc = self.parameters['afsc_vector'][0]

        if thesis_chart:
            figsize = (16, 10)
            display_title = False
            save = True
            title = 'Objective_Weights_Chart'
            label_size = 35
            yaxis_tick_size = 30
            xaxis_tick_size = 30

        chart = afsc_objective_weights_graph(self.parameters, value_parameters_dict, afsc, colors=colors, save=save,
                                             figsize=figsize, facecolor=facecolor, title=title, legend_size=legend_size,
                                             display_title=display_title, label_size=label_size, title_size=title_size,
                                             xaxis_tick_size=xaxis_tick_size, yaxis_tick_size=yaxis_tick_size,
                                             bar_color=bar_color)
        if printing:
            chart.show()

        return chart

    def create_aggregate_file(self, from_files=True, printing=None):
        """
        Create the "data_type data_name" main aggregate file with solutions, metrics, and vps
        :param printing: whether to print status updates
        :param from_files: if we want to import the data from the self.instance_files or use the instance attributes
        """
        if printing is None:
            printing = self.printing

        if printing:
            print('Creating aggregate problem instance excel file...')

        if from_files:
            solution_dict = {}
            if self.vp_dict is None:
                vp_dict = {}
            else:
                vp_dict = self.vp_dict
            for full_name in self.instance_files:
                vp_name = full_name.split(' ')[2]
                solution_name = full_name.split(' ')[3]
                filepath = paths['instances'] + full_name + '.xlsx'
                if vp_name not in vp_dict:
                    value_parameters = self.import_value_parameters(filepath=filepath, set_value_parameters=False,
                                                                    printing=False)
                    vp_dict[vp_name] = copy.deepcopy(value_parameters)
                if solution_name not in solution_dict:
                    solution = self.import_solution(filepath=filepath, set_solution=False, printing=False)
                    solution_dict[solution_name] = copy.deepcopy(solution)
            metrics_dict = {}
            for vp_name in vp_dict:
                value_parameters = vp_dict[vp_name]
                vp_dict[vp_name]['vp_weight'] = 1 / len(list(vp_dict.keys()))
                metrics_dict[vp_name] = {}
                for solution_name in solution_dict:
                    solution = solution_dict[solution_name]
                    metrics = self.measure_solution(solution, value_parameters, set_solution=False, return_z=False,
                                                    printing=False)
                    metrics_dict[vp_name][solution_name] = copy.deepcopy(metrics)
        else:
            metrics_dict = self.metrics_dict
            solution_dict = self.solution_dict
            vp_dict = self.vp_dict

        full_name = self.data_type + " " + self.data_name
        create_aggregate_instance_file(full_name, self.parameters, solution_dict, vp_dict, metrics_dict)

    # Other
    def find_solution_parameter_ineligibility(self, solution=None, filepath=None):

        if solution is None:
            if filepath is None:
                solution = self.solution
            else:
                self.import_solution(filepath)
                solution = self.solution

        find_original_solution_ineligibility(self.parameters, solution)

    # Export
    def pyomo_measures_to_excel(self, filepath=None):
        if filepath is None:
            if '.xlsx' in self.filepath:
                data_name = self.filepath[:-len('.xlsx')]
                filepath = data_name + '_X.xlsx'
            else:
                filepath = 'X_matrix.xlsx'
        pyomo_measures_to_excel(self.X, self.measure, self.value, self.parameters, self.value_parameters,
                                filepath=filepath, printing=self.printing)

    def export_to_excel(self, filepath=None, printing=None):

        if printing is None:
            printing = self.printing

        if filepath is None:
            if self.filepath is None:
                self.filepath = paths['instances'] + "New_Matching_Instance.xlsx"
            else:
                filepath = self.filepath

        # Export to excel
        data_to_excel(filepath, self.parameters, self.value_parameters, self.metrics, printing=printing)

        if printing:
            print('Exported.')
