# Import libraries
import pandas as pd

from afccp.core.comprehensive_functions import *
import datetime
import glob
import copy


class CadetCareerProblem:
    def __init__(self, data_name=None, N=1600, M=32, P=6, num_breakpoints=None, printing=False):
        """
        This is the initialization function for the AFSC/Cadet problem. We can import data using the data_name.
        We can also generate data by providing a data_name that contains "Random", "Realistic",
        or "Perfect" in the name.
        :param data_name: name of the data set. If we want to import data using the data_name, this needs to be
        the name of the excel file without the .xlsx extension
        :param N: Number of cadets to generate
        :param M: Number of AFSCs to generate
        :param P: Number of AFSC preferences to generate for each cadet
        :param num_breakpoints: Number of breakpoints to use for AFSC value functions
        :param printing: Whether we should print status updates or not
        """

        # Find out if we're dealing with "scrubbed" information or not
        if data_name is None:  # we're going to generate random data
            self.scrubbed = True
        else:
            if len(data_name) == 4:  # class year
                self.scrubbed = False
            else:
                self.scrubbed = True

        # Get list of specific data_name instances
        directory = paths_in['instances']
        self.generated_data_names = {'Random': [], 'Perfect': [], 'Realistic': []}
        real_instance_data_names = []
        for file_name in glob.iglob(directory + '*.xlsx', recursive=True):
            start_index = file_name.find(paths_in['instances']) + len(paths_in['instances']) + 1
            end_index = len(file_name) - 5
            full_name = file_name[start_index:end_index]
            sections = full_name.split(' ')
            d_name = sections[1]
            for variant in self.generated_data_names:
                if variant in d_name and d_name not in self.generated_data_names[variant]:
                    self.generated_data_names[variant].append(d_name)
            if len(d_name) in [1, 4]:  # Letter or Class Year
                real_instance_data_names.append(full_name)
        self.real_instance_data_names = np.array(real_instance_data_names)

        # Get correct data attributes
        if data_name is None:

            # If we didn't specify a data_name, we're just going to generate a random set of cadets
            self.data_name = "Random_" + str(len(self.generated_data_names["Random"]) + 1)
            self.data_type = "Generated"
            self.data_variant = "Random"
            generate = True
        else:

            # We specified a data_name
            self.data_name = data_name

            # Initially assume it's a "real" set of data
            self.data_type = "Real"
            generate = False

            # Loop through "Random", "Realistic", "Perfect"
            for data_variant in self.generated_data_names:

                # If we passed one of those three names generally, we will generate a new instance of that kind
                if data_variant == self.data_name:
                    self.data_name = data_variant + "_" + str(len(self.generated_data_names[data_variant]) + 1)
                    self.data_type = "Generated"
                    generate = True
                    break

                # If we specified a specific version ("Random_4" for example), then we'll load it in
                elif data_variant in self.data_name and '_' in self.data_name:
                    self.data_type = "Generated"
                    generate = False
                    break

        # Get data variant
        if self.data_type == "Real":

            # Is it a class year or a scrubbed class year
            if len(self.data_name) == 1:
                self.data_variant = "Scrubbed"
            else:
                self.data_variant = "Year"
        else:

            # Figure out if the data is "Random", "Realistic", or "Perfect"
            for data_variant in self.generated_data_names:
                if data_variant in self.data_name:
                    self.data_variant = data_variant

        # Create the "full name" which we piece together all the current available information
        self.full_name = self.data_type + ' ' + self.data_name   # "Real A" for example
        self.data_instance_name = copy.deepcopy(self.full_name)

        # Get correct filepath  (for loading in the instance right now)
        self.filepath_in = paths_in['instances'] + self.data_instance_name + '.xlsx'

        # Get list of instance filenames (Could be more than just the main file)
        directory = paths_in['instances']
        main_file = False  # if there is a "data_type data_name.xlsx" file
        self.instance_files = []
        for file_name in glob.iglob(directory + '*.xlsx', recursive=True):
            start_index = file_name.find(paths_in['instances']) + len(paths_in['instances']) + 1
            end_index = len(file_name) - 5
            full_name = file_name[start_index:end_index]
            sections = full_name.split(' ')
            d_name = sections[1]
            if d_name == self.data_name and len(sections) != 2:
                self.instance_files.append(file_name)
            elif d_name == self.data_name and len(sections) == 2:
                main_file = True

        # initialize more instance attributes
        self.printing = printing
        self.default_value_parameters = None
        self.value_parameters = None
        self.metrics = None
        self.x = None
        self.measure = None
        self.value = None
        self.pyomo_z = None
        self.year = None
        self.solution = None
        self.gp_parameters = None
        self.gp_df = None
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

            if main_file:  # If we're importing from the main instance file
                filepath = self.filepath_in
            else:  # If we're importing from a specific solution file (though I really want to "phase this out"
                try:
                    filepath = self.instance_files[0]
                except:
                    raise ValueError("Instance '" + self.data_name + "' not found at path " + paths_in['instances'])

            self.parameters, self.vp_dict, self.solution_dict, self.metrics_dict, self.gp_df = \
                import_aggregate_instance_file(filepath, num_breakpoints=num_breakpoints)

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

        if skip_afscs is None:
            if self.data_variant == "Year":
                skip_afscs = False
            else:
                if num_afscs < self.parameters['M']:
                    skip_afscs = False
                else:
                    skip_afscs = True

        if afsc_rotation is None:
            if skip_afscs:
                afsc_rotation = 0
            else:
                if self.data_variant == "Year":
                    if num_afscs > 18:
                        afsc_rotation = 45
                    else:
                        afsc_rotation = 0
                else:
                    if num_afscs < 25:
                        afsc_rotation = 0
                    else:
                        afsc_rotation = 45

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

        if 'cip1' in parameters:
            if 'cip2' in parameters:
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
            else:
                raise ValueError("No 'cip2' column in parameters")
        else:
            raise ValueError("No 'cip1' column in parameters")

    # Specify Value Parameters
    def import_value_parameters(self, filepath=None, vp_name=None, num_breakpoints=None, set_value_parameters=True,
                                printing=None):
        """
        Imports weight and value parameters from a problem instance excel file
        :param vp_name: name of value parameters
        :param set_value_parameters: if we want to set these value parameters as our object attribute
        :param filepath: filepath to import from (if none specified, will use filepath attribute)
        :param num_breakpoints: Number of breakpoints to use for the value functions
        :param printing: Whether we should print status updates or not
        """
        if filepath is None and len(self.instance_files) == 0:
            raise ValueError('No instance files found that contain value parameters (besides aggregate file). '
                             'Either generate new ones or load them in from the value parameters dictionary.')
        if vp_name is not None:
            full_name = self.data_type + " " + self.data_name + " " + vp_name
            if filepath is None:
                filepath = [file_name for file_name in self.instance_files if full_name in file_name][0]
        else:
            if filepath is None:
                filepath = self.instance_files[0]
            vp_name = self.instance_files[0].split(' ')[2]

        if printing is None:
            printing = self.printing

        # Import value parameters
        try:
            value_parameters = model_value_parameters_from_excel(self.parameters, filepath, printing=printing,
                                                                 num_breakpoints=num_breakpoints)
        except:
            raise ValueError('Filepath invalid for this particular problem instance.')

        value_parameters = model_value_parameters_set_additions(value_parameters)
        value_parameters = condense_value_functions(self.parameters, value_parameters)
        value_parameters = model_value_parameters_set_additions(value_parameters, printing)

        if set_value_parameters:
            self.value_parameters = value_parameters

            # Set correct names
            self.vp_name = vp_name
            self.full_name = self.data_type + " " + self.data_name + " " + self.vp_name

        if printing:
            print('Imported.')

        return value_parameters

    def set_instance_value_parameters(self, vp_name=None):
        """
        Sets the current instance value parameters to a specified set based on the vp_name. This vp_name must be
        in the value parameter dictionary
        :param printing: print status updates or not
        :param vp_name: name of value parameters to set as the instance's value parameters
        """
        if self.vp_dict is None:
            raise ValueError('Value parameter dictionary is still empty')
        else:
            if vp_name is None:
                self.vp_name = list(self.vp_dict.keys())[0]
                self.value_parameters = copy.deepcopy(self.vp_dict[self.vp_name])
            else:
                if vp_name not in self.vp_dict:
                    raise ValueError(vp_name + ' set not in value parameter dictionary')
                else:
                    self.value_parameters = copy.deepcopy(self.vp_dict[vp_name])
                    self.vp_name = vp_name

            if self.solution is not None:
                self.measure_solution()

            # Update current specific instance name
            self.full_name = self.data_type + " " + self.data_name + " " + self.vp_name

    def save_new_value_parameters_to_dict(self, value_parameters=None, vp_name=None):
        """
        Adds the set of value parameters to a dictionary as a new set (if it is a unique set)
        :param value_parameters: set of value parameters
        :param vp_name: name of the new set of value parameters (defaults to adding 1 to the last set in the dict)
        """
        if value_parameters is None:
            value_parameters = self.value_parameters

        if value_parameters is not None:
            if self.vp_dict is None:
                self.vp_dict = {}
                if vp_name is None:
                    vp_name = "VP"
                unique = True  # If this is the first set of value parameters, it's unique!
            else:
                if vp_name is None:
                    high = 1
                    for vp_name in self.vp_dict:
                        if '_' in vp_name:
                            num = int(vp_name.split('_')[1])
                            if num > high:
                                high = num
                    vp_name = "VP_" + str(high + 1)

                # Check if this new set is unique or not to get the name of the set
                unique = self.check_unique_value_parameters()

            if unique is True:  # If it's unique, we save this new set of value parameters to the dictionary
                self.vp_dict[vp_name] = copy.deepcopy(value_parameters)
                self.vp_name = vp_name
            else:  # If it's not unique, then the "unique" variable is the name of the matching set of value parameters
                self.vp_name = unique

            # Update current specific instance name
            self.full_name = self.data_type + " " + self.data_name + " " + self.vp_name
        else:
            raise ValueError('No instance value parameters detected')

    def update_value_parameters_in_dict(self, vp_name=None):
        """
        Updates a set of value parameters in the dictionary using the current instance value parameters
        :param vp_name: name of the set of value parameters to update (default current vp_name)
        """
        if self.value_parameters is not None:
            if self.vp_dict is None:
                raise ValueError('No value parameter dictionary detected')
            else:
                if vp_name is None:
                    vp_name = self.vp_name

                # Set attributes
                self.vp_dict[vp_name] = copy.deepcopy(self.value_parameters)
                self.vp_name = vp_name

                # Update current specific instance name
                self.full_name = self.data_type + " " + self.data_name + " " + self.vp_name
        else:
            raise ValueError('No instance value parameters detected')

    def check_unique_value_parameters(self, value_parameters=None, printing=False):
        """
        Take in a new set of value parameters and see if this set is in the dictionary already. Return True if the
        the set of parameters is unique, or return the name of the matching set otherwise
        :param printing: if we want to print out which value parameter sets are different
        :param value_parameters: set of value parameters (presumably the instance attributes)
        """
        if value_parameters is None:
            value_parameters = self.value_parameters

        # Assume the new set is unique until proven otherwise
        unique = True
        for vp_name in self.vp_dict:
            identical = compare_value_parameters(self.parameters, value_parameters, self.vp_dict[vp_name],
                                                 printing=printing)
            if identical:
                unique = vp_name
                break
        return unique

    def import_default_value_parameters(self, filename=None, filepath=None, no_constraints=False, num_breakpoints=24,
                                        generate_afsc_weights=True, set_to_instance=True, add_to_dict=True,
                                        vp_weight=100, printing=None):
        """
        Import default value parameter setting and generate value parameters for this instance from those
        ones.
        :param filename: filename (assumes support_paths[data variant])
        :param filepath: filepath to import from (if none specified, will use filepath attribute)
        :param num_breakpoints: Number of breakpoints to use for the value functions
        :param set_to_instance: if we want to set this set to the instance's value parameters attribute
        :param add_to_dict: if we want to add this set of value parameters to the vp dictionary
        :param vp_weight: swing weight of this entire set of value parameters relative to others
        :param printing: Whether we should print status updates or not
        :param no_constraints: If we don't want to use the predefined constraints
        :param generate_afsc_weights: If we generate the AFSC weights from the weight function, or just
        import the weights directly from the "AFSC Weight" column
        """

        if printing is None:
            printing = self.printing

        # check first if the user did not specify a filename (just name of excel file within correct folder)
        if filename is None:

            # if filename not specified, check if file path was not directly specified
            if filepath is None:  # default value parameters
                if self.data_variant == "Scrubbed":
                    filepath = support_paths['scrubbed'] + 'Value_Parameters_Defaults_' + self.data_name + '.xlsx'
                elif self.data_variant == 'Year':
                    filepath = support_paths['real'] + 'Value_Parameters_Defaults.xlsx'
                elif self.data_variant == 'Perfect':
                    filepath = support_paths['scrubbed'] + 'Value_Parameters_Defaults_Perfect.xlsx'
                else:
                    filepath = support_paths['scrubbed'] + 'Value_Parameters_Defaults_Generated.xlsx'
            else:
                pass  # we have given a filepath already, no more info is needed
        else:
            if self.data_variant == "Year":
                filepath = support_paths['real'] + filename
            else:
                filepath = support_paths['scrubbed'] + filename

            if '.xlsx' not in filepath:
                filepath += '.xlsx'

        self.default_value_parameters = default_value_parameters_from_excel(filepath, num_breakpoints=num_breakpoints,
                                                                            printing=printing)
        value_parameters = generate_value_parameters_from_defaults(
            self.parameters, generate_afsc_weights=generate_afsc_weights,
            default_value_parameters=self.default_value_parameters)

        if no_constraints:
            value_parameters['constraint_type'] = np.zeros([self.parameters['M'], value_parameters['O']])
        value_parameters = model_value_parameters_set_additions(value_parameters)
        value_parameters = condense_value_functions(self.parameters, value_parameters)
        value_parameters = model_value_parameters_set_additions(value_parameters)
        value_parameters['vp_weight'] = vp_weight

        # Set value parameters to instance attribute
        if set_to_instance:
            self.value_parameters = value_parameters
            if self.solution is not None:
                self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                        printing=printing)

        # Save new set of value parameters to dictionary
        if add_to_dict:
            self.save_new_value_parameters_to_dict(value_parameters)

        if self.printing:
            print('Imported.')

        return value_parameters

    def generate_realistic_value_parameters(self, default_value_parameters=None, constraints_df=None,
                                            deterministic=True, set_to_instance=True, add_to_dict=True,
                                            vp_weight=100, printing=None, constrain_merit=False):
        """
        This procedure only works on actual class years (scrubbed and "real"). We have data on what the quotas and
        constraints were supposed to be, as well as what was actually implemented with the original solution for each
        class year. This is how we can generate more accurate sets of value parameters based on what we know about the
        original solution.
        :param default_value_parameters: optional set of default value parameters (we would just import it
        otherwise)
        :param set_to_instance: if we want to set this set to the instance's value parameters attribute
        :param add_to_dict: if we want to add this set of value parameters to the vp dictionary
        :param vp_weight: swing weight of this entire set of value parameters relative to others
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
            if self.data_variant == "Scrubbed":
                filepath = support_paths['scrubbed'] + 'Value_Parameters_Defaults_' + self.data_name + '.xlsx'
            elif self.data_variant == 'Year':
                filepath = support_paths['real'] + 'Value_Parameters_Defaults.xlsx'
            else:
                raise ValueError('Data type must be Real not Generated.')

            default_value_parameters = default_value_parameters_from_excel(filepath)
            self.default_value_parameters = default_value_parameters

        if self.data_type == "Real":  # Not scrubbed AFSC data
            data_type = self.data_name  # Pull in the class year used
        else:
            data_type = self.data_type

        if constraints_df is None:
            if self.data_variant == "Year":
                constraints_df = import_data(
                    support_paths['real'] + 'Value_Parameter_Sets_Options.xlsx',
                    sheet_name=data_type + ' Constraint Options')
            else:
                constraints_df = import_data(
                    support_paths['scrubbed'] + 'Value_Parameter_Sets_Options.xlsx',
                    sheet_name=data_type + ' Constraint Options')

        value_parameters = value_parameter_realistic_generator(self.parameters, default_value_parameters,
                                                               constraints_df, deterministic=deterministic,
                                                               constrain_merit=constrain_merit,
                                                               data_name=data_type)
        value_parameters = model_value_parameters_set_additions(value_parameters)
        value_parameters = condense_value_functions(self.parameters, value_parameters)
        value_parameters = model_value_parameters_set_additions(value_parameters)
        value_parameters['vp_weight'] = vp_weight

        # Set value parameters to instance attribute
        if set_to_instance:
            self.value_parameters = value_parameters
            if self.solution is not None:
                self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                        printing=printing)

        # Save new set of value parameters to dictionary
        if add_to_dict:
            self.save_new_value_parameters_to_dict(value_parameters)

        if printing:
            print('Generated.')

        return value_parameters

    def export_value_parameters_as_defaults(self, filename=None, filepath=None, printing=None):
        """
        This method exports the current set of instance value parameters to a new excel file in the "default"
        value parameter format
        :param printing: status updates
        :param filename: name of the new default value parameters sheet (defaults to "Value_Parameters_Defaults_New")
        :param filepath: path to the file (defaults to support_paths[])
        """
        if printing is None:
            printing = self.printing

        if self.value_parameters is None:
            raise ValueError('No instance value parameters detected.')
        else:
            if filename is None:
                filename = "Value_Parameters_Defaults_New.xlsx"

            if filepath is None:
                if self.scrubbed:
                    filepath = support_paths['scrubbed'] + filename
                else:
                    filepath = support_paths['real'] + filename

            model_value_parameters_to_defaults(self.parameters, self.value_parameters, filepath=filepath,
                                               printing=printing)

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
            else:  # Equal
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

            else:  # Piece

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

    # Translate Parameters
    def vft_to_gp_parameters(self, use_gp_df=True, get_new_rewards_penalties=False, solver_name='cbc', executable=None,
                             provide_executable=False, printing=None):
        """
        Converts the instance parameters and value parameters to parameters used by Rebecca's model
        :param solver_name: name of solver
        :param executable: path of the solver
        :param provide_executable: if we want to provide an executable directly
        :param use_gp_df: if we want to use the dataframe of normalized rewards and penalties
        :param get_new_rewards_penalties: if we want to create a new gp df or just import the pre-determined one
        :param printing: Whether we should print status updates or not
        """

        if printing is None:
            printing = self.printing

        if use_gp_df:

            # Get basic parameters (May or may not include penalty/reward parameters
            self.gp_parameters = translate_vft_to_gp_parameters(self.parameters, self.value_parameters, self.gp_df,
                                                                use_gp_df)

            # Use generalized "GP DF"
            if self.gp_df is None:
                filepath = support_paths['scrubbed'] + 'GP_Parameters.xlsx'
                self.gp_df = import_data(filepath=filepath, sheet_name='Weights and Scaling')
                specific_gp_df = False
            else:
                specific_gp_df = True

            # Get list of constraints
            con_list = copy.deepcopy(self.gp_parameters['con'])
            con_list.append('S')
            num_constraints = len(con_list)  # should be 12

            # Either create new rewards and penalties for this specific instance
            if get_new_rewards_penalties:
                rewards, penalties = calculate_rewards_penalties(self.gp_parameters, solver_name, executable,
                                                                 provide_executable, printing)
                min_penalty = min([penalty for penalty in penalties if penalty != 0])
                min_reward = min(rewards)
                norm_penalties = np.array([min_penalty / penalty if penalty != 0 else 0 for penalty in penalties])
                norm_rewards = np.array([min_reward / reward for reward in rewards])
            else:
                if 'Raw Reward' in self.gp_df:
                    rewards, penalties = np.array(self.gp_df['Raw Reward']), np.array(self.gp_df['Raw Penalty'])
                else:
                    rewards, penalties = np.array(['Unk' for _ in range(num_constraints)]), \
                                         np.array(['Unk' for _ in range(num_constraints)])
                norm_rewards, norm_penalties = np.array(self.gp_df['Normalized Reward']), \
                                               np.array(self.gp_df['Normalized Penalty'])
            if not specific_gp_df:
                penalty_weight = [100, 100, 90, 30, 30, 25, 50, 50, 50, 50, 25, 0]
                reward_weight = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100]
            else:
                penalty_weight = np.array(self.gp_df['Penalty Weight'])
                reward_weight = np.array(self.gp_df['Reward Weight'])

            run_penalties = np.array(
                [penalty_weight[c] / norm_penalties[c] if penalties[c] != 0 else 0 for c in range(num_constraints)])
            run_rewards = np.array([reward_weight[c] / norm_rewards[c] for c in range(num_constraints)])

            self.gp_df = pd.DataFrame({'Constraint': con_list, 'Raw Penalty': penalties, 'Raw Reward': rewards,
                                       'Normalized Penalty': norm_penalties,
                                       'Normalized Reward': norm_rewards, 'Penalty Weight': penalty_weight,
                                       'Reward Weight': reward_weight, 'Run Penalty': run_penalties,
                                       'Run Reward': run_rewards})

        self.gp_parameters = translate_vft_to_gp_parameters(self.parameters, self.value_parameters, self.gp_df,
                                                            use_gp_df, printing=printing)

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

        if skip_afscs is None:
            if self.data_variant == "Real":
                skip_afscs = False
            else:
                skip_afscs = True

        if afsc_rotation is None:
            if skip_afscs:
                afsc_rotation = 0
            else:
                if self.parameters['M'] < 18:
                    afsc_rotation = 45
                else:
                    afsc_rotation = 0

        if thesis_chart:
            figsize = (16, 10)
            display_title = False
            save = True
            label_size = 35
            afsc_tick_size = 30
            yaxis_tick_size = 30
            xaxis_tick_size = 30
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

    # Import Solutions
    def import_solution(self, filepath, excel_format="Specific", set_to_instance=True, add_to_dict=True,
                        printing=None):
        """
        This method imports a solution from excel
        :param excel_format: kind of excel sheet we're importing from
        :param set_to_instance: if we want to set this solution to the instance's solution attribute
        :param add_to_dict: if we want to add this solution to the solution dictionary
        :param filepath: filepath to import the solution from
        :param printing: Whether we should print status updates or not
        :return solution
        """

        if printing is None:
            printing = self.printing

        # Import solution
        solution = import_solution_from_excel(filepath=filepath, excel_format=excel_format, printing=printing)

        # Set the solution attribute
        if set_to_instance:
            self.solution = solution
            self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                    printing=printing)

        # Add solution to solution dictionary
        if add_to_dict:
            self.add_solution_to_dictionary(solution, solution_method="Original")

        if printing:
            print('Imported.')

        # Return solution
        return solution

    def scrub_afsc_solution(self, year, filepath=None, solution_name=None, set_to_instance=True, add_to_dict=True,
                            printing=None):
        """
        Imports a real class year solution (sensitive data) and converts the real AFSCs into the scrubbed versions
        :param set_to_instance: if we want to set this solution to the instance's solution attribute
        :param add_to_dict: if we want to add this solution to the solution dictionary
        :param solution_name: name of the solution to the real class year sensitive file
        :param filepath: filepath of the solution (optional)
        :param year: year of the solution
        :param printing: Whether we should print status updates or not
        :return: solution
        """

        if printing is None:
            printing = self.printing

        if sensitive_folder:

            # Figure out how we're importing the solution
            full_name = "Real " + str(year)
            excel_format = 'Aggregate'

            # If we don't specify a filepath directly
            if filepath is None:

                # If we don't specify a solution name directly
                if solution_name is None:

                    # If the "full name" is the name of a file, we will import the solution from that file
                    if full_name in self.real_instance_data_names:
                        filepath = paths_in['instances'] + full_name + '.xlsx'
                    else:

                        # If we need to specify a solution from a specific solution excel file
                        potential_names = []
                        for potential_name in self.real_instance_data_names:

                            # Get a list of potential names for this particular data instance
                            if full_name in potential_name:
                                potential_names.append(potential_name)

                        # If there are no class year instance files, raise error
                        if len(potential_names) > 0:

                            # Get list of solutions that we have for this class year
                            solution_names = []
                            solution_full_names = []
                            for potential_name in potential_names:
                                split_list = potential_name.split(' ')
                                if len(split_list) == 4:
                                    solution_names.append(split_list[3])
                                    solution_full_names.append(potential_name)

                            # If we have at least one solution, we arbitrarily take the first one
                            if len(solution_names) > 0:
                                solution_name = solution_names[0]
                                filepath = paths_in['instances'] + solution_full_names[0] + '.xlsx'
                                excel_format = 'Specific'
                            else:
                                raise ValueError(str(year) + ' files exist, but no solutions detected.')
                        else:
                            raise ValueError('No ' + str(year) + ' files detected.')
            else:
                start_index = filepath.find(paths_in['instances']) + len(paths_in['instances']) + 1
                end_index = len(filepath) - 5
                full_name = filepath[start_index:end_index]
                sections = full_name.split(' ')
                if len(sections) == 4:
                    solution_name = sections[3]
                    excel_format = 'Specific'
                elif len(sections) != 2:
                    raise ValueError('Aggregate solution excel file not provided.')

            # Import real class year solution
            real_solution = import_solution_from_excel(filepath=filepath, solution_name=solution_name,
                                                       excel_format=excel_format)
            year_afsc_table = import_data(filepath=support_paths['real'] + "Year_AFSCs_Table.xlsx",
                                          sheet_name=str(year))
            real_afscs = np.array(year_afsc_table['AFSC'])
            old_afscs = np.sort(real_afscs)
            solution = np.zeros(self.parameters['N']).astype(int)
            for j, afsc in enumerate(real_afscs):
                _name = self.parameters['afsc_vector'][j]
                old_j = np.where(old_afscs == afsc)[0][0]
                indices = np.where(real_solution == old_j)[0]
                solution[indices] = j

            # Set the solution attribute
            if set_to_instance:
                self.solution = solution
                self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                        printing=printing)

            # Add solution to solution dictionary
            if add_to_dict:
                self.add_solution_to_dictionary(solution, solution_name="Real " + solution_name)

            if printing:
                print('Imported.')

            # Return solution
            return solution

        else:
            raise ValueError('No sensitive folder found. Cannot load real AFSC solutions')

    # Solve Models
    def generate_random_solution(self, set_to_instance=True, add_to_dict=True, printing=None):
        """
        Generate random solution by assigning cadets to AFSCs that they're eligible for
        :param set_to_instance: if we want to set this solution to the instance's solution attribute
        :param add_to_dict: if we want to add this solution to the solution dictionary
        :param printing: Whether or not to print status updates
        """
        if printing is None:
            printing = self.printing

        if printing:
            print('Generating random solution...')

        solution = np.array([np.random.choice(self.parameters['J^E'][i]) for i in self.parameters['I']])

        # Set the solution attribute
        if set_to_instance:
            self.solution = solution
            self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                    printing=printing)

        # Add solution to solution dictionary
        if add_to_dict:
            self.add_solution_to_dictionary(solution, solution_method='Greedy')

        return solution

    def stable_matching(self, set_to_instance=True, add_to_dict=True, printing=None):
        """
        This method solves the stable marriage heuristic for an initial solution
        :param printing: Whether or not to print status updates
        :param set_to_instance: if we want to set this solution to the instance's solution attribute
        :param add_to_dict: if we want to add this solution to the solution dictionary
        :return: solution vector
        """
        if printing is None:
            printing = self.printing

        solution = stable_marriage_model_solve(self.parameters, self.value_parameters, printing=printing)

        # Set the solution attribute
        if set_to_instance:
            self.solution = solution
            self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                    printing=printing)

        # Add solution to solution dictionary
        if add_to_dict:
            self.add_solution_to_dictionary(solution, solution_method='Stable')

        return solution

    def greedy_method(self, set_to_instance=True, add_to_dict=True, printing=None):
        """
        This method solves the greedy heuristic for an initial solution
        :param printing: Whether or not to print status updates
        :param set_to_instance: if we want to set this solution to the instance's solution attribute
        :param add_to_dict: if we want to add this solution to the solution dictionary
        :return: solution vector
        """
        if printing is None:
            printing = self.printing

        solution = greedy_model_solve(self.parameters, self.value_parameters, printing=printing)

        # Set the solution attribute
        if set_to_instance:
            self.solution = solution
            self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                    printing=printing)

        # Add solution to solution dictionary
        if add_to_dict:
            self.add_solution_to_dictionary(solution, solution_method='Greedy')

        return solution

    def genetic_algorithm(self, initialize=True, pop_size=10, stopping_time=60, num_crossover_points=3,
                          initial_solutions=None, mutation_rate=0.05, num_time_points=100, constraints="None",
                          penalty_scale=10, time_eval=False, num_mutations=None, percent_step=10, add_to_dict=True,
                          con_tolerance=0.95, printing=None, con_fail_dict=None, set_to_instance=True):
        """
        This is the genetic algorithm. The hyper-parameters to the algorithm can be tuned, and this is meant to be
        solved in conjunction with the pyomo model solution. Use that as the initial solution, and then we evolve
        from there
        :param set_to_instance: if we want to set this solution to the instance's solution attribute
        :param add_to_dict: if we want to add this solution to the solution dictionary
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
                    solution1 = self.stable_matching(set_to_instance=False, add_to_dict=False, printing=printing)
                    solution2 = self.greedy_method(set_to_instance=False, add_to_dict=False, printing=printing)
                    initial_solutions = np.array([self.solution, solution1, solution2])
                    con_fail_dict = self.get_constraint_fail_dictionary()
                else:
                    solution1 = self.stable_matching(set_to_instance=False, add_to_dict=False, printing=printing)
                    solution2 = self.greedy_method(set_to_instance=False, add_to_dict=False, printing=printing)
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
            time_eval_df = result[1]
            solution = result[0]
            return_object = time_eval_df, solve_time
        else:
            solution = result
            return_object = solution

        # Set the solution attribute
        if set_to_instance:
            self.solution = solution
            self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                    printing=printing)

        # Add solution to solution dictionary
        if add_to_dict:
            self.add_solution_to_dictionary(solution, solution_method="Genetic")

        return return_object

    def solve_vft_pyomo_model(self, solver_name="cbc", approximate=True, max_time=None, report=False, timing=False,
                              add_breakpoints=True, initial=None, init_from_X=False, set_to_instance=True,
                              add_to_dict=True, executable=None, provide_executable=True,
                              printing=None):
        """
        Solve the VFT model using pyomo
        :param set_to_instance: if we want to set this solution to the instance's solution attribute
        :param add_to_dict: if we want to add this solution to the solution dictionary
        :param init_from_X: if we have an X matrix to initialize the solution with
        :param initial: if this model has a warm start or not
        :param add_breakpoints: if we should add breakpoints to adjust the approximate model
        :param timing: If we want to time the model
        :param max_time: max time in seconds the solver is allowed to solve
        :param approximate: if the model is convex or not
        :param report: if we want to grab all the information to sanity check the solution
        :param solver_name: name of solver
        :param executable: path of the solver
        :param provide_executable: if we want to provide an executable directly
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
                    solution, self.x, self.measure, self.value, self.pyomo_z, solve_time = vft_model_solve(
                        model, self.parameters, self.value_parameters, solver_name=solver_name, approximate=approximate,
                        max_time=max_time, report=True, timing=True, executable=executable,
                        provide_executable=provide_executable, printing=printing)
                else:
                    solution, self.x, self.measure, self.value, self.pyomo_z = vft_model_solve(
                        model, self.parameters, self.value_parameters, solver_name=solver_name, approximate=approximate,
                        max_time=max_time, report=True, executable=executable,
                        provide_executable=provide_executable, printing=printing)
            else:
                if timing:
                    solution, solve_time = vft_model_solve(model, self.parameters, self.value_parameters,
                                                           solver_name=solver_name, approximate=approximate,
                                                           max_time=max_time, timing=True, executable=executable,
                                                           provide_executable=provide_executable, printing=printing)
                else:
                    solution = vft_model_solve(model, self.parameters, self.value_parameters, solver_name=solver_name,
                                               approximate=approximate, max_time=max_time, executable=executable,
                                               provide_executable=provide_executable, printing=printing)
        else:
            if printing:
                raise ValueError('Pyomo not available')

        # Set the solution attribute
        if set_to_instance:
            self.solution = solution
            self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                    printing=printing)

        # Add solution to solution dictionary
        if add_to_dict:
            if approximate:
                solution_method = "A-VFT"
            else:
                solution_method = "E-VFT"
            self.add_solution_to_dictionary(solution, solution_method=solution_method)

        if timing:
            return solve_time
        else:
            return solution

    def solve_original_pyomo_model(self, add_to_dict=True, set_to_instance=True, max_time=None, executable=None,
                                   provide_executable=True, printing=None):
        """
        Solve the original AFPC model using pyomo
        :param max_time: max time allowed by the solver
        :param set_to_instance: if we want to set this solution to the instance's solution attribute
        :param add_to_dict: if we want to add this solution to the solution dictionary
        :param executable: path of the solver
        :param provide_executable: if we want to provide an executable directly
        :param printing: Whether the procedure should print something
        """

        if printing is None:
            printing = self.printing

        if use_pyomo:
            model = original_pyomo_model_build(printing)
            data = convert_parameters_to_original_model_inputs(self.parameters, self.value_parameters, printing)
            solution = solve_original_pyomo_model(data, model, max_time=max_time, provide_executable=provide_executable,
                                                  executable=executable, printing=printing)
        else:
            raise ValueError('Pyomo not available')

        # Set the solution attribute
        if set_to_instance:
            self.solution = solution
            self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                    printing=printing)

        # Add solution to solution dictionary
        if add_to_dict:
            self.add_solution_to_dictionary(solution, solution_method="AFPC")

    def solve_gp_pyomo_model(self, max_time=None, solver_name='cbc', add_to_dict=True, set_to_instance=True,
                             executable=None, provide_executable=False, con_term=None,
                             get_reward=False, printing=None):
        """
        Solve the Goal Programming Model (Created by Lt. Reynolds)
        :param con_term: constraint to solve the model for (if none, it's the regular model)
        :param get_reward: if we want to solve for the reward (lambda) or penalty (mu) term
        :param executable: optional path to solver
        :param provide_executable: if we want to directly provide an executable path
        :param set_to_instance: if we want to set this solution to the instance's solution attribute
        :param add_to_dict: if we want to add this solution to the solution dictionary
        :param max_time: max solve time for the model
        :param solver_name: name of the solver
        :param printing: Whether the procedure should print something
        :return: solution
        """

        if printing is None:
            printing = self.printing

        if self.gp_parameters is None:
            self.vft_to_gp_parameters()

        r_model = gp_model_build(self.gp_parameters, con_term=con_term, get_reward=get_reward, printing=printing)
        gp_var = gp_model_solve(r_model, self.gp_parameters, max_time=max_time, solver_name=solver_name,
                                con_term=con_term, executable=executable, provide_executable=provide_executable,
                                printing=printing)
        if con_term is None:
            solution, self.x = gp_var

            # Set the solution attribute
            if set_to_instance:
                self.solution = solution
                self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                        printing=printing)

            # Add solution to solution dictionary
            if add_to_dict:
                self.add_solution_to_dictionary(solution, solution_method="GP")

            return solution
        else:
            return gp_var

    def full_vft_model_solve(self, ga_max_time=60 * 10, pyomo_max_time=10, add_to_dict=True, return_z=True,
                             executable=None, provide_executable=True, printing=None, percent_step=10,
                             ga_printing=False):
        """
        This is the main method to solve the problem instance. We first solve the pyomo Approximate model, and then
        evolve it using the GA
        :param add_to_dict: if we want to add this solution to the solution dictionary
        :param ga_max_time: the genetic algorithm's time to solve
        :param pyomo_max_time: max time to solve the pyomo model
        :param return_z: If the method should return the z value or the solution itself
        :param executable: optional path to solver
        :param provide_executable: if we want to directly provide an executable path
        :param printing: Whether the procedure should print something
        :param percent_step: what percent checkpoints we should display updates for the GA
        :param ga_printing: If we want to print status updates during the genetic algorithm
        :return: solution z
        """
        if printing is None:
            printing = self.printing

        if printing:
            now = datetime.datetime.now()
            print('Solving VFT Model for ' + str(pyomo_max_time) + ' seconds at ' + now.strftime('%H:%M:%S') + '...')
        self.solve_vft_pyomo_model(max_time=pyomo_max_time, add_to_dict=False, executable=executable,
                                   provide_executable=provide_executable, printing=False)

        if printing:
            now = datetime.datetime.now()
            print('Solution value of ' + str(round(self.metrics['z'], 4)) + ' obtained.')
            print('Solving Genetic Algorithm for ' + str(ga_max_time) + ' seconds at ' +
                  now.strftime('%H:%M:%S') + '...')
        self.genetic_algorithm(initialize=True, add_to_dict=False, stopping_time=ga_max_time, printing=ga_printing,
                               constraints='Fail', percent_step=percent_step)
        if printing:
            print('Solution value of ' + str(round(self.metrics['z'], 4)) + ' obtained.')

        # Add solution to solution dictionary
        if add_to_dict:
            self.add_solution_to_dictionary(self.solution, solution_method="AG-VFT")

        if return_z:
            return round(self.metrics['z'], 4)
        else:
            return self.solution

    # Solution Handling
    def set_instance_solution(self, solution_name=None):
        """
        Set the current instance object's solution to a solution from the dictionary
        :param solution_name: name of the solution
        """
        if self.solution_dict is None:
            raise ValueError('No solution dictionary initialized')
        else:
            if solution_name is None:
                solution_name = list(self.solution_dict.keys())[0]
            else:
                if solution_name not in self.solution_dict:
                    raise ValueError('Solution ' + solution_name + ' not in solution dictionary')

            self.solution = self.solution_dict[solution_name]
            self.solution_name = solution_name
            if self.value_parameters is not None:
                self.measure_solution()
                self.full_name = self.data_type + " " + self.data_name + " " + self.vp_name + " " + self.solution_name

            # Return solution
            return self.solution

    def add_solution_to_dictionary(self, solution=None, full_name=None, solution_name=None,
                                   solution_method="Import"):

        # If the solution is not provided, we assume it's the current solution
        if solution is None:
            solution = self.solution

        # If we provide a full excel file name, we can get the name of the solution directly
        if full_name is not None:
            split_list = full_name.split(' ')
            if len(split_list) == 4:
                if solution_name is None:
                    solution_name = split_list[3]

        # Determine name of solution
        if solution_name is None:

            # Initially assume the name is "Method"_i unless it's the original solution
            if solution_method == "Original":
                solution_name = "Original"
            else:
                count = 1
                if self.solution_dict is not None:
                    for s_name in self.solution_dict:
                        if solution_method in s_name:
                            count += 1
                if count == 1:
                    solution_name = solution_method
                else:
                    solution_name = solution_method + '_' + str(count)

        # Add solution to dictionary if it is a new solution
        if self.solution_dict is None:

            # if the solution_dict is None, we can initialize it
            self.solution_dict = {solution_name: solution}
            self.solution_name = solution_name
        else:

            # Check if this solution is a new solution
            new = True
            for s_name in self.solution_dict:
                p_i = compare_solutions(self.solution_dict[s_name], solution)
                if p_i == 1:
                    new = False
                    self.solution_name = s_name
                    break

            # If it is new, we add it to the dictionary
            if new:
                self.solution_dict[solution_name] = solution
                self.solution_name = solution_name

    # TODO: This method is irrelevant now
    def afsc_solution_to_solution(self, afsc_solution, set_to_instance=True, add_to_dict=True, solution_name=None,
                                  printing=None):
        """
        Simply converts a vector of labeled AFSCs to a vector of AFSC indices
        :param solution_name: name of the solution
        :param set_to_instance: if we want to set this solution to the instance's solution attribute
        :param add_to_dict: if we want to add this solution to the solution dictionary
        :param afsc_solution: solution vector of AFSC names (strings)
        :param printing: Whether or not we should print status updates
        :return: solution vector of AFSC indices
        """
        if printing is None:
            printing = self.printing

        solution = np.zeros(len(afsc_solution))
        for i in range(self.parameters['N']):
            solution[i] = np.where(self.parameters['afsc_vector'] == afsc_solution[i])[0]

        # Set the solution attribute
        if set_to_instance:
            self.solution = solution
            self.metrics = measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                    printing=printing)

        # Add solution to solution dictionary
        if add_to_dict:
            self.add_solution_to_dictionary(solution, solution_name=solution_name)

        return solution

    # TODO: Need to fix this method (doesn't work right now)
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

        # Evaluate Solution
        metrics = measure_solution_quality(self.x, self.parameters, self.value_parameters)
        values = metrics['objective_value']
        measures = metrics['objective_measure']

        lam, y = x_to_solution_initialization(self.parameters, self.value_parameters, measures, values)
        initialization = {'X': self.x, 'lam': lam, 'y': y, 'F_X': values}
        return initialization

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
        quota_k = np.where(self.value_parameters['objectives'] == 'Combined Quota')[0][0]
        for j in self.parameters['J']:

            for k, objective in enumerate(self.value_parameters['objectives']):

                if k in self.value_parameters['K^C'][j]:

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
        Measure a solution using a set of value parameters
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
                solution = self.x
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

    def update_metrics_dict(self):
        """
        Updates metrics dictionary from solutions and vp dictionaries
        """

        # We only update the dictionaries if we have at least one set of value parameters and one solution
        if self.vp_dict is not None and self.solution_dict is not None:

            if self.metrics_dict is None:
                self.metrics_dict = {}

            for vp_name in self.vp_dict:
                value_parameters = self.vp_dict[vp_name]
                if vp_name not in self.metrics_dict:
                    self.metrics_dict[vp_name] = {}
                for solution_name in self.solution_dict:
                    solution = self.solution_dict[solution_name]
                    if solution_name not in self.metrics_dict[vp_name]:
                        metrics = measure_solution_quality(solution, self.parameters, value_parameters)
                        self.metrics_dict[vp_name][solution_name] = copy.deepcopy(metrics)

            # Update weights on the sets of value parameters relative to each other
            sum_weights = 0
            for vp_name in self.vp_dict:
                sum_weights += self.vp_dict[vp_name]['vp_weight']
            for vp_name in self.vp_dict:
                self.vp_dict[vp_name]['vp_local_weight'] = self.vp_dict[vp_name]['vp_weight'] / sum_weights

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
            if self.data_variant == "Year":
                afsc_rotation = 45
            else:
                afsc_rotation = 0

        if skip_afscs is None:
            if self.data_variant == "Year":
                skip_afscs = False
            else:
                skip_afscs = True

        if self.data_variant == 'Year' and afsc_tick_size == 30:
            if self.parameters['M'] > 33:
                afsc_tick_size = 20
                afsc_rotation = 60
            else:
                afsc_tick_size = 25

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
        """
        This procedure conducts the pareto analysis and builds the chart
        :param step: percent step between different overall cadet weights
        :param full_solve: if we want to solve the full VFT model or just the approximate model
        :param ga_max_time: max time for the GA to solve
        :param printing: if we want to print status updates or not
        :param filepath: file path
        :param import_df: if we want to import the dataframe or not
        :param thesis_chart: if this is meant to go into the thesis
        :param save: if we want to save the chart
        :param figsize: size of the figure
        :param facecolor: color of the figure
        :param display_title: if we're displaying a title or not
        :return: chart
        """

        if filepath is None:
            filepath = paths_in['results'] + self.data_name + '_Pareto_Analysis.xlsx'

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
        """
        Conducts the "Least Squares Procedure" for sensitivity analysis. This method also builds the chart. I got
        lazy about writing all the parameters in the doc-string since most of them are already defined elsewhere.
        """
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

    def display_afsc_objective_weights_chart(self, current_set=True, afsc=None, printing=None, colors=None,
                                             save=False, figsize=(14, 6), facecolor="white", title=None,
                                             display_title=True, thesis_chart=False, title_size=None, bar_color=None,
                                             legend_size=None, label_size=20, xaxis_tick_size=15, yaxis_tick_size=15):
        """
        Displays a chart comparing the objective weights for a particular afsc across multiple sets of value parameters
        in the vp_dict
        """
        if printing is None:
            printing = self.printing

        if current_set:
            vp_dict = {self.vp_name: self.value_parameters}
        else:
            vp_dict = self.vp_dict

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

        chart = afsc_objective_weights_graph(self.parameters, vp_dict, afsc, colors=colors, save=save,
                                             figsize=figsize, facecolor=facecolor, title=title, legend_size=legend_size,
                                             display_title=display_title, label_size=label_size, title_size=title_size,
                                             xaxis_tick_size=xaxis_tick_size, yaxis_tick_size=yaxis_tick_size,
                                             bar_color=bar_color)
        if printing:
            chart.show()

        return chart

    # Export
    def create_aggregate_file(self, from_files=False, one_time_new_vp=False, printing=None):
        """
        Create the "data_type data_name" main aggregate file with solutions, metrics, and vps
        :param one_time_new_vp: this is just to recreate default value parameters for some class years
        (ignore this parameter)
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
            for filepath in self.instance_files:
                start_index = filepath.find(paths_in['instances']) + len(paths_in['instances']) + 1
                end_index = len(filepath) - 5
                full_name = filepath[start_index:end_index]
                vp_name = full_name.split(' ')[2]
                solution_name = full_name.split(' ')[3]
                if vp_name not in vp_dict:
                    if one_time_new_vp:
                        value_parameters = self.import_default_value_parameters(no_constraints=True,
                                                                                set_to_instance=False,
                                                                                add_to_dict=False)
                    else:
                        value_parameters = self.import_value_parameters(filepath=filepath, set_value_parameters=False,
                                                                        printing=False)
                    vp_dict[vp_name] = copy.deepcopy(value_parameters)
                if solution_name not in solution_dict:
                    solution = self.import_solution(filepath=filepath, set_to_instance=False, add_to_dict=False,
                                                    printing=False)
                    solution_dict[solution_name] = copy.deepcopy(solution)
            metrics_dict = {}
            for vp_name in vp_dict:
                value_parameters = vp_dict[vp_name]
                vp_dict[vp_name]['vp_weight'] = 100
                vp_dict[vp_name]['vp_local_weight'] = 1 / len(list(vp_dict.keys()))
                metrics_dict[vp_name] = {}
                for solution_name in solution_dict:
                    solution = solution_dict[solution_name]
                    metrics = self.measure_solution(solution, value_parameters, set_solution=False, return_z=False,
                                                    printing=False)
                    metrics_dict[vp_name][solution_name] = copy.deepcopy(metrics)
        else:
            self.update_metrics_dict()
            metrics_dict = self.metrics_dict
            solution_dict = self.solution_dict
            vp_dict = self.vp_dict

        create_aggregate_instance_file(self.data_instance_name, self.parameters, solution_dict, vp_dict, metrics_dict,
                                       self.gp_df)

    def export_to_excel(self, aggregate=True, printing=None):
        """
        Export data information to excel. We can either export the dictionaries back to excel or we can export
        the results for one particular solution given one set of value parameters
        :param aggregate: if we are exporting back to the aggregate excel file or not
        :param printing: if we should print status updates or not
        """
        if printing is None:
            printing = self.printing

        if aggregate:
            self.create_aggregate_file()
        else:

            if self.solution_name in self.solution_dict:

                # Check if the instance solution is the same as the one in the dictionary
                p_s = compare_solutions(self.solution, self.solution_dict[self.solution_name])
                if p_s != 1:
                    raise ValueError("Current solution: " + self.solution_name +
                                     ". This is not the same solution as found in the dictionary under that name.")

            else:
                raise ValueError("Solution " + self.solution_name + " not found in dictionary.")

            if self.vp_name in self.vp_dict:

                # Check if the instance value parameters are the same as the ones in the dictionary
                vp_same = compare_value_parameters(self.parameters, self.value_parameters, self.vp_dict[self.vp_name])
                if not vp_same:
                    raise ValueError("Current value parameters: " + self.vp_name +
                                     ". This is not the same set of value parameters as found in the "
                                     "dictionary under that name.")
            else:
                raise ValueError("Value Parameters " + self.vp_name + " not found in dictionary.")

            # Export to the right place
            self.full_name = self.data_type + " " + self.data_name + " " + self.vp_name + " " + self.solution_name
            filepath = paths_out['instances'] + self.full_name + '.xlsx'

            # Export to excel
            data_to_excel(filepath, self.parameters, self.value_parameters, self.metrics, printing=printing)

        if printing:
            print('Exported.')


try:
    from afccp.core.problem_class_add import MoreCCPMethods

    # Load in the methods from the other sheet and attach them to CadetCareerProblem
    for method_name in dir(MoreCCPMethods):
        method = getattr(MoreCCPMethods, method_name)
        if not method_name.startswith('__'):
            setattr(CadetCareerProblem, method.__name__, method)
except:
    print('Tried to import more methods for CCP, but failed. Please fix something.')
