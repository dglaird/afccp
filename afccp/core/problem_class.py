# Import libraries
from typing import Any

import os

import pandas as pd

import afccp.core.handling.value_parameter_handling
import afccp.core.handling.ccp_helping_functions
import afccp.core.visualizations.slides
import afccp.core.comprehensive_functions
import datetime
import glob
import copy

# Main Problem Class
class CadetCareerProblem:
    def __init__(self, data_name=None, N=1600, M=32, P=6, num_breakpoints=None, printing=True):
        """
        This is the AFSC/Cadet problem object. We can import data using the data_name (must be the instance folder!).
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

        # Get list of generated data name problem instances
        self.generated_data_names = {'Random': [], 'Perfect': [], 'Realistic': []}
        for file_name in glob.iglob(afccp.core.globals.paths['instances'] + '*.xlsx', recursive=True):
            start_index = file_name.find(afccp.core.globals.paths['instances']) + len(afccp.core.globals.paths['instances']) + 1  # Start of filename
            end_index = len(file_name) - 5  # Remove ".xlsx"
            full_name = file_name[start_index:end_index]  # Name of the file (not the path)
            sections = full_name.split(' ')  # Split the filename by each ' ' (space)
            d_name = sections[0]  # Second part is the "data_name"

            # Loop through each of the kinds of generated data (random, perfect, realistic)
            for variant in self.generated_data_names:

                # Ex. "Random" in "Random_1" and "Random_1" isn't already in the list
                if variant in d_name and d_name not in self.generated_data_names[variant]:
                    self.generated_data_names[variant].append(d_name)  # add this data name to the list!

        # Get correct data attributes
        if data_name is None:

            # If we didn't specify a data_name, we're just going to generate a random set of cadets
            self.data_name = "Random_" + str(len(self.generated_data_names["Random"]) + 1)
            self.data_variant = "Random"
            generate = True
        else:

            # We specified a data_name
            self.data_name = data_name
            generate = False

            # Loop through "Random", "Realistic", "Perfect"
            for data_variant in self.generated_data_names:

                # If we passed one of those three names generally, we will generate a new instance of that kind
                if data_variant == self.data_name:
                    self.data_name = data_variant + "_" + str(len(self.generated_data_names[data_variant]) + 1)
                    generate = True
                    break

                # If we specified a specific version ("Random_4" for example), then we'll load it in
                elif data_variant in self.data_name and '_' in self.data_name:
                    generate = False
                    break

        # Figure out the data variant
        self.data_variant = None
        if len(self.data_name) == 1:  # A, B, C, D, etc.
            self.data_variant = "Scrubbed"

        else:

            # Random, Realistic, or Perfect
            for data_variant in self.generated_data_names:
                if data_variant in self.data_name:
                    self.data_variant = data_variant

            # If we still haven't found the right data variant, we know it's a real class year
            if self.data_variant is None:
                self.data_variant = "Year"  # 2018, 2019, 2020, etc.

        # Get correct filepath  (for importing/exporting to)
        self.filepath = afccp.core.globals.paths['instances'] + self.data_name + '.xlsx'

        # Create a "results" folder for this problem instance
        if not os.path.exists("results/" + self.data_name):
            os.makedirs("results/" + self.data_name)

        # Create multiple "figures" folders for this problem instance
        if not os.path.exists("figures/" + self.data_name):
            os.makedirs("figures/" + self.data_name + "/value parameters")
            os.makedirs("figures/" + self.data_name + "/results")
            os.makedirs("figures/" + self.data_name + "/slides")
            os.makedirs("figures/" + self.data_name + "/data")

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
        self.info_df = None
        self.solution_dict = None
        self.metrics_dict = None
        self.solution_name = None
        self.vp_dict = None
        self.vp_name = None
        self.similarity_matrix = None

        # Check if we're generating data, and the data variant
        if self.data_variant == 'Random' and generate:

            if printing:
                print('Generating ' + self.data_name + ' problem instance...')
            parameters = afccp.core.handling.simulation_functions.simulate_model_fixed_parameters(N=N, P=P, M=M)
            self.parameters = afccp.core.handling.data_handling.model_fixed_parameters_set_additions(parameters)

        elif self.data_variant == 'Perfect' and generate:

            if printing:
                print('Generating ' + self.data_name + ' problem instance...')
            parameters, self.solution = \
                afccp.core.handling.simulation_functions.perfect_example_generator(N=N, P=P, M=M)
            self.parameters = afccp.core.handling.data_handling.model_fixed_parameters_set_additions(parameters)

        elif self.data_variant == 'Realistic' and generate:

            if printing:
                print('Generating ' + self.data_name + ' problem instance...')

            if use_sdv:
                data = afccp.core.handling.simulation_functions.simulate_realistic_fixed_data(N=N)
                cadets_fixed, afscs_fixed = \
                    afccp.core.handling.simulation_functions.convert_realistic_data_parameters(data)
                parameters = afccp.core.handling.data_handling.model_fixed_parameters_from_data_frame(
                    cadets_fixed, afscs_fixed)
            else:
                parameters = afccp.core.handling.simulation_functions.simulate_model_fixed_parameters(N=N, P=P, M=M)

            self.parameters = afccp.core.handling.data_handling.model_fixed_parameters_set_additions(parameters)
        else:

            if printing:
                print('Importing ' + self.data_name + ' problem instance...')

            # If the path exists, import the data. If not, raise an error
            if os.path.exists(self.filepath):
                self.info_df, self.parameters, self.vp_dict, self.solution_dict, self.metrics_dict, self.gp_df, \
                self.similarity_matrix = afccp.core.comprehensive_functions.import_aggregate_instance_file(
                    self.filepath, num_breakpoints=num_breakpoints)
            else:
                raise ValueError("Instance '" + self.data_name + "' not found at path '" + afccp.core.globals.paths['instances'] + "'")

        # Initialize more "functional" parameters
        self.plt_p, self.mdl_p = \
            afccp.core.handling.ccp_helping_functions.initialize_instance_functional_parameters(self.parameters["N"])

        if printing:
            if generate:
                print('Generated.')
            else:
                print('Imported.')

    # Observe "Fixed" Data
    def display_data_graph(self, p_dict={}, printing=None):
        """
        This method plots different aspects of the fixed parameters of the problem instance.
        """

        # Reset title and filename
        self.plt_p["title"], self.plt_p["filename"] = None, None

        # Update plot parameters if necessary
        for key in p_dict:
            if key in self.plt_p:
                self.plt_p[key] = p_dict[key]
            else:

                # Exception
                if key == "graph":
                    self.plt_p["data_graph"] = p_dict["graph"]

                else:
                    # If the parameter doesn't exist, we warn the user
                    print("WARNING. Specified parameter '" + str(key) + "' does not exist.")

        self.plt_p = afccp.core.handling.ccp_helping_functions.determine_afsc_plot_details(self)

        # Create the chart
        chart = afccp.core.visualizations.instance_graphs.data_graph(self)

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

        afccp.core.handling.data_handling.find_solution_ineligibility(self.parameters, solution)

    def display_all_data_graphs(self, p_dict={}, printing=None):
        """
        This method runs through all the different versions of graphs we have and saves
        them to the corresponding folder.
        """
        if printing is None:
            printing = self.printing

        if printing:
            print("Saving all data graphs to the corresponding folder...")

        # Loop through each version of each graph
        charts = []
        version_list = {"Cadet Preference": [None, "Top 3"], "Cadet Preference Analysis": [None, "AFOCD_Eligible", "Merit",
                                                                                           "USAFA", "Gender"]}
        for graph in version_list:
            for version in version_list[graph]:
                p_dict["data_graph"], p_dict["version"] = graph, version
                charts.append(self.display_data_graph(p_dict, printing=printing))

        return charts

    # Adjust Data
    def adjust_qualification_matrix(self, printing=None, report_cips_not_found=False, use_matrix=False):
        """
        This procedure simply re-runs the CIP to Qual function in case I change qualifications or I
        identify errors since some cadets in the AFPC solution may receive AFSCs for which they
        are ineligible for.
        :param use_matrix: use the matrix (loaded from excel) or the function that does it directly
        :param report_cips_not_found: if we want to get a list of CIPs of cadets that are not found in
        the full list of CIPs
        :param printing: Whether we should print status updates or not
        """
        if printing is None:
            printing = self.printing

        if printing:
            print('Adjusting qualification matrix...')

        parameters = copy.deepcopy(self.parameters)
        unknown_cips = {}

        if 'cip1' in parameters:
            if 'cip2' in parameters:

                if use_matrix:
                    if report_cips_not_found:
                        qual_matrix, unknown_cips = afccp.core.handling.preprocessing.cip_to_qual(
                            parameters['afsc_vector'], parameters['cip1'].astype(str),
                            cip2=parameters['cip2'].astype(str), report_cips_not_found=True)
                    else:
                        qual_matrix = afccp.core.handling.preprocessing.cip_to_qual(
                            parameters['afsc_vector'], parameters['cip1'].astype(str),
                            cip2=parameters['cip2'].astype(str))
                else:
                    qual_matrix = afccp.core.handling.preprocessing.cip_to_qual_direct(
                        parameters['afsc_vector'], parameters['cip1'].astype(str), cip2=parameters['cip2'].astype(str))
                parameters['qual'] = qual_matrix
                parameters['ineligible'] = (qual_matrix == 'I') * 1
                parameters['eligible'] = (parameters['ineligible'] == 0) * 1
                parameters['mandatory'] = (qual_matrix == 'M') * 1
                parameters['desired'] = (qual_matrix == 'D') * 1
                parameters['permitted'] = (qual_matrix == 'P') * 1
                parameters = afccp.core.handling.data_handling.model_fixed_parameters_set_additions(parameters)
                self.parameters = copy.deepcopy(parameters)
            else:
                raise ValueError("No 'cip2' column in parameters")
        else:
            raise ValueError("No 'cip1' column in parameters")

        if report_cips_not_found:
            cips = []
            counts = []
            for cip in unknown_cips:
                print(str(unknown_cips[cip]) + " cadets contained a CIP labeled " + cip +
                      " that was not found in the full list of CIPs.")
                cips.append(cip)
                counts.append(unknown_cips[cip])

            # Export to Excel
            df = pd.DataFrame({"Unknown CIP": cips, "Occurrences": counts})
            with pd.ExcelWriter(afccp.core.globals.paths["results"] + self.data_name + "_CIP_Report.xlsx") as writer:
                df.to_excel(writer, sheet_name="Unknown CIPs", index=False)

    def convert_utilities_to_preferences(self):
        """
        Converts the utility matrices to preferences
        """
        self.parameters = afccp.core.handling.data_handling.convert_utility_matrices_preferences(self.parameters)

    def generate_fake_afsc_preferences(self):
        """
        Uses the VFT parameters to generate simulated AFSC preferences
        """
        self.parameters = afccp.core.handling.data_handling.generate_fake_afsc_preferences(
            self.parameters, self.value_parameters)

    def convert_afsc_preferences_to_percentiles(self):
        """
        This method takes the AFSC preference lists and turns them into normalized percentiles for each cadet for each
        AFSC.
        """
        self.parameters = afccp.core.handling.data_handling.convert_afsc_preferences_to_percentiles(self.parameters)

    def convert_to_scrubbed_instance(self, new_letter, printing=None):
        """
        This method scrubs the AFSC names by sorting them by their PGL targets and creates a translated problem instance
        :param printing: If we should print status update
        :param new_letter: New letter to assign to this problem instance
        """
        if printing is None:
            printing = self.printing

        if printing:
            print("Converting problem instance '" + self.data_name + "' to new instance '" + new_letter + "'...")

        return afccp.core.comprehensive_functions.scrub_real_afscs_from_instance(self, new_letter)

    # Specify Value Parameters
    def set_instance_value_parameters(self, vp_name=None):
        """
        Sets the current instance value parameters to a specified set based on the vp_name. This vp_name must be
        in the value parameter dictionary
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
            identical = afccp.core.handling.value_parameter_handling.compare_value_parameters(
                self.parameters, value_parameters, self.vp_dict[vp_name], printing=printing)
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
        :param filename: filename
        :param filepath: filepath to import from (if none specified, will use filepath attribute)
        :param num_breakpoints: Number of breakpoints to use for the value functions
        :param set_to_instance: if we want to set this set to the instance's value parameters attribute
        :param add_to_dict: if we want to add this set of value parameters to the vp dictionary
        :param vp_weight: swing weight of this entire set of value parameters relative to others
        :param printing: Whether we should print status updates or not
        :param no_constraints: If we don't want to use the predefined constraints
        :param generate_afsc_weights: If we generate the AFSC weights from the weight function (True), or just
        import the weights directly from the "AFSC Weight" column (False)
        """

        if printing is None:
            printing = self.printing

        # check first if the user did not specify a filename (just name of excel file within correct folder)
        if filename is None:

            # if filename not specified, check if file path was not directly specified
            if filepath is None:  # default value parameters
                if self.data_variant == "Scrubbed":
                    filepath = afccp.core.globals.paths["support"] + 'Value_Parameters_Defaults_' + self.data_name + '.xlsx'
                elif self.data_variant == 'Year':
                    filename = 'Value_Parameters_Defaults_' + self.data_name + '.xlsx'
                    if filename in os.listdir(afccp.core.globals.paths["support"]):
                        filepath = afccp.core.globals.paths["support"] + filename
                    else:
                        filepath = afccp.core.globals.paths["support"] + 'Value_Parameters_Defaults.xlsx'
                elif self.data_variant == 'Perfect':
                    filepath = afccp.core.globals.paths["support"] + 'Value_Parameters_Defaults_Perfect.xlsx'
                else:
                    filepath = afccp.core.globals.paths["support"] + 'Value_Parameters_Defaults_Generated.xlsx'
            else:
                pass  # we have given a filepath already, no more info is needed
        else:
            if self.data_variant == "Year":
                filepath = afccp.core.globals.paths["support"] + filename
            else:
                filepath = afccp.core.globals.paths["support"] + filename

            if '.xlsx' not in filepath:
                filepath += '.xlsx'

        # Import "default value parameters" from excel
        self.default_value_parameters = \
            afccp.core.handling.value_parameter_handling.default_value_parameters_from_excel(
                filepath, num_breakpoints=num_breakpoints, printing=printing)

        # Generate this instance's value parameters from the defaults
        value_parameters = afccp.core.handling.value_parameter_handling.generate_value_parameters_from_defaults(
            self.parameters, generate_afsc_weights=generate_afsc_weights,
            default_value_parameters=self.default_value_parameters)

        # Add some additional components to value parameters
        if no_constraints:
            value_parameters['constraint_type'] = np.zeros([self.parameters['M'], value_parameters['O']])
        value_parameters = afccp.core.handling.value_parameter_handling.model_value_parameters_set_additions(
            self.parameters, value_parameters)
        value_parameters = afccp.core.handling.value_parameter_handling.condense_value_functions(
            self.parameters, value_parameters)
        value_parameters = afccp.core.handling.value_parameter_handling.model_value_parameters_set_additions(
            self.parameters, value_parameters)
        value_parameters['vp_weight'] = vp_weight

        # Set value parameters to instance attribute
        if set_to_instance:
            self.value_parameters = value_parameters
            if self.solution is not None:
                self.metrics = afccp.core.handling.data_handling.measure_solution_quality(
                    self.solution, self.parameters, self.value_parameters, printing=printing)

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
                filepath = afccp.core.globals.paths["support"] + 'Value_Parameters_Defaults_' + self.data_name + '.xlsx'
            elif self.data_variant == 'Year':
                filepath = afccp.core.globals.paths["support"] + 'Value_Parameters_Defaults.xlsx'
            else:
                raise ValueError('Data type must be Real not Generated.')

            default_value_parameters = \
                afccp.core.handling.value_parameter_handling.default_value_parameters_from_excel(filepath)
            self.default_value_parameters = default_value_parameters

        if self.data_type == "Real":  # Not scrubbed AFSC data
            data_type = self.data_name  # Pull in the class year used
        else:
            data_type = self.data_type

        if constraints_df is None:
            if self.data_variant == "Year":
                constraints_df = afccp.core.globals.import_data(
                    afccp.core.globals.paths["support"] + 'Value_Parameter_Sets_Options.xlsx',
                    sheet_name=data_type + ' Constraint Options')
            else:
                constraints_df = afccp.core.globals.import_data(
                    afccp.core.globals.paths["support"] + 'Value_Parameter_Sets_Options_Scrubbed.xlsx',
                    sheet_name=data_type + ' Constraint Options')

        # Generate realistic value parameters for the problem instance
        value_parameters = afccp.core.handling.value_parameter_generator.value_parameter_realistic_generator(
            self.parameters, default_value_parameters, constraints_df, deterministic=deterministic,
            constrain_merit=constrain_merit, data_name=data_type)
        value_parameters = afccp.core.handling.value_parameter_handling.model_value_parameters_set_additions(
            self.parameters, value_parameters)
        value_parameters = afccp.core.handling.value_parameter_handling.condense_value_functions(
            self.parameters, value_parameters)
        value_parameters = afccp.core.handling.value_parameter_handling.model_value_parameters_set_additions(
            self.parameters, value_parameters)
        value_parameters['vp_weight'] = vp_weight

        # Set value parameters to instance attribute
        if set_to_instance:
            self.value_parameters = value_parameters
            if self.solution is not None:
                self.metrics = afccp.core.handling.data_handling.measure_solution_quality(
                    self.solution, self.parameters, self.value_parameters, printing=printing)

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
        """
        if printing is None:
            printing = self.printing

        if self.value_parameters is None:
            raise ValueError('No instance value parameters detected.')
        else:
            if filename is None:  # I add the "_New" just so we make sure we don't accidentally overwrite the old one
                filename = "Value_Parameters_Defaults_" + self.data_name + "_New.xlsx"

            if filepath is None:
                filepath = afccp.core.globals.paths["support"] + filename

            afccp.core.handling.value_parameter_handling.model_value_parameters_to_defaults(
                self.parameters, self.value_parameters, filepath=filepath, printing=printing)

    def change_weight_function(self, cadets=True, function=None):
        """
        Changes the weight function on either cadets or AFSCs
        :param cadets: if this is for cadets (True) or AFSCs (False)
        :param function: new weight function to use
        """

        if cadets:
            if function is None:
                function = 'Linear'

            if "merit_all" in self.parameters:  # The cadets' real order of merit
                merit = self.parameters["merit_all"]
            else:  # The cadets' scaled order of merit (based solely on Non-Rated cadets)
                merit = self.parameters["merit"]

            # Update weight function
            self.value_parameters['cadet_weight_function'] = function

            # Update weights
            self.value_parameters["cadet_weight"] = \
                afccp.core.handling.value_parameter_handling.cadet_weight_function(merit, function)

        else:
            if function is None:
                function = "Curve_2"

            if "pgl" in self.parameters:  # The actual PGL target
                quota = self.parameters["pgl"]
            else:  # The "Real" Target based on surplus and such
                quota = self.parameters["quota"]

            # Update weight function
            self.value_parameters['afsc_weight_function'] = function

            # Update weights
            self.value_parameters["afsc_weight"] = \
                afccp.core.handling.value_parameter_handling.afsc_weight_function(quota, function)

    # Translate Parameters
    def vft_to_gp_parameters(self, p_dict={}, printing=None):
        """
        Converts the instance parameters and value parameters to parameters used by Rebecca's model
        """

        if printing is None:
            printing = self.printing

        # Reset certain plot parameters
        _, self.mdl_p = \
            afccp.core.handling.ccp_helping_functions.initialize_instance_functional_parameters(self.parameters["N"])

        # Update model parameters if necessary
        for key in p_dict:
            if key in self.mdl_p:
                self.mdl_p[key] = p_dict[key]
            else:

                # If the parameter doesn't exist, we warn the user
                print("WARNING. Specified parameter '" + str(key) + "' does not exist.")

        if self.mdl_p["use_gp_df"]:

            # Get basic parameters (May or may not include penalty/reward parameters
            self.gp_parameters = afccp.core.handling.value_parameter_handling.translate_vft_to_gp_parameters(
                self.parameters, self.value_parameters, self.gp_df, self.mdl_p["use_gp_df"])

            # Use generalized "GP DF"
            if self.gp_df is None:
                filepath = afccp.core.globals.paths["support"] + 'GP_Parameters.xlsx'
                self.gp_df = afccp.core.globals.import_data(filepath=filepath, sheet_name='Weights and Scaling')
                specific_gp_df = False
            else:
                specific_gp_df = True

            # Get list of constraints
            con_list = copy.deepcopy(self.gp_parameters['con'])
            con_list.append('S')
            num_constraints = len(con_list)  # should be 12

            # Either create new rewards and penalties for this specific instance
            if self.mdl_p["get_new_rewards_penalties"]:
                rewards, penalties = \
                    afccp.core.comprehensive_functions.calculate_rewards_penalties(self, printing=printing)
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

        self.gp_parameters = afccp.core.handling.value_parameter_handling.translate_vft_to_gp_parameters(
            self.parameters, self.value_parameters, self.gp_df, self.mdl_p["use_gp_df"], printing=printing)

    # Observe Value Parameters
    def show_value_function(self, afsc=None, objective=None, printing=None, label_size=25, x_point=None,
                            yaxis_tick_size=25, xaxis_tick_size=25, figsize=(12, 10), facecolor='white',
                            title=None, save=False, display_title=True):
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
        :param x_point: Optional point to plot to draw attention to (x, y coordinate)
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
            k = self.value_parameters["K^A"][0][0]
            objective = self.value_parameters['objectives'][k]

        value_function_chart = afccp.core.comprehensive_functions.plot_value_function(
            afsc, objective, self.parameters, self.value_parameters, printing=printing, label_size=label_size,
            yaxis_tick_size=yaxis_tick_size, facecolor=facecolor, xaxis_tick_size=xaxis_tick_size, figsize=figsize,
            save=save, display_title=display_title, title=title, x_point=x_point)

        if printing:
            value_function_chart.show()

        return value_function_chart

    def display_weight_function(self, cadets=True, save=False, figsize=(19, 7), facecolor='white', gui_chart=False,
                                display_title=True, title=None, label_size=35, afsc_tick_size=25,
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

        chart = afccp.core.visualizations.instance_graphs.individual_weight_graph(
            self.parameters, self.value_parameters, cadets, save, figsize, facecolor, display_title, title,
            label_size, afsc_tick_size, gui_chart, dpi, yaxis_tick_size, afsc_rotation, xaxis_tick_size,
            skip_afscs=skip_afscs)

        if printing:
            chart.show()

        return chart

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
            self.metrics = afccp.core.handling.data_handling.measure_solution_quality(
                self.solution, self.parameters, self.value_parameters, printing=printing)

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

        solution = afccp.core.solutions.heuristic_solvers.stable_marriage_model_solve(
            self.parameters, self.value_parameters, printing=printing)

        # Set the solution attribute
        if set_to_instance:
            self.solution = solution
            self.metrics = afccp.core.handling.data_handling.measure_solution_quality(self.solution, self.parameters, self.value_parameters,
                                                    printing=printing)

        # Add solution to solution dictionary
        if add_to_dict:
            self.add_solution_to_dictionary(solution, solution_method='Stable')

        return solution

    def matching_algorithm_1(self, set_to_instance=True, add_to_dict=True, printing=None):
        """
        This method solves the problem instance using "Matching Algorithm 1"
        :param printing: Whether or not to print status updates
        :param set_to_instance: if we want to set this solution to the instance's solution attribute
        :param add_to_dict: if we want to add this solution to the solution dictionary
        :return: solution vector
        """
        if printing is None:
            printing = self.printing

        solution = afccp.core.solutions.heuristic_solvers.matching_algorithm_1(self, printing=printing)

        # Set the solution attribute
        if set_to_instance:
            self.solution = solution
            self.metrics = afccp.core.handling.data_handling.measure_solution_quality(
                self.solution, self.parameters, self.value_parameters, printing=printing)

        # Add solution to solution dictionary
        if add_to_dict:
            self.add_solution_to_dictionary(solution, solution_method='MA1')

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

        solution = afccp.core.solutions.heuristic_solvers.greedy_model_solve(
            self.parameters, self.value_parameters, printing=printing)

        # Set the solution attribute
        if set_to_instance:
            self.solution = solution
            self.metrics = afccp.core.handling.data_handling.measure_solution_quality(
                self.solution, self.parameters, self.value_parameters, printing=printing)

        # Add solution to solution dictionary
        if add_to_dict:
            self.add_solution_to_dictionary(solution, solution_method='Greedy')

        return solution

    def genetic_algorithm(self, p_dict={}, printing=None):
        """
        This is the genetic algorithm. The hyper-parameters to the algorithm can be tuned, and this is meant to be
        solved in conjunction with the pyomo model solution. Use that as the initial solution, and then we evolve
        from there
        """

        if printing is None:
            printing = self.printing

        # Reset certain plot parameters
        self.mdl_p["add_to_dict"], self.mdl_p["set_to_instance"] = True, True

        # Update model parameters if necessary
        for key in p_dict:
            if key in self.mdl_p:
                self.mdl_p[key] = p_dict[key]
            else:

                # If the parameter doesn't exist, we warn the user
                print("WARNING. Specified parameter '" + str(key) + "' does not exist.")

        # Make sure we have selected a set of value parameters
        if self.value_parameters is None:
            raise ValueError("Error. No instance value parameters set.")

        # Dictionary of failed constraints across all solutions
        # (should be off by *no more than* one cadet due to rounding)
        con_fail_dict = None

        # Get a starting population of solutions if applicable!
        if self.mdl_p["initialize"]:

            if self.mdl_p["initial_solutions"] is None:

                if self.solution_dict is None:
                    raise ValueError("Error. No solutions in dictionary.")

                else:
                    if self.mdl_p["solution_names"] is None:
                        initial_solutions = np.array(
                            [self.solution_dict[solution_name] for solution_name in self.solution_dict])
                    else:
                        initial_solutions = np.array(
                            [self.solution_dict[solution_name] for solution_name in self.mdl_p["solution_names"]])

            else:
                initial_solutions = self.mdl_p["initial_solutions"]

            # Get dictionary of failed constraints
            if self.mdl_p["pyomo_constraint_based"]:
                con_fail_dict = self.get_full_constraint_fail_dictionary(initial_solutions, printing=True)
        else:
            initial_solutions = None

        if self.mdl_p["time_eval"]:
            start_time = time.perf_counter()  # Start the timer
            solution, time_eval_df = afccp.core.solutions.heuristic_solvers.genetic_algorithm(
                self, initial_solutions, con_fail_dict, printing=printing)
            solve_time = time.perf_counter() - start_time
        else:
            solution = afccp.core.solutions.heuristic_solvers.genetic_algorithm(
                self, initial_solutions, con_fail_dict, printing=printing)

        if self.mdl_p["set_to_instance"]:
            self.solution = solution
            self.metrics = afccp.core.handling.data_handling.measure_solution_quality(
                self.solution, self.parameters, self.value_parameters, printing=printing)

        # Add solution to solution dictionary
        if self.mdl_p["add_to_dict"]:
            self.add_solution_to_dictionary(solution, solution_method="Genetic")

        # Return the final solution and maybe the time evaluation dataframe if needed
        if self.mdl_p["time_eval"]:
            return solve_time, solution
        else:
            return solution

    def solve_vft_pyomo_model(self, p_dict={}, printing=None):
        """
        Solve the VFT model using pyomo
        """
        if printing is None:
            printing = self.printing

        # Reset certain plot parameters
        self.mdl_p["add_to_dict"], self.mdl_p["set_to_instance"] = True, True

        # Update model parameters if necessary
        for key in p_dict:
            if key in self.mdl_p:
                self.mdl_p[key] = p_dict[key]
            else:

                # If the parameter doesn't exist, we warn the user
                print("WARNING. Specified parameter '" + str(key) + "' does not exist.")

        # Make sure we have selected a set of value parameters
        if self.value_parameters is None:
            raise ValueError("Error. No instance value parameters set.")

        if not self.mdl_p["approximate"]:
            if self.mdl_p["solver_name"] == 'cbc':
                self.mdl_p["solver_name"] = 'ipopt'

        if self.mdl_p["warm_start"] is None:
            if self.mdl_p["init_from_X"]:
                self.mdl_p["warm_start"] = self.init_exact_solution_from_x()

        if use_pyomo:
            model = afccp.core.solutions.pyomo_models.vft_model_build(self, printing=printing)
            if self.mdl_p["report"]:
                if self.mdl_p["time_eval"]:
                    solution, self.x, self.measure, self.value, self.pyomo_z, solve_time = \
                        afccp.core.solutions.pyomo_models.vft_model_solve(self, model, printing=printing)
                else:
                    solution, self.x, self.measure, self.value, self.pyomo_z = \
                        afccp.core.solutions.pyomo_models.vft_model_solve(self, model, printing=printing)
            else:
                if self.mdl_p["time_eval"]:
                    solution, solve_time = \
                        afccp.core.solutions.pyomo_models.vft_model_solve(self, model, printing=printing)
                else:
                    solution = afccp.core.solutions.pyomo_models.vft_model_solve(self, model, printing=printing)
        else:
            if printing:
                raise ValueError('Pyomo not available')

        # Convert to integer
        solution = solution.astype(int)

        # Set the solution attribute
        if self.mdl_p["set_to_instance"]:
            self.solution = solution
            self.metrics = afccp.core.handling.data_handling.measure_solution_quality(
                self.solution, self.parameters, self.value_parameters, printing=printing)

        # Add solution to solution dictionary
        if self.mdl_p["add_to_dict"]:
            if self.mdl_p["approximate"]:
                solution_method = "A-VFT"
            else:
                solution_method = "E-VFT"
            self.add_solution_to_dictionary(solution, solution_method=solution_method)

        if self.mdl_p["time_eval"]:
            return solve_time
        else:
            return solution

    def solve_original_pyomo_model(self, p_dict={}, printing=None):
        """
        Solve the original AFPC model using pyomo
        """

        if printing is None:
            printing = self.printing

        # Reset certain plot parameters
        self.mdl_p["add_to_dict"], self.mdl_p["set_to_instance"] = True, True

        # Update model parameters if necessary
        for key in p_dict:
            if key in self.mdl_p:
                self.mdl_p[key] = p_dict[key]
            else:

                # If the parameter doesn't exist, we warn the user
                print("WARNING. Specified parameter '" + str(key) + "' does not exist.")

        if use_pyomo:
            solution = afccp.core.solutions.pyomo_models.solve_original_pyomo_model(self, printing=printing)
        else:
            raise ValueError('Pyomo not available')

        # Set the solution attribute
        if self.mdl_p["set_to_instance"]:
            self.solution = solution
            self.metrics = afccp.core.handling.data_handling.measure_solution_quality(
                self.solution, self.parameters, self.value_parameters, printing=printing)

        # Add solution to solution dictionary
        if self.mdl_p["add_to_dict"]:
            self.add_solution_to_dictionary(solution, solution_method="OG")

    def solve_gp_pyomo_model(self, p_dict={}, printing=None):
        """
        Solve the Goal Programming Model (Created by Lt. Reynolds)
        """

        if printing is None:
            printing = self.printing

        # Reset certain plot parameters
        self.mdl_p["add_to_dict"], self.mdl_p["set_to_instance"] = True, True

        # Update model parameters if necessary
        for key in p_dict:
            if key in self.mdl_p:
                self.mdl_p[key] = p_dict[key]
            else:

                # If the parameter doesn't exist, we warn the user
                print("WARNING. Specified parameter '" + str(key) + "' does not exist.")

        if self.gp_parameters is None:
            self.vft_to_gp_parameters()

        # Solve the model
        r_model = afccp.core.solutions.pyomo_models.gp_model_build(self, printing=printing)
        gp_var = afccp.core.solutions.pyomo_models.gp_model_solve(self, r_model, printing=printing)

        if self.mdl_p["con_term"] is None:
            solution, self.x = gp_var

            # Set the solution attribute
            if self.mdl_p["set_to_instance"]:
                self.solution = solution
                self.metrics = afccp.core.handling.data_handling.measure_solution_quality(
                    self.solution, self.parameters, self.value_parameters, printing=printing)

            # Add solution to solution dictionary
            if self.mdl_p["add_to_dict"]:
                self.add_solution_to_dictionary(solution, solution_method="GP")

            return solution
        else:
            return gp_var

    def full_vft_model_solve(self, p_dict={}, printing=None):
        """
        This is the main method to solve the problem instance. We first solve the pyomo Approximate model, and then
        evolve it using the GA
        """
        if printing is None:
            printing = self.printing

        # Reset certain plot parameters
        self.mdl_p["add_to_dict"], self.mdl_p["set_to_instance"] = False, True

        # Update model parameters if necessary
        for key in p_dict:
            if key in self.mdl_p:
                self.mdl_p[key] = p_dict[key]
            else:

                # If the parameter doesn't exist, we warn the user
                print("WARNING. Specified parameter '" + str(key) + "' does not exist.")

        # Get p_dict with adjustments
        p_dict = self.mdl_p
        p_dict["add_to_dict"] = False

        # Make sure we have selected a set of value parameters
        if self.value_parameters is None:
            raise ValueError("Error. No instance value parameters set.")

        # Determine population for the genetic algorithm if necessary
        if p_dict["populate"]:
            p_dict["initial_solutions"] = afccp.core.comprehensive_functions.populate_initial_ga_solutions(
                self, printing)

            # Add additional solutions if necessary
            if p_dict["solution_names"] is not None:
                for solution_name in p_dict["solution_names"]:
                    solution = self.solution_dict[solution_name]
                    p_dict["initial_solutions"] = np.vstack((p_dict["initial_solutions"], solution))

            p_dict["initialize"] = True

            if printing:
                now = datetime.datetime.now()
                print('Solving Genetic Algorithm for ' + str(self.mdl_p["ga_max_time"]) + ' seconds at ' +
                      now.strftime('%H:%M:%S') + '...')
            self.genetic_algorithm(p_dict, printing=self.mdl_p["ga_printing"])
            if printing:
                print('Solution value of ' + str(round(self.metrics['z'], 4)) + ' obtained.')

        else:

            if printing:
                now = datetime.datetime.now()
                print('Solving VFT Model for ' + str(
                    self.mdl_p["pyomo_max_time"]) + ' seconds at ' + now.strftime('%H:%M:%S') + '...')
            self.solve_vft_pyomo_model(p_dict, printing=False)

            if printing:
                now = datetime.datetime.now()
                print('Solution value of ' + str(round(self.metrics['z'], 4)) + ' obtained.')
                print('Solving Genetic Algorithm for ' + str(self.mdl_p["ga_max_time"]) + ' seconds at ' +
                      now.strftime('%H:%M:%S') + '...')
            self.genetic_algorithm(p_dict, printing=self.mdl_p["ga_printing"])
            if printing:
                print('Solution value of ' + str(round(self.metrics['z'], 4)) + ' obtained.')

        # Add solution to solution dictionary (We just assume that we add it to the dictionary
        self.add_solution_to_dictionary(self.solution, solution_method="AG-VFT")

        # Return solution
        return self.solution

    def solve_for_constraints(self, export_report=True, set_new_constraint_type=False, skip_quota=False):
        """
        This method iteratively adds constraints to the model to find which ones should be included based on
        feasibility and in order of importance
        """

        # Determine which constraints should be turned on!
        if not use_pyomo:
            raise ValueError("Pyomo is not available, model cannot be solved.")

        # If no constraints are turned on right now...
        if np.sum(self.value_parameters["constraint_type"]) == 0:
            raise ValueError("No active constraints to search for, "
                             "make sure the current set of value parameters has active constraints.")

        # Run the function!
        constraint_type, solutions_df, report_df = \
            afccp.core.comprehensive_functions.determine_model_constraints(self, skip_quota=skip_quota)

        if set_new_constraint_type:
            self.value_parameters["constraint_type"] = constraint_type
            self.save_new_value_parameters_to_dict()

        constraint_type_df = pd.DataFrame({'AFSC': self.parameters['afsc_vector']})
        for k, objective in enumerate(self.value_parameters['objectives']):
            constraint_type_df[objective] = constraint_type[:, k]

        # Export to excel
        if export_report:
            filepath = afccp.core.globals.paths["results"] + self.data_name + "_Constraint_Report.xlsx"
            with pd.ExcelWriter(filepath) as writer:
                report_df.to_excel(writer, sheet_name="Report", index=False)
                constraint_type_df.to_excel(writer, sheet_name="Constraints", index=False)
                solutions_df.to_excel(writer, sheet_name="Solutions", index=False)

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

    def compute_similarity_matrix(self, solution_names=None, set_to_instance=True):
        """
        Generates the similarity matrix for a given set of solutions
        """

        if solution_names is None:
            solution_names = list(self.solution_dict.keys())

        # Create the matrix
        num_solutions = len(solution_names)
        similarity_matrix = np.zeros((num_solutions, num_solutions))
        for row, sol_1_name in enumerate(solution_names):
            for col, sol_2_name in enumerate(solution_names):
                sol_1 = self.solution_dict[sol_1_name]
                sol_2 = self.solution_dict[sol_2_name]
                similarity_matrix[row, col] = np.sum(sol_1 == sol_2) / self.parameters["N"]  # % similarity!

        # Set this similarity matrix to be the instance attribute
        if set_to_instance:
            self.similarity_matrix = similarity_matrix

        return similarity_matrix

    def similarity_plot(self, p_dict={}, set_to_instance=True, printing=None):
        """
        Creates the solution similarity plot for the solutions specified
        """
        if printing is None:
            printing = self.printing

        if printing:
            print("Creating solution similarity plot...")

        # Update plot parameters if necessary
        for key in p_dict:
            if key in self.plt_p:
                self.plt_p[key] = p_dict[key]
            else:

                # Exception
                if key == "graph":
                    self.plt_p["results_graph"] = p_dict["graph"]

                else:
                    # If the parameter doesn't exist, we warn the user
                    print("WARNING. Specified parameter '" + str(key) + "' does not exist.")

        # Get our similarity matrix from somewhere
        if self.plt_p["new_similarity_matrix"]:
            similarity_matrix = self.compute_similarity_matrix(solution_names=self.plt_p["solution_names"],
                                                               set_to_instance=set_to_instance)
        else:
            if self.similarity_matrix is None:
                similarity_matrix = self.compute_similarity_matrix(solution_names=self.plt_p["solution_names"],
                                                                   set_to_instance=set_to_instance)
            else:
                similarity_matrix = self.similarity_matrix

        # Get the right solution names
        if self.plt_p["solution_names"] is None:
            self.plt_p["solution_names"] = list(self.solution_dict.keys())

        # Get coordinates
        coords = afccp.core.handling.data_handling.solution_similarity_coordinates(similarity_matrix)

        # Plot similarity
        chart = afccp.core.visualizations.instance_graphs.solution_similarity_graph(self, coords)

        if printing:
            chart.show()

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
        metrics = \
            afccp.core.handling.data_handling.measure_solution_quality(self.x, self.parameters, self.value_parameters)
        values = metrics['objective_value']
        measures = metrics['objective_measure']

        lam, y = afccp.core.solutions.pyomo_models.x_to_solution_initialization(
            self.parameters, self.value_parameters, measures, values)
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

    def get_full_constraint_fail_dictionary(self, solutions, printing=None):
        """
        Get a dictionary of failed constraints. This is used since the Approximate model initial solution must be
        rounded, and we may miss a few constraints by 1 cadet each. This allows the GA to not reject the solution
        initially. This is done for all solutions and we assume that they are all "feasible" (DM feasible)
        """
        if printing is None:
            printing = self.printing

        con_fail_dict = {}
        quota_k = np.where(self.value_parameters['objectives'] == 'Combined Quota')[0][0]
        for s, solution in enumerate(solutions):

            # Measure the solution
            metrics = self.measure_solution(solution=solution, set_solution=False, return_z=False,
                                            printing=False)
            for j in self.parameters['J']:

                afsc = self.parameters["afsc_vector"][j]

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

                                    if (j, k) in con_fail_dict:
                                        current_val = float(con_fail_dict[(j, k)].split(' ')[1])
                                        if new_min < current_val:

                                            if printing:
                                                print("Solution " + str(s + 1) + " AFSC " + str(afsc) +
                                                      " Objective " + objective + " Current Constraint:",
                                                      con_fail_dict[(j, k)], "New Constraint:", '> ' + str(new_min))
                                            con_fail_dict[(j, k)] = '> ' + str(new_min)
                                    else:
                                        con_fail_dict[(j, k)] = '> ' + str(new_min)
                                elif (measure * count) / target_quota > max_measure:
                                    new_max = ceil(1000 * (measure * count) / target_quota) / 1000

                                    if (j, k) in con_fail_dict:
                                        current_val = float(con_fail_dict[(j, k)].split(' ')[1])
                                        if new_max > current_val:

                                            if printing:
                                                print("Solution " + str(s + 1) + " AFSC " + str(afsc) +
                                                      " Objective " + objective + " Current Constraint:",
                                                      con_fail_dict[(j, k)], "New Constraint:", '< ' + str(new_max))
                                            con_fail_dict[(j, k)] = '< ' + str(new_max)
                                    else:
                                        con_fail_dict[(j, k)] = '< ' + str(new_max)
                            else:
                                if measure < min_measure:
                                    if (j, k) in con_fail_dict:
                                        current_val = float(con_fail_dict[(j, k)].split(' ')[1])
                                        if measure < current_val:

                                            if printing:
                                                print("Solution " + str(s + 1) + " AFSC " + str(afsc) +
                                                      " Objective " + objective + " Current Constraint:",
                                                      con_fail_dict[(j, k)], "New Constraint:", '> ' + str(measure))
                                            con_fail_dict[(j, k)] = '> ' + str(measure)
                                    else:
                                        con_fail_dict[(j, k)] = '> ' + str(measure)
                                elif measure > max_measure:
                                    if (j, k) in con_fail_dict:
                                        current_val = float(con_fail_dict[(j, k)].split(' ')[1])
                                        if measure > current_val:

                                            if printing:
                                                print("Solution " + str(s + 1) + " AFSC " + str(afsc) +
                                                      " Objective " + objective + " Current Constraint:",
                                                      con_fail_dict[(j, k)], "New Constraint:", '< ' + str(measure))
                                            con_fail_dict[(j, k)] = '< ' + str(measure)
                                    else:
                                        con_fail_dict[(j, k)] = '< ' + str(measure)

                        # Constrained Exact Measure
                        elif self.value_parameters['constraint_type'][j, k] == 4:

                            value_list = self.value_parameters['objective_value_min'][j, k].split(",")
                            min_measure = float(value_list[0].strip())
                            max_measure = float(value_list[1].strip())

                            if measure < min_measure:
                                if (j, k) in con_fail_dict:
                                    current_val = float(con_fail_dict[(j, k)].split(' ')[1])
                                    if measure < current_val:

                                        if printing:
                                            print("Solution " + str(s + 1) + " AFSC " + str(afsc) +
                                                  " Objective " + objective + " Current Constraint:",
                                                  con_fail_dict[(j, k)], "New Constraint:", '> ' + str(measure))
                                        con_fail_dict[(j, k)] = '> ' + str(measure)
                                else:
                                    con_fail_dict[(j, k)] = '> ' + str(measure)
                            elif measure > max_measure:
                                if (j, k) in con_fail_dict:
                                    current_val = float(con_fail_dict[(j, k)].split(' ')[1])
                                    if measure > current_val:

                                        if printing:
                                            print("Solution " + str(s + 1) + " AFSC " + str(afsc) +
                                                  " Objective " + objective + " Current Constraint:",
                                                  con_fail_dict[(j, k)], "New Constraint:", '< ' + str(measure))
                                        con_fail_dict[(j, k)] = '< ' + str(measure)
                                else:
                                    con_fail_dict[(j, k)] = '< ' + str(measure)

        # print(con_fail_dict)
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

        metrics = afccp.core.handling.data_handling.measure_solution_quality(
            solution, self.parameters, value_parameters, approximate=approximate, printing=printing)

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

        metrics = afccp.core.handling.data_handling.ga_fitness_function(
            solution, self.parameters, self.value_parameters, constraints=constraints, penalty_scale=penalty_scale,
            con_fail_dict=con_fail_dict, first=first, printing=printing)

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
                        metrics = afccp.core.handling.data_handling.measure_solution_quality(
                            solution, self.parameters, value_parameters)
                        self.metrics_dict[vp_name][solution_name] = copy.deepcopy(metrics)

            # Update weights on the sets of value parameters relative to each other
            sum_weights = 0
            for vp_name in self.vp_dict:
                sum_weights += self.vp_dict[vp_name]['vp_weight']
            for vp_name in self.vp_dict:
                self.vp_dict[vp_name]['vp_local_weight'] = self.vp_dict[vp_name]['vp_weight'] / sum_weights

    # Observe Results
    def display_all_results_graphs(self, p_dict={}, printing=None):
        """
        Saves all charts for the current solution and for the solutions in the solution names list if specified
        """

        if printing is None:
            printing = self.printing

        if printing:
            print("Saving all results charts to the corresponding folder...")

        # Force certain parameters
        p_dict["save"], p_dict["graph"] = True, "Measure"

        # Update plot parameters if necessary
        for key in p_dict:
            if key in self.plt_p:
                self.plt_p[key] = p_dict[key]
            else:

                # Exception
                if key == "graph":
                    self.plt_p["results_graph"] = p_dict["graph"]

                else:
                    # If the parameter doesn't exist, we warn the user
                    print("WARNING. Specified parameter '" + str(key) + "' does not exist.")

        p_dict = copy.deepcopy(self.plt_p)

        # If we specify the solution names, that means we want to view all of their individual charts as well
        # as the comparison charts
        charts = []
        if p_dict["solution_names"] is not None:

            # Show both comparison charts and regular ones
            for compare_solutions in [False, True]:
                p_dict["compare_solutions"] = compare_solutions

                if not compare_solutions:

                    # Loop through all solutions and display their charts
                    for solution_name in p_dict["solution_names"]:

                        if printing:
                            print("Saving charts for solution '" + solution_name + "'...")
                        self.set_instance_solution(solution_name)

                        for obj in self.plt_p["afsc_chart_versions"]:
                            p_dict["objective"] = obj
                            for version in self.plt_p["afsc_chart_versions"][obj]:
                                p_dict["version"] = version

                                try:
                                    charts.append(self.display_results_graph(p_dict))
                                except:
                                    pass

                else:

                    if printing:
                        print("Saving charts to compare the solutions...")

                    # Loop through each objective
                    for obj in self.value_parameters["objectives"]:

                        if obj in self.plt_p["afsc_chart_versions"]:
                            p_dict["objective"] = obj

                            try:
                                charts.append(self.display_results_graph(p_dict))
                            except:
                                pass

        else:

            if printing:
                print("Saving charts for solution '" + self.solution_name + "'...")

            if p_dict["use_useful_charts"]:

                # Loop through the subset of charts that I actually care about
                for obj, version in p_dict["desired_charts"]:

                    if printing:
                        print("<Objective '" + obj + "' version '" + version + "'>")
                    p_dict["objective"] = obj
                    p_dict["version"] = version

                    try:
                        charts.append(self.display_results_graph(p_dict))
                    except:
                        pass
            else:

                # Loop through each objective and version
                for obj in self.plt_p["afsc_chart_versions"]:

                    if printing:
                        print("<Charts for objective '" + obj + "'>")
                    p_dict["objective"] = obj
                    for version in self.plt_p["afsc_chart_versions"][obj]:
                        p_dict["version"] = version

                        try:
                            charts.append(self.display_results_graph(p_dict))
                        except:
                            pass

        return charts

    def display_results_graph(self, p_dict={}, printing=None):
        """
        Builds the AFSC Results graphs
        """

        if printing is None:
            printing = self.printing

        if printing:
            print("Saving all results graphs to folder...")

        # Reset chart functional parameters
        self.plt_p, _ = \
            afccp.core.handling.ccp_helping_functions.initialize_instance_functional_parameters(self.parameters["N"])

        # Make sure we have a solution and a set of value parameters activated
        if self.value_parameters is None:
            raise ValueError("Error. No value parameters selected")
        elif self.solution is None:
            raise ValueError("Error. No solution selected")

        # Update plot parameters if necessary
        for key in p_dict:
            if key in self.plt_p:
                self.plt_p[key] = p_dict[key]
            else:

                # Exception
                if key == "graph":
                    self.plt_p["results_graph"] = p_dict["graph"]

                else:
                    # If the parameter doesn't exist, we warn the user
                    print("WARNING. Specified parameter '" + str(key) + "' does not exist.")

        # Update plot parameters
        self.plt_p = afccp.core.handling.ccp_helping_functions.determine_afsc_plot_details(self, results_chart=True)

        # Determine which chart to show
        if self.plt_p["results_graph"] in ["Measure", "Value"]:
            chart = afccp.core.visualizations.instance_graphs.afsc_results_graph(self)
        elif self.plt_p["results_graph"] == "Cadet Utility":
            chart = afccp.core.visualizations.instance_graphs.cadet_utility_histogram(self)
        elif self.plt_p["results_graph"] == "Utility vs. Merit":
            chart = afccp.core.visualizations.instance_graphs.cadet_utility_merit_scatter(self)

        # Get necessary conditions for the multi-criteria chart
        elif self.plt_p["results_graph"] == "Multi-Criteria Comparison":
            if self.plt_p["compare_solutions"]:
                if self.plt_p["solution_names"] is None:

                    # Pick all of the solutions in the dictionary
                    self.plt_p["solution_names"] = list(self.solution_dict.keys())

                # Determine AFSCs to show
                if self.plt_p["comparison_afscs"] is None:

                    # Run through an algorithm to pick the AFSCs to show based on which ones change the most
                    self.plt_p["comparison_afscs"] = \
                        afccp.core.handling.ccp_helping_functions.pick_most_changed_afscs(self)

                # Get correct y-axis scale (scale to most cadets in biggest AFSC!)
                quota_k = np.where(self.value_parameters["objectives"] == "Combined Quota")[0][0]
                max_num = max([max(self.metrics_dict[self.vp_name][solution_name]["objective_measure"][
                                   :, quota_k]) for solution_name in self.plt_p["solution_names"]])

                # Loop through each solution and generate the charts
                for solution_name in self.plt_p["solution_names"]:

                    # Get correct title and filename
                    self.plt_p["title"] = solution_name + ": Multi-Criteria Results Across Certain AFSCs"
                    self.plt_p["filename"] = "Multi_Criteria_Chart_" + solution_name + "_AFSCs"
                    for afsc in self.plt_p["comparison_afscs"]:
                        self.plt_p["filename"] += "_" + afsc

                    # Set the correct solution
                    self.set_instance_solution(solution_name)

                    # Generate chart
                    chart = afccp.core.visualizations.instance_graphs.afsc_multi_criteria_graph(self, max_num=max_num)
            else:

                # Determine AFSCs to show
                if self.plt_p["comparison_afscs"] is None:
                    raise ValueError("No AFSCs specified to compare")
                else:  # AFSCs are specified

                    # Get correct title and filename
                    self.plt_p["title"] = self.solution_name + ": Multi-Criteria Results Across Certain AFSCs"
                    self.plt_p["filename"] = "Multi_Criteria_Chart_" + self.solution_name + "_AFSCs"
                    for afsc in self.plt_p["comparison_afscs"]:
                        self.plt_p["filename"] += "_" + afsc

                    # Generate chart
                    chart = afccp.core.visualizations.instance_graphs.afsc_multi_criteria_graph(self)

        else:
            raise ValueError("Graph '" + self.plt_p["results_graph"] + "' does not exist.")

        return chart

    def generate_slides(self, p_dict={}, printing=None):
        """
        Method to generate the results slides for a particular problem instance with solution
        """

        if printing is None:
            printing = self.printing

        if printing:
            print("Generating results slides...")

        # Reset chart functional parameters
        self.plt_p, _ = \
            afccp.core.handling.ccp_helping_functions.initialize_instance_functional_parameters(self.parameters["N"])

        # Make sure we have a solution and a set of value parameters activated
        if self.value_parameters is None:
            raise ValueError("Error. No value parameters selected")
        elif self.solution is None:
            raise ValueError("Error. No solution selected")

        # Update plot parameters if necessary
        for key in p_dict:
            if key in self.plt_p:
                self.plt_p[key] = p_dict[key]
            else:

                # Exception
                if key == "graph":
                    self.plt_p["results_graph"] = p_dict["graph"]

                else:
                    # If the parameter doesn't exist, we warn the user
                    print("WARNING. Specified parameter '" + str(key) + "' does not exist.")

        # Call the function to generate the slides
        afccp.core.visualizations.slides.generate_results_slides(self)

    # Sensitivity Analysis
    def initial_overall_weights_pareto_analysis(self, p_dict={}, printing=None):
        """
        Takes the current set of value parameters and solves the VFT approximate model solution multiple times
        given different overall weights on AFSCs and cadets. Once these solutions are determined, we can then
        run the other method "final overall_weights_pareto_analysis" to evolve all the solutions together
        given the different overall weights on cadets
        :param p_dict: more model parameters
        :param printing: whether to print something
        """

        # File we import and export to
        filepath = afccp.core.globals.paths['results'] + self.data_name + '_Pareto_Analysis.xlsx'

        if printing is None:
            printing = self.printing

        if printing:
            print("Conducting pareto analysis on problem...")

        # Update plot parameters if necessary
        for key in p_dict:
            if key in self.mdl_p:
                self.mdl_p[key] = p_dict[key]
            else:

                # Exception
                if key == "step":
                    self.mdl_p["pareto_step"] = p_dict["step"]

                else:
                    # If the parameter doesn't exist, we warn the user
                    print("WARNING. Specified parameter '" + str(key) + "' does not exist.")

        # Initialize arrays
        num_points = int(100 / self.mdl_p["pareto_step"] + 1)
        cadet_overall_values = np.zeros(num_points)
        afsc_overall_values = np.zeros(num_points)
        cadet_overall_weights = np.arange(1, 0, -(self.mdl_p["pareto_step"] / 100))
        cadet_overall_weights = np.append(cadet_overall_weights, 0)
        solutions = {}

        # Iterate over the number of points needed for the Pareto Chart
        for point in range(num_points):
            self.value_parameters['cadets_overall_weight'] = cadet_overall_weights[point]
            self.value_parameters['afscs_overall_weight'] = 1 - cadet_overall_weights[point]

            if printing:
                print("Calculating point " + str(point + 1) + " out of " + str(num_points) + "...")

            solution = self.solve_vft_pyomo_model(p_dict, printing=False)
            solution_name = str(round(cadet_overall_weights[point], 4))
            afsc_solution = np.array([self.parameters["afsc_vector"][int(j)] for j in solution])
            solutions[solution_name] = afsc_solution

            cadet_overall_values[point] = self.metrics['cadets_overall_value']
            afsc_overall_values[point] = self.metrics['afscs_overall_value']

            if printing:
                print('For an overall weight on cadets of ' + str(cadet_overall_weights[point]) +
                      ', calculated value on cadets: ' + str(round(cadet_overall_values[point], 2)) +
                      ', value on afscs: ' + str(round(afsc_overall_values[point], 2)) +
                      ', and a Z of ' + str(round(self.metrics['z'], 2)) + '.')

        # Obtain Dataframes
        pareto_df = pd.DataFrame(
            {'Weight on Cadets': cadet_overall_weights, 'Value on Cadets': cadet_overall_values,
             'Value on AFSCs': afsc_overall_values})
        solutions_df = pd.DataFrame(solutions)

        with pd.ExcelWriter(filepath) as writer:  # Export to excel
            pareto_df.to_excel(writer, sheet_name="Approximate Pareto Results", index=False)
            solutions_df.to_excel(writer, sheet_name="Initial Solutions", index=False)

    def genetic_overall_weights_pareto_analysis(self, p_dict={}, printing=None):
        """
        Takes the current set of value parameters and loads in a dataframe of solutions found using the VFT Approximate
        model. These solutions are the initial population for the GA that evolves them all together using different sets
        of value parameters (different overall weights on cadets)
        :param p_dict: more model parameters
        :param printing: whether to print something
        """

        # File we import and export to
        filepath = afccp.core.globals.paths['results'] + self.data_name + '_Pareto_Analysis.xlsx'

        if printing is None:
            printing = self.printing

        if printing:
            print("Conducting 'final' pareto analysis on problem...")

        # Solutions Dataframe
        solutions_df = afccp.core.globals.import_data(filepath, sheet_name='Initial Solutions')
        approximate_results_df = afccp.core.globals.import_data(filepath, sheet_name='Approximate Pareto Results')

        # Update plot parameters if necessary
        for key in p_dict:
            if key in self.mdl_p:
                self.mdl_p[key] = p_dict[key]
            else:

                # Exception
                if key == "step":
                    self.mdl_p["pareto_step"] = p_dict["step"]

                else:
                    # If the parameter doesn't exist, we warn the user
                    print("WARNING. Specified parameter '" + str(key) + "' does not exist.")

        # Load in the initial solutions
        initial_afsc_solutions = np.array([np.array(solutions_df[col]) for col in solutions_df])
        initial_solutions = np.array([])
        for i, afsc_solution in enumerate(initial_afsc_solutions):
            solution = np.array([np.where(self.parameters["afsc_vector"] == afsc)[0][0] for afsc in afsc_solution])
            if i == 0:
                initial_solutions = solution
            else:
                initial_solutions = np.vstack((initial_solutions, solution))

        # Initialize arrays
        num_points = int(100 / self.mdl_p["pareto_step"] + 1)
        cadet_overall_values = np.zeros(num_points)
        afsc_overall_values = np.zeros(num_points)
        cadet_overall_weights = np.arange(1, 0, -(self.mdl_p["pareto_step"] / 100))
        cadet_overall_weights = np.append(cadet_overall_weights, 0)
        solutions = {}

        # Iterate over the number of points needed for the Pareto Chart
        for point in range(num_points):

            # Current list of initial solutions gets used by the Genetic Algorithm
            self.mdl_p["initial_solutions"] = initial_solutions

            # Change Overall Weights
            self.value_parameters['cadets_overall_weight'] = cadet_overall_weights[point]
            self.value_parameters['afscs_overall_weight'] = 1 - cadet_overall_weights[point]

            if printing:
                print("Calculating point " + str(point + 1) + " out of " + str(num_points) + "...")

            solution = self.genetic_algorithm(self.mdl_p)
            solution_name = str(round(cadet_overall_weights[point], 4))
            afsc_solution = np.array([self.parameters["afsc_vector"][int(j)] for j in solution])
            solutions[solution_name] = afsc_solution

            cadet_overall_values[point] = self.metrics['cadets_overall_value']
            afsc_overall_values[point] = self.metrics['afscs_overall_value']

            # We add this solution to our initial solutions too
            initial_solutions = np.vstack((initial_solutions, solution))

            if printing:
                print('For an overall weight on cadets of ' + str(cadet_overall_weights[point]) +
                      ', calculated value on cadets: ' + str(round(cadet_overall_values[point], 2)) +
                      ', value on afscs: ' + str(round(afsc_overall_values[point], 2)) +
                      ', and a Z of ' + str(round(self.metrics['z'], 2)) + '.')

        # Obtain Dataframes
        pareto_df = pd.DataFrame(
            {'Weight on Cadets': cadet_overall_weights, 'Value on Cadets': cadet_overall_values,
             'Value on AFSCs': afsc_overall_values})
        ga_solutions_df = pd.DataFrame(solutions)

        with pd.ExcelWriter(filepath) as writer:  # Export to excel
            approximate_results_df.to_excel(writer, sheet_name="Approximate Pareto Results", index=False)
            solutions_df.to_excel(writer, sheet_name="Initial Solutions", index=False)
            pareto_df.to_excel(writer, sheet_name="GA Pareto Results", index=False)
            ga_solutions_df.to_excel(writer, sheet_name="GA Solutions", index=False)

    def show_pareto_chart(self, printing=None):
        """
        Saves the pareto chart to the figures folder
        """
        # File we import and export to
        filepath = afccp.core.globals.paths['results'] + self.data_name + '_Pareto_Analysis.xlsx'

        if printing is None:
            printing = self.printing

        if printing:
            print("Creating Pareto Chart...")

        try:
            pareto_df = afccp.core.globals.import_data(filepath, sheet_name='GA Pareto Results')
        except:
            try:
                pareto_df = afccp.core.globals.import_data(filepath, sheet_name='Approximate Pareto Results')
            except:
                raise ValueError("No Pareto Data found for instance '" + self.data_name + "'")

        return afccp.core.visualizations.instance_graphs.pareto_graph(pareto_df)

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
            value_parameters = afccp.core.solutions.pyomo_models.least_squares_procedure(
                self.parameters, self.value_parameters, self.solution, t_solution, delta=delta, printing=printing)
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

        chart = afccp.core.visualizations.instance_graphs.afsc_objective_weights_graph(
            self.parameters, vp_dict, afsc, colors=colors, save=save, figsize=figsize, facecolor=facecolor, title=title,
            legend_size=legend_size, display_title=display_title, label_size=label_size, title_size=title_size,
            xaxis_tick_size=xaxis_tick_size, yaxis_tick_size=yaxis_tick_size, bar_color=bar_color)
        if printing:
            chart.show()

        return chart

    # Export
    def create_aggregate_file(self, printing=None):
        """
        Create the "data_type data_name" main aggregate file with solutions, metrics, and vps
        """
        if printing is None:
            printing = self.printing

        if printing:
            print('Creating aggregate problem instance excel file...')

        self.update_metrics_dict()
        metrics_dict = self.metrics_dict
        solution_dict = self.solution_dict
        vp_dict = self.vp_dict

        afccp.core.comprehensive_functions.create_aggregate_instance_file(
            self.data_name, self.parameters, solution_dict, vp_dict, metrics_dict, self.gp_df, info_df=self.info_df,
            similarity_matrix=self.similarity_matrix)

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
                p_s = afccp.core.handling.data_handlingcompare_solutions(
                    self.solution, self.solution_dict[self.solution_name])
                if p_s != 1:
                    raise ValueError("Current solution: " + self.solution_name +
                                     ". This is not the same solution as found in the dictionary under that name.")

            else:
                raise ValueError("Solution " + self.solution_name + " not found in dictionary.")

            if self.vp_name in self.vp_dict:

                # Check if the instance value parameters are the same as the ones in the dictionary
                vp_same = afccp.core.handling.value_parameter_handling.compare_value_parameters(
                    self.parameters, self.value_parameters, self.vp_dict[self.vp_name])
                if not vp_same:
                    raise ValueError("Current value parameters: " + self.vp_name +
                                     ". This is not the same set of value parameters as found in the "
                                     "dictionary under that name.")
            else:
                raise ValueError("Value Parameters " + self.vp_name + " not found in dictionary.")

            # Export to the right place
            self.full_name = self.data_type + " " + self.data_name + " " + self.vp_name + " " + self.solution_name
            filepath = afccp.core.globals.paths['instances'] + self.full_name + '.xlsx'

            # Export to excel
            afccp.core.comprehensive_functions.data_to_excel(
                filepath, self.parameters, self.value_parameters, self.metrics, printing=printing)

        if printing:
            print('Exported.')


try:
    from afccp.core.matching.problem_class_add import MoreCCPMethods

    # Load in the methods from the other sheet and attach them to CadetCareerProblem
    for method_name in dir(MoreCCPMethods):
        method = getattr(MoreCCPMethods, method_name)
        if not method_name.startswith('__'):
            setattr(CadetCareerProblem, method.__name__, method)
except:
    print('Tried to import more methods for CCP, but failed. Please fix something.')
