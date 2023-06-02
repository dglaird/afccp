# Import libraries
from typing import Any

import os

import pandas as pd
import numpy as np
import math
import afccp.core.globals
import afccp.core.data.simulation_functions
import afccp.core.data.values
import afccp.core.data.ccp_helping_functions
import afccp.core.visualizations.slides
import afccp.core.visualizations.instance_graphs
import afccp.core.comprehensive_functions
import afccp.core.data.processing
import afccp.core.data.preferences
import afccp.core.solutions.handling
import datetime
import glob
import copy
import time

# Import pyomo models if library is installed
if afccp.core.globals.use_pyomo:
    import afccp.core.solutions.pyomo_models


# Main Problem Class
class CadetCareerProblem:
    def __init__(self, data_name="Random", data_version="Default", degree_qual_type="Consistent",
                 num_value_function_breakpoints=None, N=1600, M=32, P=6, printing=True):
        """
        This is the AFSC/Cadet problem object. We can import data by providing a "data_name". This is the name of the
        instance of the problem that we are solving. Usually a class year, though you can also generate data to solve by
        specifying "Random", "Realistic", or "Perfect".
        :param data_name: The name of the data set. If we want to import data using the data_name, this needs to be
        the name of the main instance sub-folder
        :param data_version: A string of some sort that indicates the version of the data we're dealing with. All the
        folders we import/export to will have this designation in parentheses and this is to manage the different
        versions of cadets/AFSCs to mitigate the headache of adjudication and also handle NRL vs All cadet scenarios
        :param degree_qual_type: A string variable indicating how we should handle degree tier qualifications. There
        are 3 kinds: "Binary"- prior to 2015, you either qualified (1) for an AFSC or didn't (0), "Relaxed"- since the
        AFOCD started, AFPC/DSY has been combining "Mandatory", "Desired", and "Permitted" tiers by those labels to
        make the model less restrictive, and lastly "Tiers"- which is something we (AFPC/DSY) are now adhering to which
        factors in both requirement level ("Mandatory", "Desired", and "Permitted") and the tier itself (1, 2, 3, 4).
        The default parameter here is "Consistent" which will just keep the method of solving the model and representing
        the data the same for whatever instance is being loaded in.
        :param num_value_function_breakpoints: Number of breakpoints to use in the value functions. If None, then we
        will load in the value functions using the breakpoints listed. If provided an integer, we will recreate the
        value functions using the "vf_strings"
        :param N: Number of cadets to generate
        :param M: Number of AFSCs to generate
        :param P: Number of AFSC preferences to generate for each cadet
        :param printing: Whether we should print status updates or not
        """

        # Data data attributes
        self.data_version = data_version  # Version of instance (in parentheses of the instance sub-folders)
        self.import_paths, self.export_paths = None, None  # Paths to various datasets we can import/export
        self.printing = printing

        # The data variant helps inform how the charts should be constructed
        if len(data_name) == 1:
            self.data_variant = "Scrubbed"
        elif len(data_name) == 4:
            self.data_variant = "Year"
        else:
            self.data_variant = "Generated"

        # Instance components (value parameters, solution, solution metrics)
        self.value_parameters, self.vp_name = None, None  # Set of weight and value parameters (and the name)
        self.solution, self.solution_name = None, None  # Array of length N (cadets) of AFSC indices (and the name)
        self.metrics = None # Dictionary of solution metrics
        self.x = None  # Solution "X" matrix (NxM binary matrix)

        # Dictionaries of instance components (sets of value parameters, solution, solution metrics)
        self.vp_dict, self.solution_dict, self.metrics_dict = None, None, None

        # Parameters from *former* Lt Rebecca Reynold's thesis
        self.gp_parameters, self.gp_df = None, None

        # If we have an instance folder already for the specified instance (we're importing it)
        if data_name in afccp.core.globals.instances_available:

            # Shorten the module name so everything fits better
            afccp_dp = afccp.core.data.processing

            # Gather information about the files we're importing and eventually exporting
            self.data_name = data_name
            self.import_paths, self.export_paths = afccp_dp.initialize_file_information(self.data_name,
                                                                                        self.data_version)

            # Print statement
            if self.printing:
                files = [self.import_paths[filename] for filename in self.import_paths]
                print("Importing '" + data_name + "' instance with datasets:\n", files)

            # Initialize dictionary of instance parameters (Information pertaining to cadets and AFSCs)
            self.parameters = {"Qual Type": degree_qual_type}

            # Import the "fixed" parameters (the information about cadets/AFSCs that, for the most part, doesn't change)
            import_data_functions = [afccp_dp.import_afscs_data, afccp_dp.import_cadets_data,
                                     afccp_dp.import_preferences_data]
            for import_function in import_data_functions:
                self.parameters = import_function(self.import_paths, self.parameters)

            # Additional sets and subsets of cadets/AFSCs need to be loaded into the instance parameters
            self.parameters = afccp_dp.parameter_sets_additions(self.parameters)

            # Import the "Goal Programming" dataframe (from Lt Rebecca Reynold's thesis)
            if "Goal Programming" in self.import_paths:
                self.gp_df = afccp.core.globals.import_csv_data(self.import_paths["Goal Programming"])
            else:  # Default "GP" file
                self.gp_df = afccp.core.globals.import_data(
                    afccp.core.globals.paths["support"] + "data/gp_parameters.xlsx")

            # Import the "Value Parameters" data dictionary
            self.vp_dict = afccp_dp.import_value_parameters_data(self.import_paths, self.parameters,
                                                                 num_value_function_breakpoints)

            # Import the "Solutions" data dictionary
            self.solution_dict = afccp_dp.import_solutions_data(self.import_paths, self.parameters)

        # This is a new problem instance that we're generating (Should be "Random", "Perfect", or "Realistic"
        else:

            # Error Handling (Must be valid data generation parameter
            if data_name not in ["Random", "Perfect", "Realistic"]:
                raise ValueError(
                    "Error. Instance name '" + data_name + "' is not a valid instance name. Instances must "
                                                           "be either generated or imported. "
                                                           "(Instance not found in folder).")

            # Determine the name of this instance (Random_1, Random_2, etc.)
            for data_variant in ["Random", "Perfect", "Realistic"]:
                if data_name == data_variant:

                    # Count how many instances we already have of this type to get the name of this new instance
                    variant_counter = 1
                    for instance_name in afccp.core.globals.instances_available:
                        if data_variant in instance_name:
                            variant_counter += 1
                    self.data_name = data_variant + "_" + str(variant_counter)

            # Print statement
            if self.printing:
                print("Generating '" + self.data_name + "' instance...")

            # For now, we can't generate data
            self.parameters = {"N": 10}  # (Just so the below function works)
            print("Didn't actually generate data yet..")

        # Initialize more "functional" parameters
        self.plt_p, self.mdl_p = \
            afccp.core.data.ccp_helping_functions.initialize_instance_functional_parameters(self.parameters["N"])

        if self.printing:
            print("Instance '" + self.data_name + "' initialized.")

    # Method helper functions
    def reset_functional_parameters(self, p_dict={}):
        """
        This method simply "resets" our instance functional parameters and then inputs the "new" ones from
        the "p_dict". This all happens in-place.
        """
        # Shorthand
        ccp_fns = afccp.core.data.ccp_helping_functions

        # Reset plot parameters and model parameters
        self.plt_p, self.mdl_p = ccp_fns.initialize_instance_functional_parameters(self.parameters["N"])

        # Update plot parameters and model parameters
        for key in p_dict:

            if key in self.plt_p:
                self.plt_p[key] = p_dict[key]
            elif key in self.mdl_p:
                self.mdl_p[key] = p_dict[key]
            else:
                print("WARNING. Specified parameter '" + str(key) + "' does not exist.")

    def solution_handling(self, solution, solution_method, x=None):
        """
        This method simply determines what to do with the solution that we've generated. I made this method
        since I had the same few lines of code re-used everywhere and decided to put it in a method.
        """

        # Set the solution attribute to the instance
        if self.mdl_p["set_to_instance"]:
            self.solution = solution
            self.x = x
            self.metrics = afccp.core.solutions.handling.evaluate_solution(
                self.solution, self.parameters, self.value_parameters, printing=self.printing)

        # Add solution to solution dictionary
        if self.mdl_p["add_to_dict"]:

            # Adjust solution method for VFT models
            if solution_method == "VFT" and self.mdl_p["approximate"]:
                solution_method = "A-VFT"
            elif solution_method == "VFT" and not self.mdl_p["approximate"]:
                solution_method = "E-VFT"
            self.add_solution_to_dictionary(solution, solution_method=solution_method)

    def error_checking(self, test="None"):
        """
        This method is here to test different conditions and raise errors where conditions are not met.
        """

        # Nested function to test if value parameters are specified
        def check_value_parameters():
            if self.value_parameters is None:
                if self.vp_dict is None:
                    raise ValueError(
                        "Error. No value parameters detected. You need to import the defaults first by using"
                        "'instance.import_default_value_parameters()' as you currently do not have a "
                        "'vp_dict' either.")
                else:
                    raise ValueError("Error. No value parameters set. You currently do have a 'vp_dict' and so all you "
                                     "need to do is run 'instance.set_instance_value_parameters()'. ")

        if test == "Value Parameters":
            check_value_parameters()
        elif test == "Pyomo Model":
            check_value_parameters()

            # We don't have Pyomo
            if not afccp.core.globals.use_pyomo:
                raise ValueError("Error. Pyomo is not currently installed and is required to run pyomo models. Please"
                                 "install this library.")

    # Visualizations
    def display_data_graph(self, p_dict={}, printing=None):
        """
        This method plots different aspects of the fixed parameters of the problem instance.
        """

        # Print statement
        if printing is None:
            printing = self.printing

        # Adjust instance plot parameters
        self.reset_functional_parameters(p_dict)
        self.plt_p = afccp.core.data.ccp_helping_functions.determine_afsc_plot_details(self)

        # Initialize the AFSC Chart object
        afsc_chart = afccp.core.visualizations.instance_graphs.AFSCsChart(self)

        # Construct the specific chart
        return afsc_chart.build(printing=printing)

    def display_all_data_graphs(self, p_dict={}, printing=None):
        """
        This method runs through all the different versions of graphs we have and saves
        them to the corresponding folder.
        """
        if printing is None:
            printing = self.printing

        if printing:
            print("Saving all data graphs to the corresponding folder...")

        # Regular Charts
        charts = []
        for graph in ["Average Utility", "USAFA Proportion", "Average Merit", "AFOCD Data", "Eligible Quota"]:
            p_dict["data_graph"] = graph
            charts.append(self.display_data_graph(p_dict, printing=printing))

        # Cadet Preference Analysis Charts
        p_dict["data_graph"] = "Cadet Preference Analysis"
        for version in range(1, 8):
            p_dict["version"] = str(version)
            charts.append(self.display_data_graph(p_dict, printing=printing))

        return charts

    # Adjust Data
    def adjust_qualification_matrix(self, printing=None):
        """
        This procedure simply re-runs the CIP to Qual function in case I change qualifications or I
        identify errors since some cadets in the AFPC solution may receive AFSCs for which they
        are ineligible for.
        """
        if printing is None:
            printing = self.printing

        if printing:
            print('Adjusting qualification matrix...')
        parameters = copy.deepcopy(self.parameters)

        # Generate new matrix
        if "cip1" in parameters:
            if "cip2" in parameters:
                qual_matrix = afccp.core.data.preprocessing.cip_to_qual_tiers(
                    parameters["afscs"][:parameters["M"]], parameters['cip1'], cip2=parameters['cip2'])
            else:
                qual_matrix = afccp.core.data.preprocessing.cip_to_qual_tiers(
                    parameters["afscs"][:parameters["M"]], parameters['cip1'])
        else:
            raise ValueError("Error. Need to update the degree tier qualification matrix to include tiers "
                             "('M1' instead of 'M' for example) but don't have CIP codes. Please incorporate this.")

        # Load data back into parameters
        parameters["qual"] = qual_matrix
        parameters["qual_type"] = "Tiers"
        parameters["ineligible"] = (np.core.defchararray.find(qual_matrix, "I") != -1) * 1
        parameters["eligible"] = (parameters["ineligible"] == 0) * 1
        parameters["exception"] = (np.core.defchararray.find(qual_matrix, "E") != -1) * 1
        for t in [1, 2, 3, 4]:
            parameters["tier " + str(t)] = (np.core.defchararray.find(qual_matrix, str(t)) != -1) * 1
        parameters = afccp.core.data.processing.parameter_sets_additions(parameters)
        self.parameters = copy.deepcopy(parameters)

    def convert_utilities_to_preferences(self, cadets_as_well=False):
        """
        Converts the utility matrices to preferences
        """
        self.parameters = afccp.core.data.preferences.convert_utility_matrices_preferences(self.parameters,
                                                                                                 cadets_as_well)

    def generate_fake_afsc_preferences(self):
        """
        Uses the VFT parameters to generate simulated AFSC preferences
        """
        self.parameters = afccp.core.data.preferences.generate_fake_afsc_preferences(
            self.parameters, self.value_parameters)

    def convert_afsc_preferences_to_percentiles(self):
        """
        This method takes the AFSC preference lists and turns them into normalized percentiles for each cadet for each
        AFSC.
        """
        self.parameters = afccp.core.data.preferences.convert_afsc_preferences_to_percentiles(self.parameters)

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
            raise ValueError("Error. No sets of value parameters (vp_dict) detected. You need to import the "
                             "defaults first by using 'instance.import_default_value_parameters()'.")
        else:
            if vp_name is None:  # Name not specified
                self.vp_name = list(self.vp_dict.keys())[0]  # Take the first set in the dictionary
                self.value_parameters = copy.deepcopy(self.vp_dict[self.vp_name])
            else:  # Name specified
                if vp_name not in self.vp_dict:
                    raise ValueError(vp_name + ' set not in value parameter dictionary. Available sets are:',
                                     self.vp_dict.keys())
                else:
                    self.value_parameters = copy.deepcopy(self.vp_dict[vp_name])
                    self.vp_name = vp_name

            if self.solution is not None:
                self.measure_solution()

    def save_new_value_parameters_to_dict(self, value_parameters=None):
        """
        Adds the set of value parameters to a dictionary as a new set (if it is a unique set)
        """
        # Grab the value parameters
        if value_parameters is None:
            value_parameters = self.value_parameters

        # Make sure this is a valid dictionary of value parameters
        if value_parameters is not None:
            if self.vp_dict is None:
                self.vp_dict, self.vp_name = {"VP": copy.deepcopy(value_parameters)}, "VP"
            else:

                # Determine new value parameters name
                vp_name, v = "VP2", 2
                while vp_name in self.vp_dict:
                    v += 1
                    vp_name = "VP" + str(v)

                # Check if this new set is unique or not to get the name of the set
                unique = self.check_unique_value_parameters()
                if unique is True: # If it's unique, we save this new set of value parameters to the dictionary
                    self.vp_dict[vp_name], self.vp_name = copy.deepcopy(value_parameters), self.vp_name
                else:  # If it's not unique, then "unique" is the name of the matching set of value parameters
                    self.vp_name = unique

        else:
            raise ValueError("Error. No value parameters set. You currently do have a 'vp_dict' and so all you "
                                     "need to do is run 'instance.set_instance_value_parameters()'. ")

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

    def check_unique_value_parameters(self, value_parameters=None, vp_name1=None, printing=False):
        """
        Take in a new set of value parameters and see if this set is in the dictionary already. Return True if the
        the set of parameters is unique, or return the name of the matching set otherwise
        """
        if value_parameters is None:
            value_parameters = self.value_parameters
            vp_name1 = self.vp_name

        if vp_name1 is None:
            vp_name1 = "VP (Unspecified)"

        # Assume the new set is unique until proven otherwise
        unique = True
        for vp_name in self.vp_dict:
            identical = afccp.core.data.values.compare_value_parameters(
                self.parameters, value_parameters, self.vp_dict[vp_name], vp_name1, vp_name, printing=printing)
            if identical:
                unique = vp_name
                break
        return unique

    def import_default_value_parameters(self, no_constraints=False, num_breakpoints=24,
                                        generate_afsc_weights=True, vp_weight=100, printing=None):
        """
        Import default value parameter settings and generate value parameters for this instance.
        """

        if printing is None:
            printing = self.printing

        # Folder/Files information
        folder_path = afccp.core.globals.paths["support"] + "value parameters defaults/"
        vp_defaults_folder = os.listdir(folder_path)
        vp_defaults_filename = "Value_Parameters_Defaults_" + self.data_name + ".xlsx"

        # Determine the path to the default value parameters we need to import
        if vp_defaults_filename in vp_defaults_folder:
            filename = vp_defaults_filename
        elif len(self.data_name) == 4:
            filename = "Value_Parameters_Defaults.xlsx"
        elif "Perfect" in self.data_name:
            filename = "Value_Parameters_Defaults_Perfect.xlsx"
        else:
            filename = "Value_Parameters_Defaults_Generated.xlsx"
        filepath = folder_path + filename

        # Module shorthand
        afccp_vp = afccp.core.data.values

        # Import "default value parameters" from excel
        dvp = afccp_vp.default_value_parameters_from_excel(filepath, num_breakpoints=num_breakpoints, printing=printing)

        # Generate this instance's value parameters from the defaults
        value_parameters = afccp_vp.generate_value_parameters_from_defaults(
            self.parameters, generate_afsc_weights=generate_afsc_weights, default_value_parameters=dvp)

        # Add some additional components to the value parameters
        if no_constraints:
            value_parameters['constraint_type'] = np.zeros([self.parameters['M'], value_parameters['O']])

        # "Condense" the value functions by removing unnecessary zeros
        value_parameters = afccp_vp.condense_value_functions(self.parameters, value_parameters)

        # Add indexed sets and subsets of AFSCs and AFSC objectives
        value_parameters = afccp_vp.value_parameters_sets_additions(self.parameters, value_parameters)

        # Weight of the value parameters as a whole
        value_parameters['vp_weight'] = vp_weight

        # Set value parameters to instance attribute
        if self.mdl_p["set_to_instance"]:
            self.value_parameters = value_parameters
            if self.solution is not None:
                self.metrics = afccp.core.solutions.handling.evaluate_solution(
                    self.solution, self.parameters, self.value_parameters, printing=printing)

        # Save new set of value parameters to dictionary
        if self.mdl_p["add_to_dict"]:
            self.save_new_value_parameters_to_dict(value_parameters)

        if self.printing:
            print('Imported.')

        return value_parameters

    def export_value_parameters_as_defaults(self, filename=None, printing=None):
        """
        This method exports the current set of instance value parameters to a new excel file in the "default"
        value parameter format
        """
        if printing is None:
            printing = self.printing

        if self.value_parameters is None:
            raise ValueError('No instance value parameters detected.')

        if filename is None:  # I add the "_New" just so we make sure we don't accidentally overwrite the old one
            filename = "Value_Parameters_Defaults_" + self.data_name + "_New.xlsx"
        filepath = afccp.core.globals.paths["support"] + "value parameters defaults/" + filename

        afccp.core.data.values.model_value_parameters_to_defaults(
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
                afccp.core.data.values.cadet_weight_function(merit, function)

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
                afccp.core.data.values.afsc_weight_function(quota, function)

    def parameter_sanity_check(self):
        """
        This method runs through all of the parameters and value parameters to sanity check them to make sure
        everything is correct and there are no issues with the data before we run the model.
        """
        afccp.core.data.processing.parameter_sanity_check(self)

    # noinspection PyDictCreation
    def vft_to_gp_parameters(self, p_dict={}, printing=None):
        """
        Converts the instance parameters and value parameters to parameters used by Rebecca's model
        """

        if printing is None:
            printing = self.printing

        # Print statement
        if printing:
            print('Translating VFT model parameters to Goal Programming Model parameters...')

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Get basic parameters (May or may not include penalty/reward parameters
        self.gp_parameters = afccp.core.data.values.translate_vft_to_gp_parameters(self)

        # Get list of constraints
        con_list = copy.deepcopy(self.gp_parameters['con'])
        con_list.append("S")
        num_constraints = len(con_list)

        # Convert gp_df to dictionary of numpy arrays ("gc" = "gp_df columns")
        gc = {col: np.array(self.gp_df[col]) for col in self.gp_df}

        # Do we want to create new rewards/penalties for this problem instance by iterating with the model?
        if self.mdl_p["get_new_rewards_penalties"]:

            gc["Raw Reward"], gc["Raw Penalty"] = afccp.core.solutions.pyomo_models.calculate_rewards_penalties(
                self, printing=printing)
            min_p = min([penalty for penalty in gc["Raw Penalty"] if penalty != 0])
            min_r = min(gc["Raw Reward"])
            gc["Normalized Penalty"] = np.array([min_p / penalty if penalty != 0 else 0 for penalty in gc["Raw Penalty"]])
            gc["Normalized Reward"] = np.array([min_r / reward for reward in gc["Raw Reward"]])

        # Rewards/Penalties used in the model "run"
        gc["Run Penalty"] = np.array(
            [gc["Penalty Weight"][c] /
             gc["Normalized Penalty"][c] if gc["Normalized Penalty"][c] != 0 else 0 for c in range(num_constraints)])
        gc["Run Reward"] = np.array([gc["Reward Weight"][c] / gc["Normalized Reward"][c] for c in range(num_constraints)])

        # Convert back to pandas dataframe using numpy array dictionary
        self.gp_df = pd.DataFrame(gc)

        # Update the "mu" and "lam" parameters with our new Reward/Penalty terms
        self.gp_parameters = afccp.core.data.values.translate_vft_to_gp_parameters(self)

    # Observe Value Parameters
    def show_value_function(self, p_dict={}, printing=None):
        """
        This method plots a specific AFSC objective value function
        """

        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)
        self.plt_p = afccp.core.data.ccp_helping_functions.determine_afsc_plot_details(self)

        # Build the chart
        value_function_chart = afccp.core.comprehensive_functions.plot_value_function(self, printing)

        if printing:
            value_function_chart.show()

        return value_function_chart

    def display_weight_function(self, p_dict={}, printing=None):
        """
        This method plots the weight function used for either cadets or AFSCs
        """

        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)
        self.plt_p = afccp.core.data.ccp_helping_functions.determine_afsc_plot_details(self)

        if printing:
            if self.plt_p["cadets_graph"]:
                print("Creating cadet weight chart...")
            else:
                print("Creating AFSC weight chart...")

        # Build the chart
        chart = afccp.core.visualizations.instance_graphs.individual_weight_graph(self)

        if printing:
            chart.show()

        return chart

    # Solve Models
    def generate_random_solution(self, p_dict={}, printing=None):
        """
        Generate random solution by assigning cadets to AFSCs that they're eligible for
        """
        if printing is None:
            printing = self.printing

        if printing:
            print('Generating random solution...')

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Generate solution
        solution = np.array([np.random.choice(self.parameters['J^E'][i]) for i in self.parameters['I']])

        # Determine what to do with the solution
        self.solution_handling(solution, solution_method="Random")

        return solution

    def stable_matching(self, p_dict={}, printing=None):
        """
        This method solves the stable marriage heuristic for an initial solution
        """
        self.error_checking("Value Parameters")
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Generate solution
        solution = afccp.core.solutions.heuristic_solvers.stable_marriage_model_solve(self, printing=printing)

        # Determine what to do with the solution
        self.solution_handling(solution, solution_method="Stable")

        return solution

    def matching_algorithm_1(self, p_dict={}, printing=None):
        """
        This method solves the problem instance using "Matching Algorithm 1"
        """
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        solution = afccp.core.solutions.heuristic_solvers.matching_algorithm_1(self, printing=printing)

        # Determine what to do with the solution
        self.solution_handling(solution, solution_method="MA1")

        return solution

    def greedy_method(self, p_dict={}, printing=None):
        """
        This method solves the greedy heuristic for an initial solution
        """
        self.error_checking("Value Parameters")
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Generate solution
        solution = afccp.core.solutions.heuristic_solvers.greedy_model_solve(self, printing=printing)

        # Determine what to do with the solution
        self.solution_handling(solution, solution_method="Greedy")
        return solution

    def genetic_algorithm(self, p_dict={}, printing=None):
        """
        This is the genetic algorithm. The hyper-parameters to the algorithm can be tuned, and this is meant to be
        solved in conjunction with the pyomo model solution. Use that as the initial solution, and then we evolve
        from there
        """
        self.error_checking("Value Parameters")
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

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

                        # Get list of initial solutions
                        initial_solutions = np.array(
                            [self.solution_dict[solution_name] for solution_name in self.solution_dict])
                        solution_names = list(self.solution_dict.keys())

                    else:

                        # If we just pass "Solution" instead of ["Solution"]
                        if type(self.mdl_p["solution_names"]) == str:
                            self.mdl_p["solution_names"] = [self.mdl_p["solution_names"]]

                        # Get list of initial solutions
                        initial_solutions = np.array(
                            [self.solution_dict[solution_name] for solution_name in self.mdl_p["solution_names"]])
                        solution_names = self.mdl_p["solution_names"]

                    if printing:
                        print("Running Genetic Algorithm with initial solutions:", solution_names)

            else:

                # Get list of initial solutions
                initial_solutions = self.mdl_p["initial_solutions"]
                if printing:
                    print("Running Genetic Algorithm with", len(initial_solutions), "initial solutions...")

            # Get dictionary of failed constraints
            if self.mdl_p["pyomo_constraint_based"]:
                con_fail_dict = self.get_full_constraint_fail_dictionary(initial_solutions, printing=printing)
        else:

            if printing:
                print("Running Genetic Algorithm with no initial solutions (not advised!)...")
            initial_solutions = None

        # Generate the solution
        solution, time_eval_df = afccp.core.solutions.heuristic_solvers.genetic_algorithm(
            self, initial_solutions, con_fail_dict, printing=printing)

        # Determine what to do with the solution
        self.solution_handling(solution, solution_method="Genetic")

        # Return the final solution and maybe the time evaluation dataframe if needed
        if self.mdl_p["time_eval"]:
            return time_eval_df, solution
        else:
            return solution

    def solve_vft_pyomo_model(self, p_dict={}, printing=None):
        """
        Solve the VFT model using pyomo
        """
        self.error_checking("Pyomo Model")
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Build the model and then solve it
        model, q = afccp.core.solutions.pyomo_models.vft_model_build(self, printing=printing)
        start_time = time.perf_counter()  # Start the timer to solve the model
        solution, x, self.mdl_p['warm_start'] = afccp.core.solutions.pyomo_models.solve_pyomo_model(
            self, model, "VFT", q=q, printing=printing)
        solve_time = round(time.perf_counter() - start_time, 2)  # Stop the timer after model is solved

        # Determine what to do with the solution
        self.solution_handling(solution, solution_method="VFT", x=x)

        # Return the solution and potentially the time it took to solve the model
        if self.mdl_p["time_eval"]:
            return solution, solve_time
        else:
            return solution

    def solve_original_pyomo_model(self, p_dict={}, printing=None):
        """
        Solve the original AFPC model using pyomo
        """
        self.error_checking("Pyomo Model")
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Build the model and then solve it
        model = afccp.core.solutions.pyomo_models.original_model_build(self, printing=printing)
        start_time = time.perf_counter()  # Start the timer to solve the model
        solution, x, self.mdl_p['warm_start'] = afccp.core.solutions.pyomo_models.solve_pyomo_model(
            self, model, "Original", printing=printing)
        solve_time = round(time.perf_counter() - start_time, 2)  # Stop the timer after model is solved

        # Determine what to do with the solution
        self.solution_handling(solution, solution_method="OG", x=x)

        # Return the solution and potentially the time it took to solve the model
        if self.mdl_p["time_eval"]:
            return solution, solve_time
        else:
            return solution

    def solve_gp_pyomo_model(self, p_dict={}, printing=None):
        """
        Solve the Goal Programming Model (Created by Lt. Reynolds)
        """
        self.error_checking("Pyomo Model")
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Convert VFT parameters to Goal Programming ("gp") parameters
        if self.gp_parameters is None:
            self.vft_to_gp_parameters(self.mdl_p)

        # Build the model and then solve it
        model = afccp.core.solutions.pyomo_models.gp_model_build(self, printing=printing)
        start_time = time.perf_counter()  # Start the timer to solve the model
        solution, x = afccp.core.solutions.pyomo_models.solve_pyomo_model(self, model, "GP", printing=printing)
        solve_time = round(time.perf_counter() - start_time, 2)  # Stop the timer after model is solved

        # Determine what to do with the solution
        self.solution_handling(solution, solution_method="GP", x=x)

        # Return the solution and potentially the time it took to solve the model
        if self.mdl_p["time_eval"]:
            return solution, solve_time
        else:
            return solution

    def full_vft_model_solve(self, p_dict={}, printing=None):
        """
        This is the main method to solve the problem instance. If we choose to develop an initial population
        of solutions, we do so using the approximate VFT model. We then evolve the solutions further using the GA.
        """
        self.error_checking("Pyomo Model")
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Determine population for the genetic algorithm if necessary
        if self.mdl_p["populate"]:
            self.mdl_p["initial_solutions"] = afccp.core.comprehensive_functions.populate_initial_ga_solutions(
                self, printing)

            # Add additional solutions if necessary
            if self.mdl_p["solution_names"] is not None:

                # In case the user specifies "Solution" instead of ["Solution"]
                if type(self.mdl_p["solution_names"]) == str:
                    self.mdl_p["solution_names"] = [self.mdl_p["solution_names"]]

                # Add additional solutions
                for solution_name in self.mdl_p["solution_names"]:
                    solution = self.solution_dict[solution_name]
                    self.mdl_p["initial_solutions"] = np.vstack((self.mdl_p["initial_solutions"], solution))

            self.mdl_p["initialize"] = True

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

    def solve_for_constraints(self, p_dict={}):
        """
        This method iteratively adds constraints to the model to find which ones should be included based on
        feasibility and in order of importance
        """
        self.error_checking("Pyomo Model")

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # If no constraints are turned on right now...
        if np.sum(self.value_parameters["constraint_type"]) == 0:
            raise ValueError("No active constraints to search for, "
                             "make sure the current set of value parameters has active constraints.")

        # Run the function!
        constraint_type, solutions_df, report_df = afccp.core.solutions.pyomo_models.determine_model_constraints(self)

        # Build constraint type dataframe
        constraint_type_df = pd.DataFrame({'AFSC': self.parameters['afscs'][:self.parameters["M"]]})
        for k, objective in enumerate(self.value_parameters['objectives']):
            constraint_type_df[objective] = constraint_type[:, k]

        # Export to excel
        filepath = self.export_paths['Analysis & Results'] + self.data_name + " " + self.vp_name + \
                   " Constraint Report (" + self.data_version + ").xlsx"
        with pd.ExcelWriter(filepath) as writer:
            report_df.to_excel(writer, sheet_name="Report", index=False)
            constraint_type_df.to_excel(writer, sheet_name="Constraints", index=False)
            solutions_df.to_excel(writer, sheet_name="Solutions", index=False)

    # Solution Handling
    def find_ineligible_cadets(self, fix_it=True):
        """
        Prints out the ID's of ineligible pairs of cadets/AFSCs
        """
        if self.solution is None:
            raise ValueError("No solution activated.")

        # Loop through each cadet to see if they're ineligible for the AFSC they're assigned to
        total_ineligible = 0
        for i, j in enumerate(self.solution):
            cadet, afsc = self.parameters['cadets'][i], self.parameters['afscs'][j]

            # Cadet is not in the set of eligible cadets for this AFSC
            if i not in self.parameters['I^E'][j]:
                total_ineligible += 1

                # Do we do anything about it yet?
                if fix_it:
                    print('Cadet', cadet, 'assigned to AFSC', afsc, 'but is ineligible for it. Adjusting qual matrix to'
                                                                    ' allow this exception.')

                    # Add exception in different parameters
                    self.parameters['qual'][i, j] = "E" + str(self.parameters['t_count'][j])
                    self.parameters['ineligible'][i, j] = 0
                    self.parameters['eligible'][i, j] = 1

                else:
                    print('Cadet', cadet, 'assigned to AFSC', afsc, 'but is ineligible for it.')

        # Printing statement
        if total_ineligible == 0:
            print("No cadets assigned to AFSCs that they're ineligible for in the current solution.")
        else:
            print(total_ineligible, "total cadets assigned to AFSCs that they're ineligible for in the current solution.")

        # Adjust sets and subsets of cadets to reflect change
        if fix_it and total_ineligible > 0:
            self.parameters = afccp.core.data.processing.parameter_sets_additions(self.parameters)

    def set_instance_solution(self, solution_name=None, printing=None):
        """
        Set the current instance object's solution to a solution from the dictionary
        """
        if printing is None:
            printing = self.printing

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
                self.metrics = self.measure_solution(printing=printing, return_z=False)

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
                p_i = afccp.core.solutions.handling.compare_solutions(self.solution_dict[s_name], solution)
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
        coords = afccp.core.data.preferences.solution_similarity_coordinates(similarity_matrix)

        # Plot similarity
        chart = afccp.core.visualizations.instance_graphs.solution_similarity_graph(self, coords)

        if printing:
            chart.show()

    def get_full_constraint_fail_dictionary(self, solutions, printing=None):
        """
        Get a dictionary of failed constraints. This is used since the Approximate model initial solution must be
        rounded, and we may miss a few constraints by 1 cadet each. This allows the GA to not reject the solution
        initially. This is done for all solutions and we assume that they are all "feasible" (DM feasible)
        """
        if printing is None:
            printing = self.printing

        # Loop through each solution to build a "constraint fail dictionary" that works on all solutions
        con_fail_dict = {}
        for s, solution in enumerate(solutions):

            # Measure the solution and then loop through all AFSCs
            metrics = self.measure_solution(solution=solution, return_z=False, printing=False)
            for j in self.parameters['J']:
                afsc = self.parameters["afscs"][j]

                # Loop through all constrained AFSC objectives
                for k in self.value_parameters["K^C"][j]:
                    objective = self.value_parameters["objectives"][k]

                    # Determine if we need to update the AFSC objective in the con_fail_dict or not
                    if (j, k) in metrics['con_fail_dict']:  # Solution specific con_fail_dict
                        ineq = metrics['con_fail_dict'][(j, k)].split(' ')[0]
                        measure_new = float(metrics['con_fail_dict'][(j, k)].split(' ')[1])
                        if (j, k) in con_fail_dict:  # "Main" con_fail_dict for all solutions
                            measure_old = float(con_fail_dict[(j, k)].split(' ')[1])

                            # Check "ineq_2" to see if somehow we failed the constraint on the other side
                            # (need to address this later)
                            ineq_2 = con_fail_dict[(j, k)].split(' ')[0]
                            if ineq != ineq_2:
                                print(
                                    "Warning. I haven't yet adjusted the con_fail_dict to work with both sides of the "
                                    "constraint. AFSC '" + afsc + "' objective '" + objective +
                                    "' has a current con_fail_dict value of '" + con_fail_dict[(j, k)] +
                                    "'. The new con_fail_dict value in solution '" + str(s) + "' is '" +
                                    metrics['con_fail_dict'][(j, k)] + "'. We need to reconcile this, but for now we'll "
                                                                       "take the latter version.")

                                # Update the dictionary
                                con_fail_dict[(j, k)] = copy.deepcopy(metrics['con_fail_dict'][(j, k)])
                                continue

                            # Increase the "ceiling" on this constraint for this AFSC objective
                            if ineq == "<" and measure_new > measure_old:

                                # Print statements
                                if printing:
                                    print("AFSC '" + afsc + "' Objective '" + objective +
                                          "' constraint maximum increased from " + str(measure_old) + " to " +
                                          str(measure_new) + ".")

                                # Update the dictionary
                                con_fail_dict[(j, k)] = "< " + str(round(measure_new, 2))

                            # Decrease the "floor" on this constraint for this AFSC objective
                            elif ineq == ">" and measure_new < measure_old:

                                # Print statements
                                if printing:
                                    print("AFSC '" + afsc + "' Objective '" + objective +
                                          "' constraint minimum decreased from " + str(measure_old) + " to " +
                                          str(measure_new) + ".")

                                # Update the dictionary
                                con_fail_dict[(j, k)] = "> " + str(round(measure_new, 2))

                        # This is a new AFSC objective to put in our con_fail_dict dictionary
                        else:
                            # Print statements
                            if printing:
                                print("AFSC '" + afsc + "' Objective '" + objective +
                                      "' current constraint:", metrics['con_fail_dict'][(j, k)])

                            # Update the dictionary
                            con_fail_dict[(j, k)] = copy.deepcopy(metrics['con_fail_dict'][(j, k)])

        return con_fail_dict

    def measure_solution(self, solution=None, value_parameters=None, approximate=False, matrix=False, printing=None,
                         return_z=True):
        """
        Evaluate a solution using VFT objective hierarchy
        """
        # Error checking, solution setting
        if solution is None:
            if matrix:
                solution = self.x
            else:
                solution = self.solution
        if value_parameters is None:
            value_parameters = self.value_parameters
        if solution is None or value_parameters is None:
            raise ValueError("Error. Solution and value parameters needed to evaluate solution.")

        # Print statement
        if printing is None:
            printing = self.printing

        # Calculate solution metrics
        metrics = afccp.core.solutions.handling.evaluate_solution(
            solution, self.parameters, value_parameters, approximate=approximate, printing=printing)

        if return_z:
            return round(metrics['z'], 4)
        else:
            return metrics

    def measure_fitness(self, solution=None, value_parameters=None, metrics=None, printing=None):
        """
        This is the fitness function method (could be slightly different depending on how the constraints are handled)
        :return: fitness score
        """
        # Error checking, solution setting
        if solution is None:
            solution = self.solution
        if value_parameters is None:
            value_parameters = self.value_parameters
        if solution is None or value_parameters is None:
            raise ValueError("Error. Solution and value parameters needed to evaluate solution.")

        # Printing statement
        if printing is None:
            printing = self.printing

        # Get the metrics
        if metrics is None:
            metrics = self.measure_solution(solution, value_parameters, return_z=False, printing=False)

        # Calculate fitness value
        z = afccp.core.solutions.handling.fitness_function(solution, self.parameters, value_parameters, self.mdl_p,
                                                           con_fail_dict=metrics['con_fail_dict'])

        # Print and return fitness value
        if printing:
            print("Fitness value calculated to be", round(z, 4))
        return z

    def update_metrics_dict(self):
        """
        Updates metrics dictionary from solutions and vp dictionaries
        """

        # We only update the dictionaries if we have at least one set of value parameters and one solution
        if self.vp_dict is not None and self.solution_dict is not None:
            if self.metrics_dict is None:
                self.metrics_dict = {}

            # Loop through each set of value parameters
            for vp_name in self.vp_dict:
                value_parameters = self.vp_dict[vp_name]
                if vp_name not in self.metrics_dict:
                    self.metrics_dict[vp_name] = {}

                # Loop through each solution
                for solution_name in self.solution_dict:
                    solution = self.solution_dict[solution_name]
                    if solution_name not in self.metrics_dict[vp_name]:
                        metrics = self.measure_solution(solution, value_parameters, return_z=False)
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

            if p_dict["use_useful_charts"]:

                # Loop through the subset of charts that I actually care about
                for obj, version in p_dict["desired_charts"]:

                    # Loop through all solutions and display their charts
                    for solution_name in p_dict["solution_names"]:
                        self.set_instance_solution(solution_name, printing=False)

                        if printing:
                            print("<Objective '" + obj + "' version '" + version + " solution" + solution_name + "'>")
                        p_dict["objective"] = obj
                        p_dict["version"] = version

                        try:
                            charts.append(self.display_results_graph(p_dict))
                        except:
                            pass

            else:

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

            if self.solution is None:
                raise ValueError("Error, no solution detected.")

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

        # Reset chart functional parameters
        self.plt_p, _ = \
            afccp.core.data.ccp_helping_functions.initialize_instance_functional_parameters(self.parameters["N"])

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
        self.plt_p = afccp.core.data.ccp_helping_functions.determine_afsc_plot_details(self, results_chart=True)

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
                        afccp.core.data.ccp_helping_functions.pick_most_changed_afscs(self)

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
            afccp.core.data.ccp_helping_functions.initialize_instance_functional_parameters(self.parameters["N"])

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
        self.error_checking("Pyomo Model")
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        if printing:
            print("Conducting pareto analysis on problem...")

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
            afsc_solution = np.array([self.parameters["afscs"][int(j)] for j in solution])
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

        # File we import and export to
        filepath = self.export_paths['Analysis & Results'] + self.data_name + " " + self.vp_name + " (" + \
                   self.data_version + ") Pareto Analysis.xlsx"
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
        filepath = self.export_paths['Analysis & Results'] + self.data_name + " " + self.vp_name + " (" + \
                   self.data_version + ") Pareto Analysis.xlsx"

        iself.error_checking("Pyomo Model")
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        if printing:
            print("Conducting 'final' pareto analysis on problem...")

        # Solutions Dataframe
        solutions_df = afccp.core.globals.import_data(filepath, sheet_name='Initial Solutions')
        approximate_results_df = afccp.core.globals.import_data(filepath, sheet_name='Approximate Pareto Results')

        # Load in the initial solutions
        initial_afsc_solutions = np.array([np.array(solutions_df[col]) for col in solutions_df])
        initial_solutions = np.array([])
        for i, afsc_solution in enumerate(initial_afsc_solutions):
            solution = np.array([np.where(self.parameters["afscs"] == afsc)[0][0] for afsc in afsc_solution])
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
            afsc_solution = np.array([self.parameters["afscs"][int(j)] for j in solution])
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
            afsc = self.parameters['afscs'][0]

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
    def export_data(self, datasets=None, printing=None):
        """
        Exports the desired problem instance datasets back to csvs.

        Parameters:
        datasets (list of str): List of datasets to export. By default, the method exports the "Value Parameters" and
            "Solutions" datasets, as these are the ones that are likely to change the most. Other possible datasets are
            "AFSCs", "Cadets", "Preferences", and "Goal Programming".
        printing (bool): Whether to print status updates or not. If not specified, the value of `self.printing` will
            be used.

        Returns:
        None

        This method exports the specified datasets using the `export_<dataset_name>_data` functions from the
        `afccp.core.data.processing` module. The exported csvs are saved in the paths specified in the
        `self.export_paths` dictionary.

        If the `self.import_paths` and `self.export_paths` attributes are not set, this method will call the
        `afccp_dp.initialize_file_information` function to create the necessary directories and set the paths.

        If the "Goal Programming" dataset is included in the `datasets` parameter and the `gp_df` attribute is not None,
        the method exports the `gp_df` dataframe to a csv using the file path specified in the `export_paths` dictionary.
        """

        # Parameter initialization
        if datasets is None:
            datasets = ["Cadets", "AFSCs", "Preferences", "Goal Programming", "Value Parameters", "Solutions"]
        if printing is None:
            printing = self.printing

        # Print statement
        if printing:
            print("Exporting datasets", datasets)

        # Shorten module name
        afccp_dp = afccp.core.data.processing

        # Check to make sure we have file data information
        for attribute in [self.import_paths, self.export_paths]:

            # If we don't have this information, that means this is a new instance to export
            if attribute is None:
                self.import_paths, self.export_paths = afccp_dp.initialize_file_information(self.data_name,
                                                                                            self.data_version)
                break

        # Export various data using the different functions
        dataset_function_dict = {"AFSCs": afccp_dp.export_afscs_data,
                                 "Cadets": afccp_dp.export_cadets_data,
                                 "Preferences": afccp_dp.export_preferences_data,
                                 "Value Parameters": afccp_dp.export_value_parameters_data,
                                 "Solutions": afccp_dp.export_solutions_data}
        for dataset in dataset_function_dict:
            if dataset in datasets:
                dataset_function_dict[dataset](self)

        # Goal Programming dataframe is an easy export (dataframe is already constructed)
        if "Goal Programming" in datasets and self.gp_df is not None:
            self.gp_df.to_csv(self.export_paths["Goal Programming"], index=False)

