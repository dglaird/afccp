# Import libraries
from typing import Any
import os
import pandas as pd
import numpy as np
import math
import datetime
import glob
import copy
import time

# afccp modules
import afccp.core.globals
import afccp.core.data.adjustments
import afccp.core.data.generation
import afccp.core.data.preferences
import afccp.core.data.processing
import afccp.core.data.support
import afccp.core.data.values
import afccp.core.solutions.algorithms
import afccp.core.solutions.handling
import afccp.core.visualizations.bubbles
import afccp.core.visualizations.charts

# Import optimization models if pyomo is installed
if afccp.core.globals.use_pyomo:
    import afccp.core.solutions.optimization
    import afccp.core.solutions.sensitivity

# Import slides script if python-pptx installed
if afccp.core.globals.use_pptx:
    import afccp.core.visualizations.slides

# Main Problem Class
class CadetCareerProblem:
    def __init__(self, data_name="Random", data_version="Default", degree_qual_type="Consistent",
                 num_value_function_breakpoints=None, N=1600, M=32, P=6, S=10, generate_extra_components=False,
                 generate_only_nrl=False, ctgan_model_name='CTGAN_2024', ctgan_pilot_sampling=False, printing=True):
        """
        Represents the AFSC/Cadet matching problem object.

        Parameters:
            data_name (str): The name of the data set. It can be an existing instance folder name or one of the following:
                             "Random", "Perfect", or "Realistic". Defaults to "Random".
            data_version (str): The version of the data set. It is used to manage different versions of cadets/AFSCs
                                and handle NRL vs All cadet scenarios. Defaults to "Default".
            degree_qual_type (str): The type of degree tier qualifications. It can be "Binary", "Relaxed", or "Tiers".
                                    Defaults to "Consistent".
            num_value_function_breakpoints (int, optional): The number of breakpoints to use in the value functions.
                                                           If None, the value functions will be loaded with the existing
                                                           breakpoints. If an integer is provided, new value functions
                                                           will be created. Defaults to None.
            N (int): Number of cadets to generate. Defaults to 1600.
            M (int): Number of AFSCs to generate. Defaults to 32.
            P (int): Number of AFSC preferences to generate for each cadet. Defaults to 6.
            S (int): Number of Bases to generate. Defaults to 10.
            generate_only_nrl (bool): Whether to generate only NRL AFSCs. Defaults to False.
            generate_extra_components (bool): Whether to generate extra components (bases/IST). Defaults to False.
            ctgan_model_name (str): Name of the CTGAN model to import
            ctgan_pilot_sampling (bool): Whether we should sample cadets in CTGAN based on the pilot preference condition
            printing (bool): Whether to print status updates or not. Defaults to True.

        This class represents the AFSC/Cadet problem object. It can import existing data or generate new instances.
        The data set can be specified by providing the name of the instance folder or using the predefined data types
        ("Random", "CTGAN"). The problem instance includes information about cadets, AFSCs, value parameters,
        and solutions.

        Example usage:
            instance = CadetCareerProblem(data_name="Random", N=200, M=5, P=5)
        """

        # Shorten the module name so everything fits better
        afccp_dp = afccp.core.data.processing

        # Data attributes
        self.data_version = data_version  # Version of instance (in parentheses of the instance sub-folders)
        self.import_paths, self.export_paths = None, None  # We initialize these attributes to 'None'
        self.printing = printing

        # The data variant helps inform how the charts should be constructed
        if len(data_name) == 1:  # "A", "B", "C", ...
            self.data_variant = "Scrubbed"
        elif len(data_name) == 4:  # "2016", "2017", "2018", ...
            self.data_variant = "Year"
        else:  # "Random_1", "Random_2", ...
            self.data_variant = "Generated"

        # Additional instance components (value parameters and solutions)
        self.value_parameters, self.vp_name = None, None  # Set of weight and value parameters (and the name)
        self.vp_dict = None  # Dictionary of value_parameters (set of sets)
        self.solution, self.solution_name = None, None  # Dictionary of solution elements (and the name)
        self.solutions = None  # Dictionary of solutions and their main attributes (x, j_array, afsc_array, etc.)

        # Parameters from *former* Lt Rebecca Reynold's thesis
        self.gp_parameters, self.gp_df = None, None

        # Update instances available (for importing)
        afccp.core.globals.instances_available = []
        for other_data_name in os.listdir(afccp.core.globals.paths["instances"]):
            if os.path.isdir(afccp.core.globals.paths["instances"] + other_data_name):
                afccp.core.globals.instances_available.append(other_data_name)

        # If we have an instance folder already for the specified instance (we're importing it)
        if data_name in afccp.core.globals.instances_available:

            # Gather information about the files we're importing and eventually exporting
            self.data_name = data_name
            self.import_paths, self.export_paths = afccp_dp.initialize_file_information(self.data_name,
                                                                                        self.data_version)

            # Print statement
            if self.printing:
                print("Importing '" + data_name + "' instance...")

            # Initialize dictionary of instance parameters (Information pertaining to cadets and AFSCs)
            self.parameters = {"Qual Type": degree_qual_type}

            # Import the "fixed" parameters (the information about cadets/AFSCs that, for the most part, doesn't change)
            import_data_functions = [afccp_dp.import_afscs_data, afccp_dp.import_cadets_data,
                                     afccp_dp.import_afsc_cadet_matrices_data, afccp_dp.import_additional_data]
            for import_function in import_data_functions:  # Here we're looping through a list of functions!
                self.parameters = import_function(self.import_paths, self.parameters)  # Python is nice like that...

            # Additional sets and subsets of cadets/AFSCs need to be loaded into the instance parameters
            self.parameters = afccp.core.data.adjustments.parameter_sets_additions(self.parameters)

            # Import the "Goal Programming" dataframe (from Lt Rebecca Reynold's thesis)
            if "Goal Programming" in self.import_paths:
                self.gp_df = afccp.core.globals.import_csv_data(self.import_paths["Goal Programming"])
            else:  # Default "GP" file
                self.gp_df = afccp.core.globals.import_data(
                    afccp.core.globals.paths["files"] + "gp_parameters.xlsx")

            # Import the "Value Parameters" data dictionary
            self.vp_dict = afccp_dp.import_value_parameters_data(self.import_paths, self.parameters,
                                                                 num_value_function_breakpoints)

            # Import the "Solutions" data dictionary
            self.solutions = afccp_dp.import_solutions_data(self.import_paths, self.parameters)

        # This is a new problem instance that we're generating (Should be "Random", "Perfect", or "Realistic")
        else:

            # Error Handling (Must be valid data generation parameter)
            if data_name not in ["Random", "CTGAN"]:
                raise ValueError(
                    "Error. Instance name '" + data_name + "' is not a valid instance name. Instances must "
                                                           "be either generated or imported. "
                                                           "(Instance not found in folder).")

            # Determine the name of this instance (Random_1, Random_2, etc.)
            for data_variant in ["Random", "CTGAN"]:
                if data_name == data_variant:

                    # Determine name of data based on what instances are already in our "instances" folder
                    variant_counter = 1
                    name_determined = False
                    while not name_determined:
                        check_name = data_variant + "_" + str(variant_counter)
                        if check_name not in afccp.core.globals.instances_available:
                            self.data_name = check_name
                            name_determined = True
                        else:
                            variant_counter += 1

            # Print statement
            if self.printing:
                print("Generating '" + self.data_name + "' instance...")

            # Want to generate a "valid" problem instance that meets the conditions below
            invalid = True
            while invalid:
                invalid = False  # "Innocent until proven guilty"

                # Generate a "Random" problem instance
                if data_name == 'Random':
                    self.parameters = afccp.core.data.generation.generate_random_instance(
                        N, M, P, S, generate_only_nrl=generate_only_nrl, generate_extra=generate_extra_components)

                    # Every cadet needs to be eligible for at least one AFSC
                    for i in range(self.parameters['N']):
                        if np.sum(self.parameters['eligible'][i, :]) == 0:
                            invalid = True  # Guilty!
                            break

                    # Every AFSC needs to have at least one cadet eligible for it
                    for j in range(self.parameters['M']):
                        if np.sum(self.parameters['eligible'][:, j]) == 0:
                            invalid = True  # Guilty!
                            break

                # Generate a "CTGAN" problem instance
                elif data_name == "CTGAN":
                    self.parameters = afccp.core.data.generation.generate_ctgan_instance(
                        N, name=ctgan_model_name, pilot_condition=ctgan_pilot_sampling)

                # We don't have that type of data available to generate
                else:
                    raise ValueError("Data type '" + data_name + "' currently not a valid method of generating"
                                                                 " data.")

            # Additional sets and subsets of cadets/AFSCs need to be loaded into the instance parameters
            self.parameters = afccp.core.data.adjustments.parameter_sets_additions(self.parameters)

        # Initialize more "functional" parameters
        self.mdl_p = afccp.core.data.support.initialize_instance_functional_parameters(
            self.parameters["N"])

        # Copy weight on GUO solution (relative to CASTLE) from "mdl_p" to "parameters"
        self.parameters['w^G'] = self.mdl_p['w^G']

        if self.printing:
            print("Instance '" + self.data_name + "' initialized.")

    # Method helper functions
    def reset_functional_parameters(self, p_dict={}):
        """
        Resets the instance functional parameters and updates them with the new values from p_dict.

        Parameters:
            p_dict (dict, optional): A dictionary containing the new parameter values to update. Defaults to an empty dict.

        This method resets the instance functional parameters of the CadetCareerProblem object to their default values.
        It then updates the parameters with new values specified in p_dict, if provided.

        The function first initializes the default functional parameters using the `initialize_instance_functional_parameters`
        function. It then iterates over the keys in p_dict and updates the corresponding parameters in self.mdl_p.

        Note that the update is performed in-place, modifying the self.mdl_p dictionary.

        If a parameter specified in p_dict does not exist in self.mdl_p, a warning message is printed.

        Example usage:
            instance = CadetCareerProblem()
            instance.reset_functional_parameters({'bar_color': 'blue', 'title_size': 18})
        """

        # Reset plot parameters and model parameters
        self.mdl_p = afccp.core.data.support.initialize_instance_functional_parameters(self.parameters["N"])

        # Update plot parameters and model parameters
        for key in p_dict:

            if key in self.mdl_p:
                self.mdl_p[key] = p_dict[key]
            else:
                print("WARNING. Specified parameter '" + str(key) + "' does not exist.")

    def solution_handling(self, solution, printing=None):
        """
        Determines what to do with the generated solution. This is a "helper method" and not intended to be called
        directly by the user.

        Parameters:
            solution (dict): The solution dictionary containing the solution elements.
            printing (str): Whether the code should print status updates or not

        This method takes a solution dictionary as input and performs the following tasks:
        - Evaluates the solution by calculating additional components.
        - Adjusts the solution method for VFT models.
        - Initializes the solutions dictionary if necessary.
        - Determines a unique name for the solution.
        - Checks if the solution is already in the solutions dictionary.
        - Adds the new solution to the solutions dictionary if it's a new solution.
        - Updates the solution name and assigns it to the instance.
        """

        if printing is None:
            printing = self.printing

        # Set the solution attribute to the instance (and calculate additional components)
        self.solution = afccp.core.solutions.handling.evaluate_solution(
            solution, self.parameters, self.value_parameters, re_calculate_x=self.mdl_p['re-calculate x'],
            printing=printing)
        self.solution['vp_name'] = self.vp_name  # Name of the value parameters currently evaluating this solution

        # Adjust solution method for VFT models
        if self.solution['method'] == "VFT" and self.mdl_p["approximate"]:
            self.solution['method'] = "A-VFT"
        elif self.solution['method'] == "VFT" and not self.mdl_p["approximate"]:
            self.solution['method'] = "E-VFT"

        # Initialize solutions dictionary if necessary
        if self.solutions is None:
            self.solutions = {}

        # Determine solution name
        if self.solution['method'] not in self.solutions:
            solution_name = self.solution['method']
        else:
            count = 2
            solution_name = self.solution['method'] + '_' + str(count)
            while solution_name in self.solutions:
                count += 1
                solution_name = self.solution['method'] + '_' + str(count)

        # Check if this solution is a new solution
        new = True
        for s_name in self.solutions:
            p_i = afccp.core.solutions.handling.compare_solutions(self.solutions[s_name]['j_array'],
                                                                  self.solution['j_array'])

            # Set the name of this solution to be the name of its equivalent that is already in the dictionary
            if p_i == 1:

                # If this is an array of unmatched cadets, we count it as unique since it's likely a SOC algorithm
                if len(np.unique(self.solution['j_array'])) == 1 and self.parameters['M'] in self.solution['j_array']:
                    pass

                else:
                    new = False
                    self.solution_name = s_name
                    self.solution['name'] = s_name
                    break

        # If it is new, we add it to the dictionary and adjust the name of the activated instance solution
        if new:
            self.solutions[solution_name] = copy.deepcopy(self.solution)
            self.solution_name = solution_name
            self.solution['name'] = solution_name

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
                                     "need to do is run 'instance.set_value_parameters()'. ")

        if test == "Value Parameters":
            check_value_parameters()
        elif test == "Pyomo Model":
            check_value_parameters()

            # We don't have Pyomo
            if not afccp.core.globals.use_pyomo:
                raise ValueError("Error. Pyomo is not currently installed and is required to run pyomo models. Please"
                                 "install this library.")

            # No Base/Training components
            if self.mdl_p['solve_extra_components'] and 'B' not in self.parameters:
                raise ValueError("Error. No base/training components found in parameters. Cannot solve "
                                 "without the extra components.")

        elif test == 'Solutions':
            if self.solutions is None:
                raise ValueError("Error. No solutions dictionary detected. You need to solve this problem first.")
            else:
                check_value_parameters()
        elif test == 'Solution':
            if self.solution is None:
                if self.solutions is None:
                    raise ValueError("Error. No solutions dictionary detected. You need to solve this problem first.")
                else:
                    raise ValueError("Error. Solutions dictionary detected but you haven't actually initialized a "
                                     "solution yet. You can do so by running 'instance.set_solution()'.")
            else:
                check_value_parameters()

    def manage_solution_folder(self):
        """
        This method creates a solution folder in the Analysis & Results directory if it doesn't already exist and
        returns a list of files in that folder.

        Returns:
            List of filenames in the solution folder.

        This method is used to manage solution folders for organizing analysis and result files.
        """

        # Make solution folder if it doesn't already exist
        folder_path = self.export_paths['Analysis & Results'] + self.solution_name
        if self.solution_name not in os.listdir(self.export_paths['Analysis & Results']):
            os.mkdir(folder_path)

        # Return list of files in the solution folder
        return os.listdir(folder_path)

    def evaluate_all_solutions(self, solution_names=None):
        """
        Evaluates all the solutions in the dictionary to acquire necessary metrics
        """

        if solution_names is None:
            solution_names = list(self.solutions.keys())

        if self.printing:
            print('Evaluating solutions in this list:', solution_names)
        for solution_name in solution_names:
            self.solutions[solution_name] = afccp.core.solutions.handling.evaluate_solution(
                self.solutions[solution_name], self.parameters, self.value_parameters, printing=False,
                re_calculate_x=self.mdl_p['re-calculate x'])

    # Adjust Data
    def calculate_qualification_matrix(self, printing=None):
        """
        This procedure re-runs the CIP to Qual function to generate or update the qualification matrix.
        The qualification matrix determines whether cadets are eligible for specific AFSCs.

        Args:
        printing (bool, optional): If True, print messages about the process. Defaults to the class's `printing` attribute.

        Raises:
        ValueError: Raised when there are no CIP codes provided.

        This method recalculates the qualification matrix based on CIP (Career Intermission Program) codes and
        AFSCs. It updates the matrix and related parameters within the class.
        """
        if printing is None:
            printing = self.printing

        if printing:
            print('Adjusting qualification matrix...')
        parameters = copy.deepcopy(self.parameters)

        # Generate new matrix
        if "cip1" in parameters:
            if "cip2" in parameters:
                qual_matrix = afccp.core.data.support.cip_to_qual_tiers(
                    parameters["afscs"][:parameters["M"]], parameters['cip1'], cip2=parameters['cip2'])
            else:
                qual_matrix = afccp.core.data.support.cip_to_qual_tiers(
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
        parameters = afccp.core.data.adjustments.parameter_sets_additions(parameters)
        self.parameters = copy.deepcopy(parameters)

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

        return afccp.core.data.adjustments.convert_instance_to_from_scrubbed(self, new_letter, translation_dict=None)

    def convert_back_to_real_instance(self, translation_dict, data_name, printing=None):
        """
        This method scrubs the AFSC names by sorting them by their PGL targets and creates a translated problem instance
        :param printing: If we should print status update
        :param translation_dict: Dictionary generated from the scrubbed instance method
        :param data_name: Data Name of the new instance
        """
        if printing is None:
            printing = self.printing

        if printing:
            print("Converting problem instance '" + self.data_name + "' back to instance '" + data_name + "'...")

        return afccp.core.data.adjustments.convert_instance_to_from_scrubbed(
            self, translation_dict=translation_dict, data_name=data_name)

    def fix_generated_data(self, printing=None):
        """
        NOTE: ONLY DO THIS FOR GENERATED DATA
        """

        if printing is None:
            printing = self.printing

        # Get "c_pref_matrix" from cadet preferences
        self.convert_utilities_to_preferences(cadets_as_well=True)

        # Generate AFSC preferences for this problem instance
        self.generate_fake_afsc_preferences()

        # Generate rated data (Separate datasets for each SOC)
        self.generate_rated_data()

        # Update qualification matrix from AFSC preferences (treating CFM lists as "gospel" except for Rated/USSF)
        self.update_qualification_matrix_from_afsc_preferences()

        # Removes ineligible cadets from all 3 matrices: degree qualifications, cadet preferences, AFSC preferences
        self.remove_ineligible_choices(printing=printing)

        # Take the preferences dictionaries and update the matrices from them (using cadet/AFSC indices)
        self.update_preference_matrices()  # 1, 2, 4, 6, 7 -> 1, 2, 3, 4, 5 (preference lists need to omit gaps)

        # Convert AFSC preferences to percentiles (0 to 1)
        self.convert_afsc_preferences_to_percentiles()  # 1, 2, 3, 4, 5 -> 1, 0.8, 0.6, 0.4, 0.2

        # The "cadet columns" are located in Cadets.csv and contain the utilities/preferences in order of preference
        self.update_cadet_columns_from_matrices()  # We haven't touched "c_preferences" and "c_utilities" until now

        # Generate fake (random) set of value parameters
        self.generate_random_value_parameters()

        # Sanity check the parameters to make sure it all looks good! (No issues found.)
        self.parameter_sanity_check()

    # Castle adjustments
    def generate_example_castle_value_curves(self, num_breakpoints: int=10):

        # Create "q" dictionary containing breakpoint information for castle value curves
        q = afccp.core.data.values.generate_realistic_castle_value_curves(self.parameters,
                                                                          num_breakpoints=num_breakpoints)

        # Save "q" dictionary for castle to parameters
        self.parameters['castle_q'] = q

    # Adjust Preferences
    def update_qualification_matrix_from_afsc_preferences(self, printing=None):
        """
        This method updates the qualification matrix to reflect cadet eligibility for AFSCs based on their preferences.

        It performs the following steps:
        1. Checks if there is an AFSC preference matrix ('a_pref_matrix') in the parameters. If not, it raises a ValueError.
        2. Iterates through each AFSC ('afscs') in the parameters.
        3. Determines cadet eligibility and ineligibility for each AFSC based on both AFSC preferences and degree qualifications.
        4. If cadet eligibility differs between preference and qualification lists, it prints a message indicating the mismatch.
        5. For Rated or USSF AFSCs, it updates the qualification matrix, making more cadets ineligible based on CFM lists.
        6. For NRL AFSCs, it handles cadets eligible based on CFM lists but not the AFOCD by giving them exceptions.
        7. For NRL AFSCs, it also handles cadets eligible based on the AFOCD but not the CFM lists by marking them as a warning.
        8. Updates the qualification matrix with these changes and updates additional sets and subsets in the parameters.

        This method helps ensure that the qualification matrix aligns with cadet preferences and the AFOCD.

        Args:
            self: The class instance containing the qualification matrix and parameters.

        Returns:
            None

        Raises:
            ValueError: If there is no AFSC preference matrix ('a_pref_matrix') in the parameters.
            :param printing: print status updates

        """

        if printing is None:
            printing = self.printing

        # Shorthand
        p = self.parameters

        if 'a_pref_matrix' not in p:
            raise ValueError("No AFSC preference matrix.")

        # Loop through each AFSC
        for j, afsc in enumerate(p['afscs'][:p['M']]):

            # Eligible & Ineligible cadets based on the CFM preference lists
            preference_eligible_cadets = np.where(p['a_pref_matrix'][:, j] > 0)[0]
            preference_ineligible_cadets = np.where(p['a_pref_matrix'][:, j] == 0)[0]

            # Eligible cadets based on their degree qualifications (and later using exceptions)
            qual_eligible_cadets = np.where(p['eligible'][:, j])[0]

            # There's a difference between the two
            if len(preference_eligible_cadets) != len(qual_eligible_cadets) and printing:
                print(j, "AFSC '" + afsc + "' has", len(preference_eligible_cadets),
                      "eligible cadets according to the AFSC preference matrix but",
                      len(qual_eligible_cadets), "according to the qual matrix.")

            # If this is a Rated or USSF AFSC, we have to make more cadets ineligible based on CFM lists
            if p['acc_grp'][j] in ['Rated', 'USSF']:

                # Only do this if we have ineligible cadets here
                if len(preference_ineligible_cadets) > 0:

                    # If there's already an ineligible tier in this AFSC, we use it
                    if "I = 0" in p['Deg Tiers'][j]:
                        val = "I" + str(p['t_count'][j])
                    else:
                        val = "I" + str(p['t_count'][j] + 1)
                        if printing:
                            print(j, "AFSC '" + afsc + "' doesn't have an ineligible tier in the Deg Tiers section"
                                                    " of the AFSCs.csv file. Please add one.")

                    # Update qual matrix if needed
                    if len(preference_ineligible_cadets) > 0 and printing:
                        print(j, "Making", len(preference_ineligible_cadets), "cadets ineligible for '" + afsc +
                              "' by altering their qualification to '" + val + "'. ")
                        self.parameters['qual'][preference_ineligible_cadets, j] = val

            # NRL AFSC
            else:

                # Cadets that are "eligible" for the AFSC based on the CFM lists but not the AFOCD
                exception_cadets = np.array([i for i in preference_eligible_cadets if i not in qual_eligible_cadets])

                # Only do this if we have exception cadets here
                if len(exception_cadets) > 0:

                    # If there's an ineligible tier, we use this tier for the cadets with the exception
                    if "I = 0" in p['Deg Tiers'][j]:
                        val = "E" + str(p['t_count'][j])
                    else:
                        val = "E" + str(p['t_count'][j] + 1)

                    # Update qual matrix
                    if printing:
                        print(j, "Giving", len(exception_cadets), "cadets an exception for '" + afsc +
                              "' by altering their qualification to '" + val + "'. ")
                    self.parameters['qual'][exception_cadets, j] = val

                # Cadets that are eligible for the AFSC based on the AFOCD but not the CFM lists
                mistake_cadets = np.array([i for i in qual_eligible_cadets if i not in preference_eligible_cadets])

                if len(mistake_cadets) > 0 and printing:
                    print(j, 'WARNING. There are', len(mistake_cadets), 'cadets that are eligible for AFSC', afsc,
                          ' according to the AFOCD but not the CFM lists. These are the cadets at indices',
                          mistake_cadets)

        # Update qual stuff
        self.parameters["ineligible"] = (np.core.defchararray.find(self.parameters['qual'], "I") != -1) * 1
        self.parameters["eligible"] = (self.parameters["ineligible"] == 0) * 1
        self.parameters["exception"] = (np.core.defchararray.find(self.parameters['qual'], "E") != -1) * 1

        # Update the additional sets and subsets for the parameters
        self.parameters = afccp.core.data.adjustments.parameter_sets_additions(self.parameters)

    def convert_utilities_to_preferences(self, cadets_as_well=False):
        """
        Convert Utility Matrices to Preference Matrices.

        This method converts utility matrices into preference matrices for AFSC assignments. It provides the option to
        convert cadet utility matrices and, if necessary, cadet rankings as well.

        Parameters:
        - cadets_as_well (bool, optional): If True, both cadet and AFSC utility matrices are converted to preferences.
          If False (default), only AFSC utility matrices are converted.

        Description:
        Utility matrices contain numerical values that represent the desirability or quality of a cadet's assignment
        to a particular AFSC. Converting these utility values into preferences is essential for the assignment process.
        Preferences are often represented as rankings or ordered lists, with higher values indicating higher preferences.

        This method ensures that both cadet and AFSC utility matrices are properly transformed into preferences, making
        them suitable for use in the assignment algorithm.
        """

        # Rest of your method code here

        self.parameters = afccp.core.data.preferences.convert_utility_matrices_preferences(self.parameters,
                                                                                                 cadets_as_well)
        self.parameters = afccp.core.data.adjustments.parameter_sets_additions(self.parameters)

    def generate_fake_afsc_preferences(self, fix_cadet_eligibility=False):
        """
        Generate Simulated AFSC Preferences using Value Focussed Thinking (VFT) Parameters.

        This method generates simulated AFSC preferences for cadets based on the VFT parameters.
        These preferences are useful for testing and analysis purposes and can be used as inputs to the assignment algorithm.

        Parameters:
        - fix_cadet_eligibility (bool, optional): If True, it fixes cadet eligibility based on VFT parameters before
        generating preferences. If False (default), preferences are generated without modifying eligibility. Use this
        option to create preferences for a specific scenario where cadet eligibility should be controlled.

        Description:
        Simulated preferences are created by modeling cadet choices using the VFT parameters.
        These preferences are essential for conducting experiments and evaluating the performance of the assignment
        algorithm under various conditions.

        This method allows you to generate preferences for a specific scenario by controlling cadet eligibility or
        generate preferences without modifying eligibility, providing flexibility for testing and analysis.
        """
        self.parameters = afccp.core.data.preferences.generate_fake_afsc_preferences(
            self.parameters, self.value_parameters, fix_cadet_eligibility=fix_cadet_eligibility)
        self.parameters = afccp.core.data.adjustments.parameter_sets_additions(self.parameters)

    def convert_afsc_preferences_to_percentiles(self):
        """
        Convert AFSC Preference Lists to Normalized Percentiles.

        This method takes the AFSC preference lists (a_pref_matrix) for each cadet and converts them into normalized percentiles
        based on the provided preferences. The resulting percentiles represent how each cadet ranks AFSCs compared to their peers.

        The percentiles are stored in the 'afsc_utility' on AFSCs Utility.csv. This conversion can be helpful for analyzing cadet
        preferences and running assignment algorithms.
        """

        if self.printing:
            print("Converting AFSC preferences (a_pref_matrix) into percentiles (afsc_utility on AFSCs Utility.csv)...")
        self.parameters = afccp.core.data.preferences.convert_afsc_preferences_to_percentiles(self.parameters)
        self.parameters = afccp.core.data.adjustments.parameter_sets_additions(self.parameters)

    def remove_ineligible_choices(self, printing=None):
        """
        Remove Ineligible Choices Based on Qualification.

        This method utilizes the qualification matrix (qual) to remove ineligible choices from both AFSC preferences (a_pref_matrix)
        and cadet preferences (c_pref_matrix). Ineligible choices are determined based on the qualification requirements, and
        this process helps ensure that the final assignments meet the qualification criteria.

        Parameters:
        - printing (bool, optional): If True, print progress and debug information. If None (default), it uses the class-level 'printing' attribute.
        """

        if printing is None:
            printing = self.printing

        if printing:
            print("Removing ineligible cadets based on any of the three eligibility sources "
                  "(c_pref_matrix, a_pref_matrix, qual)...")

        self.parameters = afccp.core.data.preferences.remove_ineligible_cadet_choices(self.parameters, printing=printing)
        self.parameters = afccp.core.data.adjustments.parameter_sets_additions(self.parameters)

    def update_cadet_columns_from_matrices(self):
        """
        Update Cadet Columns from Preference Matrix.

        This method updates the preference and utility columns for cadets based on the preference matrix (c_pref_matrix).
        The preferences are converted to preference columns, and the utility values are extracted and stored in their respective columns.
        """

        if self.printing:
            print('Updating cadet columns (Cadets.csv...c_utilities, c_preferences) from the preference matrix '
                  '(c_pref_matrix)...')

        # Update parameters
        self.parameters['c_preferences'], self.parameters['c_utilities'] = \
            afccp.core.data.preferences.get_utility_preferences_from_preference_array(self.parameters)

    def update_preference_matrices(self):
        """
        Update Preference Matrices from Preference Arrays.

        This method reconstructs the cadet preference matrices from the preference arrays by renumbering preferences to eliminate gaps.
        In preference lists, gaps may exist due to unranked choices, and this method ensures preferences are sequentially numbered.
        """

        if self.printing:
            print("Updating cadet preference matrices from the preference dictionaries. "
                  "ie. 1, 2, 4, 6, 7 -> 1, 2, 3, 4, 5 (preference lists need to omit gaps)")

        # Update parameters
        self.parameters = afccp.core.data.preferences.update_preference_matrices(self.parameters)

    def update_cadet_utility_matrices_from_cadets_data(self):
        """
        Update Cadet Utility Matrices from Cadets Data.

        This method updates the utility matrices ('utility' and 'cadet_utility') by extracting data from the "Util_1 -> Util_P" columns in Cadets.csv.
        """

        if self.printing:
            print("Updating cadet utility matrices ('utility' and 'cadet_utility') from the 'c_utilities' matrix")

        # Update parameters
        self.parameters = afccp.core.data.preferences.update_cadet_utility_matrices(self.parameters)
        self.parameters = afccp.core.data.adjustments.parameter_sets_additions(self.parameters)

    def fill_remaining_afsc_choices(self):
        """

        :return:
        """

        if self.printing:
            print("Filling remaining cadet preferences arbitrarily with the exception of the bottom choices")

        # Update parameters
        self.parameters = afccp.core.data.preferences.fill_remaining_preferences(self.parameters)
        self.parameters = afccp.core.data.adjustments.parameter_sets_additions(self.parameters)

    def create_final_utility_matrix_from_new_formula(self):
        """
        """

        if self.printing:
            print("Creating 'Final' cadet utility matrix from the new formula with different conditions...")

        # Update parameters
        self.parameters = afccp.core.data.preferences.create_final_cadet_utility_matrix_from_new_formula(self.parameters)
        self.parameters = afccp.core.data.adjustments.parameter_sets_additions(self.parameters)

    # Adjust Rated Data
    def generate_rated_data(self):
        """
        This method generates Rated data for USAFA and ROTC if it doesn't already exist. This data can then also be
        exported back as a csv for reference.
        """

        # Generate Rated Data
        self.parameters = afccp.core.data.preferences.generate_rated_data(self.parameters)
        self.parameters = afccp.core.data.adjustments.parameter_sets_additions(self.parameters)

    def construct_rated_preferences_from_om_by_soc(self):
        """
        This method takes the two OM Rated matrices (from both SOCs) and then zippers them together to
        create a combined "1-N" list for the Rated AFSCs. The AFSC preference matrix is updated as well as the
        AFSC preference lists
        """

        if self.printing:
            print("Integrating rated preferences from OM matrices for each SOC...")

        # Generate Rated Preferences
        self.parameters = afccp.core.data.preferences.construct_rated_preferences_from_om_by_soc(self.parameters)
        self.parameters = afccp.core.data.adjustments.parameter_sets_additions(self.parameters)

    # Specify Value Parameters
    def set_value_parameters(self, vp_name=None):
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

    def update_value_parameters(self, num_breakpoints=24):
        """
        Simple method to take the current set of value parameters and update their sets and subsets and all that.
        This method also updates the set of value parameters in the dictionary
        :param num_breakpoints: Number of breakpoints to use when building the value functions
        """

        # Module shorthand
        afccp_vp = afccp.core.data.values

        # Update the value functions and cadet/AFSC weights
        self.value_parameters = afccp_vp.update_value_and_weight_functions(self, num_breakpoints)

        # "Condense" the value functions by removing unnecessary zeros in the breakpoints
        self.value_parameters = afccp_vp.condense_value_functions(self.parameters, self.value_parameters)

        # Add indexed sets and subsets of AFSCs and AFSC objectives
        self.value_parameters = afccp_vp.value_parameters_sets_additions(self.parameters, self.value_parameters)

        # Update the set of value parameters in the dictionary (vp_dict attribute)
        self.update_value_parameters_in_dict()

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
                    self.vp_dict[vp_name], self.vp_name = copy.deepcopy(value_parameters), vp_name
                else:  # If it's not unique, then "unique" is the name of the matching set of value parameters
                    self.vp_name = unique

        else:
            raise ValueError("Error. No value parameters set. You currently do have a 'vp_dict' and so all you "
                                     "need to do is run 'instance.set_value_parameters()'. ")

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

    def check_unique_value_parameters(self, value_parameters=None, vp_name1=None, vp_name2=None, printing=False):
        """
        Take in a new set of value parameters and see if this set is in the dictionary already. Return True if the
        the set of parameters is unique, or return the name of the matching set otherwise
        """

        # If we specify this name, we want to compare the value parameters against this one
        if vp_name2 is not None:
            value_parameters = self.vp_dict[vp_name2]
            vp_name1 = vp_name2

        # If we don't specify "vp_name2", we check the other conditions
        else:
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

    def generate_random_value_parameters(self, num_breakpoints=24, vp_weight=100, printing=None):
        """
        Generates value parameters for a given problem instance from scratch
        """

        # Print Statement
        if printing is None:
            printing = self.printing

        # Generate random set of value parameters
        value_parameters = afccp.core.data.generation.generate_random_value_parameters(
            self.parameters, num_breakpoints=num_breakpoints)

        # Module shorthand
        afccp_vp = afccp.core.data.values

        # "Condense" the value functions by removing unnecessary zeros
        value_parameters = afccp_vp.condense_value_functions(self.parameters, value_parameters)

        # Add indexed sets and subsets of AFSCs and AFSC objectives
        value_parameters = afccp_vp.value_parameters_sets_additions(self.parameters, value_parameters)

        # Weight of the value parameters as a whole
        value_parameters['vp_weight'] = vp_weight

        # Set value parameters to instance attribute
        if self.mdl_p["set_to_instance"]:
            self.value_parameters = value_parameters

        # Save new set of value parameters to dictionary
        if self.mdl_p["add_to_dict"]:
            self.save_new_value_parameters_to_dict(value_parameters)

        return value_parameters

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
                self.solution = afccp.core.solutions.handling.evaluate_solution(
                    self.solution, self.parameters, self.value_parameters, printing=printing)

        # Save new set of value parameters to dictionary
        if self.mdl_p["add_to_dict"]:
            self.save_new_value_parameters_to_dict(value_parameters)

        if self.printing:
            print('Imported.')

        return value_parameters

    def calculate_afocd_value_parameters(self):
        """
        This method calculates the AFOCD value parameters using my own methodology on determining the
        weights and uses the AFSCs.csv dataset for the targets and constraints.
        """

        # Determine the AFOCD value parameters
        self.value_parameters = afccp.core.data.values.generate_afocd_value_parameters(
            self.parameters, self.value_parameters)

        # Update the value parameters
        self.update_value_parameters()

    def export_value_parameters_as_defaults(self, filename=None, printing=None):
        """
        This method exports the current set of instance value parameters to a new excel file in the "default"
        value parameter format
        """
        if printing is None:
            printing = self.printing

        if self.value_parameters is None:
            raise ValueError('No instance value parameters detected.')

        # Export value parameters
        if filename is None:  # I add the "_New" just so we make sure we don't accidentally overwrite the old one
            filename = "Value_Parameters_Defaults_" + self.data_name + "_New.xlsx"
        filepath = afccp.core.globals.paths["support"] + "value parameters defaults/" + filename
        afccp.core.data.values.model_value_parameters_to_defaults(self, filepath=filepath, printing=printing)

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
        This method performs a comprehensive sanity check on the parameters and value parameters
        to ensure the data is in a valid and correct state before running the model.

        It examines various parameters and their values within the class instance to identify and
        address any issues or discrepancies. The checks are designed to ensure that the data is consistent,
        within valid ranges, and suitable for use in the model.

        While the exact details of these checks are implemented in an external function or module,
        this method serves as the entry point for conducting these checks.

        It is essential to run this method before executing the model to guarantee the integrity
        of the input data and to prevent potential errors or unexpected behavior during the modeling process.

        This method is part of an object-oriented programming structure and uses 'self' to access
        the class instance's attributes and data.

        Note: The specific details of the sanity checks are defined in an external function
        or module called 'afccp.core.data.adjustments.parameter_sanity_check.'
        """

        # Call the function
        afccp.core.data.adjustments.parameter_sanity_check(self)

    # noinspection PyDictCreation  # Rebecca's model stuff!
    def vft_to_gp_parameters(self, p_dict={}, printing=None):
        """
        Translate VFT Model Parameters to *former* Lt Rebecca Reynold's Goal Programming Model Parameters.

        This method is responsible for translating various parameters and settings used in the Value Focussed Thinking (VFT) model
        into parameters suitable for the Goal Programming (GP) model. It facilitates the conversion between different modeling
        frameworks.

        Args:
            p_dict (dict, optional): A dictionary of additional parameters that can be provided to fine-tune the translation process.
                These parameters may include specific weights or settings required for the GP model. Defaults to an empty dictionary.

            printing (bool, optional): A flag to control whether to print progress information during the translation process.
                If True, it will print status updates; if False, it will run silently. Defaults to None.

        Returns:
            None

        The method updates the internal representation of parameters and settings in the instance to match the requirements
        of the Goal Programming (GP) model. It translates preference scores, rewards, and penalties according to the GP model's
        specifications, making the instance ready for goal-based optimization.

        Note:
        - The method may apply normalization to ensure that rewards and penalties are consistent with the GP model's expectations.
        - If the 'get_new_rewards_penalties' flag is set to True in the model parameters, the method may compute new rewards
          and penalties based on the instance data and preferences, creating a fresh set of values for optimization.

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

            gc["Raw Reward"], gc["Raw Penalty"] = afccp.core.solutions.optimization.calculate_rewards_penalties(
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

    # Very basic methods to generate solutions
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
        solution = {'method': 'Random',
            'j_array': np.array([np.random.choice(self.parameters['J^E'][i]) for i in self.parameters['I']])}

        # Determine what to do with the solution
        self.solution_handling(solution)

        return solution

    # Matching algorithms
    def rotc_rated_board_original(self, p_dict={}, printing=None):
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Get the solution we need
        solution = afccp.core.solutions.algorithms.rotc_rated_board_original(self, printing=printing)

        # Determine what to do with the solution
        self.solution_handling(solution)

        return solution

    def soc_rated_matching_algorithm(self, p_dict={}, printing=None):
        """
        This is the Hospitals/Residents algorithm that matches or reserves cadets to their Rated AFSCs depending on the
        source of commissioning (SOCs).
        """
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Get the solution and solution iterations we need
        combined, reserves, matches = afccp.core.solutions.algorithms.soc_rated_matching_algorithm(
            self, soc=self.mdl_p['soc'], printing=printing)

        # Determine what to do with the solution(s)
        for solution in [reserves, matches, combined]:
            self.solution_handling(solution, printing=False)  # Don't want print updates for this

        return solution

    def classic_hr(self, p_dict={}, printing=None):
        """
        This method solves the problem instance using the classical "Hospital/Residents" algorithm
        """
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Get the solution we need
        solution = afccp.core.solutions.algorithms.classic_hr(self, printing=printing)

        # Determine what to do with the solution
        self.solution_handling(solution)

        return solution

    # Optimization models
    def solve_vft_pyomo_model(self, p_dict={}, printing=None):
        """
        Solve the VFT model using Pyomo, an optimization modeling library.

        This method is responsible for solving the Value Focussed Thinking (VFT) model using Pyomo. The VFT model is a specific
        type of optimization model. It conducts the necessary preparation, builds the model, solves it, and handles the
        resulting solution. The goal is to find optimal solutions for the VFT problem.

        Args:
            p_dict (dict): A dictionary of parameters used for the model. It allows for customization of the model's
                input parameters.
            printing (bool, None): A flag to control whether to print information during the model-solving process. If set to
                True, the method will print progress and debugging information. If set to False, it will suppress printing.
                If None, the method will use the default printing setting from the class instance.

        Returns:
            solution: The solution of the VFT model, which contains the optimal values for the decision variables and other
                relevant information.

        Notes:
        - Before using this method, it's important to ensure that the class instance contains valid and appropriate data.
        - This method uses external functions for building and solving the Pyomo model, and the specifics of those functions
          are located in the 'afccp.core.solutions.optimization' module.
        """
        self.error_checking("Pyomo Model")
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Build the model and then solve it
        model, q = afccp.core.solutions.optimization.vft_model_build(self, printing=printing)
        solution = afccp.core.solutions.optimization.solve_pyomo_model(self, model, "VFT", q=q, printing=printing)

        # Determine what to do with the solution
        self.solution_handling(solution)

        # Return the solution
        return solution

    def solve_original_pyomo_model(self, p_dict={}, printing=None):
        """
        Solve the original AFPC model using pyomo
        """
        self.error_checking("Pyomo Model")
        if printing is None:
            printing = self.printing

        # One little "switch" to get the original model objective function
        p_dict['assignment_model_obj'] = "Original Utility"

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Build the model and then solve it
        model = afccp.core.solutions.optimization.assignment_model_build(self, printing=printing)
        solution = afccp.core.solutions.optimization.solve_pyomo_model(self, model, "Original", printing=printing)

        # Determine what to do with the solution
        self.solution_handling(solution)

        # Return the solution
        return solution

    def solve_guo_pyomo_model(self, p_dict={}, printing=None):
        """
        Solve the "generalized assignment problem" model with the new global utility matrix constructed
        from the AFSC and Cadet Utility matrices. This is the "GUO" model.
        """
        # One little "switch" to get the new assignment model objective function
        p_dict['assignment_model_obj'] = "Global Utility"

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Error handling
        self.error_checking("Pyomo Model")
        if printing is None:
            printing = self.printing

        # Build the model and then solve it
        model = afccp.core.solutions.optimization.assignment_model_build(self, printing=printing)
        solution = afccp.core.solutions.optimization.solve_pyomo_model(self, model, "GUO", printing=printing)

        # Determine what to do with the solution
        self.solution_handling(solution)

        # Return the solution
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
        model = afccp.core.solutions.optimization.gp_model_build(self, printing=printing)
        solution = afccp.core.solutions.optimization.solve_pyomo_model(self, model, "GP", printing=printing)

        # Determine what to do with the solution
        self.solution_handling(solution)

        # Return the solution
        return solution

    def solve_vft_main_methodology(self, p_dict={}, printing=None):
        """
        This is the main method to solve the problem instance. We first determine an initial population of solutions.
        We then evolve the solutions further using the GA.
        """
        self.error_checking("Pyomo Model")
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Determine population for the genetic algorithm
        if self.mdl_p['population_generation_model'] == 'Assignment':
            self.mdl_p["initial_solutions"] = \
                afccp.core.solutions.sensitivity.populate_initial_ga_solutions_from_assignment_model(self, printing)
        else:
            self.mdl_p["initial_solutions"] = \
                afccp.core.solutions.sensitivity.populate_initial_ga_solutions_from_vft_model(self, printing)

        # Add additional solutions if necessary
        if self.mdl_p["solution_names"] is not None:

            # In case the user specifies "Solution" instead of ["Solution"]
            if type(self.mdl_p["solution_names"]) == str:
                self.mdl_p["solution_names"] = [self.mdl_p["solution_names"]]

            # Add additional solutions
            for solution_name in self.mdl_p["solution_names"]:
                solution = self.solutions[solution_name]
                self.mdl_p["initial_solutions"] = np.vstack((self.mdl_p["initial_solutions"], solution))

        self.mdl_p["initialize"] = True  # Force the initialize parameter to be true

        if printing:
            now = datetime.datetime.now()
            print('Solving Genetic Algorithm for ' + str(self.mdl_p["ga_max_time"]) + ' seconds at ' +
                  now.strftime('%H:%M:%S') + '...')
        self.vft_genetic_algorithm(self.mdl_p, printing=self.mdl_p["ga_printing"])
        if printing:
            print('Solution value of ' + str(round(self.solution['z'], 4)) + ' obtained.')

        # Return solution
        return self.solution

    # Meta-heuristics
    def vft_genetic_algorithm(self, p_dict={}, printing=None):
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

        # Dictionary of failed constraints across all solutions (phased out since we're using APM as initial population)
        con_fail_dict = None

        # Get a starting population of solutions if applicable!
        if self.mdl_p["initialize"]:

            if self.mdl_p["initial_solutions"] is None:

                if self.solutions is None:
                    raise ValueError("Error. No solutions in dictionary.")

                else:
                    if self.mdl_p["solution_names"] is None:

                        # Get list of initial solutions
                        initial_solutions = np.array(
                            [self.solutions[solution_name]['j_array'] for solution_name in self.solutions])
                        solution_names = list(self.solutions.keys())

                    else:

                        # If we just pass "Solution" instead of ["Solution"]
                        if type(self.mdl_p["solution_names"]) == str:
                            self.mdl_p["solution_names"] = [self.mdl_p["solution_names"]]

                        # Get list of initial solutions
                        initial_solutions = np.array(
                            [self.solutions[solution_name]['j_array'] for solution_name in self.mdl_p["solution_names"]])
                        solution_names = self.mdl_p["solution_names"]

                    if printing:
                        print("Running Genetic Algorithm with initial solutions:", solution_names)

            else:

                # Get list of initial solutions
                initial_solutions = self.mdl_p["initial_solutions"]
                if printing:
                    print("Running Genetic Algorithm with", len(initial_solutions), "initial solutions...")

        else:

            if printing:
                print("Running Genetic Algorithm with no initial solutions (not advised!)...")
            initial_solutions = None

        # Generate the solution
        solution, time_eval_df = afccp.core.solutions.algorithms.vft_genetic_algorithm(
            self, initial_solutions, con_fail_dict, printing=printing)

        # Determine what to do with the solution
        self.solution_handling(solution)

        # Return the final solution and maybe the time evaluation dataframe if needed
        if self.mdl_p["time_eval"]:
            return time_eval_df, solution
        else:
            return solution

    def genetic_matching_algorithm(self, p_dict={}, printing=None):
        """
        This method solves the problem instance using "Genetic Matching Algorithm"
        """
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Force solution iteration collection to be turned off
        self.mdl_p['collect_solution_iterations'] = False

        # Get the capacities
        capacities = afccp.core.solutions.algorithms.genetic_matching_algorithm(self, printing=printing)

        # Update capacities in parameters (quota_max or quota_min)
        self.parameters[self.mdl_p['capacity_parameter']] = capacities

        # Run the matching algorithm with these capacities
        solution = afccp.core.solutions.algorithms.classic_hr(self, printing=printing)

        # Determine what to do with the solution
        self.solution_handling(solution)

        return solution

    # Solution Handling
    def incorporate_rated_algorithm_results(self, p_dict={}, printing=None):
        """
        Takes the two sets of Rated Matches and Reserves and adds that into the parameters (J^Fixed and J^Reserved)
        """
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        self.parameters = afccp.core.solutions.handling.incorporate_rated_results_in_parameters(
            self, printing=printing)

        # Shorthand
        p = self.parameters

        # Temporary stuff
        if self.mdl_p['create_new_rated_solutions']:

            name_dict = {'Rated Matches': 'J^Fixed', 'Rated Reserves': 'J^Reserved',
                         'Rated Alternates (Hard)': 'J^Alternates (Hard)',
                         'Rated Alternates (Soft)': 'J^Alternates (Soft)'}
            new_solutions = {}
            for s_name, s_param in name_dict.items():
                new_solutions[s_name] = {'method': s_name,
                                         'j_array': np.array([p['M'] for _ in p['I']]),
                                         'afsc_array': np.array(['*' for _ in p['I']])}

                # Create this solution array
                for i in p['I']:
                    if i in p[s_param]:
                        if s_param == 'J^Reserved':
                            j = p['J^Reserved'][i][len(p['J^Reserved'][i]) - 1]
                        else:
                            j = p[s_param][i]
                        new_solutions[s_name]['j_array'][i] = j
                        new_solutions[s_name]['afsc_array'][i] = p['afscs'][j]

                # Integrate this solution
                self.solution_handling(new_solutions[s_name], printing=False)

    def find_ineligible_cadets(self, solution_name=None, fix_it=True):
        """
        Prints out the ID's of ineligible pairs of cadets/AFSCs
        """

        if solution_name is None:
            if self.solution is None:
                raise ValueError("No solution activated.")
            else:
                solution = self.solution['j_array']
        else:
            solution = self.solutions[solution_name]['j_array']

        # Loop through each cadet to see if they're ineligible for the AFSC they're assigned to
        total_ineligible = 0
        for i, j in enumerate(solution):
            cadet, afsc = self.parameters['cadets'][i], self.parameters['afscs'][j]

            # Unmatched AFSC
            if j == self.parameters['M']:
                continue

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
            self.parameters = afccp.core.data.adjustments.parameter_sets_additions(self.parameters)

    def set_solution(self, solution_name=None, printing=None):
        """
        Set the current instance object's solution to a solution from the dictionary
        """
        if printing is None:
            printing = self.printing

        if self.solutions is None:
            raise ValueError('No solution dictionary initialized')
        else:
            if solution_name is None:
                solution_name = list(self.solutions.keys())[0]
            else:
                if solution_name not in self.solutions:
                    raise ValueError('Solution ' + solution_name + ' not in solution dictionary')

            self.solution = self.solutions[solution_name]
            self.solution_name = solution_name
            if self.value_parameters is not None:
                self.measure_solution(printing=printing)

    def add_solution(self, afsc_solution):
        """
        Takes a numpy array of AFSCs and adds this new solution into the solution dictionary
        """

        # Create solution dictionary
        solution = {'j_array': np.array([np.where(self.parameters['afscs'] == afsc)[0][0] for afsc in afsc_solution]),
                    'method': "Added", 'afsc_array': afsc_solution}

        # Determine what to do with the solution
        self.solution_handling(solution)

        # Return the solution
        return solution

    def compute_similarity_matrix(self, solution_names=None):
        """
        Generates the similarity matrix for a given set of solutions
        """

        if 'Similarity Solutions.csv' in os.listdir(self.export_paths['Analysis & Results']):
            solution_df = afccp.core.globals.import_csv_data(
                self.export_paths['Analysis & Results'] + 'Similarity Solutions.csv')
        else:
            raise ValueError("Error. No 'Similarity Solutions.csv' dataframe found in the 'Analysis & Results' folder. "
                             "Please create it.")

        # "Starting" Solutions: Extract solutions from dataframe and then convert to "j_array"
        solutions = {solution_name: np.array(solution_df[solution_name]) for solution_name in solution_df}
        solutions = {solution_name: np.array([np.where(
            self.parameters['afscs'] == afsc)[0][0] for afsc in solutions[solution_name]]) for solution_name in solutions}

        # If we want to add solutions to highlight in the chart
        if solution_names is not None:
            extra_solutions = {
                solution_name: self.solutions[solution_name]['j_array'] for solution_name in solution_names}
            for solution_name in extra_solutions:
                solutions[solution_name] = extra_solutions[solution_name]

        # Create the matrix
        num_solutions = len(solutions.keys())
        similarity_matrix = np.zeros((num_solutions, num_solutions))
        for row, sol_1_name in enumerate(solutions.keys()):
            for col, sol_2_name in enumerate(solutions.keys()):
                sol_1 = solutions[sol_1_name]
                sol_2 = solutions[sol_2_name]
                similarity_matrix[row, col] = np.sum(sol_1 == sol_2) / self.parameters["N"]  # % similarity!

        # Export similarity_matrix
        similarity_df = pd.DataFrame({solution: similarity_matrix[:, s] for s, solution in enumerate(solutions.keys())})
        similarity_df.to_csv(self.export_paths['Analysis & Results'] + 'Similarity Matrix.csv', index=False)

    def measure_solution(self, approximate=False, printing=None):
        """
        Evaluate a solution using the VFT objective hierarchy
        """
        # Error checking, solution setting
        if self.solution is None or self.value_parameters is None:
            raise ValueError("Error. Solution and value parameters needed to evaluate solution.")

        # Print statement
        if printing is None:
            printing = self.printing

        # Calculate solution metrics
        self.solution = afccp.core.solutions.handling.evaluate_solution(
            self.solution, self.parameters, self.value_parameters, approximate=approximate, printing=printing,
            re_calculate_x=self.mdl_p['re-calculate x'])

    def measure_fitness(self, printing=None):
        """
        This is the fitness function method (could be slightly different depending on how the constraints are handled)
        :return: fitness score
        """
        # Error checking, solution setting
        if self.solution is None or self.value_parameters is None:
            raise ValueError("Error. Solution and value parameters needed to evaluate solution.")

        # Printing statement
        if printing is None:
            printing = self.printing

        # Get the solution metrics if necessary
        if "z" not in self.solution:
            self.solution = self.measure_solution(printing=False)

        # Calculate fitness value
        z = afccp.core.solutions.handling.fitness_function(self.solution['j_array'], self.parameters,
                                                           self.value_parameters, self.mdl_p,
                                                           con_fail_dict=self.solution['con_fail_dict'])

        # Print and return fitness value
        if printing:
            print("Fitness value calculated to be", round(z, 4))
        return z

    # Matching Algorithm Iterations
    def export_solution_iterations(self, printing=None):
        """
        Exports iterations of a matching algorithm to excel
        """
        if printing is None:
            printing = self.printing

        # Error handling
        if 'iterations' not in self.solution:
            raise ValueError("Error. No solution iterations detected.")

        # Update the solution iterations dictionary
        if 'sequence' not in self.solution['iterations']:
            self.manage_bubbles_parameters()

        # 'Sequence' Folder
        folder_path = self.export_paths['Analysis & Results'] + 'Cadet Board/'
        if self.solution['iterations']['sequence'] not in os.listdir(folder_path):
            os.mkdir(folder_path + self.solution['iterations']['sequence'])

        # Create proposals dataframe
        if 'proposals' in self.solution['iterations']:
            proposals_df = pd.DataFrame({'Cadet': self.parameters['cadets']})
            for s in self.solution['iterations']['proposals']:
                solution = self.solution['iterations']['proposals'][s]
                afsc_solution = [self.parameters['afscs'][j] for j in solution]
                proposals_df['Iteration ' + str(s + 1)] = afsc_solution

            # Export file
            filepath = folder_path + self.solution['iterations']['sequence'] + '/Solution Iterations (Proposals).csv'
            proposals_df.to_csv(filepath, index=False)

            if printing:
                print('Proposal iterations exported to', filepath)

        # Create matches dataframe
        if 'matches' in self.solution['iterations']:
            solutions_df = pd.DataFrame({'Cadet': self.parameters['cadets']})
            for s in self.solution['iterations']['matches']:
                solution = self.solution['iterations']['matches'][s]
                afsc_solution = [self.parameters['afscs'][j] for j in solution]
                solutions_df['Iteration ' + str(s + 1)] = afsc_solution

            # Export file
            filepath = folder_path + self.solution['iterations']['sequence'] + '/Solution Iterations (Matches).csv'
            solutions_df.to_csv(filepath, index=False)

            if printing:
                print('Solution iterations exported to', filepath)

        if printing:
            print('Done.')

    def import_solution_iterations(self, sequence_name, printing=None):
        """
        Exports iterations of a matching algorithm to excel
        """
        if printing is None:
            printing = self.printing

        # 'Sequence' Folder
        folder_path = self.export_paths['Analysis & Results'] + 'Cadet Board/'
        if sequence_name not in os.listdir(folder_path):
            raise ValueError("Error. Sequence '" + str(sequence_name) + "' not found in 'Cadet Board' figure.")
        sequence_folder_path = self.export_paths['Analysis & Results'] + 'Cadet Board/' + sequence_name
        sequence_folder = os.listdir(sequence_folder_path)
        self.solution['iterations'] = {'type': sequence_name.split(',')[2].strip()}

        # Current solution should match the sequence name
        if self.solution_name not in sequence_name:
            raise ValueError("Error. Current solution is '" + self.solution_name +
                             "' which is not found in the provided sequence name of '" + sequence_name + "'.")

        # Get proposals if applicable
        if 'Solution Iterations (Proposals).csv' in sequence_folder:
            filepath = sequence_folder_path + '/Solution Iterations (Proposals).csv'
            proposals_df = afccp.core.globals.import_csv_data(filepath)
            self.solution['iterations']['proposals'] = {}
            for s, col in enumerate(proposals_df.columns[1:]):
                afsc_solution = np.array(proposals_df[col])
                solution = np.array([np.where(self.parameters['afscs'] == afsc)[0][0] for afsc in afsc_solution])
                self.solution['iterations']['proposals'][s] = solution

            if printing:
                print('Proposal iterations imported from', filepath)

        # Get matches if applicable
        if 'Solution Iterations (Matches).csv' in sequence_folder:
            filepath = sequence_folder_path + '/Solution Iterations (Matches).csv'
            matches_df = afccp.core.globals.import_csv_data(filepath)
            self.solution['iterations']['matches'] = {}
            self.solution['iterations']['names'] = {}
            for s, col in enumerate(matches_df.columns[1:]):
                afsc_solution = np.array(matches_df[col])
                solution = np.array(
                    [np.where(self.parameters['afscs'] == afsc)[0][0] for afsc in afsc_solution])
                self.solution['iterations']['matches'][s] = solution
                self.solution['iterations']['names'][s] = "Iteration " + str(s + 1)

            # Last solution iteration
            self.solution['iterations']['last_s'] = s

            if printing:
                print('Solution iterations imported from', filepath)

    def manage_bubbles_parameters(self, p_dict={}):
        """
        Handles the solution iterations that we should already have as an attribute of the problem instance
        """

        # Error Checking
        self.error_checking("Solution")

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        # Cadets/AFSCs solved for is by default "All"
        if "cadets_solved_for" not in self.solution:
            self.solution['cadets_solved_for'] = "All"
        if "afscs_solved_for" not in self.solution:
            self.solution['afscs_solved_for'] = "All"
        self.mdl_p['afscs_solved_for'] = self.solution['afscs_solved_for']  # Update AFSCs solved for in mdl_p

        # Determine which AFSCs to show in this visualization
        self.mdl_p = afccp.core.data.support.determine_afscs_in_image(self.parameters, self.mdl_p)

        # Determine what kind of cadet/AFSC board figure and/or animation we're building
        if 'iterations' in self.solution:

            # Adjust certain elements for Rated stuff
            if 'Rated' in self.solution['iterations']['type']:
                self.solution['afscs_solved_for'] = 'Rated'
                if "USAFA" in self.solution_name:
                    self.solution['cadets_solved_for'] = 'USAFA Rated'
                elif "ROTC" in self.solution_name:
                    self.solution['cadets_solved_for'] = 'ROTC Rated'

            # Determine name of this BubbleChart sequence
            self.solution['iterations']['sequence'] = \
                self.data_name + ', ' + self.solution['cadets_solved_for'] + ' Cadets, ' + \
                self.solution['iterations']['type'] + ', ' + self.solution['afscs_solved_for'] + \
                ' AFSCs' + ', ' + self.solution['name']
            if self.data_version != 'Default':
                self.solution['iterations']['sequence'] += ' (' + self.data_version + ')'
            self.solution['iterations']['sequence'] += ' ' + str(self.mdl_p['M']) + " AFSCs Displayed"

        # Single solution
        else:

            # Create solution folder if necessary
            self.manage_solution_folder()

    # Data Visualizations
    def display_data_graph(self, p_dict={}, printing=None):
        """
        This method plots different aspects of the fixed parameters of the problem instance.
        """

        # Print statement
        if printing is None:
            printing = self.printing

        # Adjust instance plot parameters
        self.reset_functional_parameters(p_dict)
        self.mdl_p = afccp.core.data.support.determine_afsc_plot_details(self)

        # Initialize the AFSC Chart object
        afsc_chart = afccp.core.visualizations.charts.AFSCsChart(self)

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

    # Value Parameter Visualizations
    def show_value_function(self, p_dict={}, printing=None):
        """
        This method plots a specific AFSC objective value function
        """

        if printing is None:
            printing = self.printing

        # Shorthand
        p, vp = self.parameters, self.value_parameters

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)
        self.mdl_p = afccp.core.data.support.determine_afsc_plot_details(self)
        ip = self.mdl_p  # More shorthand

        if printing:
            print('Creating value function chart for objective ' + ip['objective'] + ' for AFSC ' + ip['afsc'])

        # Determine AFSC and objective shown in this chart
        j, k = np.where(p["afscs"] == ip["afsc"])[0][0], np.where(vp["objectives"] == ip["objective"])[0][0]

        # ValueFunctionChart specific parameters
        vfc = ip['ValueFunctionChart']
        vfc['x_label'] = afccp.core.globals.obj_label_dict[ip['objective']]  # Get x label for this objective

        # Value Function specific coordinates to plot
        if vfc['x_pt'] is not None:
            vfc['y_pt'] = afccp.core.solutions.handling.value_function(vp['a'][j][k], vp['f^hat'][j][k], vp['r'][j][k],
                                                                       vfc["x_pt"])

        # Determine x and y arrays
        if ip['smooth_value_function']:
            x_arr = (np.arange(1001) / 1000) * vp['a'][j][k][vp['r'][j][k] - 1]
            y_arr = np.array([afccp.core.solutions.handling.value_function(
                vp['a'][j][k], vp['f^hat'][j][k], vp['r'][j][k], x) for x in x_arr])
        else:
            x_arr, y_arr = vp['a'][j][k], vp['f^hat'][j][k]

        # Title and filepath for this value function!
        vfc['title'] = ip['afsc'] + ' ' + ip['objective'] + ' Value Function'
        vfc['filepath'] = self.export_paths['Analysis & Results'] + \
                          'Value Functions/' + self.data_name + ' ' + vfc['title'] + ' (' + self.vp_name + ').png'

        # Create and return the chart
        return afccp.core.visualizations.charts.ValueFunctionChart(x_arr, y_arr, vfc)

    def display_weight_function(self, p_dict={}, printing=None):
        """
        This method plots the weight function used for either cadets or AFSCs
        """

        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)
        self.mdl_p = afccp.core.data.support.determine_afsc_plot_details(self)

        # Make the folder
        if 'Value Parameters' not in os.listdir(self.export_paths['Analysis & Results']):
            os.mkdir(self.export_paths['Analysis & Results'] + 'Value Parameters')

        if printing:
            if self.mdl_p["cadets_graph"]:
                print("Creating cadet weight chart...")
            else:
                print("Creating AFSC weight chart...")

        # Build the chart
        chart = afccp.core.visualizations.charts.individual_weight_graph(self)

        if printing:
            chart.show()

        return chart

    # Results Visualizations
    def display_all_results_graphs(self, p_dict={}, printing=None):
        """
        Saves all charts for the current solution and for the solutions in the solution names list if specified
        """

        if printing is None:
            printing = self.printing

        # Create solution folder if necessary
        self.manage_solution_folder()

        # Determine what kind of results charts we're creating
        if 'solution_names' not in p_dict:  # Regular Solution Charts
            desired_charts = 'desired_charts'
            if printing:
                print("Saving all solution results charts to the corresponding folder...")

        else:  # Solution Comparison Charts
            desired_charts = 'desired_comparison_charts'
            p_dict['results_graph'] = 'Solution Comparison'
            if printing:
                print("Saving all solution comparison charts to the corresponding folder...")

            # Evaluate the solutions to get metrics
            self.evaluate_all_solutions(p_dict['solution_names'])

        # Loop through the subset of AFSC charts that I actually care about
        charts = []
        for obj, version in self.mdl_p[desired_charts]:
            if printing:
                print("<Objective '" + obj + "' version '" + version + "'>")

            # Build the figure
            if obj in self.value_parameters['objectives'] or obj == 'Extra':
                p_dict["objective"] = obj
                p_dict["version"] = version
                p_dict['macro_chart_kind'] = 'AFSC Chart'
                charts.append(self.display_results_graph(p_dict))
            else:
                if printing:
                    print("Objective '" + obj + "' passed since it isn't in our set of objectives.")

        # Loop through the subset of "other charts" that I care about
        if self.mdl_p['results_graph'] != "Solution Comparison":  # Only for a solution-specific chart
            for kind, version in self.mdl_p['desired_other_charts']:
                if printing:
                    print("<Other Charts '" + kind + "' version '" + version + "'>")

                # Build the figure
                p_dict['objective'] = "Extra"
                p_dict["version"] = version
                p_dict['macro_chart_kind'] = 'Accessions Group'
                charts.append(self.display_results_graph(p_dict))


        return charts

    def display_cadet_individual_utility_graph(self, p_dict={}, printing=None):
        """
        Builds the cadet utility graph for a particular cadet
        """

        # Print statement
        if printing is None:
            printing = self.printing

        # Adjust instance plot parameters
        self.reset_functional_parameters(p_dict)

        # Build the chart
        afccp.core.visualizations.charts.CadetUtilityGraph(self)

    def display_results_graph(self, p_dict={}, printing=None):
        """
        Builds the AFSC Results graphs
        """

        # Print statement
        if printing is None:
            printing = self.printing

        # Adjust instance plot parameters
        self.reset_functional_parameters(p_dict)
        self.mdl_p = afccp.core.data.support.determine_afsc_plot_details(self, results_chart=True)

        # Error handling
        if self.mdl_p['results_graph'] == 'Solution Comparison':
            self.error_checking('Solutions')
            chart_type = 'Comparison'
        else:
            self.error_checking('Solution')
            chart_type = 'Solution'

        # Determine which chart to create
        if self.mdl_p["macro_chart_kind"] == "AFSC Chart":

            # Initialize the AFSC Chart object
            afsc_chart = afccp.core.visualizations.charts.AFSCsChart(self)

            # Construct the specific chart
            return afsc_chart.build(chart_type=chart_type, printing=printing)

        elif self.mdl_p["macro_chart_kind"] == "Accessions Group":

            # Initialize the AFSC Chart object
            acc_chart = afccp.core.visualizations.charts.AccessionsGroupChart(self)

            # Construct the specific chart
            return acc_chart.build(chart_type=chart_type, printing=printing)

    def generate_results_slides(self, p_dict={}, printing=None):
        """
        Method to generate the results slides for a particular problem instance with solution
        """

        if printing is None:
            printing = self.printing

        if printing:
            print("Generating results slides...")

        # Adjust instance plot parameters
        self.reset_functional_parameters(p_dict)
        self.error_checking('Solution')

        # Call the function to generate the slides
        if afccp.core.globals.use_pptx:
            afccp.core.visualizations.slides.generate_results_slides(self)
        else:
            print('PPTX library not installed.')

        if printing:
            print('Done.')

    def generate_comparison_slides(self, p_dict={}, printing=None):
        """
        Method to generate the results slides for a particular problem instance with solution
        """

        if printing is None:
            printing = self.printing

        if printing:
            print("Generating comparison slides...")

        if 'Comparison Charts' not in os.listdir(self.export_paths['Analysis & Results']):
            raise ValueError("Error. No 'Comparison Charts' folder found in the 'Analysis & Results' folder. You need to"
                             " put all charts you'd like to compile into a slide-deck in this folder.")

        # Adjust instance plot parameters
        self.reset_functional_parameters(p_dict)
        self.error_checking('Solutions')

        # Call the function to generate the slides
        if afccp.core.globals.use_pptx:
            afccp.core.visualizations.slides.generate_comparison_slides(self)
        else:
            print('PPTX library not installed.')

        if printing:
            print('Done.')

    def generate_animation_slides(self, p_dict={}, printing=None):
        """
        Method to generate the animation slides for a particular problem instance and solution iterations
        """

        if printing is None:
            printing = self.printing

        if printing:
            print("Generating animation slides...")

        # Manage the solution iterations
        self.manage_bubbles_parameters(p_dict)

        # Call the function to generate the slides
        if afccp.core.globals.use_pptx:
            afccp.core.visualizations.slides.create_animation_slides(self)
        else:
            print('PPTX library not installed.')

        if printing:
            print('Done.')

    def generate_comparison_slide_components(self, p_dict={}, printing=None):
        """
        Method to do all the steps of generating the specific solution comparison charts I want
        """

        if printing is None:
            printing = self.printing

        if "solution_names" not in p_dict:
            raise ValueError("Error. In order to run this comparison method, the argument 'solution_names' must be "
                             "passed within 'p_dict'. This needs to be a list of solution names.")

        if printing:
            print("Generating comparison charts for the solutions:", p_dict['solution_names'])

        # Create the comparison charts folder if necessary
        if 'Comparison Charts' not in os.listdir(self.export_paths['Analysis & Results']):
            os.mkdir(self.export_paths['Analysis & Results'] + 'Comparison Charts')

        # Adjust instance plot parameters
        self.reset_functional_parameters(p_dict)
        self.error_checking('Solutions')

        # Save all solution comparison charts to the "Comparison Charts" folder
        self.display_all_results_graphs(p_dict, printing)

        # Cadet Utility Histogram
        self.display_utility_histogram(p_dict, folder='Comparison Charts')

        # Create pareto frontier plots
        self.show_pareto_chart(folder='Comparison Charts')  # without solutions
        self.show_pareto_chart(p_dict, folder='Comparison Charts',
                               solution_names=p_dict['solution_names'])  # with solutions

        # Compute similarity matrix and then calculate the similarity plot between all the solutions
        self.compute_similarity_matrix(solution_names=p_dict['solution_names'])
        self.similarity_plot(p_dict, folder='Comparison Charts')

        if printing:
            print('Done.')

    def generate_bubbles_chart(self, p_dict={}, printing=None):
        """
        Method to generate the "BubbleChart" figure by calling the BubbleChart class and applying the parameters
        as specified in the ccp helping functions.
        """

        if printing is None:
            printing = self.printing

        # Manage the solution iterations
        self.manage_bubbles_parameters(p_dict)

        # Print updates
        if printing:
            print('Creating Bubbles Chart...')

        # Call the figure object
        cadet_board = afccp.core.visualizations.bubbles.BubbleChart(self, printing=printing)
        cadet_board.main()

        # Only build the animation slides if we're saving iteration frames
        if self.mdl_p['save_iteration_frames']:

            # Generate the slides to go with this
            self.generate_animation_slides(p_dict, printing)

    def display_utility_histogram(self, p_dict={}, printing=None, folder="Results Charts"):
        """
        This method plots the cadet utility histogram
        """

        # Print statement
        if printing is None:
            printing = self.printing
        if printing:
            print("Creating cadet utility histogram...")

        # Adjust instance plot parameters
        self.reset_functional_parameters(p_dict)
        self.mdl_p = afccp.core.data.support.determine_afsc_plot_details(self, results_chart=True)

        # Evaluate the solutions to get metrics
        if self.mdl_p['solution_names'] is not None:
            self.evaluate_all_solutions(self.mdl_p['solution_names'])

        # Filepath for plot
        filepath = self.export_paths['Analysis & Results'] + folder + "/"

        # Construct the chart
        return afccp.core.visualizations.charts.cadet_utility_histogram(self, filepath=filepath)

    def solve_cadet_board_model_direct(self, filepath):
        """
        This method runs the cadet board model directly from the parameters in the csv at the specified
        filepath. We read and write back to this filepath
        """

        # Run function
        afccp.core.solutions.optimization.solve_cadet_board_model_direct_from_board_parameters(self, filepath)

    def similarity_plot(self, p_dict={}, printing=None, folder="Results Charts"):
        """
        Creates the solution similarity plot for the solutions specified
        """
        if printing is None:
            printing = self.printing

        if printing:
            print("Creating solution similarity plot...")

        # Adjust instance plot parameters
        self.reset_functional_parameters(p_dict)
        self.mdl_p = afccp.core.data.support.determine_afsc_plot_details(self, results_chart=True)

        # Import similarity matrix
        if 'Similarity Solutions.csv' in os.listdir(self.export_paths['Analysis & Results']):
            similarity_df = afccp.core.globals.import_csv_data(
                self.export_paths['Analysis & Results'] + 'Similarity Matrix.csv')
        else:
            raise ValueError("Error. No 'Similarity Matrix.csv' dataframe found in the 'Analysis & Results' folder. "
                             "Please create it.")

        # Extract similarity matrix information
        solution_names = np.array(similarity_df.keys())
        similarity_matrix = np.array(similarity_df)

        # Get coordinates
        coords = afccp.core.solutions.handling.similarity_coordinates(similarity_matrix)

        # Filepath for plot
        filepath = self.export_paths['Analysis & Results'] + folder + "/"

        # Plot similarity
        return afccp.core.visualizations.charts.solution_similarity_graph(self, coords, solution_names,
                                                                          filepath=filepath)

    # Sensitivity Analysis
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
        constraint_type, solutions_df, report_df = afccp.core.solutions.sensitivity.determine_model_constraints(self)

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

            # Build the model and then solve it
            model, q = afccp.core.solutions.optimization.vft_model_build(self, printing=printing)
            solution = afccp.core.solutions.optimization.solve_pyomo_model(self, model, "VFT", q=q, printing=printing)
            solution = afccp.core.solutions.handling.evaluate_solution(solution, self.parameters, self.value_parameters)
            solution_name = str(round(cadet_overall_weights[point], 4))

            # Extract solution information
            solutions[solution_name] = solution['afsc_array']
            cadet_overall_values[point] = solution['cadets_overall_value']
            afsc_overall_values[point] = solution['afscs_overall_value']

            if printing:
                print('For an overall weight on cadets of ' + str(cadet_overall_weights[point]) +
                      ', calculated value on cadets: ' + str(round(cadet_overall_values[point], 2)) +
                      ', value on afscs: ' + str(round(afsc_overall_values[point], 2)) +
                      ', and a Z of ' + str(round(solution['z'], 2)) + '.')

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

    def castle_pareto_analysis(self, p_dict={}, printing=None):
        """
        Takes the current set of value parameters and solves the GUO-Castle model with different weights on CASTLE
        and GUO.
        :param p_dict: more model parameters
        :param printing: whether to print something
        """
        self.error_checking("Pyomo Model")
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        p_dict['solve_castle_guo'] = True  # Just in case we didn't put that in ;)
        self.reset_functional_parameters(p_dict)

        if printing:
            print("Conducting CASTLE-MARKET pareto analysis on problem...")

        # Initialize arrays
        num_points = int(100 / self.mdl_p["pareto_step"] + 1)
        castle_values = np.zeros(num_points)
        guo_values = np.zeros(num_points)
        castle_weights = np.arange(1, 0, -(self.mdl_p["pareto_step"] / 100))
        castle_weights = np.append(castle_weights, 0)
        solutions = {}

        # Iterate over the number of points needed for the Pareto Chart
        for point in range(num_points):
            self.mdl_p['w^G'] = 1 - castle_weights[point]

            if printing:
                print("Calculating point " + str(point + 1) + " out of " + str(num_points) + "...")

            # Build the model and then solve it
            model = afccp.core.solutions.optimization.assignment_model_build(self, printing=printing)
            solution = afccp.core.solutions.optimization.solve_pyomo_model(self, model, "GUO", printing=printing)
            solution = afccp.core.solutions.handling.evaluate_solution(solution, self.parameters, self.value_parameters)
            solution_name = str(round(cadet_overall_weights[point], 4))

            # Extract solution information
            solutions[solution_name] = solution['afsc_array']
            castle_values[point] = solution['z^CASTLE (Values)']
            guo_values[point] = solution['z^gu']

            if printing:
                print('For an overall weight on CASTLE of ' + str(castle_weights[point]) +
                      ', calculated value on CASTLE: ' + str(round(castle_values[point], 2)) +
                      ', value on GUO: ' + str(round(guo_values[point], 2)) +
                      ', and a Z^CASTLE-MARKET of ' + str(round(solution['z^CASTLE'], 2)) + '.')

        # Obtain Dataframes
        pareto_df = pd.DataFrame(
            {'Weight on CASTLE': castle_weights, 'Value on CASTLE': castle_values,
             'Value on GUO': guo_values})
        solutions_df = pd.DataFrame(solutions)

        # File we import and export to
        filepath = self.export_paths['Analysis & Results'] + self.data_name + " " + self.vp_name + " (" + \
                   self.data_version + ") Pareto Analysis (CASTLE).xlsx"
        with pd.ExcelWriter(filepath) as writer:  # Export to excel
            pareto_df.to_excel(writer, sheet_name="CASTLE Results", index=False)
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

        self.error_checking("Pyomo Model")
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

            # Change Overall Weights
            self.value_parameters['cadets_overall_weight'] = cadet_overall_weights[point]
            self.value_parameters['afscs_overall_weight'] = 1 - cadet_overall_weights[point]

            if printing:
                print("Calculating point " + str(point + 1) + " out of " + str(num_points) + "...")

            # Solve the genetic algorithm
            solution, time_eval_df = afccp.core.solutions.algorithms.vft_genetic_algorithm(
                self, initial_solutions, printing=printing)
            solution = afccp.core.solutions.handling.evaluate_solution(solution, self.parameters, self.value_parameters)
            solution_name = str(round(cadet_overall_weights[point], 4))

            # Extract solution information
            solutions[solution_name] = solution['afsc_array']
            cadet_overall_values[point] = solution['cadets_overall_value']
            afsc_overall_values[point] = solution['afscs_overall_value']

            # We add this solution to our initial solutions too
            initial_solutions = np.vstack((initial_solutions, solution['j_array']))

            if printing:
                print('For an overall weight on cadets of ' + str(cadet_overall_weights[point]) +
                      ', calculated value on cadets: ' + str(round(cadet_overall_values[point], 2)) +
                      ', value on afscs: ' + str(round(afsc_overall_values[point], 2)) +
                      ', and a Z of ' + str(round(solution['z'], 2)) + '.')

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

    def overall_weights_pareto_analysis_utility(self, p_dict={}, printing=None):
        """
        Conduct pareto analysis on the "global utility" function using the assignment problem model
        """

        self.error_checking("Pyomo Model")
        if printing is None:
            printing = self.printing

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        if printing:
            print("Conducting 'utility' pareto analysis on problem...")

        # Force the correct objective function
        self.mdl_p['assignment_model_obj'] = 'Global Utility'

        # Initialize arrays
        num_points = int(100 / self.mdl_p["pareto_step"] + 1)
        cadet_overall_utilities = np.zeros(num_points)
        afsc_overall_utilities = np.zeros(num_points)
        cadet_overall_weights = np.arange(1, 0, -(self.mdl_p["pareto_step"] / 100))
        cadet_overall_weights = np.append(cadet_overall_weights, 0)
        solutions = {}

        # Iterate over the number of points needed for the Pareto Chart
        for point in range(num_points):
            w = cadet_overall_weights[point]

            # Update global utility matrix
            self.value_parameters['global_utility'] = np.zeros([self.parameters['N'], self.parameters['M'] + 1])
            for j in self.parameters['J']:
                self.value_parameters['global_utility'][:, j] = w * self.parameters['cadet_utility'][:, j] + \
                                                                (1 - w) * self.parameters['afsc_utility'][:, j]

            if printing:
                print("Calculating point " + str(point + 1) + " out of " + str(num_points) + "...")

            # Build & solve the model
            model = afccp.core.solutions.optimization.assignment_model_build(self)
            solution = afccp.core.solutions.optimization.solve_pyomo_model(self, model, "Assignment", printing=False)
            solution = afccp.core.solutions.handling.evaluate_solution(solution, self.parameters, self.value_parameters)
            solution_name = str(round(cadet_overall_weights[point], 4))

            # Extract solution information
            solutions[solution_name] = solution['afsc_array']
            cadet_overall_utilities[point] = solution['cadet_utility_overall']
            afsc_overall_utilities[point] = solution['afsc_utility_overall']

            if printing:
                print('For an overall weight on cadets of ' + str(cadet_overall_weights[point]) +
                      ', calculated utility on cadets: ' + str(round(cadet_overall_utilities[point], 2)) +
                      ', utility on afscs: ' + str(round(afsc_overall_utilities[point], 2)) +
                      ', and a global utility (z^gu) of ' + str(round(solution['z^gu'], 2)) + '.')

        # Obtain Dataframes
        pareto_df = pd.DataFrame(
            {'Weight on Cadets': cadet_overall_weights, 'Utility on Cadets': cadet_overall_utilities,
             'Utility on AFSCs': afsc_overall_utilities})
        solutions_df = pd.DataFrame(solutions)

        # File we import and export to
        filepath = self.export_paths['Analysis & Results'] + self.data_name + " " + self.vp_name + " (" + \
                   self.data_version + ") Pareto Analysis (Utility).xlsx"
        with pd.ExcelWriter(filepath) as writer:  # Export to excel
            pareto_df.to_excel(writer, sheet_name="Utility Pareto Results", index=False)
            solutions_df.to_excel(writer, sheet_name="Initial Solutions", index=False)

    def show_pareto_chart(self, printing=None, utility_version=True, solution_names=None, folder="Results Charts"):
        """
        Saves the pareto chart to the figures folder
        """

        if printing is None:
            printing = self.printing

        if printing:
            print("Creating Pareto Chart...")

        if utility_version:
            l_word = 'Utility'

            # File we import and export to
            filepath = self.export_paths['Analysis & Results'] + self.data_name + " " + self.vp_name + " (" + \
                       self.data_version + ") Pareto Analysis (Utility).xlsx"

            pareto_df = afccp.core.globals.import_data(filepath, sheet_name='Utility Pareto Results')
        else:
            l_word = 'Value'

            # File we import and export to
            filepath = self.export_paths['Analysis & Results'] + self.data_name + " " + self.vp_name + " (" + \
                       self.data_version + ") Pareto Analysis.xlsx"

            try:
                pareto_df = afccp.core.globals.import_data(filepath, sheet_name='GA Pareto Results')
            except:
                try:
                    pareto_df = afccp.core.globals.import_data(filepath, sheet_name='Approximate Pareto Results')
                except:
                    raise ValueError("No Pareto Data found for instance '" + self.data_name + "'")

        return afccp.core.visualizations.charts.pareto_graph(self, pareto_df, l_word=l_word, solution_names=solution_names,
                                                             filepath=self.export_paths['Analysis & Results'] + folder + '/')

    def what_if_analysis(self, p_dict={}, printing=None):
        """
        This method takes in an AFSC/cadet problem instance and performs some "What If" analysis based on the items listed
        in "What If List.csv". We manipulate the "value parameters" to meet these pre-defined conditions and then evaluate
        the model with the new constraints. We can then create a pareto frontier by modifying the weights on cadets/AFSCs.
        These results are all exported to a sub-folder called "What If" in the Analysis & Results folder.
        """
        if printing is None:
            printing = self.printing

        self.error_checking("Pyomo Model")

        # Reset instance model parameters
        self.reset_functional_parameters(p_dict)

        if "What If List.csv" not in os.listdir(self.export_paths['Analysis & Results']):
            raise ValueError("Error. File 'What If List.csv' required for this analysis. It needs to be located"
                             " in the 'Analysis & Results' folder.")

        if printing:
            print("Conducting 'What If?' analysis on this problem instance...")

        # Run the "what if" analysis function
        afccp.core.solutions.sensitivity.optimization_what_if_analysis(self, printing)

    def solve_pgl_capacity_sensitivity(self, p_dict={}, printing=None):
        """
        Docstring
        """

        if printing is None:
            printing = self.printing

        self.error_checking("Pyomo Model")

        # Adjust instance plot parameters
        self.reset_functional_parameters(p_dict)
        self.mdl_p = afccp.core.data.support.determine_afsc_plot_details(self, results_chart=True)

        # Run the function
        afccp.core.solutions.sensitivity.solve_pgl_capacity_sensitivity(self, p_dict, printing)

    def generate_pgl_capacity_charts(self, p_dict={}, printing=None):
        """
        Docstring
        """

        if printing is None:
            printing = self.printing

        # Adjust instance plot parameters
        self.reset_functional_parameters(p_dict)
        self.mdl_p = afccp.core.data.support.determine_afsc_plot_details(self, results_chart=True)

        # Run the function
        afccp.core.solutions.sensitivity.generate_pgl_capacity_charts(self, p_dict, printing)

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
            datasets = ["Cadets", "AFSCs", "Preferences", "Goal Programming", "Value Parameters",
                        "Solutions", "Additional", "Base Solutions", "Course Solutions"]
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
                                 "Preferences": afccp_dp.export_afsc_cadet_matrices_data,
                                 "Value Parameters": afccp_dp.export_value_parameters_data,
                                 "Solutions": afccp_dp.export_solutions_data,
                                 "Additional": afccp_dp.export_additional_data}
        for dataset in dataset_function_dict:
            if dataset in datasets:
                dataset_function_dict[dataset](self)

        # Goal Programming dataframe is an easy export (dataframe is already constructed)
        if "Goal Programming" in datasets and self.gp_df is not None:
            self.gp_df.to_csv(self.export_paths["Goal Programming"], index=False)

    def export_solution_results(self, printing=None):
        """
        This function exports the metrics for one solution back to excel for review
        """
        if printing is None:
            printing = self.printing

        # Make sure we have a solution
        self.error_checking('Solution')

        # Create solution folder if necessary
        self.manage_solution_folder()

        # Filepath to export to
        filename = self.data_name + " " + self.solution_name + " (" + self.vp_name + ").xlsx"
        filepath = self.export_paths['Analysis & Results'] + self.solution_name + '/' + filename

        # Print statement
        if printing:
            print("Exporting solution", self.solution_name, "results to " + filepath + "...")

        # Export results
        afccp.core.data.processing.export_solution_results_excel(self, filepath)

        if printing:
            print("Done.")



