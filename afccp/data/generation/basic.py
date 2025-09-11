"""
`afccp.data.generation.basic`
=============================

Provides foundational random instance and parameter generation functions for the AFCCP framework.
This module is designed to simulate realistic cadet–AFSC assignment problem instances, including
utility structures, quotas, preferences, and optional extensions (bases, training, CASTLE-level curves).

Main Capabilities
-----------------
- **Instance generation** (`generate_random_instance`):
  Creates randomized cadet/AFSC datasets with merit scores, quotas, eligibility tiers, and preference matrices.
  Supports constraints such as "NRL only" generation or inclusion of extra base/training components.

- **Value parameter generation** (`generate_random_value_parameters`):
  Produces randomized Value-Focused Thinking (VFT) objective weights, targets, and value functions.
  Uses AFCCP's `values` submodule to build piecewise-linear approximations of non-linear functions.

- **Extra component generation** (`generate_extra_components`):
  Adds bases, base capacities, training preferences, and course schedules for more complex problem variants.

- **CASTLE integration** (`generate_realistic_castle_value_curves`):
  Generates concave utility curves for CASTLE-level AFSC groupings to support strategic simulations.

Dependencies
------------
- **afccp.data.preferences** — Builds cadet preference lists and utilities.
- **afccp.data.adjustments** — Adds parameter set modifications for alignment with AFCCP models.
- **afccp.data.values** — Creates weight functions and value function breakpoints.
- **afccp.data.support** — Provides shared helper utilities for data preparation.
- **numpy**, **random**, **datetime**, **string**, **copy** — For stochastic generation and data shaping.

Use Cases
---------
- Rapid prototyping of AFCCP algorithms without relying on sensitive or incomplete real-world data.
- Benchmarking and stress-testing optimization methods.
- Building reproducible examples for tutorials, workshops, and documentation.

See Also
--------
- [`data.preferences`](../../../afccp/reference/data/preferences/#data.preferences)
- [`data.adjustments`](../../../afccp/reference/data/adjustments/#data.adjustments)
- [`data.values`](../../../afccp/reference/data/values/#data.values)
- [`data.support`](../../../afccp/reference/data/support/#data.support)
"""
import random
import numpy as np
import string
import copy
import datetime
import warnings
warnings.filterwarnings('ignore')  # prevent red warnings from printing

# afccp modules
import afccp.globals
import afccp.data.preferences
import afccp.data.adjustments
import afccp.data.values
import afccp.data.support


# _______________________________________________BASIC FUNCTIONS________________________________________________________
def generate_random_instance(N=1600, M=32, P=6, S=6, generate_only_nrl=False, generate_extra=False):
    """
    This procedure takes in the specified parameters (defined below) and then simulates new random "fixed" cadet/AFSC
    input parameters. These parameters are then returned and can be used to solve the VFT model.
    :param N: number of cadets
    :param M: number of AFSCs
    :param P: number of preferences allowed
    :param S: number of Bases
    :param generate_only_nrl: Only generate NRL AFSCs (default to False)
    :param generate_extra: Whether to generate extra components (bases/IST). Defaults to False.
    :return: model fixed parameters
    """

    # Initialize parameter dictionary
    # noinspection PyDictCreation
    p = {'N': N, 'P': P, 'M': M, 'num_util': P, 'cadets': np.arange(N), 'I': np.arange(N), 'J': np.arange(M),
         'usafa': np.random.choice([0, 1], size=N, p=[2 / 3, 1 / 3]), 'merit': np.random.rand(N)}

    # Generate various features of the cadets
    p['merit_all'] = p['merit']
    p['assigned'] = np.array(['' for _ in range(N)])
    p['soc'] = np.array(['USAFA' for _ in range(p['N'])])
    p['soc'][np.where(p['usafa'] == 0)[0]] = 'ROTC'

    # Calculate quotas for each AFSC
    p['pgl'], p['usafa_quota'], p['rotc_quota'] = np.zeros(M), np.zeros(M), np.zeros(M)
    p['quota_min'], p['quota_max'] = np.zeros(M), np.zeros(M)
    p['quota_e'], p['quota_d'] = np.zeros(M), np.zeros(M)
    for j in range(M):

        # Get PGL target
        p['pgl'][j] = max(10, np.random.normal(1000 / M, 100))

    # Scale PGL and force integer values and minimum of 1
    p['pgl'] = np.around((p['pgl'] / np.sum(p['pgl'])) * N * 0.8)
    indices = np.where(p['pgl'] == 0)[0]
    p['pgl'][indices] = 1

    # Sort PGL by size
    p['pgl'] = np.sort(p['pgl'])[::-1]

    # USAFA/ROTC Quotas
    p['usafa_quota'] = np.around(np.random.rand(M) * 0.3 + 0.1 * p['pgl'])
    p['rotc_quota'] = p['pgl'] - p['usafa_quota']

    # Min/Max
    p['quota_min'], p['quota_max'] = p['pgl'], np.around(p['pgl'] * (1 + np.random.rand(M) * 0.9))

    # Target is a random integer between the minimum and maximum targets
    target = np.around(p['quota_min'] + np.random.rand(M) * (p['quota_max'] - p['quota_min']))
    p['quota_e'], p['quota_d'] = target, target

    # Generate AFSCs
    p['afscs'] = np.array(['R' + str(j + 1) for j in range(M)])

    # Determine what "accessions group" each AFSC is in
    if generate_only_nrl:
        p['acc_grp'] = np.array(["NRL" for _ in range(M)])
    else:

        # If there are 3 or more AFSCs, we want all three accessions groups represented
        if M >= 3:
            invalid = True
            while invalid:

                # If we have 6 or fewer, limit USSF to just one AFSC
                if M <= 6:
                    p['acc_grp'] = ['USSF']
                    for _ in range(M - 1):
                        p['acc_grp'].append(np.random.choice(['NRL', 'Rated']))
                else:
                    p['acc_grp'] = [np.random.choice(['NRL', 'Rated', 'USSF']) for _ in range(M)]

                # Make sure we have at least one AFSC from each accession's group
                invalid = False  # "Innocent until proven guilty"
                for grp in ['NRL', 'Rated', 'USSF']:
                    if grp not in p['acc_grp']:
                        invalid = True
                        break

                # If we have 4 or more AFSCs, make sure we have at least two Rated
                if M >= 4:
                    if p['acc_grp'].count('Rated') < 2:
                        invalid = True
            p['acc_grp'] = np.array(p['acc_grp'])  # Convert to numpy array

        # If we only have one or two AFSCs, they'll all be NRL
        else:
            p['acc_grp'] = np.array(["NRL" for _ in range(M)])

    # Add an "*" to the list of AFSCs to be considered the "Unmatched AFSC"
    p["afscs"] = np.hstack((p["afscs"], "*"))

    # Add degree tier qualifications to the set of parameters
    def generate_degree_tier_qualifications():
        """
        I made this nested function, so I could have a designated section to generate degree qualifications and such
        """

        # Determine degree tiers and qualification information
        p['qual'] = np.array([['P1' for _ in range(M)] for _ in range(N)])
        p['Deg Tiers'] = np.array([[' ' * 10 for _ in range(4)] for _ in range(M)])
        for j in range(M):

            if p['acc_grp'][j] == 'Rated':  # All Degrees eligible for Rated
                p['qual'][:, j] = np.array(['P1' for _ in range(N)])
                p['Deg Tiers'][j, :] = ['P = 1', 'I = 0', '', '']

                # Pick 20% of the cadets at random to be ineligible for this Rated AFSC
                indices = random.sample(list(np.arange(N)), k=int(0.2 * N))
                p['qual'][indices, j] = 'I2'
            else:
                # Determine what tiers to use on this AFSC
                if N < 100:
                    random_number = np.random.rand()
                    if random_number < 0.2:
                        tiers = ['M1', 'I2']
                        p['Deg Tiers'][j, :] = ['M = 1', 'I = 0', '', '']
                    elif 0.2 < random_number < 0.4:
                        tiers = ['D1', 'P2']
                        target_num = round(np.random.rand(), 2)
                        p['Deg Tiers'][j, :] = ['D > ' + str(target_num), 'P < ' + str(1 - target_num), '', '']
                    elif 0.4 < random_number < 0.6:
                        tiers = ['P1']
                        p['Deg Tiers'][j, :] = ['P = 1', '', '', '']
                    else:
                        tiers = ['M1', 'P2']
                        target_num = round(np.random.rand(), 2)
                        p['Deg Tiers'][j, :] = ['M > ' + str(target_num), 'P < ' + str(1 - target_num), '', '']
                else:
                    random_number = np.random.rand()
                    if random_number < 0.1:
                        tiers = ['M1', 'I2']
                        p['Deg Tiers'][j, :] = ['M = 1', 'I = 0', '', '']
                    elif 0.1 < random_number < 0.2:
                        tiers = ['D1', 'P2']
                        target_num = round(np.random.rand(), 2)
                        p['Deg Tiers'][j, :] = ['D > ' + str(target_num), 'P < ' + str(1 - target_num), '', '']
                    elif 0.2 < random_number < 0.3:
                        tiers = ['P1']
                        p['Deg Tiers'][j, :] = ['P = 1', '', '', '']
                    elif 0.3 < random_number < 0.4:
                        tiers = ['M1', 'P2']
                        target_num = round(np.random.rand(), 2)
                        p['Deg Tiers'][j, :] = ['M > ' + str(target_num), 'P < ' + str(1 - target_num), '', '']
                    elif 0.4 < random_number < 0.5:
                        tiers = ['M1', 'D2', 'P3']
                        target_num_1 = round(np.random.rand() * 0.7, 2)
                        target_num_2 = round(np.random.rand() * (1 - target_num_1) * 0.8, 2)
                        target_num_3 = round(1 - target_num_1 - target_num_2, 2)
                        p['Deg Tiers'][j, :] = ['M > ' + str(target_num_1), 'D > ' + str(target_num_2),
                                                'P < ' + str(target_num_3), '']
                    elif 0.5 < random_number < 0.6:
                        tiers = ['D1', 'D2', 'P3']
                        target_num_1 = round(np.random.rand() * 0.7, 2)
                        target_num_2 = round(np.random.rand() * (1 - target_num_1) * 0.8, 2)
                        target_num_3 = round(1 - target_num_1 - target_num_2, 2)
                        p['Deg Tiers'][j, :] = ['D > ' + str(target_num_1), 'D > ' + str(target_num_2),
                                                'P < ' + str(target_num_3), '']
                    elif 0.6 < random_number < 0.7:
                        tiers = ['M1', 'D2', 'I3']
                        target_num = round(np.random.rand(), 2)
                        p['Deg Tiers'][j, :] = ['M > ' + str(target_num), 'D < ' + str(1 - target_num), 'I = 0', '']
                    elif 0.7 < random_number < 0.8:
                        tiers = ['M1', 'P2', 'I3']
                        target_num = round(np.random.rand(), 2)
                        p['Deg Tiers'][j, :] = ['M > ' + str(target_num), 'P < ' + str(1 - target_num), 'I = 0', '']
                    else:
                        tiers = ['M1', 'D2', 'P3', 'I4']
                        target_num_1 = round(np.random.rand() * 0.7, 2)
                        target_num_2 = round(np.random.rand() * (1 - target_num_1) * 0.8, 2)
                        target_num_3 = round(1 - target_num_1 - target_num_2, 2)
                        p['Deg Tiers'][j, :] = ['M > ' + str(target_num_1), 'D > ' + str(target_num_2),
                                                'P < ' + str(target_num_3), 'I = 0']

                # Generate the tiers for the cadets
                c_tiers = np.random.randint(0, len(tiers), N)
                p['qual'][:, j] = np.array([tiers[c_tiers[i]] for i in range(N)])

        # NxM qual matrices with various features
        p["ineligible"] = (np.core.defchararray.find(p['qual'], "I") != -1) * 1
        p["eligible"] = (p["ineligible"] == 0) * 1
        for t in [1, 2, 3, 4]:
            p["tier " + str(t)] = (np.core.defchararray.find(p['qual'], str(t)) != -1) * 1
        p["mandatory"] = (np.core.defchararray.find(p['qual'], "M") != -1) * 1
        p["desired"] = (np.core.defchararray.find(p['qual'], "D") != -1) * 1
        p["permitted"] = (np.core.defchararray.find(p['qual'], "P") != -1) * 1

        # NEW: Exception to degree qualification based on CFM ranks
        p["exception"] = (np.core.defchararray.find(p['qual'], "E") != -1) * 1

        # Initialize information for AFSC degree tiers
        p["t_count"] = np.zeros(p['M']).astype(int)
        p["t_proportion"] = np.zeros([p['M'], 4])
        p["t_leq"] = (np.core.defchararray.find(p["Deg Tiers"], "<") != -1) * 1
        p["t_geq"] = (np.core.defchararray.find(p["Deg Tiers"], ">") != -1) * 1
        p["t_eq"] = (np.core.defchararray.find(p["Deg Tiers"], "=") != -1) * 1
        p["t_mandatory"] = (np.core.defchararray.find(p["Deg Tiers"], "M") != -1) * 1
        p["t_desired"] = (np.core.defchararray.find(p["Deg Tiers"], "D") != -1) * 1
        p["t_permitted"] = (np.core.defchararray.find(p["Deg Tiers"], "P") != -1) * 1

        # Loop through each AFSC
        for j, afsc in enumerate(p["afscs"][:p['M']]):

            # Loop through each potential degree tier
            for t in range(4):
                val = p["Deg Tiers"][j, t]

                # Empty degree tier
                if 'M' not in val and 'D' not in val and 'P' not in val and 'I' not in val:
                # if val in ["nan", "", ''] or pd.isnull(val):
                    t -= 1
                    break

                # Degree Tier Proportion
                p["t_proportion"][j, t] = val.split(" ")[2]

            # Num tiers
            p["t_count"][j] = t + 1

        return p   # Return updated parameters
    p = generate_degree_tier_qualifications()

    # Cadet preferences
    utility = np.random.rand(N, M)  # random utility matrix
    max_util = np.max(utility, axis=1)
    p['utility'] = np.around(utility / np.array([[max_util[i]] for i in range(N)]), 2)

    # Get cadet preferences
    p["c_pref_matrix"] = np.zeros([p["N"], p["M"]]).astype(int)
    for i in range(p['N']):

        # Sort the utilities to get the preference list
        utilities = p["utility"][i, :p["M"]]
        sorted_indices = np.argsort(utilities)[::-1]
        preferences = np.argsort(
            sorted_indices) + 1  # Add 1 to change from python index (at 0) to rank (start at 1)
        p["c_pref_matrix"][i, :] = preferences

    # Create the "column data" preferences and utilities
    p['c_preferences'], p['c_utilities'] = afccp.data.preferences.update_cadet_columns_from_matrices(p)
    p['c_preferences'] = p['c_preferences'][:, :P]
    p['c_utilities'] = p['c_utilities'][:, :P]

    # If we want to generate extra components to match with, we do so here
    if generate_extra:
        p['S'] = S
        p = generate_extra_components(p)

    # Update set of parameters
    p = afccp.data.adjustments.parameter_sets_additions(p)

    return p  # Return updated parameters


def generate_random_value_parameters(parameters, num_breakpoints=24):
    """
    Generate Random Value Parameters for a Cadet-AFSC Assignment Problem.

    This function constructs a randomized set of value-focused thinking (VFT) parameters for a given cadet-AFSC
    matching instance. These include AFSC weights, cadet weights, value function definitions, and constraint structures
    across defined objectives. It supports a mix of manually assigned logic and randomized components and can be
    used to simulate plausible input conditions for testing the assignment algorithm.

    Parameters
    ----------
    parameters : dict
        The problem instance parameters, including cadet/AFSC info, merit scores, eligibility, quotas, and utilities.
    num_breakpoints : int, optional
        Number of breakpoints to use in piecewise linear value functions, by default 24.

    Returns
    -------
    dict
        A dictionary `vp` containing generated value parameters, including objectives, weights, constraints,
        value functions, and breakpoints.

    Examples
    --------
    ```python
    vp = generate_random_value_parameters(parameters, num_breakpoints=16)
    ```

    See Also
    --------
    - [`generate_afocd_value_parameters`](../../../afccp/reference/data/values/#data.values.generate_afocd_value_parameters):
      Adds tiered AFOCD objectives and fills in default VFT structure for a given instance.
    - [`create_segment_dict_from_string`](../../../afccp/reference/data/values/#data.values.create_segment_dict_from_string):
      Parses string definitions into nonlinear segment dictionaries for value functions.
    - [`value_function_builder`](../../../afccp/reference/data/values/#data.values.value_function_builder):
      Linearizes nonlinear value functions using a fixed number of breakpoints.
    - [`cadet_weight_function`](../../../afccp/reference/data/values/#data.values.cadet_weight_function):
      Creates weights across cadets based on merit scores and function type.
    - [`afsc_weight_function`](../../../afccp/reference/data/values/#data.values.afsc_weight_function):
      Creates weights across AFSCs based on projected gains/losses and selected function type.
    """

    # Shorthand
    p = parameters

    # Objective to parameters lookup dictionary (if the parameter is in "p", we include the objective)
    objective_lookups = {'Norm Score': 'a_pref_matrix', 'Merit': 'merit', 'USAFA Proportion': 'usafa',
                         'Combined Quota': 'quota_d', 'USAFA Quota': 'usafa_quota', 'ROTC Quota': 'rotc_quota',
                         'Utility': 'utility', 'Mandatory': 'mandatory',
                         'Desired': 'desired', 'Permitted': 'permitted'}
    for t in ["1", "2", "3", "4"]:  # Add in AFOCD Degree tiers
        objective_lookups["Tier " + t] = "tier " + t

    # Add the AFSC objectives that are included in this instance (check corresponding parameters using dict above)
    objectives = np.array([objective for objective in objective_lookups if objective_lookups[objective] in p])

    # Initialize set of value parameters
    vp = {'objectives': objectives, 'cadets_overall_weight': np.random.rand(), 'O': len(objectives),
          'K': np.arange(len(objectives)), 'num_breakpoints': num_breakpoints, 'cadets_overall_value_min': 0,
          'afscs_overall_value_min': 0}
    vp['afscs_overall_weight'] = 1- vp['cadets_overall_weight']

    # Generate AFSC and cadet weights
    weight_functions = ['Linear', 'Direct', 'Curve_1', 'Curve_2', 'Equal']
    vp['cadet_weight_function'] = np.random.choice(weight_functions)
    vp['afsc_weight_function'] = np.random.choice(weight_functions)
    vp['cadet_weight'] = afccp.data.values.cadet_weight_function(p['merit_all'], func= vp['cadet_weight_function'])
    vp['afsc_weight'] = afccp.data.values.afsc_weight_function(p['pgl'], func = vp['afsc_weight_function'])

    # Stuff that doesn't matter here
    vp['cadet_value_min'], vp['afsc_value_min'] = np.zeros(p['N']), np.zeros(p['N'])
    vp['USAFA-Constrained AFSCs'], vp['Cadets Top 3 Constraint'] = '', ''
    vp['USSF OM'] = False

    # Initialize arrays
    vp['objective_weight'], vp['objective_target'] = np.zeros([p['M'], vp['O']]), np.zeros([p['M'], vp['O']])
    vp['constraint_type'] = np.zeros([p['M'], vp['O']])
    vp['objective_value_min'] = np.array([[' ' * 20 for _ in vp['K']] for _ in p['J']])
    vp['value_functions'] = np.array([[' ' * 200 for _ in vp['K']] for _ in p['J']])

    # Initialize breakpoints
    vp['a'] = [[[] for _ in vp['K']] for _ in p["J"]]
    vp['f^hat'] = [[[] for _ in vp['K']] for _ in p["J"]]

    # Initialize objective set
    vp['K^A'] = {}

    # Get AFOCD Tier objectives
    vp = afccp.data.values.generate_afocd_value_parameters(p, vp)
    vp['constraint_type'] = np.zeros([p['M'], vp['O']])  # Turn off all the constraints again

    # Loop through all AFSCs
    for j in p['J']:

        # Loop through all AFSC objectives
        for k, objective in enumerate(vp['objectives']):

            maximum, minimum, actual = None, None, None
            if objective == 'Norm Score':
                vp['objective_weight'][j, k] = (np.random.rand() * 0.2 + 0.3) * 100  # Scale up to 100
                vp['value_functions'][j, k] = 'Min Increasing|0.3'
                vp['objective_target'][j, k] = 1

            if objective == 'Merit':
                vp['objective_weight'][j, k] = (np.random.rand() * 0.4 + 0.05) * 100
                vp['value_functions'][j, k] = 'Min Increasing|-0.3'
                vp['objective_target'][j, k] = p['sum_merit'] / p['N']
                actual = np.mean(p['merit'][p['I^E'][j]])

            elif objective == 'USAFA Proportion':
                vp['objective_weight'][j, k] = (np.random.rand() * 0.3 + 0.05) * 100
                vp['value_functions'][j, k] = 'Balance|0.15, 0.15, 0.1, 0.08, 0.08, 0.1, 0.6'
                vp['objective_target'][j, k] = p['usafa_proportion']
                actual = len(p['I^D'][objective][j]) / len(p['I^E'][j])

            elif objective == 'Combined Quota':
                vp['objective_weight'][j, k] = (np.random.rand() * 0.8 + 0.2) * 100
                vp['value_functions'][j, k] = 'Quota_Normal|0.2, 0.25, 0.2'
                vp['objective_target'][j, k] = p['quota_d'][j]

                # Get bounds and turn on this constraint
                minimum, maximum = p['quota_min'][j], p['quota_max'][j]
                vp['objective_value_min'][j, k] = str(int(minimum)) + ", " + str(int(maximum))
                vp['constraint_type'][j, k] = 2

            elif objective == 'USAFA Quota':
                vp['objective_weight'][j, k] = 0
                vp['value_functions'][j, k] = 'Min Increasing|0.3'
                vp['objective_target'][j, k] = p['usafa_quota'][j]

                # Bounds on this constraint (but leave it off)
                vp['objective_value_min'][j, k] = str(int(p['usafa_quota'][j])) + ", " + \
                                                  str(int(p['quota_max'][j]))

            elif objective == 'ROTC Quota':
                vp['objective_weight'][j, k] = 0
                vp['value_functions'][j, k] = 'Min Increasing|0.3'
                vp['objective_target'][j, k] = p['rotc_quota'][j]

                # Bounds on this constraint (but leave it off)
                vp['objective_value_min'][j, k] = str(int(p['rotc_quota'][j])) + ", " + \
                                                  str(int(p['quota_max'][j]))

            # If we care about this objective, we load in its value function breakpoints
            if vp['objective_weight'][j, k] != 0:

                # Create the non-linear piecewise exponential segment dictionary
                segment_dict = afccp.data.values.create_segment_dict_from_string(
                    vp['value_functions'][j, k], vp['objective_target'][j, k],
                    minimum=minimum, maximum=maximum, actual=actual)

                # Linearize the non-linear function using the specified number of breakpoints
                vp['a'][j][k], vp['f^hat'][j][k] = afccp.data.values.value_function_builder(
                    segment_dict, num_breakpoints=num_breakpoints)

        # Scale the objective weights for this AFSC, so they sum to 1
        vp['objective_weight'][j] = vp['objective_weight'][j] / sum(vp['objective_weight'][j])
        vp['K^A'][j] = np.where(vp['objective_weight'][j] != 0)[0]

    return vp


# __________________________________________BASE & TRAINING ADDITIONS___________________________________________________
def generate_extra_components(parameters):
    """
    Generate additional components (bases, training courses, and timing factors)
    for a CadetCareerProblem instance.

    This function augments the problem parameters with synthetic **bases** (locations),
    **base capacities**, **cadet base preferences**, **training courses**, and
    **training start distributions**. It also assigns weights to AFSC, base, and
    training preferences, enabling richer downstream optimization scenarios.

    Parameters
    ----------
    parameters : dict
        The problem parameter dictionary for a `CadetCareerProblem` instance.
        Must contain:
        - `M` : int
            Number of AFSCs.
        - `N` : int
            Number of cadets.
        - `S` : int
            Number of bases to generate.
        - `pgl` : np.ndarray
            PGL targets per AFSC.
        - `acc_grp` : np.ndarray
            Accession group labels per AFSC (e.g., "Rated", "USSF", "NRL").
        - `usafa` : np.ndarray
            Indicator for USAFA cadets.

    Returns
    -------
    dict
        Updated parameters dictionary with additional fields:
        - `afsc_assign_base` : np.ndarray
            Flags for AFSCs assigned to bases.
        - `bases` : np.ndarray
            Names of generated bases.
        - `base_min`, `base_max` : np.ndarray
            Min/max base capacities per AFSC.
        - `base_preferences` : dict
            Cadet-level base preference lists.
        - `b_pref_matrix`, `base_utility` : np.ndarray
            Matrices encoding cadet base preferences and utilities.
        - `baseline_date` : datetime.date
            Baseline date for training course scheduling.
        - `training_preferences`, `training_threshold`, `base_threshold` : np.ndarray
            Randomized cadet-level training/base thresholds and preferences.
        - `weight_afsc`, `weight_base`, `weight_course` : np.ndarray
            Weights for AFSC vs base vs course assignment importance.
        - `training_start` : np.ndarray
            Cadet training start dates (distribution differs for USAFA vs ROTC).
        - `courses`, `course_start`, `course_min`, `course_max` : dict
            Course identifiers, schedules, and capacities by AFSC.
        - `T` : np.ndarray
            Number of courses per AFSC.

    Workflow
    --------
    1. **Base Assignment**
        - Randomly selects which AFSCs require base-level assignments.
        - Generates base names from Excel-style column naming (`A`, `B`, ..., `AA`, etc.).
        - Distributes base capacities (`base_min`, `base_max`) across AFSCs.

    1. **Cadet Base Preferences**
        - Randomly assigns each cadet preferences over bases.
        - Generates a preference matrix (`b_pref_matrix`) and base utilities (`base_utility`).

    1. **Training Preferences**
        - Creates training preference labels (`Early` vs `Late`) and thresholds.
        - Allocates random weights for AFSC, base, and training course priorities.

    1. **Training Start Dates**
        - USAFA cadets start late May.
        - ROTC cadets follow a spring/late graduation distribution.

    1. **Training Courses**
        - Generates course identifiers (random strings of letters).
        - Randomizes start dates and max capacities.
        - Computes `T`, the number of courses per AFSC.

    Notes
    -----
    - `baseline_date` is set to **Jan 1 of the year after the current system year**.
    - Weights are normalized per cadet to sum to 1.
    - Utility values are randomized but ensure first-choice base has utility 1.0.

    Examples
    --------
    ```python
    p = {'M': 5, 'N': 100, 'S': 3,
         'pgl': np.array([10, 20, 15, 30, 25]),
         'acc_grp': np.array(["NRL", "Rated", "NRL", "USSF", "NRL"]),
         'usafa': np.random.randint(0, 2, size=100)}
    p = generate_extra_components(p)
    p.keys()
    ```

    Example Output:
    ```
    dict_keys([... 'bases', 'base_preferences', 'training_start', 'courses', 'T' ...])
    ```
    """

    # Shorthand
    p = parameters

    # Get list of ordered letters (based on Excel column names)
    alphabet = list(string.ascii_uppercase)
    excel_columns = copy.deepcopy(alphabet)
    for letter in alphabet:
        for letter_2 in alphabet:
            excel_columns.append(letter + letter_2)

    # Determine which AFSCs we assign bases for
    p['afsc_assign_base'] = np.zeros(p['M']).astype(int)
    for j in range(p['M']):
        if p['acc_grp'][j] != "Rated" and np.random.rand() > 0.3:
            p['afsc_assign_base'][j] = 1

    # Name the bases according to the Excel columns (just a method of generating unique ordered letters)
    p['bases'] = np.array(["Base " + excel_columns[b] for b in range(p['S'])])

    # Get capacities for each AFSC at each base
    p['base_min'] = np.zeros((p['S'], p['M'])).astype(int)
    p['base_max'] = np.zeros((p['S'], p['M'])).astype(int)
    afscs_with_base_assignments = np.where(p['afsc_assign_base'])[0]
    for j in afscs_with_base_assignments:
        total_max = p['pgl'][j] * 1.5
        base_max = np.array([np.random.rand() for _ in range(p['S'])])
        base_max = (base_max / np.sum(base_max)) * total_max
        p['base_max'][:, j] = base_max.astype(int)
        p['base_min'][:, j] = (base_max * 0.4).astype(int)

    # Generate random cadet preferences for bases
    bases = copy.deepcopy(p['bases'])
    p['base_preferences'] = {}
    p['b_pref_matrix'] = np.zeros((p['N'], p['S'])).astype(int)
    p['base_utility'] = np.zeros((p['N'], p['S']))
    for i in range(p['N']):
        random.shuffle(bases)
        num_base_pref = np.random.choice(np.arange(2, p['S'] + 1))
        p['base_preferences'][i] = np.array([np.where(p['bases'] == base)[0][0] for base in bases[:num_base_pref]])

        # Convert to base preference matrix
        p['b_pref_matrix'][i, p['base_preferences'][i]] = np.arange(1, len(p['base_preferences'][i]) + 1)

        utilities = np.around(np.random.rand(num_base_pref), 2)
        p['base_utility'][i, p['base_preferences'][i]] = np.sort(utilities)[::-1]
        p['base_utility'][i, p['base_preferences'][i][0]] = 1.0  # First choice is always utility of 1!

    # Get the baseline starting date (January 1st of the year we're classifying)
    next_year = datetime.datetime.now().year + 1
    p['baseline_date'] = datetime.date(next_year, 1, 1)

    # Generate training preferences for each cadet
    p['training_preferences'] = np.array(
        [random.choices(['Early', 'Late'], weights=[0.9, 0.1])[0] for _ in range(p['N'])])

    # Generate base/training "thresholds" for when these preferences kick in (based on preferences for AFSCs)
    p['training_threshold'] = np.array([np.random.choice(np.arange(p['M'] + 1)) for _ in range(p['N'])])
    p['base_threshold'] = np.array([np.random.choice(np.arange(p['M'] + 1)) for _ in range(p['N'])])

    # Generate weights for AFSCs, bases, and courses
    p['weight_afsc'], p['weight_base'], p['weight_course'] = np.zeros(p['N']), np.zeros(p['N']), np.zeros(p['N'])
    for i in range(p['N']):

        # Force some percentage of cadets to make their threshold the last possible AFSC (this means these don't matter)
        if np.random.rand() > 0.8:
            p['base_threshold'][i] = p['M']
        if np.random.rand() > 0.7:
            p['training_threshold'][i] = p['M']

        # Generate weights for bases, training (courses), and AFSCs
        if p['base_threshold'][i] == p['M']:
            w_b = 0
        else:
            w_b = np.random.triangular(0, 50, 100)
        if p['training_threshold'][i] == p['M']:
            w_c = 0
        else:
            w_c = np.random.triangular(0, 20, 100)
        w_a = np.random.triangular(0, 90, 100)

        # Scale weights so that they sum to one and load into arrays
        p['weight_afsc'][i] = w_a / (w_a + w_b + w_c)
        p['weight_base'][i] = w_b / (w_a + w_b + w_c)
        p['weight_course'][i] = w_c / (w_a + w_b + w_c)

    # Generate training start dates for each cadet
    p['training_start'] = []
    for i in range(p['N']):

        # If this cadet is a USAFA cadet
        if p['usafa'][i]:

            # Make it May 28th of this year
            p['training_start'].append(datetime.date(next_year, 5, 28))

        # If it's an ROTC cadet, we sample from two different distributions (on-time and late grads)
        else:

            # 80% should be in spring
            if np.random.rand() < 0.8:
                dt = datetime.date(next_year, 4, 15) + datetime.timedelta(int(np.random.triangular(0, 30, 60)))
                p['training_start'].append(dt)

            # 20% should be after
            else:
                dt = datetime.date(next_year, 6, 1) + datetime.timedelta(int(np.random.triangular(0, 30*5, 30*6)))
                p['training_start'].append(dt)
    p['training_start'] = np.array(p['training_start'])

    # Generate training courses for each AFSC
    p['courses'], p['course_start'], p['course_min'], p['course_max'] = {}, {}, {}, {}
    p['course_count'] = np.zeros(p['M'])
    for j in range(p['M']):

        # Determine total number of training slots to divide up
        total_max = p['pgl'][j] * 1.5

        # Determine number of courses to generate
        if total_max <= 3:
            T = 1
        elif total_max <= 10:
            T = np.random.choice([1, 2])
        elif total_max < 25:
            T = np.random.choice([2, 3])
        elif total_max < 100:
            T = np.random.choice([3, 4, 5])
        else:
            T = np.random.choice([4, 5, 6, 7, 8, 9])

        # Course minimums and maximums
        random_nums = np.random.rand(T)
        p['course_max'][j] = np.around(total_max * (random_nums / np.sum(random_nums))).astype(int)
        p['course_min'][j] = np.zeros(T).astype(int)

        # Generate course specific information
        p['courses'][j], p['course_start'][j] = [], []
        current_date = p['baseline_date'] + datetime.timedelta(int(np.random.triangular(30*5, 30*9, 30*11)))
        for _ in range(T):

            # Course names (random strings of letters)
            num_letters = random.choice(np.arange(4, 10))
            p['courses'][j].append(''.join(random.choices(alphabet, k=num_letters)))

            # Course start date
            p['course_start'][j].append(current_date)

            # Get next course start date
            current_date += datetime.timedelta(int(np.random.triangular(30, 30*4, 30*6)))

        # Convert to numpy arrays
        for param in ['courses', 'course_start', 'course_max', 'course_min']:
            p[param][j] = np.array(p[param][j])

    # Number of training courses per AFSC
    p['T'] = np.array([len(p['courses'][j]) for j in range(p['M'])])

    # Return updated parameters
    return p


# ______________________________________________CASTLE INTEGRATION______________________________________________________
def generate_concave_curve(num_points, max_x):
    """
    Generates x and y coordinates for a concave function.

    Args:
        num_points (int): Number of points to generate.
        max_x (float): Maximum value along the x-axis.

    Returns:
        tuple: (x_values, y_values) as numpy arrays.
    """
    x_values = np.linspace(0, max_x, num_points)
    y_values = 1 - np.exp(-x_values / (max_x / 6))  # Adjust curvature
    return x_values, y_values


def generate_realistic_castle_value_curves(parameters, num_breakpoints: int = 10):
    """
    Generate Concave Value Curves for CASTLE AFSCs.

    Creates piecewise linear approximations of realistic concave value functions for each CASTLE-level AFSC.
    These curves are used to evaluate the marginal utility of inventory across AFSCs, enabling smooth
    optimization and modeling in the CASTLE simulation.

    Parameters:
        parameters (dict): Problem instance parameters containing CASTLE AFSC groups and PGL values.
        num_breakpoints (int, optional): Number of breakpoints to use in the piecewise value curve.
            Defaults to 10.

    Returns:
        dict: A dictionary `q` containing the following keys for each CASTLE AFSC:
            - `'a'`: Array of x-values (inventory levels).
            - `'f^hat'`: Array of corresponding y-values (utility).
            - `'r'`: Number of breakpoints.
            - `'L'`: Index array of breakpoints.

    Example:
        ```python
        q = generate_realistic_castle_value_curves(parameters, num_breakpoints=12)
        x_vals = q['a']['21A']       # x-values for AFSC 21A
        y_vals = q['f^hat']['21A']   # corresponding utility values
        ```

    See Also:
        - [`generate_concave_curve`](../../../afccp/reference/data/generation/#data.generation.generate_concave_curve):
          Generates a concave (diminishing returns) curve with specified number of points and max range.
    """
    # Shorthand
    p = parameters

    # Define "q" dictionary for value function components
    q = {'a': {}, 'f^hat': {}, 'r': {}, 'L': {}}
    for afsc in p['castle_afscs']:
        # Sum up the PGL targets for all "AFPC" AFSCs grouped for this "CASTLE" AFSC
        pgl = np.sum(p['pgl'][p['J^CASTLE'][afsc]])

        # Generate x and y coordinates for concave shape
        x, y = generate_concave_curve(num_points=num_breakpoints, max_x=pgl * 2)

        # Save breakpoint information to q dictionary
        q['a'][afsc], q['f^hat'][afsc] = x, y
        q['r'][afsc], q['L'][afsc] = len(x), np.arange(len(x))

    return q
