# Import Libraries
import time
import numpy as np
import random
import copy
import afccp.core.globals
import afccp.core.solutions.handling

# Old useless algorithms
def greedy_model_solve(instance, printing=False):
    """
    This is a simple greedy algorithm that matches each cadet according to the highest valued AFSC determined by the
    initial overall utility matrix.
    """
    if printing:
        print('Solving Greedy Model...')

    # Shorthand
    p, vp = instance.parameters, instance.value_parameters

    # Create AFSC Utility Matrix
    if "afsc_utility" in p:
        afsc_utility = p["afsc_utility"]
    else:
        afsc_utility = np.zeros([p["N"], p["M"]])
        for objective in ['Merit', 'Mandatory', 'Desired', 'Permitted', 'Utility']:
            if objective in vp["objectives"]:
                k = np.where(vp["objectives"] == objective)[0][0]
                if objective == 'Merit':
                    merit = np.tile(np.where(p['merit'] > p['sum_merit'], 1,
                                             p['merit'] / p['sum_merit']), [p["M"], 1]).T
                    afsc_utility += merit * vp["objective_weight"][:, k].T
                else:
                    afsc_utility += p[objective.lower()][:, :p["M"]] * vp["objective_weight"][:, k].T

    # Overall Utility Matrix
    overall_utility = vp['afscs_overall_weight'] * (afsc_utility.T * vp['afsc_weight'][:, np.newaxis]).T + \
                      vp['cadets_overall_weight'] * p['cadet_utility'][:, :p["M"]] * vp['cadet_weight'][:, np.newaxis] + \
                      p['ineligible'] * -100

    # Begin Greedy Algorithm to match the cadets using overall utility matrix
    solution = np.zeros(p["N"])
    for i in p["I"]:

        # Match cadet to AFSC which adds the highest overall value
        solution[i] = np.argmax(overall_utility[i, :])

    return solution


# Matching algorithms
def matching_algorithm_1(instance, capacities=None, printing=True):
    """
    This is the Hospitals/Residents algorithm that matches cadets and AFSCs across all rated, space, and NRL positions.
    """
    if printing:
        print("Solving the deferred acceptance algorithm (1)...")

    # Shorthand
    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p

    # Algorithm initialization
    if capacities is None:
        total_slots = p[mdl_p['capacity_parameter']]
    else:  # In case this is used in a genetic algorithm
        total_slots = capacities

    # Array to keep track of what AFSC choice in their list the cadets are proposing to (python index at 0)
    cadet_proposal_choice = np.zeros(p['N']).astype(int)  # Everyone proposes to their first choice initially

    # Dictionary of parameters used for the "CadetBoardFigure" object (animation)
    if mdl_p['collect_solution_iterations']:
        solution_iterations = {'proposals': {}, 'solutions': {}, 'iteration_names': {}, 'type': 'MA1'}

    # Begin the simple Hospital/Residents Algorithm
    total_rejections = np.zeros(p['M'])  # Number of rejections for each AFSC
    total_matched = np.zeros(p['M'])  # Number of accepted cadets for each AFSC
    exhausted_cadets = []  # Will contain the cadets that have exhausted (been rejected by) all of their preferences
    iteration = 0  # First iteration of the algorithm
    while np.sum(total_matched) + len(exhausted_cadets) < p['N']:  # Stopping conditions

        # Cadets propose to their top choice that hasn't been rejected
        exhausted_cadets = np.where(cadet_proposal_choice >= p['num_cadet_choices'])[0]
        proposals = np.array([p['cadet_preferences'][i][cadet_proposal_choice[i]] if i not in exhausted_cadets
                              else p['M'] for i in p['I']])

        # Solution Iteration components (Proposals) and print statement
        if mdl_p['collect_solution_iterations']:
            solution_iterations['proposals'][iteration] = copy.deepcopy(proposals)
        if mdl_p['ma_printing']:
            print("\nIteration", iteration + 1)
            counts = {p['afscs'][j]: len(np.where(proposals == j)[0]) for j in p['J']}

        # Initialize matches information for this iteration
        total_matched = np.zeros(p['M'])

        # AFSCs accept their best cadets and reject the others
        for j in p['J']:

            # Loop through their preferred cadets from top to bottom
            iteration_rejections = 0
            for i in p['afsc_preferences'][j]:

                # If the cadet is proposing to this AFSC, we have two options
                if proposals[i] == j:

                    # We haven't hit capacity, so we accept this cadet
                    if total_matched[j] < total_slots[j]:
                        total_matched[j] += 1

                    # We're at capacity, so we reject this cadet
                    else:

                        # Essentially "delete" the preference from the cadet's list
                        cadet_proposal_choice[i] += 1
                        proposals[i] = p['M']  # index of the unmatched AFSC (*)

                        # Collect additional information
                        if mdl_p['ma_printing']:
                            iteration_rejections += 1
                            total_rejections[j] += 1

        # Solution Iteration components
        if mdl_p['collect_solution_iterations']:
            solution_iterations['solutions'][iteration] = copy.deepcopy(proposals)
            solution_iterations['iteration_names'][iteration] = 'Iteration ' + str(iteration + 1)

        # Specific matching algorithm print statement
        if mdl_p['ma_printing']:
            print('Proposals:', counts)
            print('Matched', {p['afscs'][j]: total_matched[j] for j in p['J']})
            print('Rejected', {p['afscs'][j]: total_rejections[j] for j in p['J']})

        iteration += 1 # Next iteration!

    # Last solution iteration
    solution_iterations['last_s'] = iteration - 1
    return proposals, solution_iterations


# SOC specific algorithms
def rotc_rated_board_original(instance, printing=False):
    """
    The function assigns Rated AFSCs to ROTC cadets based on their preferences and the existing quotas for each
    AFSC using the current rated board algorithm.
    """

    if printing:
        print("Running status quo ROTC rated algorithm...")

    # Shorthand
    p = instance.parameters

    # Cadets/AFSCs and their preferences
    cadet_indices = p['Rated Cadets']['rotc']  # indices of the cadets in the full set of cadets
    cadets, N = np.arange(len(cadet_indices)), len(cadet_indices)
    rated_J = np.array([j for j in p['J^Rated'] if '_U' not in p['afscs'][j]])
    afscs, M = p['afscs'][rated_J], len(rated_J)
    afsc_indices = np.array([np.where(p['afscs'] == afsc)[0][0] for afsc in afscs])
    afsc_om = {afscs[j]: p['rr_om_matrix'][:, j] for j in range(M)}  # list of OM percentiles for each AFSC
    afsc_interest = {afscs[j]: p['rr_interest_matrix'][:, j] for j in range(M)}  # list of Rated interest from cadets
    eligible = {afscs[j]: p['rr_om_matrix'][:, j] > 0 for j in range(M)}  # binary eligibility column for each AFSC

    # Dictionary of dictionaries of cadets within each order of merit "level" for each AFSC
    om_cl = {"High": (0.4, 1), "Med": (0.2, 0.4), "Low": (0.1, 0.2)}
    om = {afsc: {level: np.where((afsc_om[afsc] >= om_cl[level][0]) &
                                 (afsc_om[afsc] < om_cl[level][1]))[0] for level in om_cl} for afsc in afscs}

    # Dictionary of dictionaries of cadets with each interest "level" for each AFSC
    interest_levels = ["High", "Med", "Low", "None"]
    interest = {afsc: {level: np.where(afsc_interest[afsc] == level)[0] for level in interest_levels} for afsc in afscs}

    # Algorithm initialization
    total_slots = {afscs[idx]: p['rotc_quota'][j] for idx, j in enumerate(afsc_indices)}
    total_matched = {afsc: 0 for afsc in afscs}  # Number of matched cadets for each AFSC
    matching = {afsc: True for afsc in afscs}  # Used in stopping conditions
    assigned_afscs = {cadet: "" for cadet in cadets}

    # Dictionary of parameters used for the "CadetBoardFigure" object (animation)
    solution_iterations = {'solutions': {}, 'iteration_names': {}, 'type': 'ROTC Rated Board',
                           'cadets_solved_for': 'ROTC Rated', 'afscs_solved_for': 'Rated'}

    # Re-order AFSCs if necessary
    if instance.mdl_p['rotc_rated_board_afsc_order'] is not None:
        afscs = instance.mdl_p['rotc_rated_board_afsc_order']  # Need to be ordered list of ROTC Rated AFSCs

    # Phases of the Rated board where each tuple represents the level for (OM, Interest)
    phases = [("High", "High"), ("High", "Med"), ("Med", "High"), ("Med", "Med"), ("Low", "High"), ("High", "Low"),
              ("Low", "Med"), ("Med", "Low"), ("Low", "Low"), ("High", "None"), ("Med", "None"), ("Low", "None")]
    phase_num = 0
    s = 0  # for solution iterations
    while any(matching.values()) and phase_num < len(phases):
        phase = phases[phase_num]
        om_level, interest_level = phase[0], phase[1]
        phase_num += 1
        if printing:
            print("\nPhase", phase_num, om_level, "OM", "&", interest_level, "Interest")

        # Loop through each Rated AFSC
        for afsc in afscs:

            # Get the ordered list of cadets we're considering in this phase
            phase_cadets = np.intersect1d(om[afsc][om_level], interest[afsc][interest_level])
            om_phase_cadets = afsc_om[afsc][phase_cadets]
            indices = np.argsort(om_phase_cadets)[::-1]
            ordered_cadets = phase_cadets[indices]

            # Loop through each eligible cadet to assign them this AFSC if applicable
            eligible_cadets = np.where(eligible[afsc])[0]
            counter = 0
            for cadet in ordered_cadets:

                # The cadet has to be eligible for the AFSC to be considered
                if cadet not in eligible_cadets:
                    continue

                # If we didn't already assign them an AFSC, and we've still got open slots left, the cadet gets matched
                if assigned_afscs[cadet] == "" and total_matched[afsc] < total_slots[afsc]:
                    assigned_afscs[cadet] = afsc
                    total_matched[afsc] += 1
                    counter += 1

                # We've reached capacity for this AFSC
                if total_matched[afsc] == total_slots[afsc]:
                    matching[afsc] = False
                    break

            if counter != 0 and printing:
                print(afsc, "Phase Matched:", counter, "  --->   Total Matched:", total_matched[afsc], "/",
                      total_slots[afsc])

            # Solution iteration components (convert to full solution)
            afsc_solution = np.array([" " * 10 for _ in p['I']])
            for cadet, i in enumerate(cadet_indices):
                if assigned_afscs[cadet] in afscs:
                    afsc_solution[i] = assigned_afscs[cadet]
            indices = np.where(afsc_solution == " " * 10)[0]
            afsc_solution[indices] = "*"
            solution_iterations['solutions'][s] = np.array([np.where(p['afscs'] == afsc)[0][0] for afsc in afsc_solution])
            solution_iterations['iteration_names'][s] = \
                "Phase " + str(phase_num) + " (" + om_level + ", " + interest_level + ") [" + afsc + "]"
            s += 1

    solution_iterations['last_s'] = s - 1

    # Convert it back to a full solution with all cadets (anyone not matched to a Rated AFSC is unmatched)
    afsc_solution = np.array([" " * 10 for _ in p['I']])
    for cadet, i in enumerate(cadet_indices):
        if assigned_afscs[cadet] in afscs:
            afsc_solution[i] = assigned_afscs[cadet]
    indices = np.where(afsc_solution == " " * 10)[0]
    afsc_solution[indices] = "*"
    solution = np.array([np.where(p['afscs'] == afsc)[0][0] for afsc in afsc_solution])
    return solution, solution_iterations


def soc_rated_matching_algorithm(instance, soc='usafa', printing=True):
    """
    This is the Hospitals/Residents algorithm that matches or reserves cadets to their Rated AFSCs based on the SOC.
    """
    if printing:
        print("Solving the rated matching algorithm for " + soc.upper() + " cadets...")

    # Shorthand
    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p

    # Slight change to Rated AFSCs (Remove SOC specific slots)
    if soc == 'usafa':
        rated_J = np.array([j for j in p['J^Rated'] if '_R' not in p['afscs'][j]])
    else:
        rated_J = np.array([j for j in p['J^Rated'] if '_U' not in p['afscs'][j]])

    # Algorithm initialization
    total_slots = {j: p[soc + "_quota"][j] for j in rated_J}
    cadets = p['Rated Cadets'][soc]
    N = len(cadets)

    # "REAL" first choice of the cadet
    first_choice = {i: p['cadet_preferences'][i][0] for i in cadets}

    # Dictionary to keep track of what AFSC choice in their list the cadets are proposing to
    cadet_proposal_choice = {i: 0 for i in cadets}  # Initially all propose to their top Rated preference!

    # Dictionary of parameters used for the "CadetBoardFigure" object (animation)
    if mdl_p['collect_solution_iterations']:
        solution_iterations = {'proposals': {}, 'solutions': {}, 'matches': {}, 'reserves': {}, 'iteration_names': {},
                               'type': 'Rated SOC HR', 'cadets_solved_for': soc.upper() + ' Rated',
                               'afscs_solved_for': 'Rated'}

    # Begin the simple Hospital/Residents Algorithm
    total_rejections = {j: 0 for j in rated_J}  # Number of rejections for each AFSC
    total_matched = {j: 0 for j in rated_J}  # Number of accepted cadets for each AFSC
    exhausted_cadets = []  # Will contain the cadets that have exhausted (been rejected by) all of their preferences
    iteration = 0  # First iteration of the algorithm
    while sum([total_matched[j] for j in rated_J]) + len(exhausted_cadets) < N:  # Stopping conditions

        # Cadets propose to their top choice that hasn't been rejected
        proposals = {i: p['Rated Choices'][
            soc][i][cadet_proposal_choice[i]] if i not in exhausted_cadets else p['M'] for i in cadets}
        proposal_array = np.array([proposals[i] if i in cadets else p['M'] for i in p['I']])
        counts = {p['afscs'][j]: len(np.where(proposal_array == j)[0]) for j in rated_J}

        # Solution Iteration components (Proposals) and print statement
        if mdl_p['collect_solution_iterations']:
            solution_iterations['proposals'][iteration] = proposal_array
        if mdl_p['ma_printing']:
            print("\nIteration", iteration + 1)

        # Initialize matches information for this iteration
        total_matched = {j: 0 for j in rated_J}

        # AFSCs accept their best cadets and reject the others
        for j in rated_J:

            # Loop through their preferred cadets from top to bottom
            iteration_rejections = 0
            for i in p['afsc_preferences'][j]:
                if i not in cadets:
                    continue  # Other SOC (we don't care about them right now)

                # If the cadet is proposing to this AFSC, we have two options
                if proposals[i] == j:

                    # We haven't hit capacity, so we accept this cadet
                    if total_matched[j] < total_slots[j]:
                        total_matched[j] += 1

                    # We're at capacity, so we reject this cadet
                    else:

                        # Essentially "delete" the preference from the cadet's list
                        cadet_proposal_choice[i] += 1
                        proposals[i] = p['M']  # index of the unmatched AFSC (*)

                        # Collect additional information
                        if mdl_p['ma_printing']:
                            iteration_rejections += 1
                            total_rejections[j] += 1

        # Solution Iteration components
        if mdl_p['collect_solution_iterations']:

            # Rated "solution" (containing matched and reserved cadets)
            solution_array = np.array([proposals[i] if i in cadets else p['M'] for i in p['I']])
            matches, reserves = [], []
            for i in cadets:
                if proposals[i] in rated_J:
                    if first_choice[i] == proposals[i]:
                        matches.append(i)
                    else:
                        reserves.append(i)
            solution_iterations['matches'][iteration] = np.array(matches)  # List of cadets that are matched
            solution_iterations['reserves'][iteration] = np.array(reserves)  # List of cadets that have reserved slots
            solution_iterations['solutions'][iteration] = copy.deepcopy(solution_array)
            solution_iterations['iteration_names'][iteration] = 'Iteration ' + str(iteration + 1)

        # Specific matching algorithm print statement
        if mdl_p['ma_printing']:
            print('Proposals:', counts)
            print('Matched', {p['afscs'][j]: total_matched[j] for j in p['J']})
            print('Rejected', {p['afscs'][j]: total_rejections[j] for j in p['J']})

        # Check exhausted cadets
        exhausted_cadets = []
        for i in cadets:
            if cadet_proposal_choice[i] >= p['Num Rated Choices'][soc][i]:
                exhausted_cadets.append(i)

        iteration += 1 # Next iteration!

    # Last solution iteration
    solution_iterations['last_s'] = iteration - 1

    # Final solution (matched cadets) with also another "solution" with reserved cadets
    solution = np.zeros(p['N']).astype(int)
    reserves = np.zeros(p['N']).astype(int)
    for i in p['I']:
        solution[i], reserves[i] = p['M'], p['M']  # Default to unmatched
        if i in cadets:
            if proposals[i] in rated_J:
                if first_choice[i] == proposals[i]:
                    solution[i] = proposals[i]
                else:
                    reserves[i] = proposals[i]

    # Return solution, reserved array, and solution iterations
    return solution, reserves, solution_iterations


# Meta-heuristics
def genetic_algorithm(instance, initial_solutions=None, con_fail_dict=None, printing=False):
    """
    This is the genetic algorithm. The hyper-parameters to the algorithm can be tuned, and this is meant to be
    solved in conjunction with the pyomo model solutions. Use several VFT pyomo solutions in the initial population to
    this genetic algorithm
    """

    def multi_point_crossover(genome1, genome2):
        """
        Take two parent genomes, crossover the genes at multiple points and return two offspring solutions
        :param genome1: first parent genome
        :param genome2: second parent genome
        :return: offspring
        """
        points = np.sort(np.random.choice(crossover_positions, size=mp["num_crossover_points"], replace=False))
        start_points = np.append(0, points)
        stop_points = np.append(points, p['N'] - 1)
        child1 = np.zeros(p['N']).astype(int)
        child2 = np.zeros(p['N']).astype(int)
        flip = 1
        for i in range(len(start_points)):
            if flip == 1:
                child1[start_points[i]:stop_points[i] + 1] = genome2[start_points[i]:stop_points[i] + 1]
                child2[start_points[i]:stop_points[i] + 1] = genome1[start_points[i]:stop_points[i] + 1]
            else:
                child1[start_points[i]:stop_points[i] + 1] = genome1[start_points[i]:stop_points[i] + 1]
                child2[start_points[i]:stop_points[i] + 1] = genome2[start_points[i]:stop_points[i] + 1]

            flip = flip * -1

        return child1, child2

    def mutation(genome):
        """
        Takes a genome, and picks a random cadet index to mutate with some probability.
        This means we can swap an AFSC for one cadet individually
        :param genome: solution vector
        :return: mutated genome
        """
        for _ in range(mp["num_mutations"]):
            i = np.random.randint(low=0, high=p['N'])  # Random cadet
            j = np.random.choice(p['J^E'][i])  # Random AFSC
            genome[i] = j if (np.random.uniform() < mp["mutation_rate"]) else genome[i]

        return genome

    # Shorthand
    p = instance.parameters
    vp = instance.value_parameters
    mp = instance.mdl_p

    # Rank Selection Parameters
    rank_weights = (np.arange(1, mp["pop_size"] + 1)[::-1]) ** 1.2
    rank_weights = rank_weights / sum(rank_weights)
    rank_choices = np.arange(mp["pop_size"])

    # Multi-Point Crossover Parameters
    crossover_positions = np.arange(1, p['N'] - 1)

    # Initialize Population
    population = np.array([[np.random.choice(p['J^E'][i]) for i in p['I']] for _ in range(mp["pop_size"])]).astype(int)
    if initial_solutions is not None:

        # Get fitness of initial_solutions in case there are feasibility issues or there are too many initial solutions
        num_initial = len(initial_solutions)
        fitness = np.zeros(num_initial)
        for s, chromosome in enumerate(initial_solutions):
            fitness[s] = afccp.core.solutions.handling.fitness_function(chromosome, p, vp, mp, con_fail_dict)

        # Sort Initial solutions by Fitness
        sorted_indices = fitness.argsort()[::-1]
        initial_solutions = initial_solutions[sorted_indices]

        # Insert these solutions into the population
        for s, chromosome in enumerate(initial_solutions):

            # Make sure there aren't too many initial solutions
            if s < mp["pop_size"]:
                population[s] = chromosome

    # Initialize Fitness Scores
    fitness = np.zeros(mp["pop_size"])
    for s, chromosome in enumerate(population):
        fitness[s] = afccp.core.solutions.handling.fitness_function(chromosome, p, vp, mp, con_fail_dict)

    # Sort Population by Fitness
    sorted_indices = fitness.argsort()[::-1]
    fitness = fitness[sorted_indices]
    population = population[sorted_indices]

    # Print updates
    if mp["ga_printing"]:
        step = mp["ga_max_time"] / (100 / mp["percent_step"])
        checkpoint = step
        completed = 0
        print('Initial Fitness Scores', fitness)

    # Time Evaluation Initialization
    if mp["time_eval"]:
        time_step = mp["ga_max_time"] / mp["num_time_points"]
        step_num = 1
        times = [0]
        scores = [fitness[0]]

    # Main Loop
    start_time = time.perf_counter()
    eval_times = []
    gen_times = []
    generating = True
    generation = 1
    while generating:

        # Start time
        gen_start_time = time.perf_counter()

        # Evaluate Population
        for index in range(2, mp["pop_size"]):
            fitness[index] = afccp.core.solutions.handling.fitness_function(population[index], p, vp, mp, con_fail_dict)
        eval_times.append(time.perf_counter() - gen_start_time)

        # Sort Population by Fitness
        sorted_indices = fitness.argsort()[::-1]
        fitness = fitness[sorted_indices]
        population = population[sorted_indices]
        best_score = fitness[0]

        # Printing updates
        if mp["ga_printing"]:
            if (time.perf_counter() - start_time) > checkpoint:
                completed += mp["percent_step"]
                checkpoint += step
                print(str(completed) + "% complete. Best solution value: " + str(round(best_score, 4)))
                avg_eval_time = np.mean(np.array(eval_times[0:generation]))
                print('Average evaluation time for ' + str(mp["pop_size"]) + ' solutions: ' +
                      str(round(avg_eval_time, 4)) + ' seconds.')
                avg_gen_time = np.mean(np.array(gen_times[0:generation]))
                print('Average generation time: ' + str(round(avg_gen_time, 4)) + ' seconds.')

        # Create next generation
        next_generation = population[0:2]  # the best two solutions are kept for the next generation
        for twins in range(int((mp["pop_size"] / 2) - 1)):  # create the offspring

            # Select parents for mating
            index_1, index_2 = np.random.choice(rank_choices, size=2, replace=False, p=rank_weights)
            parent_1, parent_2 = population[index_1], population[index_2]

            # Apply crossover function
            offspring_1, offspring_2 = multi_point_crossover(parent_1, parent_2)

            # Mutate genomes of offspring
            offspring_1 = mutation(offspring_1)
            offspring_2 = mutation(offspring_2)

            # Add this pair to the next generation
            offsprings = np.vstack((offspring_1, offspring_2))
            next_generation = np.vstack((next_generation, offsprings))

        # Time Eval
        if mp["time_eval"]:
            if (time.perf_counter() - start_time) > (time_step * step_num):
                times.append(time.perf_counter() - start_time)
                scores.append(best_score)
                step_num += 1

        # Check stopping criteria
        if (time.perf_counter() - start_time) > mp["ga_max_time"]:
            if mp["ga_printing"]:
                end_time = round(time.perf_counter() - start_time, 2)
                print('End time reached in ' + str(end_time) + ' seconds.')
            generating = False

        # Next Generation
        population = next_generation
        gen_times.append(time.perf_counter() - gen_start_time)
        generation += 1

    # Time Eval
    if mp["time_eval"]:

        # Create time_eval_df
        time_eval_df = pd.DataFrame({'Time': times, 'Objective Value': scores})

        if printing:
            print(time_eval_df)
        return population[0], time_eval_df

    else:

        # Return best solution
        return population[0], None


def genetic_matching_algorithm(instance, printing=False):
    """
    CHatGPT GMA docstring
    """

    # Shorthand
    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p

    # Define functions
    def initialize_population():
        """
        Function to initialize all the "chromosomes" for this GA
        """
        population = np.array([np.zeros(p['M']) for _ in range(mdl_p['gma_pop_size'])])
        for c in range(mdl_p['gma_pop_size']):
            for j in p['J']:
                capacity = int(random.choice(capacity_options[j]))
                population[c, j] = capacity

        return population

    def fitness_function(chromosome):
        """
        Evaluates the chromosome (capacities for MA1)
        """

        # Run the algorithm using these capacities
        solution, _ = matching_algorithm_1(instance, capacities=chromosome, printing=False)

        # Evaluate blocking pairs
        return afccp.core.solutions.handling.calculate_blocking_pairs(p, solution, only_return_count=True)

    def multi_point_crossover(genome1, genome2):
        """
        Take two parent genomes, crossover the genes at multiple points and return two offspring solutions
        :param genome1: first parent genome
        :param genome2: second parent genome
        :return: offspring
        """
        points = np.sort(np.random.choice(crossover_positions, size=mdl_p["gma_num_crossover_points"], replace=False))
        start_points = np.append(0, points)
        stop_points = np.append(points, p['M'] - 1)
        child1 = np.zeros(p['M']).astype(int)
        child2 = np.zeros(p['M']).astype(int)
        flip = 1
        for i in range(len(start_points)):
            if flip == 1:
                child1[start_points[i]:stop_points[i] + 1] = genome2[start_points[i]:stop_points[i] + 1]
                child2[start_points[i]:stop_points[i] + 1] = genome1[start_points[i]:stop_points[i] + 1]
            else:
                child1[start_points[i]:stop_points[i] + 1] = genome1[start_points[i]:stop_points[i] + 1]
                child2[start_points[i]:stop_points[i] + 1] = genome2[start_points[i]:stop_points[i] + 1]

            flip = flip * -1

        return child1, child2

    def mutation(genome):
        """
        Takes a genome, and picks a random cadet index to mutate with some probability.
        This means we can swap an AFSC for one cadet individually
        :param genome: solution vector
        :return: mutated genome
        """
        for _ in range(mdl_p["gma_mutations"]):
            j = np.random.choice(p['J'])  # Random AFSC

            # Pick random new capacity for AFSC j
            min, max = p['quota_min'][j], p['quota_max'][j]
            capacity_options = np.arange(min, max + 1).astype(int)
            capacity = int(random.choice(capacity_options))
            genome[j] = capacity if (np.random.uniform() < mdl_p["gma_mutation_rate"]) else genome[j]

        return genome

    # Determine range of capacities for all AFSCs
    capacity_options = {}
    for j in p['J']:
        min, max = p['quota_min'][j], p['quota_max'][j]
        capacity_options[j] = np.arange(min, max + 1).astype(int)

    # Rank Selection Parameters
    rank_weights = (np.arange(1, mdl_p["gma_pop_size"] + 1)[::-1]) ** 1.2
    rank_weights = rank_weights / sum(rank_weights)
    rank_choices = np.arange(mdl_p["gma_pop_size"])

    # Multi-Point Crossover Parameters
    crossover_positions = np.arange(1, p['M'] - 1)

    # Initialize population
    population = initialize_population()

    # Initialize fitness scores
    fitness = np.zeros(mdl_p['gma_pop_size'])
    for c in range(mdl_p['gma_pop_size']):
        fitness[c] = fitness_function(population[c])

    # Sort Population by Fitness
    sorted_indices = fitness.argsort()
    fitness = fitness[sorted_indices]
    population = population[sorted_indices]

    # Main Loop
    start_time = time.perf_counter()
    generation = 0
    generating = True
    while generating:

        # Evaluate population
        for c in range(2, mdl_p['gma_pop_size']):
            fitness[c] = fitness_function(population[c])

        # Sort Population by Fitness
        sorted_indices = fitness.argsort()
        fitness = fitness[sorted_indices]
        population = population[sorted_indices]

        # Print statements
        if mdl_p['gma_printing']:
            print('Generation', generation, 'Fitness', fitness)

        # Create next generation
        next_generation = population[:2]  # the best two solutions are kept for the next generation
        for twins in range(int((mdl_p["gma_pop_size"] / 2) - 1)):  # create the offspring

            # Select parents for mating
            c1, c2 = np.random.choice(rank_choices, size=2, replace=False, p=rank_weights)
            parent_1, parent_2 = population[c1], population[c2]

            # Apply crossover function
            offspring_1, offspring_2 = multi_point_crossover(parent_1, parent_2)

            # Mutate genomes of offspring
            offspring_1 = mutation(offspring_1)
            offspring_2 = mutation(offspring_2)

            # Add this pair to the next generation
            offsprings = np.vstack((offspring_1, offspring_2))
            next_generation = np.vstack((next_generation, offsprings))

        # Next Generation
        population = next_generation
        generation += 1

        # Stopping conditions
        if mdl_p['stopping_conditions'] == 'Time':
            if (time.perf_counter() - start_time) < mdl_p['gma_max_time']:
                generating = False
        elif mdl_p['stopping_conditions'] == 'Generations':
            if generation >= mdl_p['gma_num_generations']:
                generating = False

    # Return solution (running algorithm using the best capacities)
    solution, _ = matching_algorithm_1(instance, capacities=population[0], printing=False)
    return solution


