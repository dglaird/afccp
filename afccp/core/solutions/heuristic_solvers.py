# Import Libraries
import time
import numpy as np
import random
import copy
import afccp.core.globals
import afccp.core.solutions.handling


def stable_marriage_model_solve(instance, printing=False):
    """
    This is a stable marriage implementation. This procedure takes in the parameters and value_parameters for the
    specified problem, then formulates the problem as a stable marriage problem and solves it. The procedure returns
    the solution.
    """
    if printing:
        print('Solving stable marriage model...')

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
                      vp['cadets_overall_weight'] * p['utility'][:, :p["M"]] * vp['cadet_weight'][:, np.newaxis] + \
                      p['ineligible'] * -100

    # Calculate AFSC preferences
    afsc_preference_lists = [[] for _ in p["J"]]
    for j in p["J"]:

        # Sort AFSC utility column j in descending order
        afsc_preference_lists[j] = np.argsort(afsc_utility[:, j])[::-1]

        # Calculate number of eligible cadets
        num_eligible = len(np.where(p['ineligible'][:, j] == 0)[0])

        # Create incomplete preference list using only eligible cadets
        afsc_preference_lists[j] = afsc_preference_lists[j][:num_eligible]

    # Calculate Cadet preferences
    cadet_preference_list = np.argsort(-p['utility'], axis=-1)
    matches = np.zeros(p["N"]) - 1  # 1-D array of matched AFSC indices (initially -1 indicating no match)
    rejections = np.zeros([p["N"], p["M"]])  # matrix indicating if cadet i has been rejected by their p preference
    pref_proposals = np.zeros(p["N"]) - 1  # 1-D array of cadet preference indices indicating proposal preference p
    afsc_matches = [[] for _ in p["J"]]  # list of matched cadets to AFSC j

    # Begin Stable Marriage Algorithm
    for _ in p["J"]:
        afsc_proposal_lists = [[] for _ in p["J"]]  # list of list of cadets proposing to AFSCs
        afsc_proposal_ranks = [[] for _ in p["J"]]  # rank associated with the cadets that are proposing to AFSC j
        afsc_proposal_indices = [[] for _ in p["J"]]  # index associated with the cadets that are proposing to AFSC j

        # cadets propose to their highest choice AFSC that has not rejected them
        for i in p["I"]:

            rejected = True  # initially rejected until accepted
            x = 0  # start with first choice
            while rejected:
                if rejections[i, x] == 0:
                    rejected = False
                    pref_proposals[i] = x
                    afsc_proposal_lists[cadet_preference_list[i, x]] = \
                        np.append(afsc_proposal_lists[cadet_preference_list[i, x]], int(i))  # Add cadet i to list of
                    # cadets proposing to that particular AFSC (obtained through looking up the index of cadet i's
                    # x choice)
                else:
                    x += 1

        # AFSCs accept their best offers as long as they're under their quotas, and reject others
        for j in p["J"]:

            # Loop through all cadets that are proposing to AFSC j
            for i in afsc_proposal_lists[j]:
                if i in afsc_preference_lists[j]:  # if the cadet is eligible for the AFSC
                    if len(afsc_proposal_lists[j]) > p['pgl'][j]:
                        afsc_proposal_ranks[j] = np.append(afsc_proposal_ranks[j],
                                                           np.where(afsc_preference_lists[j] == i)[0][0])
                    afsc_proposal_indices[j] = np.append(afsc_proposal_indices[j], int(i))
                else:
                    rejections[int(i), int(pref_proposals[int(i)])] = 1  # This cadet has been rejected by the AFSC

            if len(afsc_proposal_lists[j]) > p['pgl'][j]:

                if len(afsc_proposal_ranks[j]) != 0:
                    # This line sorts the indices of cadets who are proposing to AFSC j according to AFSC j's ordinal
                    # preference for them
                    sorted_indices = afsc_proposal_indices[j][afsc_proposal_ranks[j].argsort()]
                    afsc_matches[j] = sorted_indices[0:int(p['pgl'][j])]

                    # reject excess cadets above quota
                    rejected_cadets = sorted_indices[int(p['pgl'][j]):len(sorted_indices)]
                    for r in rejected_cadets:
                        rejections[int(r), int(pref_proposals[int(r)])] = 1  # This cadet has been rejected by the AFSC

                else:
                    afsc_matches[j] = []

            else:

                if len(afsc_proposal_ranks[j]) != 0:
                    afsc_matches[j] = afsc_proposal_indices[j]
                else:
                    afsc_matches[j] = []

    for j in p["J"]:
        for i in afsc_matches[j]:
            matches[int(i)] = int(j)

    # Begin Greedy Algorithm to match the rest of the cadets using overall utility matrix
    unmatched = np.where(matches == -1)[0]
    for i in unmatched:

        # Match cadet to AFSC which adds the highest overall value
        matches[i] = np.argmax(overall_utility[i, :])

    return matches


def matching_algorithm_1_old(instance, printing=True):
    """
    This is the Hospitals/Residents algorithm that matches cadets and AFSCs across all rated, space, and NRL positions.
    """
    if printing:
        print("Solving the deferred acceptance algorithm (1)...")

    # Shorthand
    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p

    # Create Cadet preferences
    cadet_preferences = {}
    for i, cadet in enumerate(p['cadets']):
        cadet_sorted_preferences = np.argsort(p['c_pref_matrix'][i, :])
        cadet_preferences[cadet] = []
        for j in cadet_sorted_preferences:
            if j in p['J^E'][i] and p['c_pref_matrix'][i, j] != 0:
                cadet_preferences[cadet].append(p['afscs'][j])

    # Create AFSC preferences
    afsc_preferences = {}
    for j, afsc in enumerate(p['afscs'][:p['M']]):
        afsc_sorted_preferences = np.argsort(p['a_pref_matrix'][:, j])
        afsc_preferences[afsc] = []
        for i in afsc_sorted_preferences:
            if i in p['I^E'][j] and p['a_pref_matrix'][i, j] != 0:
                afsc_preferences[afsc].append(p['cadets'][i])

    # Algorithm initialization
    total_slots = {p['afscs'][j]: p[mdl_p['capacity_parameter']][j] for j in p['J']}  # capacity targets
    total_rejections = {p['afscs'][j]: 0 for j in p['J']}  # Rejections for each AFSC

    # Dictionary of parameters used for the "CadetBoardFigure" object (animation)
    solution_iterations = {'proposals': {}, 'solutions': {}, 'iteration_names': {}}

    # Begin the simple Hospital/Residents Algorithm
    for iteration in range(p['M']):

        if mdl_p['ma_printing']:
            print("\nIteration", iteration + 1)

        # Cadets propose to their top choice that hasn't been rejected
        proposals = {cadet: "*" for cadet in p['cadets']}
        for cadet in p['cadets']:
            if len(cadet_preferences[cadet]) > 0:  # Making sure we haven't run out of preferences
                proposals[cadet] = cadet_preferences[cadet][0]  # (Current first choice)

        # Solution Iteration components (Proposals)
        afsc_solution = np.array([proposals[cadet] for cadet in p['cadets']])
        solution_iterations['proposals'][iteration] = \
            np.array([np.where(p['afscs'] == afsc)[0][0] for afsc in afsc_solution])
        counts = {afsc: len(np.where(afsc_solution == afsc)[0]) for afsc in p['afscs']}

        # Initialize matches information
        total_matched = {p['afscs'][j]: 0 for j in p['J']}  # Number of matched cadets for each AFSC

        # AFSCs accept their best cadets and reject the others
        for afsc in p['afscs'][:p['M']]:

            # Loop through their preferred cadets from top to bottom
            iteration_rejections = 0
            for cadet in afsc_preferences[afsc]:

                # If the cadet is proposing to this AFSC, we have two options
                if proposals[cadet] == afsc:

                    # We haven't hit capacity, so we accept this cadet
                    if total_matched[afsc] < total_slots[afsc]:
                        total_matched[afsc] += 1

                    # We're at capacity, so we reject this cadet
                    else:

                        # Delete this AFSC (first choice) from the cadet's options
                        proposals[cadet] = "*"
                        cadet_preferences[cadet].remove(afsc)
                        iteration_rejections += 1
                        total_rejections[afsc] += 1

        # Solution Iteration components (Current Matches)
        afsc_solution = np.array([proposals[cadet] for cadet in p['cadets']])
        solution_iterations['solutions'][iteration] = \
            np.array([np.where(p['afscs'] == afsc)[0][0] for afsc in afsc_solution])
        solution_iterations['iteration_names'][iteration] = 'Iteration ' + str(iteration + 1)

        # Specific matching algorithm print statement
        if mdl_p['ma_printing']:
            print('Proposals:', counts)
            print('Matched', total_matched)
            print('Rejected', total_rejections)

    # Last solution iteration
    solution_iterations['last_s'] = iteration

    # Return the solution and iterations
    afsc_solution = np.array([proposals[cadet] for cadet in p['cadets']])
    solution = np.array([np.where(p['afscs'] == afsc)[0][0] for afsc in afsc_solution])
    return solution, solution_iterations


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
                      vp['cadets_overall_weight'] * p['utility'][:, :p["M"]] * vp['cadet_weight'][:, np.newaxis] + \
                      p['ineligible'] * -100

    # Begin Greedy Algorithm to match the cadets using overall utility matrix
    solution = np.zeros(p["N"])
    for i in p["I"]:

        # Match cadet to AFSC which adds the highest overall value
        solution[i] = np.argmax(overall_utility[i, :])

    return solution


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
    afscs, M = p['afscs_acc_grp']['Rated'], len(p['afscs_acc_grp']['Rated'])
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


