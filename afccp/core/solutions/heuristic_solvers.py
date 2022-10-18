# Import Libraries
import time
import numpy as np
from afccp.core.globals import *
from afccp.core.handling.data_handling import value_function


def stable_marriage_model_solve(parameters, value_parameters, printing=False):
    """
    This is a stable marriage implementation. This procedure takes in the parameters and value_parameters for the
    specified problem, then formulates the problem as a stable marriage problem and solves it. The procedure returns
    the solution.
    :param printing: Whether the procedure should print something
    :param parameters: fixed cadet/AFSC input parameters
    :param value_parameters: weight/value parameters
    :return: solution in vector form
    """
    if printing:
        print('Solving stable marriage model...')

    # Load Parameters
    N = parameters['N']
    M = parameters['M']
    objectives = value_parameters['objectives']
    objective_weight = value_parameters['objective_weight']

    # Create AFSC Utility Matrix
    afsc_utility = np.zeros([N, M])
    for objective in ['Merit', 'Mandatory', 'Desired', 'Permitted', 'Utility']:
        if objective in objectives:
            k = np.where(objectives == objective)[0][0]
            if objective == 'Merit':
                merit = np.tile(np.where(parameters['merit'] > parameters['sum_merit'], 1,
                                         parameters['merit'] / parameters['sum_merit']), [M, 1]).T
                afsc_utility += merit * objective_weight[:, k].T
            else:
                afsc_utility += parameters[objective.lower()] * objective_weight[:, k].T

    # Overall Utility Matrix
    overall_utility = value_parameters['afscs_overall_weight'] * \
                      (afsc_utility.T * value_parameters['afsc_weight'][:, np.newaxis]).T + \
                      value_parameters['cadets_overall_weight'] * \
                      parameters['utility'] * value_parameters['cadet_weight'][:, np.newaxis] + \
                      parameters['ineligible'] * -100

    # Calculate AFSC preferences
    afsc_preference_lists = [[] for _ in range(M)]
    for j in range(M):

        # Sort AFSC utility column j in descending order
        afsc_preference_lists[j] = np.argsort(afsc_utility[:, j])[::-1]

        # Calculate number of eligible cadets
        num_eligible = len(np.where(parameters['ineligible'][:, j] == 0)[0])

        # Create incomplete preference list using only eligible cadets
        afsc_preference_lists[j] = afsc_preference_lists[j][:num_eligible]

    # Calculate Cadet preferences
    cadet_preference_list = np.argsort(-parameters['utility'], axis=-1)
    matches = np.zeros(N) - 1  # 1-D array of matched AFSC indices (initially -1 indicating no match)
    rejections = np.zeros([N, M])  # matrix indicating if cadet i has been rejected by their p preference
    pref_proposals = np.zeros(N) - 1  # 1-D array of cadet preference indices indicating proposal preference p
    afsc_matches = [[] for _ in range(M)]  # list of matched cadets to AFSC j

    # Begin Stable Marriage Algorithm
    for _ in range(M):
        afsc_proposal_lists = [[] for _ in range(M)]  # list of list of cadets proposing to AFSCs
        afsc_proposal_ranks = [[] for _ in range(M)]  # rank associated with the cadets that are proposing to AFSC j
        afsc_proposal_indices = [[] for _ in range(M)]  # index associated with the cadets that are proposing to AFSC j

        # cadets propose to their highest choice AFSC that has not rejected them
        for i in range(N):

            rejected = True  # initially rejected until accepted
            p = 0  # start with first choice
            while rejected:
                if rejections[i, p] == 0:
                    rejected = False
                    pref_proposals[i] = p
                    afsc_proposal_lists[cadet_preference_list[i, p]] = \
                        np.append(afsc_proposal_lists[cadet_preference_list[i, p]], int(i))  # Add cadet i to list of
                    # cadets proposing to that particular AFSC (obtained through looking up the index of cadet i's
                    # p choice)
                else:
                    p += 1

        # AFSCs accept their best offers as long as they're under their quotas, and reject others
        for j in range(M):

            # Loop through all cadets that are proposing to AFSC j
            for i in afsc_proposal_lists[j]:
                if i in afsc_preference_lists[j]:  # if the cadet is eligible for the AFSC
                    if len(afsc_proposal_lists[j]) > parameters['quota'][j]:
                        afsc_proposal_ranks[j] = np.append(afsc_proposal_ranks[j],
                                                           np.where(afsc_preference_lists[j] == i)[0][0])
                    afsc_proposal_indices[j] = np.append(afsc_proposal_indices[j], int(i))
                else:
                    rejections[int(i), int(pref_proposals[int(i)])] = 1  # This cadet has been rejected by the AFSC

            if len(afsc_proposal_lists[j]) > parameters['quota'][j]:

                if len(afsc_proposal_ranks[j]) != 0:
                    # This line sorts the indices of cadets who are proposing to AFSC j according to AFSC j's ordinal
                    # preference for them
                    sorted_indices = afsc_proposal_indices[j][afsc_proposal_ranks[j].argsort()]
                    afsc_matches[j] = sorted_indices[0:int(parameters['quota'][j])]

                    # reject excess cadets above quota
                    rejected_cadets = sorted_indices[int(parameters['quota'][j]):len(sorted_indices)]
                    for r in rejected_cadets:
                        rejections[int(r), int(pref_proposals[int(r)])] = 1  # This cadet has been rejected by the AFSC

                else:
                    afsc_matches[j] = []

            else:

                if len(afsc_proposal_ranks[j]) != 0:
                    afsc_matches[j] = afsc_proposal_indices[j]
                else:
                    afsc_matches[j] = []

    for j in range(M):
        for i in afsc_matches[j]:
            matches[int(i)] = int(j)

    # Begin Greedy Algorithm to match the rest of the cadets using overall utility matrix
    unmatched = np.where(matches == -1)[0]
    for i in unmatched:

        # Match cadet to AFSC which adds the highest overall value
        matches[i] = np.argmax(overall_utility[i, :])

    return matches


def matching_algorithm_1(instance, printing=True):
    """
    This is my (Lt. Laird)'s first attempt at a working matching algorithm using the preference lists
    :param printing: whether to print something out or not
    :param instance: problem instance to solve
    :return: solution
    """
    if printing:
        print("Solving the deferred acceptance algorithm (1)...")

    # Load parameters
    p = instance.parameters

    if "c_pref_matrix" not in p or "a_pref_matrix" not in p:
        raise ValueError("No preference matrices found in the parameters. Cannot run the algorithm.")

    # Sort the preferences for cadets and remove AFSCs they're not eligible for
    c_preferences = [[] for _ in p["I"]]
    for i in p["I"]:
        sorted_preferences = np.argsort(p["c_pref_matrix"][i, :])
        c_preferences[i] = [j for j in sorted_preferences if j in p["J^E"][i]]

    # Sort the preferences for AFSCs and remove cadets that aren't eligible for them
    a_preferences = [[] for _ in p["J"]]
    for j in p["J"]:
        sorted_preferences = np.argsort(p["a_pref_matrix"][:, j])
        a_preferences[j] = [i for i in sorted_preferences if i in p["I^E"][j]]

    return np.zeros(p["N"]).astype(int)


def greedy_model_solve(parameters, value_parameters, printing=False):
    """
    This is a simple greedy algorithm that matches each cadet according to the highest valued AFSC determined by the
    initial overall utility matrix.
    :param printing: Whether the procedure should print something or not
    :param parameters: fixed cadet/AFSC input parameters
    :param value_parameters: weight/value parameters
    :return: solution in vector form
    """
    if printing:
        print('Solving Greedy Model...')

    # Load Parameters
    N = parameters['N']
    M = parameters['M']
    objectives = value_parameters['objectives']
    objective_weight = value_parameters['objective_weight']

    # Create AFSC Utility Matrix
    afsc_utility = np.zeros([N, M])
    for objective in ['Merit', 'Mandatory', 'Desired', 'Permitted', 'Utility']:
        if objective in objectives:
            k = np.where(objectives == objective)[0][0]
            if objective == 'Merit':
                merit = np.tile(np.where(parameters['merit'] > parameters['sum_merit'], 1,
                                         parameters['merit'] / parameters['sum_merit']), [M, 1]).T
                afsc_utility += merit * objective_weight[:, k].T
            else:
                afsc_utility += parameters[objective.lower()] * objective_weight[:, k].T

    # Overall Utility Matrix
    overall_utility = value_parameters['afscs_overall_weight'] * \
                      (afsc_utility.T * value_parameters['afsc_weight'][:, np.newaxis]).T + \
                      value_parameters['cadets_overall_weight'] * \
                      parameters['utility'] * value_parameters['cadet_weight'][:, np.newaxis] + \
                      parameters['ineligible'] * -100

    # Begin Greedy Algorithm to match the cadets using overall utility matrix
    solution = np.zeros(N)
    for i in range(N):

        # Match cadet to AFSC which adds the highest overall value
        solution[i] = np.argmax(overall_utility[i, :])

    return solution


def genetic_algorithm(instance, initial_solutions=None, con_fail_dict=None, printing=False):
    """
    This is the genetic algorithm. The hyper-parameters to the algorithm can be tuned, and this is meant to be
    solved in conjunction with the pyomo model solutions. Use several VFT pyomo solutions in the initial population to
    this genetic algorithm
    """

    # Function Definitions
    def fitness_function(chromosome):
        """
        This function takes in a chromosome (solution vector) and evaluates it. I tried to make it as efficient
        as possible, since it is a very time-consuming function (relatively speaking)
        :param chromosome: solution vector
        :return: fitness score
        """
        # We assume the solution is feasible until proven otherwise
        failed = False

        # Make sure cadets are assigned to the AFSCs they need to be assigned to (fixed variables)
        if p["J^Fixed"] is not None:
            for i in p["J^Fixed"]:
                chromosome[i] = p["J^Fixed"][i]

        # 5% cap on total percentage of USAFA cadets allowed into certain AFSCs
        if vp["J^USAFA"] is not None:

            # This is a pretty arbitrary constraint and will only be used for real class years
            cap = 0.05 * mp["real_usafa_n"]

            u_count = 0
            for j in vp["J^USAFA"]:

                # list of indices of assigned cadets
                cadets = np.where(chromosome == j)[0]
                usafa_cadets = np.intersect1d(p['I^D']['USAFA Proportion'][j], cadets)
                u_count += len(usafa_cadets)

            if u_count > int(cap + 1):
                failed = True

        afsc_value = np.zeros(p['M'])
        for j in p['J']:

            # Initialize objective measures and values
            measure = np.zeros(vp['O'])
            value = np.zeros(vp['O'])

            # list of indices of assigned cadets
            cadets = np.where(chromosome == j)[0]

            # Only calculate measures for AFSCs with at least one cadet
            count = len(cadets)
            usafa_count = count
            if count > 0:

                if "USAFA Proportion" in vp["objectives"]:
                    usafa_cadets = np.intersect1d(p['I^D']['USAFA Proportion'][j], cadets)
                    usafa_count = len(usafa_cadets)

                # Loop through all AFSC objectives
                for k in vp['K^A'][j]:
                    objective = vp['objectives'][k]
                    if objective == 'Merit':
                        measure[k] = np.mean(p['merit'][cadets])
                    elif objective == 'Utility':
                        measure[k] = np.mean(p['utility'][cadets, j])
                    elif objective == 'Combined Quota':
                        measure[k] = count
                    elif objective == 'USAFA Quota':
                        measure[k] = usafa_count
                    elif objective == 'ROTC Quota':
                        measure[k] = count - usafa_count
                    elif objective in vp['K^D']:
                        measure[k] = len(np.intersect1d(p['I^D'][objective][j], cadets)) / count
                    elif objective == "Norm Score":
                        best_sum = np.sum(c for c in range(count))
                        worst_range = range(p["num_eligible"][j] - count, p["num_eligible"][j])
                        worst_sum = np.sum(c for c in worst_range)
                        achieved_sum = np.sum(p["a_pref_matrix"][cadets, j])
                        measure[k] = 1 - (achieved_sum - best_sum) / (worst_sum - best_sum)

                    # Assign AFSC objective value
                    value[k] = value_function(vp['a'][j][k], vp['f^hat'][j][k], vp['r'][j][k], measure[k])

                    # AFSC Objective Constraints
                    if k in vp['K^C'][j]:

                        # Check if this is a constrained approximate measure or exact measure
                        if vp["constraint_type"][j, k] == 3:

                            # PGL should be more "forgiving" as a constraint
                            if "pgl" in p:
                                constrained_measure = (measure[k] * count) / p['pgl'][j]
                            else:
                                constrained_measure = (measure[k] * count) / p['quota'][j]
                        else:
                            constrained_measure = measure[k]

                        # Use the "real" constraint (potentially different as a result of approximate model)
                        constrained_min, constrained_max = objective_min[j, k], objective_max[j, k]
                        if con_fail_dict is not None:
                            if (j, k) in con_fail_dict:
                                split_list = con_fail_dict[(j, k)].split(' ')
                                if split_list[0] == '>':
                                    constrained_min = float(split_list[1])
                                    constrained_max = objective_max[j, k]
                                else:
                                    constrained_min = objective_min[j, k]
                                    constrained_max = float(split_list[1])

                        # Constraint penalties
                        if constrained_measure < constrained_min or constrained_measure > constrained_max:

                            # We failed this constraint, exit the fitness function!
                            if con_fail_dict is not None:
                                failed = True
                                break

                if failed:
                    break
                else:

                    # Calculate AFSC value
                    afsc_value[j] = np.dot(vp['objective_weight'][j, :], value)
                    if j in vp['J^C']:
                        if afsc_value[j] < vp['afsc_value_min'][j]:
                            failed = True
                            break

            else:
                failed = True
                break

        if not failed:
            cadet_value = np.array([p['utility'][i, int(chromosome[i])] for i in p['I']])
            for i in vp['I^C']:
                if cadet_value[i] < vp['cadet_value_min'][i]:
                    failed = True
                    break

        if not failed:
            z = vp['cadets_overall_weight'] * np.dot(vp['cadet_weight'], cadet_value) + \
                vp['afscs_overall_weight'] * np.dot(vp['afsc_weight'], afsc_value)
        else:
            z = 0

        return z

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
            i = np.random.randint(low=0, high=p['N'])
            new_j = np.random.choice(p['J^E'][i])
            genome[i] = new_j if (np.random.uniform() < mp["mutation_rate"]) else genome[i]

        return genome

    if printing:
        print("Running Genetic Algorithm...")

    # Shorthand
    p = instance.parameters
    vp = instance.value_parameters
    mp = instance.mdl_p

    # Obtain objective minimums and maximums
    objective_min = np.zeros([p['M'], vp['O']])
    objective_max = np.zeros([p['M'], vp['O']])
    for j in p['J']:
        for k in vp['K^C'][j]:
            if vp['constraint_type'][j, k] == 1 or vp['constraint_type'][j, k] == 2:
                objective_min[j, k] = float(vp['objective_value_min'][j, k])
            elif vp['constraint_type'][j, k] == 3 or vp['constraint_type'][j, k] == 4:
                value_list = vp['objective_value_min'][j, k].split(",")
                objective_min[j, k] = float(value_list[0].strip())
                objective_max[j, k] = float(value_list[1].strip())

    # Printing updates
    if mp["ga_printing"]:
        step = mp["ga_max_time"] / (100 / mp["percent_step"])
        checkpoint = step
        completed = 0

    # Rank Selection Parameters
    rank_weights = (np.arange(1, mp["pop_size"] + 1)[::-1]) ** 1.2
    rank_weights = rank_weights / sum(rank_weights)
    rank_choices = np.arange(mp["pop_size"])

    # Multi-Point Crossover Parameters
    crossover_positions = np.arange(1, p['N'] - 1)

    # Initialize Population
    population = np.array([[np.random.choice(p['J^E'][i]) for i in p['I']] for _ in range(mp["pop_size"])]).astype(int)
    if initial_solutions is not None:

        # Get fitness of initial_solutions in case there are issues with feasibility or there are
        # too many initial solutions
        num_initial = len(initial_solutions)
        fitness = np.zeros(num_initial)
        for s, chromosome in enumerate(initial_solutions):
            fitness[s] = fitness_function(chromosome)

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
    for i, chromosome in enumerate(population):
        fitness[i] = fitness_function(chromosome)

    # Sort Population by Fitness
    sorted_indices = fitness.argsort()[::-1]
    fitness = fitness[sorted_indices]
    population = population[sorted_indices]

    if mp["ga_printing"]:
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
            fitness[index] = fitness_function(population[index])
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
        return population[0]
