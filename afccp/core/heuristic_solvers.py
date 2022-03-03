# Import Libraries
import time
import numpy as np
from afccp.core.globals import *


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
    for iteration in range(M):
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


def genetic_algorithm(parameters, value_parameters, pop_size=6, stopping_time=60, num_crossover_points=3,
                      initial_solutions=None, mutation_rate=0.1, num_time_points=100, constraints="Penalty",
                      penalty_scale=1.3, time_eval=True, num_mutations=None, percent_step=10, con_tolerance=0.95,
                      con_fail_dict=None, printing=True):
    """
    This is the genetic algorithm. The hyper-parameters to the algorithm can be tuned, and this is meant to be
    solved in conjunction with the pyomo model solution. Use that as the initial solution, and then we evolve
    from there
    :param con_fail_dict: dictionary of failed constraints in the approximate model that we adhere to
    :param con_tolerance: constraint fail tolerance (we can meet X % of the constraint or above and be ok)
    :param penalty_scale: how to scale the penalties to further decrease the fitness value
    :param constraints: how we handle fitness in terms of constraints (penalty, fail, or other)
    :param parameters: cadet/AFSC parameters
    :param value_parameters: weight and value parameters
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

    # Function Definitions
    def fitness_function(chromosome, first=False):
        """
        This function takes in a chromosome (solution vector) and evaluates it. I tried to make it as efficient
        as possible, since it is a very time-consuming function (relatively speaking)
        :param chromosome: solution vector
        :return: fitness score
        """
        failed = False
        penalty = 0
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
                if not ignore_uc:
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
                    elif objective in p['K^D']:
                        measure[k] = len(np.intersect1d(p['I^D'][objective][j], cadets)) / count

                    # Assign AFSC objective value
                    value[k] = value_function(vp['a'][j][k], vp['f_a'][j][k], vp['r'][j][k], measure[k])

                    # AFSC Objective Constraints
                    if k in vp['K^C'][j]:

                        # We're really only ever going to constrain the approximate measure for Mandatory
                        if objective == 'Mandatory':
                            constrained_measure = (measure[k] * count) / p['quota'][j]
                        else:
                            constrained_measure = measure[k]

                        # Use the real constraint (potentially different as a result of approximate model)
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
                        if (constrained_measure < constrained_min or constrained_measure > constrained_max) \
                                and not first:

                            if con_fail_dict is not None and constraints == 'Fail':
                                failed = True
                                break
                            else:
                                if constrained_measure < constrained_min:
                                    p_con_met = constrained_measure / constrained_min
                                else:
                                    p_con_met = constrained_max / constrained_measure

                                if objective == 'USAFA Proportion':
                                    adj_con_tolerance = 0.9
                                elif objective == 'Combined Quota':
                                    adj_con_tolerance = min((constrained_min - 1) / constrained_min,
                                                            constrained_max / (constrained_max + 1))
                                elif objective == 'Mandatory':
                                    adj_con_tolerance = min((constrained_min - (1 / p['quota'][j])) / constrained_min,
                                                            constrained_max / (constrained_max + (1 / p['quota'][j])))
                                else:
                                    adj_con_tolerance = con_tolerance

                                # Either we reduce z by some penalty or z is set to 0
                                if constraints == 'Penalty' and p_con_met < adj_con_tolerance:
                                    penalty += vp['afscs_overall_weight'] * vp['afsc_weight'][j]
                                elif constraints == 'Fail' and p_con_met < adj_con_tolerance:
                                    failed = True
                                    break

                if failed:
                    break
                else:

                    # Calculate AFSC value
                    afsc_value[j] = np.dot(vp['objective_weight'][j, :], value)
                    if j in vp['J^C']:
                        if afsc_value[j] < vp['afsc_value_min'][j]:

                            # Either we reduce z by some penalty or z is set to 0
                            if constraints == 'Penalty':
                                penalty += vp['afscs_overall_weight'] * vp['afsc_weight'][j]
                            elif constraints == 'Fail':
                                failed = True
                                break

            else:

                # Either we reduce z by some penalty or z is set to 0
                if constraints == 'Penalty':
                    penalty += vp['afscs_overall_weight'] * vp['afsc_weight'][j]
                elif constraints == 'Fail':
                    failed = True
                    break

        if not failed:
            cadet_value = np.array([p['utility'][i, chromosome[i]] for i in p['I']])
            for i in vp['I^C']:
                if cadet_value[i] < vp['cadet_value_min'][i]:

                    # Either we reduce z by some penalty or z is set to 0
                    if constraints == 'Penalty':
                        penalty += vp['cadets_overall_weight'] * vp['cadet_weight'][i]
                    elif constraints == 'Fail':
                        failed = True
                        break

        if not failed:
            z = vp['cadets_overall_weight'] * np.dot(vp['cadet_weight'], cadet_value) + \
                vp['afscs_overall_weight'] * np.dot(vp['afsc_weight'], afsc_value)
        else:
            z = 0

        if constraints == 'Penalty':
            penalized_z = z - penalty ** (1 / penalty_scale)
            return z, penalized_z
        else:
            return z

    def multi_point_crossover(genome1, genome2):
        """
        Take two parent genomes, crossover the genes at multiple points and return two offspring solutions
        :param genome1: first parent genome
        :param genome2: second parent genome
        :return: offspring
        """
        points = np.sort(np.random.choice(crossover_positions, size=num_crossover_points, replace=False))
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
        for _ in range(num_mutations):
            i = np.random.randint(low=0, high=p['N'])
            new_j = np.random.choice(p['J^E'][i])
            genome[i] = new_j if (np.random.uniform() < mutation_rate) else genome[i]

        return genome

    if printing:
        print("Running Genetic Algorithm...")

    # Shorthand
    p = parameters
    vp = value_parameters

    # Obtain objective minimums and maximums
    objective_min = np.zeros([p['M'], p['O']])
    objective_max = np.zeros([p['M'], p['O']])
    soc_counts = np.zeros(p['M'])
    for j in p['J']:

        # If we care about USAFA Count and ROTC Count
        if 'USAFA Count' in vp['objectives'][vp['K^A'][j]]:
            soc_counts[j] = 1

        for k in vp['K^C'][j]:
            if vp['objective_constraint_type'][j, k] == 1 or vp['objective_constraint_type'][j, k] == 2:
                objective_min[j, k] = float(value_parameters['objective_value_min'][j, k])
            elif vp['objective_constraint_type'][j, k] == 3 or vp['objective_constraint_type'][j, k] == 4:
                value_list = value_parameters['objective_value_min'][j, k].split(",")
                objective_min[j, k] = float(value_list[0].strip())
                objective_max[j, k] = float(value_list[1].strip())

    # If USAFA counts are used
    if sum(soc_counts) == 0:
        ignore_uc = True
    else:
        ignore_uc = False

    # Number of Mutations
    if num_mutations is None:
        num_mutations = int(np.ceil(p['N'] / 75))

    # Printing updates
    if printing:
        step = stopping_time / (100 / percent_step)
        checkpoint = step
        completed = 0

    # Rank Selection Parameters
    rank_weights = (np.arange(1, pop_size + 1)[::-1]) ** 1.2
    rank_weights = rank_weights / sum(rank_weights)
    rank_choices = np.arange(pop_size)

    # Multi-Point Crossover Parameters
    crossover_positions = np.arange(1, p['N'] - 1)

    # Initialize Population
    population = np.array([[np.random.choice(p['J^E'][i]) for i in p['I']] for _ in range(pop_size)]).astype(int)
    if initial_solutions is not None:
        for i, solution in enumerate(initial_solutions):
            population[i] = solution

    # Initialize Fitness Scores
    fitness = np.zeros(pop_size)
    if constraints == 'Penalty':
        real_z = np.zeros(pop_size)
    for i, chromosome in enumerate(population):
        if constraints == 'Penalty':
            real_z[i], fitness[i] = fitness_function(chromosome, first=True)
        else:
            fitness[i] = fitness_function(chromosome, first=True)

    # Sort Population by Fitness
    sorted_indices = fitness.argsort()[::-1]
    fitness = fitness[sorted_indices]
    population = population[sorted_indices]
    if constraints == 'Penalty':
        real_z = real_z[sorted_indices]

    # Time Evaluation Initialization
    if time_eval:
        time_step = stopping_time / num_time_points
        step_num = 1
        times = [0]
        if constraints == 'Penalty':
            scores = [real_z[0]]
        else:
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
        for index in range(2, pop_size):
            if constraints == 'Penalty':
                real_z[index], fitness[index] = fitness_function(population[index])
            else:
                fitness[index] = fitness_function(population[index])
        eval_times.append(time.perf_counter() - gen_start_time)

        # Sort Population by Fitness
        sorted_indices = fitness.argsort()[::-1]
        fitness = fitness[sorted_indices]
        population = population[sorted_indices]
        if constraints == 'Penalty':
            real_z = real_z[sorted_indices]
            best_score = real_z[0]
        else:
            best_score = fitness[0]

        # Printing updates
        if printing:
            if (time.perf_counter() - start_time) > checkpoint:
                completed += percent_step
                checkpoint += step
                print(str(completed) + "% complete. Best solution value: " + str(round(best_score, 4)))
                avg_eval_time = np.mean(np.array(eval_times[0:generation]))
                print('Average evaluation time for ' + str(pop_size) + ' solutions: ' +
                      str(round(avg_eval_time, 4)) + ' seconds.')
                avg_gen_time = np.mean(np.array(gen_times[0:generation]))
                print('Average generation time: ' + str(round(avg_gen_time, 4)) + ' seconds.')

        # Create next generation
        next_generation = population[0:2]  # the best two solutions are kept for the next generation
        for twins in range(int((pop_size / 2) - 1)):  # create the offspring

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
        if time_eval:
            if (time.perf_counter() - start_time) > (time_step * step_num):
                times.append(time.perf_counter() - start_time)
                scores.append(best_score)
                step_num += 1

        # Check stopping criteria
        if (time.perf_counter() - start_time) > stopping_time:
            if printing:
                end_time = round(time.perf_counter() - start_time, 2)
                print('End time reached in ' + str(end_time) + ' seconds.')
            generating = False

        # Next Generation
        population = next_generation
        gen_times.append(time.perf_counter() - gen_start_time)
        generation += 1

    # Time Eval
    if time_eval:

        # Create time_eval_df
        time_eval_df = pd.DataFrame({'Time': times, 'Objective Value': scores})

        if printing:
            print(time_eval_df)
        return population[0], time_eval_df

    else:

        # Return best solution
        return population[0]
