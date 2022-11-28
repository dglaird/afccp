import afccp.core.comprehensive_functions


class MoreCCPMethods:

    def new_method(self):
        print('This is a new method for the CadetCareerProblem class')
        print('I can access all of the stuff from the other one and create new methods')
        print('N:', self.parameters['N'])  # Pycharm isn't going to like this, but this will work once CCP is defined

    def test_ian_function_method(self):
        test_function_1(self)

    def classic_hr_alg(self):
        func_classic_hr_alg_1_0(self)


    def hr_alg_1_0_0(self):
        func_hr_alg_1_0_0(self)

    def hr_alg_1_0_1(self):
        func_hr_alg_1_0_1(self)

    def hr_alg_1_0_1_1(self):
        func_hr_alg_1_0_1_1(self)

    def hr_alg_1_0_2(self):
        func_hr_alg_1_0_2(self)

    def hr_alg_1_0_2_1(self):
        func_hr_alg_1_0_2_1(self)

    def hr_alg_1_0_2_1_1(self):
        func_hr_alg_1_0_2_1_1(self)

    def hr_alg_1_0_2_2(self):
        func_hr_alg_1_0_2_2(self)

    def hr_alg_1_1(self):
        func_hr_alg_1_1(self)

    def hr_alg_1_1_1(self):
        func_hr_alg_1_1_1(self)

    def hr_alg_1_1_1_1(self):
        func_hr_alg_1_1_1_1(self)

    def hr_alg_1_1_2(self):
        func_hr_alg_1_1_2(self)

    def hr_alg_1_1_2_1(self):
        func_hr_alg_1_1_2_1(self)

    def hr_alg_1_1_2_2(self):
        func_hr_alg_1_1_2_2(self)

    def hr_alg_1_2(self):
        func_hr_alg_1_2(self)

    def hr_alg_1_3(self):
        func_hr_alg_1_3(self)

    def hr_lq_alg_2_0(self):
        func_hr_lq_alg_2_0(self)

    def hr_lq_alg_2_1(self):
        func_hr_lq_alg_2_1(self)

    def hr_alg_function_test(self, capacity_type):
        alg_function_test(self, capacity_type)

    def algorithm_1_func_for_thesis_pca(self, capacity_type, scoring_files):
        algorithm_1_for_thesis(self, capacity_type, scoring_files)

    def algorithm_2_func_for_thesis(self, capacity_type, scoring_files):
        algorithm_2_for_thesis(self, capacity_type, scoring_files)

    def algorithm_3_func_for_thesis(self,capacity_type, scoring_files):
        algorithm_3_for_thesis(self, capacity_type, scoring_files)
# Trying to call on the algorithm that I make by functions so I can modify the capacities when calling as an instance
# method.

    def algorithm_4_func_for_thesis(self, capacity_type, scoring_files):
        algorithm_4_for_thesis(self, capacity_type, scoring_files)

    def hr_alg_function_test_cap_mod(self):
        alg_mod_capacities = alg_function_test(self, instance, capacity_type)
        AFSC_scores_list = AFSC_scoring_data_structure(instance)
        cadet_scores_list, unmatched_ordered_cadets_list = cadet_scoring_data_structure(instance)
        cadet_score_from_AFSC, cadet_scoring, cadet_ranking, afsc_scoring, afsc_ranking = build_ranking_data_structures(
            instance)
        afsc_capacities = set_afsc_capacities(instance, capacity_type)
        afsc_matches = afsc_matches_build(instance)

        start_time = time.time()
        Classical_HR_algorithm_time = time.time()
        # initialize parameters
        unmatched_ordered_cadets = unmatched_ordered_cadets_list  # all cadets sorted decreasing by merit
        cannot_match = []  # cadets with empty ranking lists
        M = {}  # matches
        blocking_pairs = []
        iter_num = 0
        iter_limit = 20000
        next_cadet = True  # tracks if it is a new cadet or the same one that got rejected

        while len(unmatched_ordered_cadets) > 0 and iter_num <= iter_limit:  # while there are still cadets to match
            iter_num += 1
            if next_cadet:
                c = unmatched_ordered_cadets[0]  # next cadet to match
                unmatched_ordered_cadets.remove(c)  # take cadet out of unmatched list
            if len(cadet_ranking[c]) == 0:  # if no more AFSCs in ranking list
                cannot_match.append(c)  # add them to cannot match list
                next_cadet = True
                continue  # go to beginning of while loop
            a = cadet_ranking[c][0]  # highest ranking AFSC for cadet c

            if len(afsc_matches[a]) < afsc_capacities[a]:  # if not at capacity
                M[c] = a  # add match to M
                afsc_matches[a].append(c)  # insert into AFSC a's matches
                next_cadet = True  # move to next cadet
                continue  # go to beginning of while loop

            else:  # if at capacity
                c_ = afsc_matches[a][-1]  # set c_ as the lowest ranking cadet from AFSC list
                for c_hat in afsc_matches[a]:
                    if afsc_scoring[a][c_hat] < afsc_scoring[a][c_]:
                        c_ = c_hat
                if afsc_scoring[a][c_] > afsc_scoring[a][c]:  # if c_ ranks higher than c
                    cadet_ranking[c].remove(a)  # remove a from c's ranking list
                    next_cadet = False  # keep trying to match this cadet
                    continue  # go to beginning of while loop

                else:  # if c ranks higher than c_
                    afsc_matches[a].remove(c_)  # remove c_ from AFSC a's matches
                    cadet_ranking[c_].remove(a)  # Removing AFSC from cadet c_'s list
                    M.pop(c_)  # remove c_ from M - Gives error that can't use pop to remove a string.
                    M[c] = a  # add match to M
                    next_cadet = False
                    afsc_matches[a].append(c)  # insert into AFSC a's matches
                    c = c_
                    if len(unmatched_ordered_cadets) == 0 and next_cadet == False:  # added to get the last cadet added as unmatched
                        cannot_match.append(c)  # added to get the last cadet added as unmatched
                    continue  # go to beginning of while loop

        print("Total iterations:", iter_num)

        print("Time to run matching algorithm:", Classical_HR_algorithm_time - start_time)

        print('Length of cadets still remaining to match:', len(unmatched_ordered_cadets))
        print('Length of perm unmatched cadets:', len(cannot_match))

        # print("Total time:", time.time() - start_time)

        # Trying to loop through those who could not be matched to match them with the career field that wants them.
        print('# unmatched', len(cannot_match))

        return M, cannot_match, afsc_matches


    def func_alg_AFSC_choose_best_cadet_to_match_all(self):
        alg_AFSC_choose_best_cadet_to_match_all(self)

    def func_OLEA_build(self):
        return OLEA_Scoring_Build_func(self)

    def func_Tiered_Degree_Scoring_Build(self):
        return Tiered_Degree_Scoring_Build(self)


    def evaluate_classic_hr_method(self):
            """
            Evaluates the current solution using the H/R Algorithm metrics
            :return:
            """
            # hr_metrics = classic_hr_alg_evaluate(self.solution, self)
            hr_metrics = classic_hr_alg_evaluate(self.metrics, self)
            self.hr_metrics = hr_metrics

    def export_match_to_excel(self):
        """
        Exports current matching to excel column
        """
        match_to_excel(self)


    def export_solution_to_excel(self):
        """
        Exports the current solution to excel with H/R Algorithm metrics
        :return:
        """
        ian_export(self)