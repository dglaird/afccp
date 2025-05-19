# Import Libraries
from afccp.main import CadetCareerProblem
import pandas as pd
import datetime


def test_solve_times():
    """
    Tests solve times
    """

    # size_list = [(100, 6, 0), (200, 6, 0), (1200, 6, 0), (2500, 16, 0)]
    size_list = [(500, 10, 6), (1500, 14, 6), (3000, 14, 10)]
    # model_solver_list = [('GUO', 'cbc'), ('A-VFT', 'cbc'), ('E-VFT', 'ipopt')]
    model_solver_list = [('GUO', 'cbc')]
    instances_per_size = 5
    max_solve_time = 60

    df_columns = {'Iteration': [], 'Size (NxMxS)': [], 'Instance': [], 'Model': [], 'Solver': [], 'Result': [],
                  'Solve Time (s)': [], 'Pyomo Z': [], 'Real X Exact VFT Z': [], 'Real X Approx VFT Z': [],
                  'Rounded X Exact VFT Z': [], 'Real X GUO Z': [], 'Rounded X GUO Z': [], 'Integer X': [],
                  'Integer Y': [], 'Integer V': [], 'Integer Q': [], 'Fixed, Reserved, Alternates': []}

    # Loop through each problem size in the list
    iteration = 0
    for size in size_list:

        N, M, S = size[0], size[1], size[2]
        size_name = str(N) + 'x' + str(M) + 'x' + str(S)

        print('\nSize', size_name)

        # Loop through each iteration of this problem size
        for i in range(instances_per_size):

            print('\nInstance', i + 1)

            # Create the problem instance
            if S == 0:
                instance = CadetCareerProblem('Random', N=N, M=M, P=M, printing=False)
            else:
                instance = CadetCareerProblem('Random', N=N, M=M, P=M, S=S, generate_extra_components=True,
                                              printing=False)
            instance.fix_generated_data(printing=False)
            instance.set_value_parameters()
            instance.soc_rated_matching_algorithm({'soc': 'usafa'})
            instance.soc_rated_matching_algorithm({'soc': 'rotc'})
            instance.incorporate_rated_algorithm_results()

            # Loop through each model & solver name in the list
            for model_name, solver_name in model_solver_list:
                iteration += 1

                # Initialize "p_dict" for model controls
                p_dict = {'solver_name': solver_name, 'pyomo_max_time': max_solve_time, 're-calculate x': False}

                # Change dictionary based on certain features
                if solver_name == 'couenne':
                    p_dict['provide_executable'] = True
                if S != 0:
                    p_dict['solve_extra_components'] = True

                # Add standard values
                df_columns['Iteration'].append(iteration), df_columns['Instance'].append(i + 1)
                df_columns['Size (NxMxS)'].append(size_name), df_columns['Model'].append(model_name),
                df_columns['Solver'].append(solver_name)

                print(model_name, solver_name, 'solving at', datetime.datetime.now().strftime("%H:%M:%S"))

                # Solve the model
                try:
                    if model_name == 'GUO':
                        instance.solve_guo_pyomo_model(p_dict)
                    elif model_name == 'A-VFT':
                        p_dict['approximate'] = True
                        instance.solve_vft_pyomo_model(p_dict)
                    elif model_name == 'E-VFT':
                        p_dict['approximate'] = False
                        instance.solve_vft_pyomo_model(p_dict)

                    # Solve Time
                    if instance.solution['solve_time'] >= max_solve_time:
                        df_columns['Result'].append('Time Limit Reached')
                    else:
                        df_columns['Result'].append('Solved')

                    # Add values
                    df_columns['Solve Time (s)'].append(instance.solution['solve_time'])
                    df_columns['Pyomo Z'].append(instance.solution['pyomo_obj_value'])
                    df_columns['Fixed, Reserved, Alternates'].append(
                        instance.solution['cadets_fixed_correctly'] + ', ' +
                        instance.solution['cadets_reserved_correctly'] + ', ' + instance.solution['alternate_list_metric'])

                    # Check if we have integer values
                    df_columns['Integer X'].append(str(instance.solution['x_integer']))
                    if model_name != 'GUO':
                        df_columns['Integer Y'].append(str(instance.solution['y_integer']))
                    else:
                        df_columns['Integer Y'].append('N/A')
                    if S != 0:
                        df_columns['Integer V'].append(str(instance.solution['v_integer']))
                        df_columns['Integer Q'].append(str(instance.solution['q_integer']))
                    else:
                        df_columns['Integer V'].append('N/A')
                        df_columns['Integer Q'].append('N/A')

                    # Get objective values
                    df_columns['Real X GUO Z'].append(round(instance.solution['z^gu'], 4))
                    df_columns['Real X Exact VFT Z'].append(round(instance.solution['z'], 4))

                    # Get approximate VFT Z
                    instance.measure_solution(approximate=True)
                    df_columns['Real X Approx VFT Z'].append(round(instance.solution['z'], 4))

                    # Get rounded X matrix objective values
                    instance.mdl_p['re-calculate x'] = True
                    instance.measure_solution()
                    df_columns['Rounded X Exact VFT Z'].append(round(instance.solution['z'], 4))
                    df_columns['Rounded X GUO Z'].append(round(instance.solution['z^gu'], 4))
                    print(model_name, solver_name, 'solved in', instance.solution['solve_time'], 'seconds.')

                    print_str = df_columns['Result'][iteration - 1]
                    for element in ['Real X GUO Z', 'Real X Exact VFT Z', 'Real X Approx VFT Z', 'Rounded X Exact VFT Z',
                                    'Rounded X GUO Z']:
                        print_str += ', ' + element + ': ' + str(df_columns[element][iteration - 1])
                    print(model_name, solver_name, print_str)

                # Solver failed!
                except:

                    # Add values
                    df_columns['Result'].append('Failed'), df_columns['Real X Approx VFT Z'].append(0)
                    df_columns['Solve Time (s)'].append(0), df_columns['Rounded X GUO Z'].append(0)
                    df_columns['Pyomo Z'].append(0), df_columns['Rounded X Exact VFT Z'].append(0)
                    df_columns['Real X Exact VFT Z'].append(0), df_columns['Real X GUO Z'].append(0)
                    df_columns['Integer X'].append('N/A'), df_columns['Integer Y'].append('N/A'), \
                    df_columns['Integer V'].append('N/A'), df_columns['Integer Q'].append('N/A')
                    df_columns['Fixed, Reserved, Alternates'].append('N/A')
                    print(model_name, solver_name, 'Failed')

    # Put dataframe together
    df = pd.DataFrame(df_columns)

    # Export to csv
    df.to_csv('Solve Time Test.csv', index=False)



