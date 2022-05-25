import random

from afccp.core.problem_class import *
from afccp.research.laird.research_graphs import *


# Thesis Appendix: Value Parameters
def class_year_value_parameter_instance_generator(years=None, instances=None, printing=True, constrain_merit=True):
    """
    This procedure creates model instances using realistically generated value parameter sets.
    We take a class year's fixed parameters and generate value parameters for that class year.
    :param constrain_merit: if we should constrain the lower bound on merit
    :param years: list of years to generate instances for
    :param instances: list of instance indices to generate per class year
    :param printing: whether the procedure should print something
    :return: None. Exports to excel
    """

    if printing:
        print('Generating class year problem instances...')
        print('')
        print('')

    if years is None:  # year options are 2016, 2017, 2018, 2019, 2020, 2021
        years = [2021]

    if instances is None:
        instances = np.arange(1, 3).astype(int)  # You can generate as many instances as you want

    options_filename = paths['Data Processing Support'] + 'Value_Parameter_Sets_Options_Real.xlsx'
    default_filepath = paths['Data Processing Support'] + 'Value_Parameters_Defaults_Real.xlsx'
    default_value_parameters = default_value_parameters_from_excel(filepath=default_filepath)

    for year in years:

        if printing:
            print('')
            print('')
            print_str = 22 * ' '
            print_str += '<' + str(year) + '>'
            print_str += (60 - len(print_str)) * ' '
            print(print_str)
            print('')

        # Load class year data
        constraint_options = import_data(options_filename, sheet_name=str(year) + ' Constraint Options')
        filepath = paths['Problem Instances'] + str(year) + '_Original.xlsx'

        # Loop through all value parameter sets
        for i in instances:

            if printing:
                print('')
                print_str = 20 * '='
                print_str += '<Instance ' + str(i) + '>'
                print_str += (56 - len(print_str)) * '='
                print(print_str)

            # Create Instance
            data_name = str(year) + '_VP_' + str(i)
            instance = CadetCareerProblem(data_name, filepath)

            # Generate Value Parameters
            instance.generate_realistic_value_parameters(default_value_parameters, constraint_options,
                                                         deterministic=False, constrain_merit=constrain_merit)

            # Export to Excel
            year_filepath = paths['Problem Instances'] + 'Class Year VP Instances\\' + '1111aaaa' + data_name + '.xlsx'
            instance.export_to_excel(year_filepath)


def create_value_function_methodology_example(function='Merit', segment=None):
    """
    Displays the example value function graph used in the thesis
    """

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white', tight_layout=True)
    x_ticklabels = None
    if function == 'Merit':
        function = 'Balance'
        measure = 'Average Merit'
        title = 'average_merit_value_function'
        target = 0.5
        left_bm, right_bm = 0.1, 0.14

        colors = {0.4: 'red', 0.5: 'black', 0.6: 'blue'}
        linestyles = {0.4: 'dashed', 0.5: 'solid', 0.6: 'dotted'}

        rhos = {0.4: [0.1, 0.08, 0.1, 0.1], 0.5: [0.08, 0.1, 0.1, 0.15], 0.6: [0.05, 0.06, 0.06, 0.15]}
        buffer_ys = {0.4: 0.8, 0.5: 0.7, 0.6: 0.7}
        for actual in [0.4, 0.5, 0.6]:
            rho1, rho2, rho3, rho4 = rhos[actual][0], rhos[actual][1], rhos[actual][2], rhos[actual][3]
            buffer_y = buffer_ys[actual]
            vf_string = function + "|" + str(left_bm) + ", " + str(right_bm) + ", " \
                        + str(rho1) + ", " + str(rho2) + ", " + str(rho3) + ", " + str(rho4) + ", " + str(buffer_y)

            segment_dict = create_segment_dict_from_string(vf_string, target, actual=actual)

            # Grab function points
            a, f_a = value_function_builder(segment_dict, num_breakpoints=100)

            # Plot graph
            ax.plot(a, f_a, color=colors[actual], linewidth=4, label=actual, linestyle=linestyles[actual])

        # Tick marks
        x_ticks = [0, 0.5, 1]
        y_ticks = [1]

        # Plot critical line
        ax.plot((0.5, 0.5), (0, 1), color='black', linewidth=3, alpha=0.5, linestyle='--')

        # Legend
        legend = ax.legend(edgecolor='black', fontsize=30, loc='upper left', title='Eligible Cadets\nAverage Merit')
        legend.get_title().set_fontsize('30')

    elif function == 'USAFA Proportion':
        function = 'Balance'
        measure = 'USAFA Proportion'
        title = 'usafa_proportion_value_function'
        target = 0.24
        left_bm, right_bm = 0.12, 0.12

        colors = {0.1: 'red', 0.25: 'black', 0.4: 'blue'}
        linestyles = {0.1: 'dashed', 0.25: 'solid', 0.4: 'dotted'}

        rho = {0.1: 0.08, 0.25: 0.1, 0.4: 0.12}
        buffer_ys = {0.1: 0.7, 0.25: 0.8, 0.4: 0.75}
        for actual in [0.1, 0.25, 0.4]:
            rho1, rho2, rho3, rho4 = rho[actual], rho[actual], rho[actual], rho[actual]
            buffer_y = buffer_ys[actual]
            vf_string = function + "|" + str(left_bm) + ", " + str(right_bm) + ", " \
                        + str(rho1) + ", " + str(rho2) + ", " + str(rho3) + ", " + str(rho4) + ", " + str(buffer_y)

            segment_dict = create_segment_dict_from_string(vf_string, target, actual=actual)

            # Grab function points
            a, f_a = value_function_builder(segment_dict, num_breakpoints=100)

            # Plot graph
            ax.plot(a, f_a, color=colors[actual], linewidth=4, label=actual, linestyle=linestyles[actual])

        # Tick marks
        x_ticks = [0, 0.24, 1]
        y_ticks = [1]

        # Plot critical line
        ax.plot((target, target), (0, 1), color='black', linewidth=3, alpha=0.5, linestyle='--')

        # Legend
        legend = ax.legend(edgecolor='black', fontsize=30, loc='upper right', title='Eligible Cadets\nUSAFA Proportion')
        legend.get_title().set_fontsize('30')

    elif function == 'Combined Quota':
        measure = 'Number of Cadets Assigned'
        title = 'combined_quota_value_function'
        target = 50
        indiff_ub = 1.3
        con_ub = 1.5

        colors = {'Method 1': 'black', 'Method 1 (Adjust)': 'red', 'Method 2': 'blue'}
        linestyles = {'Method 1': 'solid', 'Method 1 (Adjust)': 'dashed', 'Method 2': 'dotted'}

        rho = {'Method 1': [0.25, 0.1], 'Method 1 (Adjust)': [0.3, 0.05], 'Method 2': [0.2, 0.1, 0.05]}
        buffer_ys = {'Method 2': 0.6}
        domain_max = 0.2
        for q_type in ['Method 1', 'Method 1 (Adjust)', 'Method 2']:

            if q_type != 'Method 2':

                rho1, rho2 = rho[q_type][0], rho[q_type][1]
                f_type = 'Quota_Normal'
                vf_string = f_type + "|" + str(domain_max) + ", " + str(rho1) + ", " + str(rho2)

                if q_type == 'Method 1 (Adjust)':
                    segment_dict = create_segment_dict_from_string(vf_string, target, maximum=con_ub)
                else:
                    segment_dict = create_segment_dict_from_string(vf_string, target, maximum=indiff_ub)
            else:
                rho1, rho2, rho3 = rho[q_type][0], rho[q_type][1], rho[q_type][2]
                buffer_y = buffer_ys[q_type]
                f_type = 'Quota_Over'
                vf_string = f_type + "|" + str(domain_max) + ", " \
                            + str(rho1) + ", " + str(rho2) + ", " + str(rho3) + ", " + str(buffer_y)

                segment_dict = create_segment_dict_from_string(vf_string, target, maximum=indiff_ub, actual=con_ub)

            # Grab function points
            a, f_a = value_function_builder(segment_dict, num_breakpoints=100)

            # Plot graph
            ax.plot(a, f_a, color=colors[q_type], linewidth=4, label=q_type, linestyle=linestyles[q_type])

        # Tick marks
        x_ticks = [0, target, int(indiff_ub * target), int(con_ub * target)]
        y_ticks = [0.6, 1]

        # # Plot critical lines
        # for mult in [1, indiff_ub, con_ub]:
        #     ax.plot((target * mult, target * mult), (0, 1), color='black', linewidth=3, alpha=0.4,
        #             linestyle=(0, (3, 1, 1, 1)))

        # Legend
        # legend = ax.legend(edgecolor='black', fontsize=30, loc='upper left')
        # legend.get_title().set_fontsize('30')

    elif function == 'AFOCD':
        measure = 'Degree Tier Proportion'
        title = 'afocd_value_function'
        targets = {'< 0.2': 0.2, '> 0.4': 0.4, '> 0.8': 0.8}

        colors = {'< 0.2': 'red', '> 0.4': 'blue', '> 0.8': 'black'}
        linestyles = {'< 0.2': 'dashed', '> 0.4': 'dotted', '> 0.8': 'solid'}

        rho = {'< 0.2': 0.1, '> 0.4': 0.1, '> 0.8': 0.1}
        for label in targets:

            if label == '< 0.2':
                function = 'Min Decreasing'
            else:
                function = 'Min Increasing'
            vf_string = function + "|" + str(rho[label])

            segment_dict = create_segment_dict_from_string(vf_string, targets[label])

            # Grab function points
            a, f_a = value_function_builder(segment_dict, num_breakpoints=100)

            # Plot graph
            ax.plot(a, f_a, color=colors[label], linewidth=4, label=label, linestyle=linestyles[label])

        # Tick marks
        x_ticks = [0, 0.2, 0.4, 0.8, 1]
        y_ticks = [1]

        # Legend
        legend = ax.legend(edgecolor='black', fontsize=30, loc='lower right', title='Degree Tier\nObjective')
        legend.get_title().set_fontsize('30')

    elif function == 'measure':

        function = 'Balance'
        measure = 'Measure'
        if segment is None:
            title = 'value_function_methodology_example'
        else:
            title = 'Value_Function_Segment_' + str(segment)
        target = 0.5
        actual = 0.5
        left_bm, right_bm = 0.2, 0.2
        rho1, rho2, rho3, rho4 = 0.1, 0.08, 0.08, 0.1
        max_x = 1
        buffer_y = 0.5
        vf_string = function + "|" + str(left_bm) + ", " + str(right_bm) + ", " \
                    + str(rho1) + ", " + str(rho2) + ", " + str(rho3) + ", " + str(rho4) + ", " + str(buffer_y)
        segment_dict = create_segment_dict_from_string(vf_string, target, actual=actual)

        # Grab function points
        a, f_a = value_function_builder(segment_dict, num_breakpoints=32)

        if segment in [1, 2, 3, 4]:
            start = int(0 + (segment - 1) * 25)
            end = int(26 + (segment - 1) * 25)
            a, f_a = a[start:end], f_a[start:end]

            # Plot graph
            ax.plot(a, f_a, color='black', linewidth=4)

            # Gather coordinates
            x1 = target - left_bm
            x2 = target + right_bm
            y = buffer_y

            if segment == 1:
                ax.plot((x1, x1), (0, y), c="black", linestyle="--", linewidth=2)
                ax.plot((0, x1), (y, y), c="black", linestyle="--", linewidth=2)
                ax.text(x=x1 / 2, y=y / 2, s='1', fontsize=30, c='black')
            elif segment == 2:
                ax.plot((x1, x1), (y, 1), c="black", linestyle="--", linewidth=2)
                ax.plot((x1, target), (y, y), c="black", linestyle="--", linewidth=2)
                ax.plot((x1, target), (1, 1), c="black", linestyle="--", linewidth=2)
                ax.plot((target, target), (y, 1), c="black", linestyle="--", linewidth=2)
                ax.text(x=x1 + (target - x1) / 2, y=y + (1 - y) / 2, s='2', fontsize=30, c='black')
            elif segment == 3:
                ax.plot((x2, x2), (y, 1), c="black", linestyle="--", linewidth=2)
                ax.plot((target, x2), (y, y), c="black", linestyle="--", linewidth=2)
                ax.plot((target, x2), (1, 1), c="black", linestyle="--", linewidth=2)
                ax.plot((target, target), (y, 1), c="black", linestyle="--", linewidth=2)
                ax.text(x=target + (x2 - target) / 2, y=y + (1 - y) / 2, s='3', fontsize=30, c='black')
            else:
                ax.plot((x2, x2), (0, y), c="black", linestyle="--", linewidth=2)
                ax.plot((x2, max_x), (y, y), c="black", linestyle="--", linewidth=2)
                ax.text(x=x2 + (max_x - x2) / 2, y=y / 2, s='4', fontsize=30, c='black')

            # Tick marks
            y_ticks = [y, 1]
            plt.xlim(0, max_x)
            x_ticks = np.array([0, round(x1, 2), target, round(x2, 2), max_x])
            x_ticklabels = [0, 3, 5, 7, 10]

        else:

            # Plot graph
            ax.plot(a, f_a, color='black', linewidth=4)
            ax.scatter(a, f_a, color='black', s=120)

            # Gather coordinates
            x1 = target - left_bm
            x2 = target + right_bm
            y = buffer_y

            # Piecewise divider lines
            ax.plot((x1, x1), (0, 1), c="black", linestyle="--", linewidth=2)
            ax.plot((target, target), (y, 1), c="black", linestyle="--", linewidth=2)
            ax.plot((x2, x2), (0, 1), c="black", linestyle="--", linewidth=2)
            ax.plot((max_x, max_x), (0, y), c="black", linestyle="--", linewidth=2)
            ax.plot((0, max_x), (y, y), c="black", linestyle="--", linewidth=2)
            ax.plot((x1, x2), (1, 1), c="black", linestyle="--", linewidth=2)

            # Plot divider text
            ax.text(x=x1 / 2, y=y / 2, s='1', fontsize=30, c='black')
            ax.text(x=x1 + (target - x1) / 2, y=y + (1 - y) / 2, s='2', fontsize=30, c='black')
            ax.text(x=target + (x2 - target) / 2, y=y + (1 - y) / 2, s='3', fontsize=30, c='black')
            ax.text(x=x2 + (max_x - x2) / 2, y=y / 2, s='4', fontsize=30, c='black')

            # Plot main breakpoints
            bp_x = [x1, target, x2]
            bp_y = [y, 1, y]
            ax.scatter(bp_x, bp_y, color='black', s=300)

            # Tick marks
            x_ticks = np.array([0, round(x1, 2), target, round(x2, 2), max_x])
            x_ticklabels = [0, 3, 5, 7, 10]
            y_ticks = [y, 1]

    elif function == 'Standard':

        measure = 'Measure'
        title = 'Expon_Function'
        target = 10
        vf_string = 'Min Increasing|-2'
        segment_dict = create_segment_dict_from_string(vf_string, target)

        # Grab function points
        a, f_a = value_function_builder(segment_dict, num_breakpoints=100)

        # Plot graph
        ax.plot(a, f_a, color='black', linewidth=4)

        # Tick marks
        x_ticks = np.array([0, 10])
        x_ticklabels = x_ticks
        y_ticks = [0.5, 1]

    # Set tick marks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    if x_ticklabels is None:
        x_ticklabels = x_ticks
    ax.set_xticklabels(x_ticklabels)

    # Adjust axes and ticks
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.yaxis.label.set_size(35)
    ax.set_ylabel('Value')
    ax.xaxis.label.set_size(35)
    ax.set_xlabel(measure)
    ax.margins(x=0)
    ax.margins(y=0)
    plt.ylim(0, 1.1)

    fig.savefig(paths_out['figures'] + title + '.png')
    fig.show()


def value_function_examples(function='Merit', segment=None, actual=None):
    """
    Displays the example value function graph used in the thesis
    """

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white', tight_layout=True)
    x_ticklabels = None
    if function == 'Merit':

        if actual is None:
            actual = 0.4

        function = 'Balance'
        measure = 'Average Merit'
        title = 'average_merit_value_function_' + str(actual)
        target = 0.5
        left_bm, right_bm = 0.1, 0.14

        colors = {0.4: 'red', 0.5: 'black', 0.6: 'blue'}
        linestyles = {0.4: 'dashed', 0.5: 'solid', 0.6: 'dotted'}

        rhos = {0.4: [0.1, 0.08, 0.1, 0.1], 0.5: [0.08, 0.1, 0.1, 0.15], 0.6: [0.05, 0.06, 0.06, 0.15]}
        buffer_ys = {0.4: 0.8, 0.5: 0.7, 0.6: 0.7}

        rho1, rho2, rho3, rho4 = rhos[actual][0], rhos[actual][1], rhos[actual][2], rhos[actual][3]
        buffer_y = buffer_ys[actual]
        vf_string = function + "|" + str(left_bm) + ", " + str(right_bm) + ", " \
                    + str(rho1) + ", " + str(rho2) + ", " + str(rho3) + ", " + str(rho4) + ", " + str(buffer_y)

        segment_dict = create_segment_dict_from_string(vf_string, target, actual=actual)

        # Grab function points
        a, f_a = value_function_builder(segment_dict, num_breakpoints=100)

        # Plot graph
        ax.plot(a, f_a, color=colors[actual], linewidth=4, label=actual, linestyle=linestyles[actual])

        # Tick marks
        x_ticks = [0, 0.5, 1]
        y_ticks = [1]

        # Plot critical line
        ax.plot((0.5, 0.5), (0, 1), color='black', linewidth=3, alpha=0.5, linestyle='--')

        # Legend
        legend = ax.legend(edgecolor='black', fontsize=30, loc='upper left', title='Eligible Cadets\nAverage Merit')
        legend.get_title().set_fontsize('30')

    elif function == 'USAFA Proportion':

        if actual is None:
            actual = 0.1

        function = 'Balance'
        measure = 'USAFA Proportion'
        title = 'usafa_proportion_value_function_' + str(actual)
        target = 0.24
        left_bm, right_bm = 0.12, 0.12

        colors = {0.1: 'red', 0.25: 'black', 0.4: 'blue'}
        linestyles = {0.1: 'dashed', 0.25: 'solid', 0.4: 'dotted'}

        rho = {0.1: 0.08, 0.25: 0.1, 0.4: 0.12}
        buffer_ys = {0.1: 0.7, 0.25: 0.8, 0.4: 0.75}
        rho1, rho2, rho3, rho4 = rho[actual], rho[actual], rho[actual], rho[actual]
        buffer_y = buffer_ys[actual]
        vf_string = function + "|" + str(left_bm) + ", " + str(right_bm) + ", " \
                    + str(rho1) + ", " + str(rho2) + ", " + str(rho3) + ", " + str(rho4) + ", " + str(buffer_y)

        segment_dict = create_segment_dict_from_string(vf_string, target, actual=actual)

        # Grab function points
        a, f_a = value_function_builder(segment_dict, num_breakpoints=100)

        # Plot graph
        ax.plot(a, f_a, color=colors[actual], linewidth=4, label=actual, linestyle=linestyles[actual])

        # Tick marks
        x_ticks = [0, 0.24, 1]
        y_ticks = [1]

        # Plot critical line
        ax.plot((target, target), (0, 1), color='black', linewidth=3, alpha=0.5, linestyle='--')

        # Legend
        legend = ax.legend(edgecolor='black', fontsize=30, loc='upper right', title='Eligible Cadets\nUSAFA Proportion')
        legend.get_title().set_fontsize('30')

    elif function == 'Combined Quota':
        measure = 'Number of Cadets Assigned'
        if segment is None:
            segment = 0

        if actual is None:
            actual = 50
        title = 'combined_quota_value_function_' + segment

        rhos = {1: round(random.random() * 0.05 + 0.05, 2), 2: round(random.random() * 0.2 + 0.4, 2),
                3: round(random.random() * 0.2 + 0.7, 2), 4: round(random.random() * 0.1 + 0.15, 2)}
        y1 = round(random.random() * 0.1 + 0.8, 2)
        y2 = round(random.random() * 0.1 + 0.85, 2)
        minimum, maximum = 50, 70
        target = 50
        pref_target = actual
        vf_string = "Quota_Direct|" + str(rhos[1]) + ", " + str(rhos[2]) + ", " + \
                    str(rhos[3]) + ", " + str(rhos[4]) + ", " + str(y1) + ", " + str(y2) + ", " + str(pref_target)
        segment_dict = create_segment_dict_from_string(vf_string, target, maximum=maximum, minimum=minimum)
        colors = {1: 'black', 2: 'red', 3: 'blue'}
        linestyles = {1: 'solid', 2: 'dashed', 3: 'dotted'}

        # Grab function points
        a, f_a = value_function_builder(segment_dict, num_breakpoints=100)

        # Plot graph
        ax.plot(a, f_a, color=colors[actual], linewidth=4, label=q_type, linestyle=linestyles[q_type])

        for q_type in ['Method 1', 'Method 1 (Adjust)', 'Method 2']:

            if q_type != 'Method 2':

                rho1, rho2 = rho[q_type][0], rho[q_type][1]
                f_type = 'Quota_Normal'
                vf_string = f_type + "|" + str(domain_max) + ", " + str(rho1) + ", " + str(rho2)

                if q_type == 'Method 1 (Adjust)':
                    segment_dict = create_segment_dict_from_string(vf_string, target, maximum=con_ub)
                else:
                    segment_dict = create_segment_dict_from_string(vf_string, target, maximum=indiff_ub)
            else:
                rho1, rho2, rho3 = rho[q_type][0], rho[q_type][1], rho[q_type][2]
                buffer_y = buffer_ys[q_type]
                f_type = 'Quota_Over'
                vf_string = f_type + "|" + str(domain_max) + ", " \
                            + str(rho1) + ", " + str(rho2) + ", " + str(rho3) + ", " + str(buffer_y)

                segment_dict = create_segment_dict_from_string(vf_string, target, maximum=indiff_ub, actual=con_ub)

            # Grab function points
            a, f_a = value_function_builder(segment_dict, num_breakpoints=100)

            # Plot graph
            ax.plot(a, f_a, color=colors[q_type], linewidth=4, label=q_type, linestyle=linestyles[q_type])

        # Tick marks
        x_ticks = [0, target, int(indiff_ub * target), int(con_ub * target)]
        y_ticks = [0.6, 1]

        # # Plot critical lines
        # for mult in [1, indiff_ub, con_ub]:
        #     ax.plot((target * mult, target * mult), (0, 1), color='black', linewidth=3, alpha=0.4,
        #             linestyle=(0, (3, 1, 1, 1)))

        # Legend
        # legend = ax.legend(edgecolor='black', fontsize=30, loc='upper left')
        # legend.get_title().set_fontsize('30')

    elif function == 'AFOCD':
        measure = 'Degree Tier Proportion'
        title = 'afocd_value_function'
        targets = {'< 0.2': 0.2, '> 0.4': 0.4, '> 0.8': 0.8}

        colors = {'< 0.2': 'red', '> 0.4': 'blue', '> 0.8': 'black'}
        linestyles = {'< 0.2': 'dashed', '> 0.4': 'dotted', '> 0.8': 'solid'}

        rho = {'< 0.2': 0.1, '> 0.4': 0.1, '> 0.8': 0.1}
        for label in targets:

            if label == '< 0.2':
                function = 'Min Decreasing'
            else:
                function = 'Min Increasing'
            vf_string = function + "|" + str(rho[label])

            segment_dict = create_segment_dict_from_string(vf_string, targets[label])

            # Grab function points
            a, f_a = value_function_builder(segment_dict, num_breakpoints=100)

            # Plot graph
            ax.plot(a, f_a, color=colors[label], linewidth=4, label=label, linestyle=linestyles[label])

        # Tick marks
        x_ticks = [0, 0.2, 0.4, 0.8, 1]
        y_ticks = [1]

        # Legend
        legend = ax.legend(edgecolor='black', fontsize=30, loc='lower right', title='Degree Tier\nObjective')
        legend.get_title().set_fontsize('30')

    elif function == 'measure':

        function = 'Balance'
        measure = 'Measure'
        if segment is None:
            title = 'value_function_methodology_example'
        else:
            title = 'Value_Function_Segment_' + str(segment)
        target = 0.5
        actual = 0.5
        left_bm, right_bm = 0.2, 0.2
        rho1, rho2, rho3, rho4 = 0.1, 0.08, 0.08, 0.1
        max_x = 1
        buffer_y = 0.5
        vf_string = function + "|" + str(left_bm) + ", " + str(right_bm) + ", " \
                    + str(rho1) + ", " + str(rho2) + ", " + str(rho3) + ", " + str(rho4) + ", " + str(buffer_y)
        segment_dict = create_segment_dict_from_string(vf_string, target, actual=actual)

        # Grab function points
        a, f_a = value_function_builder(segment_dict, num_breakpoints=32)

        if segment in [1, 2, 3, 4]:
            start = int(0 + (segment - 1) * 25)
            end = int(26 + (segment - 1) * 25)
            a, f_a = a[start:end], f_a[start:end]

            # Plot graph
            ax.plot(a, f_a, color='black', linewidth=4)

            # Gather coordinates
            x1 = target - left_bm
            x2 = target + right_bm
            y = buffer_y

            if segment == 1:
                ax.plot((x1, x1), (0, y), c="black", linestyle="--", linewidth=2)
                ax.plot((0, x1), (y, y), c="black", linestyle="--", linewidth=2)
                ax.text(x=x1 / 2, y=y / 2, s='1', fontsize=30, c='black')
            elif segment == 2:
                ax.plot((x1, x1), (y, 1), c="black", linestyle="--", linewidth=2)
                ax.plot((x1, target), (y, y), c="black", linestyle="--", linewidth=2)
                ax.plot((x1, target), (1, 1), c="black", linestyle="--", linewidth=2)
                ax.plot((target, target), (y, 1), c="black", linestyle="--", linewidth=2)
                ax.text(x=x1 + (target - x1) / 2, y=y + (1 - y) / 2, s='2', fontsize=30, c='black')
            elif segment == 3:
                ax.plot((x2, x2), (y, 1), c="black", linestyle="--", linewidth=2)
                ax.plot((target, x2), (y, y), c="black", linestyle="--", linewidth=2)
                ax.plot((target, x2), (1, 1), c="black", linestyle="--", linewidth=2)
                ax.plot((target, target), (y, 1), c="black", linestyle="--", linewidth=2)
                ax.text(x=target + (x2 - target) / 2, y=y + (1 - y) / 2, s='3', fontsize=30, c='black')
            else:
                ax.plot((x2, x2), (0, y), c="black", linestyle="--", linewidth=2)
                ax.plot((x2, max_x), (y, y), c="black", linestyle="--", linewidth=2)
                ax.text(x=x2 + (max_x - x2) / 2, y=y / 2, s='4', fontsize=30, c='black')

            # Tick marks
            y_ticks = [y, 1]
            plt.xlim(0, max_x)
            x_ticks = np.array([0, round(x1, 2), target, round(x2, 2), max_x])
            x_ticklabels = [0, 3, 5, 7, 10]

        else:

            # Plot graph
            ax.plot(a, f_a, color='black', linewidth=4)
            ax.scatter(a, f_a, color='black', s=120)

            # Gather coordinates
            x1 = target - left_bm
            x2 = target + right_bm
            y = buffer_y

            # Piecewise divider lines
            ax.plot((x1, x1), (0, 1), c="black", linestyle="--", linewidth=2)
            ax.plot((target, target), (y, 1), c="black", linestyle="--", linewidth=2)
            ax.plot((x2, x2), (0, 1), c="black", linestyle="--", linewidth=2)
            ax.plot((max_x, max_x), (0, y), c="black", linestyle="--", linewidth=2)
            ax.plot((0, max_x), (y, y), c="black", linestyle="--", linewidth=2)
            ax.plot((x1, x2), (1, 1), c="black", linestyle="--", linewidth=2)

            # Plot divider text
            ax.text(x=x1 / 2, y=y / 2, s='1', fontsize=30, c='black')
            ax.text(x=x1 + (target - x1) / 2, y=y + (1 - y) / 2, s='2', fontsize=30, c='black')
            ax.text(x=target + (x2 - target) / 2, y=y + (1 - y) / 2, s='3', fontsize=30, c='black')
            ax.text(x=x2 + (max_x - x2) / 2, y=y / 2, s='4', fontsize=30, c='black')

            # Plot main breakpoints
            bp_x = [x1, target, x2]
            bp_y = [y, 1, y]
            ax.scatter(bp_x, bp_y, color='black', s=300)

            # Tick marks
            x_ticks = np.array([0, round(x1, 2), target, round(x2, 2), max_x])
            x_ticklabels = [0, 3, 5, 7, 10]
            y_ticks = [y, 1]

    elif function == 'Standard':

        measure = 'Measure'
        title = 'Expon_Function'
        target = 10
        vf_string = 'Min Increasing|-2'
        segment_dict = create_segment_dict_from_string(vf_string, target)

        # Grab function points
        a, f_a = value_function_builder(segment_dict, num_breakpoints=100)

        # Plot graph
        ax.plot(a, f_a, color='black', linewidth=4)

        # Tick marks
        x_ticks = np.array([0, 10])
        x_ticklabels = x_ticks
        y_ticks = [0.5, 1]

    # Set tick marks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    if x_ticklabels is None:
        x_ticklabels = x_ticks
    ax.set_xticklabels(x_ticklabels)

    # Adjust axes and ticks
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.yaxis.label.set_size(35)
    ax.set_ylabel('Value')
    ax.xaxis.label.set_size(35)
    ax.set_xlabel(measure)
    ax.margins(x=0)
    ax.margins(y=0)
    plt.ylim(0, 1.1)

    fig.savefig(paths_out['figures'] + title + '.png')
    fig.show()


def cadet_weight_function_chart(rho=-0.3):
    """
    Displays the all cadet weight function choices shown in the thesis appendix
    :param rho: rho parameter for exponential function
    :return: fig
    """

    # Create Chart
    figsize = (12, 10)
    facecolor = 'white'
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor, tight_layout=True)
    # ax.set_aspect('equal', adjustable='box')

    # Get data
    x = np.arange(1001) / 1000
    functions = {'Equal': {'linestyle': 'solid', 'color': 'black'}, 'Linear': {'linestyle': 'dotted', 'color': 'blue'},
                 'Exponential': {'linestyle': 'dashed', 'color': 'red'}}

    # Create chart
    for func in functions:
        if func == 'Equal':
            y = np.repeat(0.5, len(x))
        elif func == 'Linear':
            y = x
        elif func == 'Exponential':
            y = [(1 - exp(-i / rho)) / (1 - exp(-1 / rho)) for i in x]
        ax.plot(x, y, color=functions[func]['color'], linestyle=functions[func]['linestyle'], label=func,
                linewidth=4)

    # Set ticks and labels
    y_ticks = [0.5, 1]
    ax.set_yticks(y_ticks)
    x_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.set_xticks(x_ticks)

    # Adjust axes and ticks
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.yaxis.label.set_size(35)
    ax.set_ylabel('Swing Weight')
    ax.xaxis.label.set_size(35)
    ax.set_xlabel('Percentile')
    ax.margins(x=0)
    ax.margins(y=0)

    # Legend
    ax.legend(edgecolor='black', fontsize=35, loc='upper left')

    title = "Cadet Weight Function Graph"
    fig.savefig(paths['Charts & Figures'] + title + '.png')
    fig.show()


# Thesis Other
def create_latex_table_from_dataframe(df=None, table_name=None, df_type='Other', printing=True):
    """
    Writes the latex code for a table of the specified dataframe to a .txt file
    :param table_name: name of table
    :param df_type: type of table (default to solver size table)
    :param printing: whether the procedure should print something
    :return: None (writes to .txt file)
    """
    if df is None:

        if table_name is None:
            table_name = 'Thesis_4.2.1_Size_Table'
            df_type = 'Size'

        # Load dataframe
        df = import_data(paths['Tables'] + table_name + '.xlsx')

    else:

        table_name = 'Example_Table'

    if printing:
        print('Writing latex code for table using this dataframe:')
        print('')
        print(df)

    # Gather useful info from dataframe
    headers = list(df.columns)
    num_cols = len(headers)
    num_rows = len(df)

    # If this is the chart for size analysis or data metrics, I span the headers across 2 rows
    if df_type == 'Size':
        caption = 'Averaged Solver Results on Different Sized Problem Instances'
        label = 'size_table'
        multi_row_header = True
        table_align = 'l' + "c" * (num_cols - 1)
    elif df_type == 'Data Metrics':
        caption = 'Evaluation Metrics for Averaged Synthetic Data and Real Data'
        label = 'avg_datametrics'
        multi_row_header = True
        table_align = 'l' + "c" * (num_cols - 1)
    else:
        caption = 'Example Table'
        label = 'example_table'
        multi_row_header = False
        table_align = "c" * (num_cols)

    lines = ["\\begin{table}[!b]", "\t\\centering", "\t\\caption{" + caption + "} \\label{" + label + "}",
             "\t{\\small", "\t\\begin{tabular}{" + table_align + "}", "\t\t\\hline",
             "\t\t\\hline"]

    # Correct underscores in headers
    for c, header in enumerate(headers):
        if '_' in header:
            header_split = header.split('_')
            headers[c] = header_split[0] + '\\textunderscore ' + header_split[1]

    # Create headers
    if multi_row_header:
        headers1 = []
        headers2 = []

        # Loop through all the headers
        for header in headers:

            # Two lines for the headers here
            split_header = header.split(' ')
            if len(split_header) == 1:
                headers1.append(split_header[0])
                headers2.append("")
            elif len(split_header) == 2:
                headers1.append(split_header[0])
                headers2.append(split_header[1])
            else:
                headers1.append(split_header[0])
                header2 = ""
                for word in split_header[1:]:
                    header2 += word + " "
                headers2.append(header2[:-1])
        str1 = ""
        str2 = ""
        str3 = ""
        for c, header in enumerate(headers[:-1]):

            if c == 0:
                if table_align == 'l' + "c" * (num_cols - 1):
                    str1 += "\\multicolumn{1}{c}{\\multirow{2}{*}{" + headers1[c] + "}}" + " & "
                else:
                    str1 += "\\multirow{2}{*}{" + headers1[c] + "}" + " & "
            else:
                str1 += "\\multirow{2}{*}{" + headers1[c] + "}" + " & "
            str2 += "& "
            str3 += headers2[c] + " & "
        str1 += "\\multirow{2}{*}{" + headers1[len(headers) - 1] + "}" + "\\\\"
        str2 += "\\\\[-6pt]  % This is just to add a little more separation between the words"
        str3 += headers2[c] + " \\\\"
        lines.append("\t\t" + str1)
        lines.append("\t\t" + str2)
        lines.append("\t\t" + str3)
    else:
        str1 = ""
        for header in headers[:-1]:
            str1 += header + " & "
        str1 += headers[len(headers) - 1] + "\\\\"
        lines.append("\t\t" + str1)

    # Append the headers and additional horizontal lines
    lines.append("\t\t\\hline")
    lines.append("\t\t\\hline")

    # Loop through the contents of the dataframe and write them into latex
    for i in range(num_rows):
        row_str = ""
        for j in range(num_cols - 1):

            # Change the underscores to "\times" for latex for the "Size" column
            if j == 0 and df_type == 'Size':
                size_split = df.iloc[i, j].split('_')
                row_str += "$" + size_split[0] + " \\times " + size_split[1] + "$ & "

            # Adjust the rounding for the elements of the size table
            elif df_type == 'Size':
                num = df.iloc[i, j]
                if int(num) >= 100:
                    num = int(num)
                row_str += str(num) + ' & '

            # Otherwise just place the same entry
            else:
                row_str += str(df.iloc[i, j]) + ' & '

        # Add this line of latex code
        row_str += str(df.iloc[i, num_cols - 1]) + " \\\\"
        lines.append("\t\t" + row_str)
        lines.append("\t\t\\hline")

    # Closing lines
    lines.append("\t\t\\hline")
    lines.append("\t\\end{tabular}}")
    lines.append("\\end{table}")

    # Print the latex code to the console
    if printing:
        print('')
        print('Latex code output:')
        print('')
        for line in lines:
            print(line)

    # Write latex code to text file
    filepath = paths['Tables'] + table_name + '.txt'
    with open(filepath, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')


# Thesis Defense
def create_value_function_optimization_example(slide=1):
    """
    Displays the example breakpoint graph used in the thesis
    :return: None
    """
    F_bp = [0, 0.2, 0.5, 0.8, 1]
    F_v = [0, 0.1, 1, 0.1, 0]
    x, y = F_bp, F_v

    fig, ax = plt.subplots(figsize=(11, 10), facecolor='white', tight_layout=True)
    ax.plot(x, y, color="black", linewidth=3, zorder=1)
    ax.scatter(x, y, color='black', s=150, zorder=2)
    x_ticks = F_bp
    ax.set_xticks(x_ticks)
    y_ticks = [0.1, 1]
    ax.set_yticks(y_ticks)

    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.yaxis.label.set_size(35)
    ax.set_ylabel('Value')
    ax.xaxis.label.set_size(35)
    ax.set_xlabel('Measure')
    ax.margins(x=0)
    ax.margins(y=0)
    plt.ylim(0, 1.1)
    plt.xlim(0, 1)

    if slide == 2:
        ax.scatter(x, y, color='blue', s=150, zorder=2)
        f_text = '$%s$' % '\\mathcal{L}_{jk}' + ' = {1, 2, 3, 4, 5}'
        ax.text(x=0.6, y=1, s=f_text, fontsize=30, c='black')
    if slide == 3:
        f_text = '$%s$' % 'r_{jk}' + ' = 5'
        ax.text(x=0.6, y=1, s=f_text, fontsize=30, c='black')
    elif slide == 4:

        # Placeholder text
        a_text = '$%s$' % 'a_{jk2}'
        ax.text(x=0.5, y=0.5, s=a_text, fontsize=30, c='black')
        x_tick_labels = ['0.0', '', '0.5', '0.8', '1.0']
        ax.set_xticklabels(x_tick_labels)

        # Placeholder Y
        f_text = '$%s$' % '\\hat{f}_{jk2}'
        ax.text(x=0.5, y=0.25, s=f_text, fontsize=30, c='black')
        y_tick_labels = ['', '1.0']
        ax.set_yticklabels(y_tick_labels)

        # Breakpoint indicator lines
        ax.plot((0.2, 0.2), (0, 0.1), c="blue", linestyle="--", linewidth=3, zorder=1)
        ax.plot((0, 0.2), (0.1, 0.1), c="blue", linestyle="--", linewidth=3, zorder=1)
        ax.scatter(0.2, 0.1, color='blue', s=150, zorder=2)

    elif slide == 5:

        x_ticks = [0, 0.2, 0.4, 0.5, 0.8, 1]
        ax.set_xticks(x_ticks)
        ax.plot((0.5, 0.5), (0, 0.1), c='black', alpha=0.5, linestyle='--', linewidth=2, zorder=1)
        ax.plot((0.4, 0.4), (0, 0.1), c='black', alpha=0.5, linestyle='--', linewidth=2, zorder=1)
        ax.plot((0.2, 0.2), (0, 0.1), c='black', alpha=0.5, linestyle='--', linewidth=2, zorder=1)
        ax.plot((0.2, 0.4), (0.1, 0.1), c='black', alpha=1, linestyle='--', linewidth=3, zorder=1)
        ax.plot((0.4, 0.5), (0.1, 0.1), c='red', alpha=1, linestyle='--', linewidth=3, zorder=1)
        ax.scatter(0.4, 0.1, color='red', s=150, zorder=2)

    elif slide == 6:

        x_ticks = [0, 0.2, 0.4, 0.5, 0.8, 1]
        ax.set_xticks(x_ticks)
        ax.plot((0, 0.2), (1, 1), c='black', alpha=0.5, linestyle='--', linewidth=2, zorder=1)
        ax.plot((0, 0.2), (0.1, 0.1), c='black', alpha=0.5, linestyle='--', linewidth=2, zorder=1)
        ax.plot((0, 0.2), (0.7, 0.7), c='black', alpha=0.5, linestyle='--', linewidth=2, zorder=1)
        ax.plot((0.2, 0.2), (0.1, 0.7), c='black', alpha=1, linestyle='--', linewidth=3, zorder=1)
        ax.plot((0.2, 0.2), (0.7, 1), c='red', alpha=1, linestyle='--', linewidth=3, zorder=1)
        ax.scatter(0.2, 0.7, color='red', s=150, zorder=2)
        y_ticks = [0.1, 0.7, 1]
        ax.set_yticks(y_ticks)

    elif slide == 7:

        # add measure
        ax.scatter(0.4, 0.7, color='red', s=150, zorder=2)
        #
        # Indicator lines
        ax.plot((0.4, 0.4), (0, 0.7), c="black", linestyle="--", linewidth=3, zorder=1, alpha=0.5)
        ax.plot((0, 0.4), (0.7, 0.7), c="black", linestyle="--", linewidth=3, zorder=1, alpha=0.5)

        x_ticks = [0, 0.2, 0.4, 0.5, 0.8, 1]
        ax.set_xticks(x_ticks)
        y_ticks = [0.1, 0.7, 1]
        ax.set_yticks(y_ticks)

    # Save slide
    fig.savefig(paths['Charts & Figures'] + 'breakpoint_method_slide_' + str(slide) + '.png')
    fig.show()
