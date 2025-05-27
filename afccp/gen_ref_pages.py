"""Generate the code reference pages and structured navigation for AFCCP."""

import sys
import os
from pathlib import Path
import mkdocs_gen_files

# ------------------------ CONFIG ------------------------ #

CADET_METHOD_GROUPS = {
    "Generated Data Corrections": [
        'fix_generated_data',
        'convert_utilities_to_preferences',
        'generate_fake_afsc_preferences',
        'generate_rated_data',
        'generate_random_value_parameters',
        'generate_example_castle_value_curves',
    ],
    'Main Data Corrections': [
        'make_all_initial_real_instance_modifications',
        'import_default_value_parameters',
        'construct_rated_preferences_from_om_by_soc',
        'update_qualification_matrix_from_afsc_preferences',
        'fill_remaining_afsc_choices',
        'remove_ineligible_choices',
        'update_preference_matrices',
        'update_first_choice_cadet_utility_to_one',
        'convert_afsc_preferences_to_percentiles',
        'update_cadet_columns_from_matrices',
        'update_cadet_utility_matrices_from_cadets_data',
        'modify_rated_cadet_lists_based_on_eligibility'
    ],
    'Other Adjustments': [
        'parameter_sanity_check',
        'create_final_utility_matrix_from_new_formula',
        'set_ots_must_matches',
        'calculate_qualification_matrix'
    ],
    'Value Parameter Specifications': [
        'set_value_parameters',
        'update_value_parameters',
        'calculate_afocd_value_parameters',
        'export_value_parameters_as_defaults',
        'change_weight_function',
        'vft_to_gp_parameters'
    ],
    'Solution Generation & Algorithms': [
        'generate_random_solution',
        'rotc_rated_board_original',
        'allocate_ots_candidates_original_method',
        'soc_rated_matching_algorithm',
        'classic_hr'
    ],
    'Optimization Models & Meta-Heuristics': [
        'solve_vft_pyomo_model',
        'solve_original_pyomo_model',
        'solve_guo_pyomo_model',
        'solve_gp_pyomo_model',
        'solve_vft_main_methodology',
        'vft_genetic_algorithm',
        'genetic_matching_algorithm'
    ],
    'Solution Handling': [
        'measure_solution',
        'measure_fitness',
        'set_solution',
        'add_solution',
        'incorporate_rated_algorithm_results',
        'find_ineligible_cadets'
    ],
    'Export Data': [
        'export_data',
        'export_solution_results'
    ],
    'Data Visualizations': [
        'display_data_graph',
        'display_all_data_graphs',
        'show_value_function',
        'display_weight_function'
    ],
    'Results Visualizations': [
        'display_all_results_graphs',
        'display_cadet_individual_utility_graph',
        'display_results_graph',
        'generate_results_slides',
        'generate_comparison_slides',
        'generate_animation_slides',
        'generate_comparison_slide_components',
        'display_utility_histogram',
        'generate_bubbles_chart'
    ]
}


# ------------------------ SETUP ------------------------ #

dir_path = os.getcwd() + '/'
index = dir_path.find('afccp')
dir_path = dir_path[:index + 6]
os.chdir(dir_path)

sys.path.insert(0, os.path.abspath("afccp"))

ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "afccp/afccp"
OUTPUT_DIR = Path("reference")
nav = mkdocs_gen_files.Nav()

# ------------------------ FILE LOOP ------------------------ #

for path in sorted(SRC_DIR.rglob("*.py")):
    if 'executables' in str(path) or 'instances' in str(path) or path.name == 'setup.py':
        continue

    if path.name == "__init__.py":
        relative = path.relative_to(SRC_DIR).parent
        if relative == Path("."):
            module_path = Path("afccp")
            parts = [module_path.name]
            module_name = "afccp"
            doc_path = Path("index.md")
        else:
            module_path = relative
            parts = list(module_path.parts)
            module_name = ".".join(parts)
            doc_path = relative.with_suffix(".md")
        full_doc_path = OUTPUT_DIR / doc_path

    elif path.name == "main.py":
        # Overview file
        class_path = OUTPUT_DIR / "main" / "cadetcareerproblem_overview.md"
        nav["main/CadetCareerProblem – Overview"] = class_path.as_posix()
        with mkdocs_gen_files.open(class_path, "w") as fd:
            print("::: afccp.main.CadetCareerProblem", file=fd)
            print("    options:", file=fd)
            print("      members:", file=fd)
            print("      filters:", file=fd)
            print("        - '__init__'", file=fd)
            print("        - '!.*'", file=fd)
            print(file=fd)
        mkdocs_gen_files.set_edit_path(class_path, Path("afccp/afccp/main.py"))

        for group, methods in CADET_METHOD_GROUPS.items():
            safe_name = group.lower().replace(" ", "_")
            group_path = OUTPUT_DIR / "main" / f"cadetcareerproblem_{safe_name}.md"
            nav[f"main/CadetCareerProblem – {group}"] = group_path.as_posix()
            with mkdocs_gen_files.open(group_path, "w") as fd:
                print(f"# CadetCareerProblem – {group} Methods\n", file=fd)
                for method in methods:
                    # print(f"### `{method}`\n", file=fd)
                    print(f"::: afccp.main.CadetCareerProblem.{method}", file=fd)
                    print("    options:", file=fd)
                    print("      heading_level: 3", file=fd)
                    print("      show_root_heading: true", file=fd)
                    print("      show_root_full_path: false", file=fd)
                    print(file=fd)
            mkdocs_gen_files.set_edit_path(group_path, Path("afccp/afccp/main.py"))

        continue  # Skip default handling for main.py

    else:
        module_path = path.relative_to(SRC_DIR).with_suffix("")
        parts = list(module_path.parts)
        module_name = ".".join(parts)
        doc_path = path.relative_to(SRC_DIR).with_suffix(".md")
        full_doc_path = OUTPUT_DIR / doc_path

    if not parts:
        continue
    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        print(f"::: {module_name}", file=fd)
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(ROOT_DIR))

# ------------------------ BUILD SUMMARY ------------------------ #

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

print("\n✅ API reference and structured navigation generated successfully!")