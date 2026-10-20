"""
One Market Model Feature Registry

Centralized definitions of all modeling features,
constraints, and priorities supported by AFCCP.
"""
import numpy as np


def resolve_meet_afsc_pgl_feature(instance):
    """
    Resolve the Meet AFSC PGL Targets feature.

    Ensures all AFSC Combined Quota constraints are active when the feature is
    enabled in the checklist. If all constraints are already active, automatically
    enables the checklist. Otherwise, preserves the existing configuration.

    Returns:
        str: Formatted feature resolution report with color-coded statuses.
    """

    # ___________________________________________________INITIALIZATION_________________________________________________

    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p
    checklist = instance.feature_checklist
    feature_id = 'meet_afsc_pgl'
    feature_name = instance.model_features[feature_id]['name']

    # Standardized status formatting
    colors = {'ACTIVE': '\033[92m', 'FULLY ACTIVE': '\033[92m', 'PARTIALLY ACTIVE': '\033[93m', 'INACTIVE': '\033[90m'}
    color = lambda status: f"{colors.get(status, '')}{status}\033[0m"
    status_line = lambda label, status: f"    {label}: {color(status)}"

    # Identify Combined Quota constraints and their initial activation states
    k = np.where(vp['objectives'] == 'Combined Quota')[0][0]
    active_indices = np.where(vp['constraint_type'][:, k] > 0)[0]
    num_on, num_total = len(active_indices), p['M']

    checklist_initial = bool(checklist[feature_id])
    parameters_active = num_on == num_total
    parameter_status = 'FULLY ACTIVE' if parameters_active else ('PARTIALLY ACTIVE' if num_on else 'INACTIVE')
    checklist_status = 'ACTIVE' if checklist_initial else 'INACTIVE'

    # ___________________________________________________RESOLVE FEATURE_________________________________________________

    if checklist_initial:
        if not parameters_active:
            vp['constraint_type'][:, k] = 2
            resolution_status = 'ACTIVE'
            resolution = f'Checklist activated. Enabled all {num_total} AFSC quota constraints.'
        else:
            resolution_status = 'ACTIVE'
            resolution = 'Feature already fully configured. No changes required.'

    elif parameters_active:
        checklist[feature_id] = True
        resolution_status = 'ACTIVE'
        resolution = 'All AFSC quota constraints already active. Activated checklist.'

    else:
        resolution_status = 'INACTIVE'
        resolution = 'Feature remains disabled. Existing parameter configuration preserved.'

    # ___________________________________________________CONSTRUCT REPORT_________________________________________________

    report = [
        f'[{feature_name}]',
        '  Initial State:',
        status_line('Checklist', checklist_status),
        status_line('Parameters', parameter_status),
        '  Details:',
        f'    Active AFSC Quotas: {num_on}/{num_total}'
    ]

    # Include initially active AFSCs when the constraints were only partially enabled.
    if 0 < num_on < num_total:
        active_afscs = [str(mdl_p['afscs'][j]) for j in active_indices]
        report.append(f"    Initially Active AFSCs: {', '.join(active_afscs)}")

    report.extend([
        '  Resolution:',
        status_line('Status', resolution_status),
        f'    {resolution}'
    ])

    return '\n'.join(report)


def resolve_meet_rated_pgl_feature(instance):
    """
    Resolve the Meet Rated PGL Targets feature.

    Ensures all Rated AFSC Combined Quota constraints are active when the feature
    is enabled in the checklist. If all rated constraints are already active,
    automatically enables the checklist. Otherwise, preserves the existing configuration.

    Returns:
        str: Formatted feature resolution report with color-coded statuses.
    """

    # ___________________________________________________INITIALIZATION_________________________________________________

    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p
    checklist = instance.feature_checklist
    feature_id = 'meet_rated_pgl'
    feature_name = instance.model_features[feature_id]['name']

    # Standardized status formatting
    colors = {'ACTIVE': '\033[92m', 'FULLY ACTIVE': '\033[92m', 'PARTIALLY ACTIVE': '\033[93m', 'INACTIVE': '\033[90m'}
    color = lambda status: f"{colors.get(status, '')}{status}\033[0m"
    status_line = lambda label, status: f"    {label}: {color(status)}"

    # Identify Rated AFSC Combined Quota constraints and their initial activation states
    k = np.where(vp['objectives'] == 'Combined Quota')[0][0]
    rated_indices = np.asarray(p['J^Rated'], dtype=int)
    active_indices = rated_indices[vp['constraint_type'][rated_indices, k] > 0]
    num_on, num_total = len(active_indices), len(rated_indices)

    checklist_initial = bool(checklist[feature_id])
    parameters_active = num_on == num_total
    parameter_status = 'FULLY ACTIVE' if parameters_active else ('PARTIALLY ACTIVE' if num_on else 'INACTIVE')
    checklist_status = 'ACTIVE' if checklist_initial else 'INACTIVE'

    # ___________________________________________________RESOLVE FEATURE_________________________________________________

    if checklist_initial:
        if not parameters_active:
            vp['constraint_type'][rated_indices, k] = 2
            resolution_status = 'ACTIVE'
            resolution = f'Checklist activated. Enabled all {num_total} Rated AFSC quota constraints.'
        else:
            resolution_status = 'ACTIVE'
            resolution = 'Feature already fully configured. No changes required.'

    elif parameters_active:
        checklist[feature_id] = True
        resolution_status = 'ACTIVE'
        resolution = 'All Rated AFSC quota constraints already active. Activated checklist.'

    else:
        resolution_status = 'INACTIVE'
        resolution = 'Feature remains disabled. Existing parameter configuration preserved.'

    # ___________________________________________________CONSTRUCT REPORT_________________________________________________

    report = [
        f'[{feature_name}]',
        '  Initial State:',
        status_line('Checklist', checklist_status),
        status_line('Parameters', parameter_status),
        '  Details:',
        f'    Active Rated AFSC Quotas: {num_on}/{num_total}'
    ]

    # Include initially active Rated AFSCs when constraints were only partially enabled.
    if 0 < num_on < num_total:
        active_afscs = [str(mdl_p['afscs'][j]) for j in active_indices]
        report.append(f"    Initially Active Rated AFSCs: {', '.join(active_afscs)}")

    report.extend([
        '  Resolution:',
        status_line('Status', resolution_status),
        f'    {resolution}'
    ])

    return '\n'.join(report)


def resolve_meet_rated_soc_pgl_feature(instance):
    """
    Resolve the Meet Rated SOC PGL Targets feature.

    Enforces Combined Quota constraints for all Rated AFSCs and additional
    SOC-specific quota constraints for Rated AFSCs without SOC suffixes.

    Synchronizes the feature checklist with the model parameters and reports
    initial activation states, missing constraints, and final resolution.

    Returns:
        str: Formatted feature resolution report with color-coded statuses.
    """

    # ___________________________________________________INITIALIZATION_________________________________________________

    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p
    checklist = instance.feature_checklist
    feature_id = 'meet_rated_soc_pgl'
    feature_name = instance.model_features[feature_id]['name']

    # Standardized status formatting
    colors = {'ACTIVE': '\033[92m', 'FULLY ACTIVE': '\033[92m',
              'PARTIALLY ACTIVE': '\033[93m', 'INACTIVE': '\033[90m',
              'ERROR': '\033[91m'}
    color = lambda status: f"{colors.get(status, '')}{status}\033[0m"
    status_line = lambda label, status: f"    {label}: {color(status)}"

    # Identify Rated AFSCs and quota objective indices
    rated_indices = np.asarray(p['J^Rated'], dtype=int)
    combined_k = np.where(vp['objectives'] == 'Combined Quota')[0][0]
    soc_indices = {
        soc: np.where(vp['objectives'] == f'{soc.upper()} Quota')[0][0]
        for soc in p['SOCs']
    }

    # Determine which constraints are required and currently active
    active_indices, missing_constraints = [], []

    for j in rated_indices:
        afsc = str(p['afscs'][j])

        # All Rated AFSCs require Combined Quota constraints
        required = [('Combined Quota', combined_k)]

        # Non-SOC-specific Rated AFSCs also require individual SOC quotas
        if not afsc.endswith(('_U', '_R', '_O')):
            required.extend((f'{soc.upper()} Quota', k) for soc, k in soc_indices.items())

        # Check each required constraint
        missing = [(name, k) for name, k in required if vp['constraint_type'][j, k] <= 0]

        if missing:
            missing_constraints.extend((j, name, k) for name, k in missing)
        else:
            active_indices.append(j)

    # Determine initial activation states
    checklist_initial = bool(checklist[feature_id])
    num_on, num_total = len(active_indices), len(rated_indices)
    parameters_active = num_on == num_total

    parameter_status = 'FULLY ACTIVE' if parameters_active else ('PARTIALLY ACTIVE' if num_on else 'INACTIVE')
    checklist_status = 'ACTIVE' if checklist_initial else 'INACTIVE'

    # ___________________________________________________RESOLVE FEATURE________________________________________________

    if checklist_initial:
        if not parameters_active:

            # Activate only the missing required quota constraints
            for j, name, k in missing_constraints:
                vp['constraint_type'][j, k] = 2
            resolution_status = 'ACTIVE'
            resolution = (f'Checklist activated. Enabled {len(missing_constraints)} missing '
                          f'quota constraints across {num_total - num_on} Rated AFSCs.')

        else:
            resolution_status = 'ACTIVE'
            resolution = 'Feature already fully configured. No changes required.'

    elif parameters_active:
        checklist[feature_id] = True
        resolution_status = 'ACTIVE'
        resolution = 'All required Rated SOC quota constraints already active. Activated checklist.'

    else:
        resolution_status = 'INACTIVE'
        resolution = 'Feature remains disabled. Existing parameter configuration preserved.'

    # _________________________________________________CONSTRUCT REPORT_________________________________________________

    report = [
        f'[{feature_name}]',
        '  Initial State:',
        status_line('Checklist', checklist_status),
        status_line('Parameters', parameter_status),
        '  Details:',
        f'    Fully Configured Rated AFSCs: {num_on}/{num_total}',
        f'    Initially Missing Quota Constraints: {len(missing_constraints)}'
    ]

    # Include individual missing constraints when partially configured
    if missing_constraints:
        report.append('    Initially Missing Constraints:')
        for j, name, k in missing_constraints:
            report.append(f"      - {p['afscs'][j]}: {name}")

    report.extend([
        '  Resolution:',
        status_line('Status', resolution_status),
        f'    {resolution}'
    ])

    return '\n'.join(report)


def resolve_rated_stability_soc_feature(instance):
    """
    Resolve the Rated Reserves by SOC feature and synchronize its checklist with model parameters.

    The rated matching process consists of two stages:

    STAGE 1 — SOC-Specific Hospital/Residents Matching
        The soc_rated_matching_algorithm() method generates preliminary rated matches
        and reserves for each commissioning source.

        Required solutions:
            - Rated {SOC} HR (Matches)
            - Rated {SOC} HR (Reserves)

        These preliminary solutions are validated against current SOC-specific rated quotas.

    STAGE 2 — Incorporation and Augmentation
        The incorporate_rated_algorithm_results() instance method incorporates the
        preliminary solutions into the model parameters.

        Its underlying incorporate_rated_results_in_parameters() function:
            1. Adds preliminary matches to J^Fixed.
            2. Adds preliminary reserves to J^Reserved.
            3. Calls augment_rated_algorithm_results() for each SOC.
            4. Creates alternate lists and associated optimization sets.
            5. Converts inevitable reserved assignments into fixed assignments.

        Consequently, a cadet originally reserved by Stage 1 may legitimately appear
        in J^Fixed rather than J^Reserved after Stage 2.

    VALIDATION
        - Preliminary matched cadets must appear in J^Fixed with the correct AFSC.
        - Preliminary reserved cadets must either:
            a. Remain in J^Reserved, with the expected rated AFSC as the final
               element of their permitted preference array; or
            b. Have been converted to J^Fixed for that rated AFSC.
        - Required alternate-list optimization structures must exist for each SOC.
        - Additional fixed or reserved cadets are permitted.
        - Final solution names 'Rated Matches' and 'Rated Reserves' are reported
          when available, but are not required because the supplied incorporation
          functions establish final assignments directly in the parameters.

    RESOLUTION
        - If the checklist is active, generate missing preliminary solutions.
        - Validate preliminary SOC/AFSC quotas before incorporation.
        - If the checklist is active and parameters are incomplete, execute the
          existing incorporation method, including its augmentation stage.
        - If parameters are already fully incorporated, activate the checklist.
        - If neither the checklist nor parameters activate the feature, leave it inactive.
        - Report validation or execution failures without deleting existing parameters.

    Args:
        instance (CadetCareerProblem): Current problem instance.

    Returns:
        str: Formatted feature resolution report with ANSI-colored status labels.
    """

    # ___________________________________________________INITIALIZATION_________________________________________________

    # Shorthand and feature information
    p, mdl_p = instance.parameters, instance.mdl_p
    checklist = instance.feature_checklist
    feature_id = 'rated_stability_soc'
    feature_name = instance.model_features[feature_id]['name']

    # Standardized report formatting
    colors = {'ACTIVE': '\033[92m', 'FULLY ACTIVE': '\033[92m', 'PARTIALLY ACTIVE': '\033[93m',
              'INACTIVE': '\033[90m', 'ERROR': '\033[91m'}
    color = lambda status: f"{colors.get(status, '')}{status}\033[0m"
    status_line = lambda label, status, explanation=None: f"    {label}: {color(status)}" + (f" — {explanation}" if explanation else "")

    # Report components and original feature state
    details, actions, issues = [], [], []
    checklist_initial = bool(checklist[feature_id])
    resolution_status = 'INACTIVE'

    # Identify the SOCs included in the current model configuration.
    all_socs = [str(s).lower() for s in p['SOCs']]
    socs = [str(s).lower() for s in (mdl_p['socs_to_use'] if mdl_p.get('socs_to_use') is not None else p['SOCs'])]
    excluded_socs = sorted(set(all_socs) - set(socs))
    invalid_socs = sorted(set(socs) - set(all_socs))

    # Stage 1 solutions are SOC-specific; final solution names are optional.
    required = {soc: {kind: f"Rated {soc.upper()} HR ({kind})" for kind in ('Matches', 'Reserves')} for soc in socs}
    final_solution_names = ['Rated Matches', 'Rated Reserves']

    # Alternate-list structures produced by augment_rated_algorithm_results().
    alternate_keys = ['J^Alternates (Hard)', 'J^Alternates (Soft)']
    alternate_keys += [f'{prefix} [{soc}]' for soc in socs
                       for prefix in ('J^Preferred', 'I^Preferred', 'I^Alternate')]

    # Report SOC scope, including any excluded or invalid commissioning sources.
    if set(socs) == set(all_socs):
        details.append(f"    SOC Scope: All ({', '.join(s.upper() for s in socs)})")
    else:
        details.extend([f"    SOC Scope: {', '.join(s.upper() for s in socs)}",
                        f"    Full SOC Set: {', '.join(s.upper() for s in all_socs)}"])
        if excluded_socs: details.append(f"    Excluded SOCs: {', '.join(s.upper() for s in excluded_socs)}")
        if invalid_socs: issues.append(f"Unrecognized SOCs: {', '.join(s.upper() for s in invalid_socs)}")

    # ___________________________________________________HELPER FUNCTIONS_________________________________________________

    def check_availability():
        """
        Identify available preliminary and optional final rated solutions.

        Returns:
            tuple: Present and missing Stage 1 solutions, followed by available
                and missing optional final solution names.
        """
        solutions = instance.solutions or {}
        initial_names = [name for kinds in required.values() for name in kinds.values()]
        present = [name for name in initial_names if name in solutions]
        missing = [name for name in initial_names if name not in solutions]
        final_present = [name for name in final_solution_names if name in solutions]
        final_missing = [name for name in final_solution_names if name not in solutions]

        return present, missing, final_present, final_missing

    def validate_quotas():
        """
        Validate preliminary SOC-specific rated assignments against current quotas.

        For each selected SOC and rated AFSC:
            Quota == Preliminary Matches + Preliminary Reserves

        Returns:
            tuple: All quota validation records and records containing mismatches.
        """
        records, mismatches = [], []

        for soc in socs:
            matched = np.asarray(instance.solutions[required[soc]['Matches']]['j_array'])
            reserved = np.asarray(instance.solutions[required[soc]['Reserves']]['j_array'])

            for j in p['J^Rated']:
                j = int(j)
                quota = int(p[f'{soc}_quota'][j])
                num_matched, num_reserved = int(np.count_nonzero(matched == j)), int(np.count_nonzero(reserved == j))

                record = {'soc': soc, 'j': j, 'quota': quota, 'matched': num_matched, 'reserved': num_reserved,
                          'total': num_matched + num_reserved, 'difference': num_matched + num_reserved - quota}

                records.append(record)
                if record['difference']: mismatches.append(record)

        return records, mismatches

    def check_incorporation():
        """
        Validate the final model parameters against preliminary rated matching results.

        The augmentation stage can convert preliminary reservations into inevitable
        matches. Therefore, preliminary reserved cadets are considered incorporated
        if they remain correctly reserved OR are fixed to their reserved rated AFSC.

        Preliminary matched cadets must remain fixed to their matched AFSC.

        Reserved preference arrays are validated by their LAST element, which identifies
        the rated AFSC whose slot was reserved. Earlier elements represent more-preferred
        AFSCs that the cadet may still receive during GUO optimization.

        Additional assignments in J^Fixed and J^Reserved are permitted.

        Returns:
            tuple:
                incorporated (bool): Whether all required assignments and alternate
                    structures are present.
                records (list): SOC-specific incorporation statistics.
                problems (list): Missing or incorrect assignments and structures.
        """
        fixed, reserved = p.get('J^Fixed'), p.get('J^Reserved')
        records, problems = [], []

        # Missing parameter dictionaries indicate that incorporation has not occurred.
        if not isinstance(fixed, dict) or not isinstance(reserved, dict):
            return False, records, [('PARAMETERS', None, 'J^Fixed or J^Reserved is unavailable')]

        # Validate preliminary matches and reserves against their final parameter states.
        for soc in socs:
            matched_array = np.asarray(instance.solutions[required[soc]['Matches']]['j_array'])
            reserved_array = np.asarray(instance.solutions[required[soc]['Reserves']]['j_array'])

            # Both arrays use global cadet indices; p['M'] indicates an unassigned cadet.
            matched_cadets = np.where(matched_array != p['M'])[0].tolist()
            reserved_cadets = np.where(reserved_array != p['M'])[0].tolist()

            # Preliminary matches must remain fixed to their assigned AFSC.
            matched_correct = [i for i in matched_cadets if i in fixed and fixed[i] == matched_array[i]]
            matched_missing = [i for i in matched_cadets if i not in fixed]
            matched_incorrect = [i for i in matched_cadets if i in fixed and fixed[i] != matched_array[i]]

            # Preliminary reserves may remain reserved or become inevitable fixed matches.
            reserved_correct, converted_to_fixed = [], []
            reserved_missing, reserved_incorrect = [], []

            for i in reserved_cadets:
                j = reserved_array[i]

                # Augmentation may convert a reserved slot into a fixed rated assignment.
                if i in fixed:
                    if fixed[i] == j: converted_to_fixed.append(i)
                    else: reserved_incorrect.append(i)

                # Otherwise, the rated AFSC must be the last element of the reservation array.
                elif i in reserved:
                    choices = np.atleast_1d(reserved[i])
                    if len(choices) and choices[-1] == j: reserved_correct.append(i)
                    else: reserved_incorrect.append(i)

                else:
                    reserved_missing.append(i)

            # Record incorporation statistics for reporting.
            records.append({'soc': soc, 'matched': len(matched_cadets), 'matched_correct': len(matched_correct),
                            'reserved': len(reserved_cadets), 'reserved_correct': len(reserved_correct),
                            'converted': len(converted_to_fixed), 'reserved_missing': len(reserved_missing),
                            'reserved_incorrect': len(reserved_incorrect)})

            # Collect all missing and incorrect assignments rather than stopping at the first.
            problems.extend((soc, i, 'MATCH MISSING') for i in matched_missing)
            problems.extend((soc, i, 'MATCH INCORRECT') for i in matched_incorrect)
            problems.extend((soc, i, 'RESERVATION MISSING') for i in reserved_missing)
            problems.extend((soc, i, 'RESERVATION INCORRECT') for i in reserved_incorrect)

        # Validate that augmentation created the alternate-list optimization structures.
        for key in alternate_keys:
            if key not in p: problems.append(('ALTERNATES', None, f'Missing parameter: {key}'))

        # Each SOC should have alternate-list sets for every rated AFSC.
        for soc in socs:
            for prefix in ('J^Preferred', 'I^Preferred', 'I^Alternate'):
                key = f'{prefix} [{soc}]'
                if key in p:
                    missing_afscs = [int(j) for j in p['J^Rated'] if j not in p[key]]
                    problems.extend((soc, None, f'{key} missing AFSC {p["afscs"][j]}') for j in missing_afscs)

        return not problems, records, problems

    def report_incorporation(records):
        """Append final rated matching and reservation statistics to the report."""
        if not records: return

        details.append('    Final Rated Incorporation:')

        for r in records:
            soc = r['soc'].upper()
            reserved_total = r['reserved_correct'] + r['converted']
            details.append(f"      {soc}: {r['matched_correct']}/{r['matched']} matches fixed; "
                           f"{reserved_total}/{r['reserved']} reservations incorporated")

            # Converted reservations are an expected outcome of Algorithm 2, not an error.
            if r['converted']: details.append(f"        Inevitable Matches: {r['converted']} reservations converted to fixed assignments")

    # ___________________________________________________INITIAL STATE_________________________________________________

    # Inspect existing preliminary and optional final solutions.
    present, missing, final_present, final_missing = check_availability()
    initial_status = 'INACTIVE' if not present else ('PARTIALLY ACTIVE' if missing else 'FULLY ACTIVE')
    final_status = 'INACTIVE' if not final_present else ('PARTIALLY ACTIVE' if final_missing else 'FULLY ACTIVE')

    details.append(status_line('SOC Matching Solutions', initial_status, f'{len(present)}/{len(present) + len(missing)} available'))
    details.append(status_line('Final Rated Solutions (Optional)', final_status, f'{len(final_present)}/2 available'))

    # Include missing preliminary solution names, but do not treat missing optional final solutions as errors.
    if missing:
        details.append('    Missing SOC Solutions:')
        details.extend(f'      - {name}' for name in missing)

    # Determine whether existing model parameters already reflect completed augmentation.
    initially_incorporated = False
    if not missing and not invalid_socs:
        initially_incorporated, _, _ = check_incorporation()

    parameter_status = 'FULLY ACTIVE' if initially_incorporated else ('PARTIALLY ACTIVE'
                       if p.get('J^Fixed') or p.get('J^Reserved') else 'INACTIVE')
    checklist_status = 'ACTIVE' if checklist_initial else 'INACTIVE'

    # ___________________________________________________RESOLVE FEATURE_________________________________________________

    # STEP 1: Reject invalid SOC configurations before running algorithms.
    if invalid_socs:
        checklist[feature_id] = False
        resolution_status = 'ERROR'
        actions.append('Feature disabled because the SOC configuration is invalid.')

    else:

        # STEP 2: Generate missing preliminary SOC matching solutions when requested.
        # Only SOCs with missing results are rerun.
        preliminary_rerun = False

        if missing and checklist_initial:
            socs_to_run = [soc for soc, kinds in required.items() if any(name in missing for name in kinds.values())]

            for soc in socs_to_run:
                try:
                    instance.soc_rated_matching_algorithm({'soc': soc, 'ma_printing': False})
                    actions.append(f'Executed SOC rated matching algorithm for {soc.upper()}.')
                    preliminary_rerun = True

                except Exception as exc:
                    issues.append(f'{soc.upper()} matching algorithm failed: {type(exc).__name__}: {exc}')

            present, missing, final_present, final_missing = check_availability()

        # STEP 3: Handle missing preliminary solutions.
        if missing:
            checklist[feature_id] = False

            if checklist_initial:
                resolution_status = 'ERROR'
                issues.append('Required SOC matching solutions remain unavailable.')
            else:
                resolution_status = 'INACTIVE'
                actions.append('Feature not requested; missing SOC solutions were not generated.')

        else:

            # STEP 4: Validate SOC-specific rated quotas before incorporating results.
            quota_records, mismatches = validate_quotas()
            details.append(f"    SOC Quota Validation: {len(quota_records) - len(mismatches)}/{len(quota_records)} passed")

            if mismatches:
                checklist[feature_id] = False
                resolution_status = 'ERROR'
                actions.append('Feature disabled: preliminary rated matching results do not match current quotas.')
                issues.append(f'{len(mismatches)} SOC/AFSC quota mismatches detected:')

                for r in mismatches:
                    afsc = str(p['afscs'][r['j']])
                    issues.append(f"  {r['soc'].upper()} / {afsc}: Quota={r['quota']}, Matches={r['matched']}, "
                                  f"Reserves={r['reserved']}, Total={r['total']}, Difference={r['difference']:+d}")

            else:

                # STEP 5: Inspect final incorporation and augmentation results.
                # A preliminary algorithm rerun requires incorporation to be repeated.
                incorporated, records, problems = check_incorporation()

                # CASE A: Valid results have already been incorporated and augmented.
                # Synchronize the checklist without rerunning the incorporation method.
                if incorporated and not preliminary_rerun:
                    checklist[feature_id] = True
                    resolution_status = 'ACTIVE'
                    actions.append('Rated matching results already incorporated and augmented. No changes required.'
                                   if checklist_initial else 'Valid rated matching parameters detected. Activated checklist.')

                # CASE B: Checklist requests the feature, but incorporation is incomplete or outdated.
                # The existing method handles BOTH initial incorporation and alternate-list augmentation.
                elif checklist_initial:

                    try:
                        instance.incorporate_rated_algorithm_results()
                        actions.append('Executed rated results incorporation and alternate-list augmentation.')

                        # Refresh parameter reference in case the instance method replaced its dictionary.
                        p = instance.parameters
                        incorporated_after, updated_records, remaining = check_incorporation()

                        if incorporated_after:
                            checklist[feature_id] = True
                            resolution_status = 'ACTIVE'
                            actions.append('All preliminary matches and reserves correctly reflected in final model parameters.')

                            # Report any reservations converted into inevitable matches.
                            num_converted = sum(r['converted'] for r in updated_records)
                            if num_converted: actions.append(f'{num_converted} preliminary reservations converted to inevitable fixed matches.')

                        else:
                            checklist[feature_id] = False
                            resolution_status = 'ERROR'
                            issues.append('Rated results incorporation or augmentation produced incomplete parameters.')
                            issues.extend(f'  {soc.upper()}: Cadet {i} — {reason}.' if i is not None
                                          else f'  {soc.upper()}: {reason}.' for soc, i, reason in remaining)

                    except Exception as exc:
                        checklist[feature_id] = False
                        resolution_status = 'ERROR'
                        issues.append(f'Rated results incorporation failed: {type(exc).__name__}: {exc}')

                # CASE C: Feature is inactive and incorporation is incomplete.
                # Preserve existing parameters without executing the incorporation method.
                else:
                    checklist[feature_id] = False
                    resolution_status = 'INACTIVE'
                    actions.append('Rated results are not fully incorporated; checklist remains disabled.')

    # _______________________________________FINAL INCORPORATION REPORT_________________________________________________

    # Report the final parameter state after any rated incorporation and augmentation.
    # This ensures the displayed counts reflect completed changes rather than initial values.
    if not missing and not invalid_socs and resolution_status == 'ACTIVE':
        _, final_records, _ = check_incorporation()
        report_incorporation(final_records)

    # ___________________________________________________CONSTRUCT REPORT_______________________________________________

    # Construct the standardized report with color-coded statuses.
    report = [
        f'[{feature_name}]',
        '  Initial State:',
        status_line('Checklist', checklist_status),
        status_line('Parameters', parameter_status),
        '  Details:',
        *details,
        '  Resolution:',
        status_line('Status', resolution_status),
        *(f'    {action}' for action in actions)
    ]

    # Include the Issues section only when inconsistencies or execution failures occur.
    if issues:
        report.extend(['  Issues:', *(f'    {issue}' for issue in issues)])

    return '\n'.join(report)


def resolve_ussf_om_balance_feature(instance):
    """
    Resolve the USSF Order of Merit Balance feature.

    Synchronizes the feature checklist with the legacy 'USSF OM' model parameter
    and applies the configured merit threshold. Requires USSF to be included
    in the instance's AFSC accession groups.
    """

    # ___________________________________________________INITIALIZATION_________________________________________________

    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p
    checklist = instance.feature_checklist
    feature_id = 'ussf_om_balance'
    feature_name = instance.model_features[feature_id]['name']
    threshold = instance.model_features[feature_id]['threshold']

    # Standardized status formatting
    colors = {'ACTIVE': '\033[92m', 'FULLY ACTIVE': '\033[92m', 'PARTIALLY ACTIVE': '\033[93m',
              'INACTIVE': '\033[90m', 'ERROR': '\033[91m'}
    color = lambda status: f"{colors.get(status, '')}{status}\033[0m"
    status_line = lambda label, status: f"    {label}: {color(status)}"

    # Determine initial feature states
    checklist_initial = bool(checklist[feature_id])
    ussf_available = 'USSF' in p['afscs_acc_grp']
    parameter_initial = bool(mdl_p['USSF OM']) and ussf_available

    checklist_status = 'ACTIVE' if checklist_initial else 'INACTIVE'
    parameter_status = 'ACTIVE' if parameter_initial else 'INACTIVE'

    details, actions, issues = [], [], []

    # ___________________________________________________RESOLVE FEATURE________________________________________________

    if not ussf_available:
        checklist[feature_id] = False
        resolution_status = 'ERROR' if checklist_initial or mdl_p['USSF OM'] else 'INACTIVE'
        actions.append('Feature disabled because USSF is not included in the instance.')
        if resolution_status == 'ERROR': issues.append('USSF accession group is unavailable.')

    elif checklist_initial:
        checklist[feature_id] = True
        mdl_p['USSF OM'] = True
        mdl_p['ussf_merit_bound'] = threshold
        resolution_status = 'ACTIVE'
        actions.append('Feature already enabled. Merit threshold synchronized.' if parameter_initial
                       else 'Checklist activated. Enabled USSF OM balancing and applied merit threshold.')

    elif parameter_initial:
        checklist[feature_id] = True
        mdl_p['ussf_merit_bound'] = threshold
        resolution_status = 'ACTIVE'
        actions.append('USSF OM balancing already enabled in parameters. Activated checklist and synchronized threshold.')

    else:
        resolution_status = 'INACTIVE'
        actions.append('Feature remains disabled. Existing parameter configuration preserved.')

    # _________________________________________________CONSTRUCT REPORT_________________________________________________

    details.append(f"    USSF Accession Group: {'Available' if ussf_available else 'Unavailable'}")
    details.append(f"    Target USSF OM: 50.0% ± {threshold:.1%} ({0.5 - threshold:.1%}–{0.5 + threshold:.1%})")
    if parameter_initial: details.append(f"    Initial Merit Bound: {mdl_p.get('ussf_merit_bound', 'Not set')}")

    report = [
        f'[{feature_name}]',
        '  Initial State:',
        status_line('Checklist', checklist_status),
        status_line('Parameters', parameter_status),
        '  Details:',
        *details,
        '  Resolution:',
        status_line('Status', resolution_status),
        *(f'    {action}' for action in actions)
    ]

    if issues: report.extend(['  Issues:', *(f'    {issue}' for issue in issues)])

    return '\n'.join(report)


def resolve_usafa_support_cap_feature(instance):
    """
    Resolve the USAFA Support AFSC Cap feature.

    Synchronizes the feature checklist with the legacy model parameters
    'USAFA-Constrained AFSCs' and 'usafa_support_afsc_cap'.

    The feature registry defines the authoritative constrained AFSCs and percentage
    cap. Reports genuine data anomalies while treating initially missing constraints
    as normal when the feature has not yet been activated.

    Returns:
        str: Formatted feature resolution report with color-coded statuses.
    """

    # ___________________________________________________INITIALIZATION_________________________________________________

    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p
    checklist = instance.feature_checklist
    feature_id = 'usafa_support_cap'
    feature_name = instance.model_features[feature_id]['name']
    percent_cap = instance.model_features[feature_id]['percent_cap']
    constrained_afscs = list(instance.model_features[feature_id]['constrained_afscs'])

    # Standardized status formatting
    colors = {'ACTIVE': '\033[92m', 'FULLY ACTIVE': '\033[92m', 'PARTIALLY ACTIVE': '\033[93m',
              'INACTIVE': '\033[90m', 'ERROR': '\033[91m'}
    color = lambda status: f"{colors.get(status, '')}{status}\033[0m"
    status_line = lambda label, status: f"    {label}: {color(status)}"

    # Determine initial feature states and identify configuration discrepancies.
    checklist_initial = bool(checklist[feature_id])
    existing_afscs = list(mdl_p.get('USAFA-Constrained AFSCs') or [])
    existing_cap = mdl_p.get('usafa_support_afsc_cap')

    missing_afscs = sorted(set(constrained_afscs) - set(existing_afscs))
    additional_afscs = sorted(set(existing_afscs) - set(constrained_afscs))
    invalid_afscs = sorted(set(constrained_afscs) - set(p['afscs']))
    cap_mismatch = existing_cap is not None and existing_cap != percent_cap

    parameter_initial = bool(existing_afscs) and not missing_afscs and not additional_afscs and not cap_mismatch
    parameter_status = 'FULLY ACTIVE' if parameter_initial else ('PARTIALLY ACTIVE' if existing_afscs else 'INACTIVE')
    checklist_status = 'ACTIVE' if checklist_initial else 'INACTIVE'

    # Calculate the maximum permitted USAFA cadets under the configured cap.
    num_usafa = len(p['usafa_cadets'])
    max_usafa = int(np.floor(num_usafa * percent_cap))

    details, actions, issues = [], [], []

    # ___________________________________________________RESOLVE FEATURE_________________________________________________

    # Invalid registry AFSCs prevent the feature from being configured correctly.
    if invalid_afscs:
        checklist[feature_id] = False
        resolution_status = 'ERROR'
        issues.append(f"Constrained AFSCs not found in instance: {', '.join(invalid_afscs)}")
        actions.append('Feature disabled because the registry contains invalid AFSCs.')

    elif checklist_initial:
        checklist[feature_id] = True
        mdl_p['USAFA-Constrained AFSCs'] = constrained_afscs
        mdl_p['usafa_support_afsc_cap'] = percent_cap
        resolution_status = 'ACTIVE'
        actions.append('Feature already fully configured. No changes required.' if parameter_initial
                       else 'Checklist activated. Synchronized constrained AFSCs and percentage cap.')

    elif parameter_initial:
        checklist[feature_id] = True
        resolution_status = 'ACTIVE'
        actions.append('Support AFSC cap already configured. Activated checklist.')

    else:
        resolution_status = 'INACTIVE'
        actions.append('Feature remains disabled. Existing parameter configuration preserved.')

    # ___________________________________________________CONSTRUCT REPORT_________________________________________________

    details.append(f"    Constrained AFSCs: {len(existing_afscs)}/{len(constrained_afscs)} initially configured")

    # Only report both percentage caps when they differ.
    if cap_mismatch:
        details.append(f"    Percentage Cap Discrepancy: Parameter={existing_cap:.1%}, Registry={percent_cap:.1%}")

    # Unexpected AFSCs are worth reporting when the existing configuration is nonempty.
    if additional_afscs:
        issues.append(f"Unexpected constrained AFSCs in parameters: {', '.join(additional_afscs)}")

    report = [
        f'[{feature_name}]',
        '  Initial State:',
        status_line('Checklist', checklist_status),
        status_line('Parameters', parameter_status),
        '  Details:',
        *details,
        '  Resolution:',
        status_line('Status', resolution_status),
        *(f'    {action}' for action in actions)
    ]

    # Show the final enforced constraints only when the feature is active.
    if resolution_status == 'ACTIVE':
        report.extend([
            f"    Constrained AFSCs: {', '.join(constrained_afscs)}",
            f"    Percentage Cap: {percent_cap:.1%}",
            f"    Maximum USAFA Cadets: {max_usafa:,} of {num_usafa:,}"
        ])

    # Include genuine anomalies, not routine activation changes.
    if issues: report.extend(['  Issues:', *(f'    {issue}' for issue in issues)])

    return '\n'.join(report)


def resolve_high_om_receive_preference_feature(instance):
    """
    Resolve the High OM Receive Preference feature.

    Identifies cadets within the configured top merit percentile and restricts
    them to their first N AFSC preferences. These restrictions are stored for
    subsequent resolution into J^Constrained, where manual overrides take precedence.

    Unlike other features, no existing parameter configuration is checked.
    The feature checklist directly determines activation.

    Returns:
        str: Formatted feature resolution report with color-coded statuses.
    """

    # ___________________________________________________INITIALIZATION_________________________________________________

    p, vp, mdl_p = instance.parameters, instance.value_parameters, instance.mdl_p
    checklist = instance.feature_checklist
    feature_id = 'high_om_receive_preference'
    feature_name = instance.model_features[feature_id]['name']
    percentile_cutoff = instance.model_features[feature_id]['percentile_cutoff']
    preference_number = instance.model_features[feature_id]['preference_number']

    # Standardized status formatting
    colors = {'ACTIVE': '\033[92m', 'FULLY ACTIVE': '\033[92m', 'PARTIALLY ACTIVE': '\033[93m',
              'INACTIVE': '\033[90m', 'ERROR': '\033[91m'}
    color = lambda status: f"{colors.get(status, '')}{status}\033[0m"
    status_line = lambda label, status: f"    {label}: {color(status)}"

    checklist_initial = bool(checklist[feature_id])
    checklist_status = 'ACTIVE' if checklist_initial else 'INACTIVE'

    # __________________________________________________RESOLVE FEATURE_________________________________________________

    # The checklist directly controls this feature; no parameter reconciliation required.
    p['high_om_preference'] = checklist_initial
    p['high_om_percentile_cutoff'] = percentile_cutoff
    p['high_om_preference_number'] = preference_number

    if checklist_initial:

        # Identify cadets within the top X% of the merit distribution.
        p['I^High OM Constrained'] = np.where(p['merit'] >= (1 - percentile_cutoff))[0]

        # Python slicing [:3] returns preferences at indices 0, 1, and 2.
        p['J^High OM Constrained AFSCs'] = {
            int(i): np.asarray(p['cadet_preferences'][i][:preference_number], dtype=int)
            for i in p['I^High OM Constrained']
        }

        resolution_status = 'ACTIVE'
        resolution = (f'Identified {len(p["I^High OM Constrained"]):,} cadets within the '
                      f'top {percentile_cutoff:.0%} of merit and assigned their top '
                      f'{preference_number} AFSC preferences.')

    else:
        p['I^High OM Constrained'] = np.array([], dtype=int)
        p['J^High OM Constrained AFSCs'] = {}

        resolution_status = 'INACTIVE'
        resolution = 'Feature disabled. No High OM preference constraints generated.'

    # _________________________________________________CONSTRUCT REPORT_________________________________________________

    report = [
        f'[{feature_name}]',
        '  Initial State:',
        status_line('Checklist', checklist_status),
        '  Details:',
        f'    Merit Percentile Cutoff: Top {percentile_cutoff:.0%}',
        f'    Guaranteed Preference Range: Top {preference_number}',
        '  Resolution:',
        status_line('Status', resolution_status),
        f'    {resolution}'
    ]

    return '\n'.join(report)

