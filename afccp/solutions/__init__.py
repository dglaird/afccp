"""
The `solutions` module provides the full infrastructure for solving cadet–AFSC matching problems in the
AFCCP system. It includes a range of algorithms (HR, GA, GUO, etc.), constraint handling logic, Pyomo-based
model construction and solver interfaces, and extensive support for sensitivity analysis and "what-if"
scenarios.

This module is central to the AFCCP ecosystem—it orchestrates the end-to-end optimization workflow:
generating initial solutions, building and solving models, evaluating results, and exploring trade-offs
through iterative constraints or shifting objectives.

Submodules
----------

- **[`solutions.algorithms`](../../../afccp/reference/solutions/algorithms/#solutions.algorithms)**
  Implements baseline and advanced matching algorithms including:
    - Classic and enhanced **Hospital/Residents** (HR) matching
    - **GA/GMA**: Genetic algorithms with cadet–AFSC crossover, mutation, and selection logic
    - **Hybrid solutions**: Merge utility-based models with HR/GMA results for feasible fallback options
- **[`solutions.handling`](../../../afccp/reference/solutions/handling/#solutions.handling)**
  Provides tools for:
    - **Evaluating objective values**: VFT scores, fairness metrics, quota and eligibility violations
    - **Constraint diagnostics**: Identifies which constraints were violated or binding
    - **Preference reconstruction**: Converts `j_array` solutions into readable cadet assignments
- **[`solutions.optimization`](../../../afccp/reference/solutions/optimization/#solutions.optimization)**
  Contains the core **Pyomo model** logic and solver interfaces for:
    - **VFT models** (exact and approximate)
    - **GUO** (Global Utility Optimization) and goal programming models
    - Constraint injection, warm-starting, and solution persistence
    - Full compatibility with real-world constraints like AFOCD tiers, base/training thresholds, and USAFA rules
- **[`solutions.sensitivity`](../../../afccp/reference/solutions/sensitivity/#solutions.sensitivity)**
  Tools for **constraint iteration**, **GA initialization**, and **what-if analysis**, including:
    - Automated constraint relaxation workflows
    - PGL capacity adjustment loops and visualizations
    - "What If List.csv" batch simulation and delta metric reporting

Typical Workflow
----------------

1. **Generate Initial Solution(s)**
    - Use [`algorithms`](../../../afccp/reference/solutions/algorithms/#solutions.algorithms) to run HR or GA on initial cadet–AFSC preferences
    - Or pre-seed GA populations using VFT/Assignment heuristics from `sensitivity`
1. **Build and Solve Optimization Model**
    - Construct VFT or GUO model in [`optimization`](../../../afccp/reference/solutions/optimization/#solutions.optimization)
    - Add constraints based on cadet thresholds, base/training requirements, or fairness tiers
1. **Evaluate Solution**
    - Use [`handling`](../../../afccp/reference/solutions/handling/#solutions.handling) to check feasibility, score each objective, and extract readable results
1. **Run Sensitivity Analysis**
    - Apply `sensitivity` to modify constraints, quotas, or preference structures
    - Generate what-if metrics and comparative charts

See Also
--------
- [`data.preferences`](../../../afccp/reference/data/preferences/#data.preferences):
  Cadet/AFSC eligibility, tier data, and OM-based utility scores

- [`data.values`](../../../afccp/reference/data/values/#data.values):
  Solver weight configurations for VFT and GUO objectives

- [`CadetCareerProblem`](../../../afccp/reference/main/#main.CadetCareerProblem):
  Main interface to run all solution workflows via `.solve_*()` methods
"""