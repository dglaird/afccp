"""
The `data.generation` module provides synthetic data creation tools for the AFCCP system,
designed to support testing, experimentation, and scenario modeling for cadet-to-AFSC assignment problems.
It includes both simple and advanced generators that simulate cadet preferences, AFSC eligibility,
base assignments, training courses, and other problem parameters.

This module allows users to rapidly produce valid input datasets for `CadetCareerProblem`
that mimic either minimal input assumptions or realistic programmatic constraints.

Submodules
----------

- **[`data.generation.basic`](../../../reference/data/generation/basic/#data.generation.basic)**
  Provides minimal synthetic data generation pipelines with:
    - Uniform or random utility scores
    - Simplified cadet and AFSC attributes
    - Basic constraint setups for quick prototyping
- **[`data.generation.realistic`](../../../reference/data/generation/realistic/#data.generation.realistic)**
  Generates highly realistic cadet datasets based on:
    - Real CIP distributions and merit-based AFSC interest
    - Custom samplers for AFSC/cadet utilities
    - Condition-based sampling to satisfy rare AFSC degree quotas

Functionality
-------------

The `data.generation` module supports:
- **Randomized dataset creation**: from scratch or based on conditional rules
- **AFSC-specific quota balancing**: especially for underrepresented AFSCs like `62EXE`
- **Cadet utility modeling**: using KDE-based samplers from empirical data
- **Preference shaping**: structured base, training, and AFSC preference profiles
- **Training course simulation**: schedules, capacities, and cadet matching windows

Typical Use Cases
-----------------

- Unit testing optimization models with diverse synthetic populations
- Stress-testing sensitivity analysis pipelines on edge-case cadet distributions
- Exploring rare AFSC scenarios using CIP-based data generation
- Prototyping utility functions, preference matrices, and training alignment logic

See Also
--------

- [`data.processing`](../../../reference/data/processing/#data.processing):
  Tools to clean and restructure raw real-world data before generation

- [`data.preferences`](../../../reference/data/preferences/#data.preferences):
  Utility score generation and AFSC–cadet preference logic

- [`CadetCareerProblem`](../../../reference/core/ccp/#core.ccp.CadetCareerProblem):
  Primary object that consumes synthetic datasets generated here via `.load_data()` and `.solve_*()` methods
"""
from afccp.data.generation.basic import *

# Import 'realistic' generation functions if we have the SDV module
from afccp.globals import use_sdv
if use_sdv:
    from afccp.data.generation.realistic import *