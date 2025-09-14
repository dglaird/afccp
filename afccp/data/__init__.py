"""
The `data` module provides a robust framework for ingesting, transforming, and preparing datasets
used throughout the Air Force Cadet Career Problem (AFCCP) workflow. It serves as the data backbone
of the entire system, organizing all core inputs (cadet, AFSC, and parameter data) and enabling
downstream optimization, evaluation, and visualization.

The module is composed of six submodules, each handling a distinct category of functionality:

- **adjustments**:
    - Provides tools to dynamically modify AFCCP data after loading.
    - Enables addition/removal of records, overrides to standard parameters, and data corrections.
    - Used for setting policy-like constraints (e.g., mandatory AFSC matches, special case cadets).
- **preferences**:
    - Constructs utility matrices and choice rankings for cadets.
    - Implements preference formulas and customizations for rated vs. non-rated pipelines.
    - Supports business-hour conversions, tier weighting, and preference smoothing.
- **processing**:
    - Handles raw data ingestion and conversion into AFCCP-compatible formats.
    - Includes import functions for cadets, AFSCs, matrices, and external policy data.
    - Validates data structures and resolves ID and naming consistency issues.
- **values**:
    - Houses the value parameters and weights used by scoring and optimization functions.
    - Loads, stores, and exports weights related to cadet utility, AFSC metrics, and fairness goals.
    - Includes VFT-style parameter managers for weighting different objectives.
- **support**:
    - Contains utility functions and helpers used across data submodules.
    - Provides tools for CIP-to-AFSC tier logic, matrix reshaping, name formatting, and more.
- **generation**:
    - Offers synthetic data generation tools for testing AFCCP workflows.
    - Can simulate cadets, AFSCs, matrices, and full problem instances with controlled parameters.
    - Useful for benchmarking, tutorial development, and automated testing.

This module ensures that data entering the AFCCP pipeline is well-structured, customizable, and
aligned with downstream model expectations. It separates concerns cleanly across ingestion,
preparation, transformation, and preference modeling.

See Also
--------
- [`data.adjustments`](../../../afccp/reference/data/adjustments/#data.adjustments)
- [`data.preferences`](../../../afccp/reference/data/preferences/#data.preferences)
- [`data.processing`](../../../afccp/reference/data/processing/#data.processing)
- [`data.values`](../../../afccp/reference/data/values/#data.values)
- [`data.support`](../../../afccp/reference/data/support/#data.support)
- [`data.generation`](../../../afccp/reference/data/generation/#data.generation)
"""