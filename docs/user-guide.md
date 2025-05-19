# Getting Started with AFCCP

Welcome to the Air Force Cadet Career Problem (AFCCP) documentation. This guide walks you through a simple example of how to use the AFCCP model, including inputs, logic, and interpreting results.

---

## 📦 1. Load Required Packages

```python
import numpy as np
import pandas as pd
from afccp import CadetCareerProblem
```

---

## 📋 2. Define a Simple Example Dataset

Here we construct a small set of cadets and AFSCs to walk through the core matching logic.

```python
cadets = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "preferences": [["11X", "13X", "17X"],
                    ["13X", "11X", "17X"],
                    ["17X", "11X", "13X"]]
})

afscs = pd.DataFrame({
    "afsc": ["11X", "13X", "17X"],
    "slots": [1, 1, 1]
})
```

---

## 🧠 3. Run the Matching Model

The model takes the cadet preferences and available slots and performs an assignment using a merit-based preference system.

```python
model = CadetCareerProblem(cadets=cadets, afscs=afscs)
model.solve()
assignments = model.get_assignments()
assignments
```

### ✅ Output

```text
  name     assigned_afsc
0  Alice            11X
1    Bob            13X
2 Charlie           17X
```

---

## 🔍 4. Visualize the Results

You can also visualize the match using built-in plotting tools.

```python
model.plot_assignments()
```

This creates a visual match between cadets and their assigned AFSCs using arrows and preferences.

---

## 📌 Summary

In this quickstart:
- We initialized a toy dataset of cadets and AFSCs.
- We ran the assignment model and viewed the results.
- We generated a plot to visualize the match.

To run more realistic examples, refer to [Model Overview](model/overview.md) or the [API Reference](api_reference.md).