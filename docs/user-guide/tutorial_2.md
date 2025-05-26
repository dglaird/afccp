# Tutorial 2: Data Overview

In this second tutorial, we'll explain the contents of the `data` module within `afccp`, as well as how the data itself 
is structured in the [CadetCareerProblem](../reference/main.md#cadetcareerproblem) object.

---

## 1. Data Module

A `CadetCareerProblem` instance comes with a lot of data, and so the largest section of this tutorial is here to 
talk about that data. I'll outline where the data is coming from (the various csvs), how the data is captured within 
the `CadetCareerProblem` class, and where the code is that manages all of it. 

### afccp.data

The "data" module of afccp.core contains the scripts and functions that all have something to do with processing the 
many sets and parameters of the problem. At a high level the modules are setup like this: