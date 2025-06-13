# Optimization
This section is here to describe the various optimization model components we have to solve the `CadetCareerProblem`. 
At the time I'm writing this tutorial, the VFT methodology doesn't work on the current combined classification 
problem (Rated, USSF, NRL). The model used now and for the foreseeable future is the GUO model.
This is the global utility optimization model that OLEA has advertised, and it's represented in the code as the 
"assignment problem model". We'll discuss that soon.

## VFT Model
For more information on the VFT model, please review my <a href="https://scholar.afit.edu/cgi/viewcontent.cgi?article=6348&context=etd" target="_blank">thesis</a>. 
It describes in detail the difference between the "Approximate" model and the "Exact" model. The VFT model was my main
focus while at AFIT, and I've summarized that work through this poster:

<p align="center">
  <img src="/afccp/images/user-guide/output.gif" width="1200px">
</p>

Here, I will solve this random instance using both the "Approximate" model and the "Exact" model.

```python
# Solve with the "Approximate" model  (These three controls I've added are also the defaults!)
s = instance.solve_vft_pyomo_model({"approximate": True, "pyomo_max_time": 10, "solver_name": "cbc"})
```
??? note "💻 Console Output"
    ```text
    Building VFT Model...
    Solving Approximate VFT Model instance with solver cbc...
    Start Time: June 09 2025 01:56:48 PM.
    ✅ Solver complete: |████████████████████████████████████████| 100.0%              
    Model solved in 7.04 seconds at June 09 2025 01:56:55 PM. Pyomo reported objective value: 0.8883
    New Solution Evaluated.
    Measured exact VFT objective value: 0.8969.
    Global Utility Score: 0.7988. 3 / 3 AFSCs fixed. 4 / 4 AFSCs reserved. 7 / 7 alternate list scenarios respected.
    Blocking pairs: 2. Unmatched cadets: 0.
    Matched cadets: 20/20. N^Match: 20. Ineligible cadets: 0.
    ```

```python
# Solve with the "Exact" model (On a small problem, this actually works quite well!)
s = instance.solve_vft_pyomo_model({"approximate": False, "pyomo_max_time": 20, "solver_name": "ipopt"})
```
??? note "💻 Console Output"
    ```text
    Building VFT Model...
    Solving Exact VFT Model instance with solver ipopt...
    Start Time: June 09 2025 01:56:33 PM.
    ✅ Solver complete: |████████████████████████████████████████| 100.0%              
    Model solved in 3.03 seconds at June 09 2025 01:56:36 PM. Pyomo reported objective value: 0.9445
    New Solution Evaluated.
    Measured exact VFT objective value: 0.8885.
    Global Utility Score: 0.8291. 3 / 3 AFSCs fixed. 4 / 4 AFSCs reserved. 7 / 7 alternate list scenarios respected.
    Blocking pairs: 1. Unmatched cadets: 0.
    Matched cadets: 20/20. N^Match: 20. Ineligible cadets: 0.
    ```

```python
instance.solutions.keys()
```
??? note "💻 Console Output"
    ```text
    dict_keys(['Random', 'Random_2', 'HR', 'Rated ROTC HR (Reserves)', 'Rated ROTC HR (Matches)', 'Rated ROTC HR', 'Rated USAFA HR (Reserves)', 'Rated USAFA HR (Matches)', 'Rated Matches', 'Rated Alternates (Hard)', 'Rated Alternates (Soft)', 'ROTCRatedBoard', 'VFT_Genetic', 'A-VFT', 'E-VFT'])
    ```

We just solved the VFT model twice (once with the "Approximate Model" and another with the "Exact Model"). 
As you can probably tell, both solutions are represented from two different methods: "A-VFT" and "E-VFT". 
I'll let you guess which one is which. Since it is a very small problem, they both do exactly what they're 
supposed to: the Exact model finds the optimal solution to the "real" (Exact) VFT objective function 
(beating the Approximate model) and they both produce integer solutions (something that isn't true when the problem 
gets bigger). Additionally, these models are deterministic, meaning they will produce the same solution every time. 
This is not true for the GA solutions!

!!! info "💡 Calculations are off"
    When I was working through my thesis, I was very meticulous to ensure all the calculations were correct and my own
    method of evaluating solutions with the [`evaluate_solution()`](../../../afccp/reference/solutions/handling/#solutions.handling.evaluate_solution)
    function would perfectly match the objective value produced by `pyomo`. As you can see from above, this does not seem
    to be the case. To be honest I don't know why the VFT model is not the same but I don't have the time to explore this
    issue again. I DO know, however, that the GUO values do match up with what pyomo calculates! Just not the VFT one.
    
The next section describes my VFT methodology using the Approximate model.

## VFT Main Methodology
I had this weird "con_fail_dict" (constraint fail dictionary) feature of the model to allow some AFSC objective constraints to be broken by AT MOST the amount they were broken as a product of the non-integer pyomo output of the VFT Approximate Model. This will be made more clear in the third bullet below.

My current pyomo VFT model does not produce integer solutions using cbc. I really don't understand why this is and it's certainly the motivation for future AFIT research using that model. My methodology that I came up with (only for NRL cadets/AFSCs at the time) is as follows:

1. [Solve the VFT Model] Solve the VFT Approximate Model for 10 seconds using the cbc solver. The time limit of 10 seconds allows us to find a pretty good solution (not optimal, but close-ish enough) to the Exact Model. As a reminder, the Exact Model is the real objective function using the true number of assigned cadets, whereas the Approximate Model approximates that number (specifically, it uses the "Estimated" number of cadets from AFSCs.csv). The optimal solution to the Approximate Model, therefore, is not the optimal solution to the Exact Model. One other reason to cut the Approximate Model short is that we were never going to find the optimal solution to the "real" problem anyway! 

2. [Round the $X$ variables] Ok, so as a result of stopping it at 10 seconds we don't get integer values for all the variables we need. I therefore have to round them to make sure cadets receive one and only one AFSC. By doing this, I am oftentimes breaking constraints. This was generally ok, however, since constraints were primarily meant as guidelines to ensure quality distributions across AFSCs. The class of 2023 and 2024 were very different problems in many ways, and the fact that we had ~150 extra cadets above the PGL in 2023 allowed for this to be less of an issue. Since we're 4 short in 2024, rounding these variables could have produced other problems.

3. [Create "con_fail_dict"] As mentioned previously, this constraint fail dictionary was used to determine the extent to which we broke the AFSC objective constraints from whatever solution(s) pyomo produced. This dictionary is then used in the GA to allow more flexibility when evaluating the solutions (or "chromosomes") fitness values. Fitness is based directly on the true (Exact, not Approximate) VFT objective function, and constraint violations result in a fitness score of 0. I played around with providing a tolerance but ultimately decided to aggressively restrict constraint violations. This is why con_fail_dict is so important, since it allows the initial solutions to be feasible. 

4. [Initialize GA Population] In my "real" solution methodology, I solve the VFT model several different times with slightly different settings on the overall weights for cadets/AFSCs. This essentially creates my initial population of solutions that I use to then create the "con_fail_dict" from. These solutions are then fed into the GA.

5. [Run GA] 

## 🧮 Assignment Problem Model- "Global Utility Optimization" (GUO)
The GUO model refers to solving the generalized assignment problem formulation that seeks the optimal assignment of 
a set of $n$ workers, $\mathcal{I}$, to a set of $m$ jobs, $\mathcal{J}$, such that the total cost is minimized. Let:

- $c_{ij}$ = cost of assigning worker $i$ to job $j$
- $d_j$ = capacity of job $j$
- \( x_{ij} = \text{1 if worker } i \text{ is assigned to job } j; \text{ 0 otherwise} \)

### Objective (Simplified):

Minimize total cost:

$$
\min \sum_{i=1}^{n} \sum_{j=1}^{m} c_{ij} x_{ij}
$$

### Subject to (Simplified):

Each worker is assigned to exactly one job:

$$
\sum_{j=1}^{m} x_{ij} = 1 \quad \forall i \in \{1, \dots, n\}
$$

Each job does not surpass its capacity:

$$
\sum_{i=1}^{n} x_{ij} \leq d_j \quad \forall j \in \{1, \dots, m\}
$$

Binary constraints:

$$
x_{ij} \in \{0, 1\} \quad \forall i \in \mathcal{I}, j \in \mathcal{J}
$$

Again, this is the "generalized" formulation of this academic problem. Like the "original model" (the model AFPC used 
up until the class of 2023), there are many additional constraints that can be added to this model. The addition of 
these constraints is one key difference between "GUO" in our context and the generalized framework. Another key 
difference is the utility matrix itself. Unlike the original AFPC model, this "new" utility matrix is much more direct 
with its representation of quality for the career fields. This is the merged cadet/AFSC `global_utility` matrix 
I've referenced in the Data section of this tutorial. Quick note: rather than a "min cost" function, it's a 
"max utility" function. This is a small, but necessary, clarification!

```python
# Solve the GUO model!
s = instance.solve_guo_pyomo_model()
```
??? note "💻 Console Output"
    ```text
    Building assignment problem (GUO) model...
    Done. Solving model...
    Solving GUO Model instance with solver cbc...
    Start Time: June 09 2025 03:15:04 PM.
    Model solved in 0.21 seconds at June 09 2025 03:15:05 PM. Pyomo reported objective value: 0.8364
    New Solution Evaluated.
    Measured exact VFT objective value: 0.8718.
    Global Utility Score: 0.8364. 3 / 3 AFSCs fixed. 4 / 4 AFSCs reserved. 7 / 7 alternate list scenarios respected.
    Blocking pairs: 4. Unmatched cadets: 0.
    Matched cadets: 20/20. N^Match: 20. Ineligible cadets: 0.
    ```

As you can see, it solves very fast on a small problem! On a larger problem, too, it still solves pretty quickly. 
Although the value functions and weights from the VFT model aren't used in the GUO model, the constraints certainly are! 
The value parameters, therefore, are still very important in this model as they control the constraints applied to the 
problem. This model can be found in the 
[`assignment_model_build()`](../../../afccp/reference/solutions/optimization/#solutions.optimization.assignment_model_build) function.
If you look at that function, you can see an additional 
[`common_optimization_handling()`](../../../afccp/reference/solutions/optimization/#solutions.optimization.common_optimization_handling) 
function that does exactly what it sounds like: handles the features of the pyomo model that are common across all of my 
optimization models (VFT, original, GUO). Mostly, this function handles the definition of the "`x`" variable and the 
many potential constraints associated with it. This includes the fixed/reserved slot constraints I discussed in the 
[`SOC Rated Algorithm`](../../../../afccp/user-guide/tutorial_7/#soc-rated-algorithm) section previously.

```python
# Run the USAFA/ROTC rated algorithm
s = instance.soc_rated_matching_algorithm({"soc": "usafa"})  # "s =" prevents lots of output
s = instance.soc_rated_matching_algorithm({"soc": "rotc"})  # "s =" prevents lots of output

# Integrate the Rated algorithm solutions into "instance.parameters" 
instance.incorporate_rated_algorithm_results()

# Solve the GUO model with the rated algorithm results!
s = instance.solve_guo_pyomo_model({"USSF OM": True})
```
??? note "💻 Console Output"
    ```text
    Solving the rated matching algorithm for USAFA cadets...
    Solving the rated matching algorithm for ROTC cadets...
    Incorporating rated algorithm results...
    Rated SOC Algorithm Results:
    USAFA Fixed Cadets: 1, USAFA Reserved Cadets: 0, ROTC Fixed Cadets: 2, ROTC Reserved Cadets: 4
    USAFA Rated Alternates: 0, ROTC Rated Alternates: 7
    Building assignment problem (GUO) model...
    Done. Solving model...
    Solving GUO Model instance with solver cbc...
    Start Time: June 09 2025 07:44:24 PM.
    Model solved in 0.11 seconds at June 09 2025 07:44:24 PM. Pyomo reported objective value: 0.8324
    New Solution Evaluated.
    Measured exact VFT objective value: 0.8753.
    Global Utility Score: 0.8324. 3 / 3 AFSCs fixed. 4 / 4 AFSCs reserved. 7 / 7 alternate list scenarios respected.
    Blocking pairs: 2. Unmatched cadets: 0.
    Matched cadets: 20/20. N^Match: 20. Ineligible cadets: 0.
    ```

Quick note: Turning on the "USSF OM" constraint above ensures that the overall distribution of OM of the Space Force 
is around where it is with the Air Force. When running this officially, you should turn this constraint on like I did 
above. The mdl_p control "ussf_merit_bound" determines how wide of a range we can have around 50%. 
It is defaulted to 0.03, which means that USSF OM can be within 47%-53%.

As a reminder, for rated cadet-AFSC matches, the optimization model does not get to decide entirely based on global utility. 
We adhere to the rated OM list exclusively! The SOC rated matching algorithm must be run once, and then every time we execute 
[`instance.incorporate_rated_algorithm_results()`](../../../afccp/reference/solutions/handling/#solutions.handling.incorporate_rated_algorithm_results), 
the alternate list sets and parameters are added by default. 

```python
# These are the default controls, so alternate lists should always be included unless otherwise specified!
instance.incorporate_rated_algorithm_results({'rated_alternates': True,
                                              'alternate_list_iterations_printing': False})
```
??? note "💻 Console Output"
    ```text
    Incorporating rated algorithm results...
    Rated SOC Algorithm Results:
    USAFA Fixed Cadets: 1, USAFA Reserved Cadets: 0, ROTC Fixed Cadets: 2, ROTC Reserved Cadets: 4
    USAFA Rated Alternates: 0, ROTC Rated Alternates: 7
    ```

If we want to see the rated alternate algorithm a little more clearly, we can toggle the 
"alternate_list_iterations_printing" parameter.

```python
# These are the default controls, so alternate lists should always be included unless otherwise specified!
instance.incorporate_rated_algorithm_results({'alternate_list_iterations_printing': True})
```
??? note "💻 Console Output"
    ```text
    Incorporating rated algorithm results...

    SOC: USAFA
    
    Iteration 0
    Possible {'R2': 0, 'R4': 0}
    Matched {'R2': 1, 'R4': 0}
    Reserved {'R2': 0, 'R4': 0}
    Alternates (Hard) {'R2': 0, 'R4': 0}
    Alternates (Soft) {'R2': 0, 'R4': 0}
    Iteration 1
    Possible {'R2': 0, 'R4': 0}
    Matched {'R2': 1, 'R4': 0}
    Reserved {'R2': 0, 'R4': 0}
    Alternates (Hard) {'R2': 0, 'R4': 0}
    Alternates (Soft) {'R2': 0, 'R4': 0}
    
    SOC: ROTC
    
    Iteration 0
    Possible {'R2': 4, 'R4': 6}
    Matched {'R2': 1, 'R4': 1}
    Reserved {'R2': 3, 'R4': 1}
    Alternates (Hard) {'R2': 0, 'R4': 0}
    Alternates (Soft) {'R2': 4, 'R4': 6}
    Iteration 1
    Possible {'R2': 4, 'R4': 6}
    Matched {'R2': 1, 'R4': 1}
    Reserved {'R2': 3, 'R4': 1}
    Alternates (Hard) {'R2': 0, 'R4': 0}
    Alternates (Soft) {'R2': 4, 'R4': 6}
    Rated SOC Algorithm Results:
    USAFA Fixed Cadets: 1, USAFA Reserved Cadets: 0, ROTC Fixed Cadets: 2, ROTC Reserved Cadets: 4
    USAFA Rated Alternates: 0, ROTC Rated Alternates: 7
    ```

Again, not much happens here as a result of the small, random dataset. 

```python
# Re-import module and data 
from afccp.main import CadetCareerProblem
instance = CadetCareerProblem("Random_1")
instance.set_value_parameters()
```
??? note "💻 Console Output"
    ```text
    Importing 'Random_1' instance...
    Instance 'Random_1' initialized.
    ```

## Assignment Problem Model- "Original" AFPC model
To be true to this problem's history, I have coded up the original AFPC model formulation. However, this isn't 
entirely accurate since it relies on my value parameters for constraints and can be solved in an almost identical 
manner to GUO. The only difference here is the objective function! We create the original utility matrix that 
AFPC used up until 2023 for AFSC NRL classification and solve the model with the same features as GUO. If you 
look at the optimization.py script at the assignment_model_build() function, you will notice that I simply have 
an "if" statement to differentiate the two models. Here it is:

```python
s = instance.solve_original_pyomo_model()  # Not recommended in anyway, just here as an artifact!
```
??? note "💻 Console Output"
    ```text
    Building original assignment problem model...
    Done. Solving model...
    Solving Original Model instance with solver cbc...
    Start Time: June 09 2025 07:53:30 PM.
    Model solved in 0.1 seconds at June 09 2025 07:53:30 PM. Pyomo reported objective value: 141.9145
    New Solution Evaluated.
    Measured exact VFT objective value: 0.8408.
    Global Utility Score: 0.7979. 0 / 0 AFSCs fixed. 0 / 0 AFSCs reserved. 0 / 0 alternate list scenarios respected.
    Blocking pairs: 3. Unmatched cadets: 0.
    Matched cadets: 20/20. N^Match: 20. Ineligible cadets: 0.
    ```

## Goal Programming Model
This section is here to talk about former Lt Rebecca Reynold's optimization model. I have it coded up to follow her 
methodology to the best of my ability and it lives in the code as well. Again, this is here primarily for 
historical purposes to capture academic contributions to this problem! You are free to look at all the ins and outs 
of this model and see how the data is represented and utilized. 

Quick note, part of the process of running her goal programming model involves translating my parameters & 
value parameters into her own specific parameters that she uses. Additionally, when she and I took on the 
thesis, we handled AFOCD objectives through the lens of "Mandatory", "Desired", and "Permitted" rather than 
"Tier 1 -> Tier 4". Her model uses the requirement levels (M, D, P) rather than the tiers in the same way DSY used 
that through c/2023. Little nuanced thing but since making this tutorial I have updated the random value 
parameter generation function to still generate these objectives (Mandatory, Desired, Permitted) so this translation 
function will work. Moral of the story: I just need to regenerate a set of value parameters to include 
these objectives which will allow me to translate my parameters to hers, and ultimately solve the model.

```python
# Generate random set of value parameters (will contain M, D, P objectives)
instance.generate_random_value_parameters()

# We now have two sets of value parameters!
print(instance.vp_dict.keys())
```
??? note "💻 Console Output"
    ```text
    dict_keys(['VP', 'VP2'])
    ```

```python
# Solve the goal-programming model!
s = instance.solve_gp_pyomo_model({'USSF OM': True})

# Re-activate the first set of value parameters
instance.set_value_parameters("VP")  # VP_2 evaluated the model by default (we want VP)
```
??? note "💻 Console Output"
    ```text
    Translating VFT model parameters to Goal Programming Model parameters...
    Building GP Model...
    Model built.
    Solving GP Model instance with solver cbc...
    Start Time: June 09 2025 07:55:35 PM.
    Model solved.
    New Solution Evaluated.
    Measured exact VFT objective value: 0.8266.
    Global Utility Score: 0.7732. 0 / 0 AFSCs fixed. 0 / 0 AFSCs reserved. 0 / 0 alternate list scenarios respected.
    Blocking pairs: 3. Unmatched cadets: 0.
    Matched cadets: 20/20. N^Match: 20. Ineligible cadets: 0.
    
    Solution Evaluated: GP.
    Measured exact VFT objective value: 0.873.
    Global Utility Score: 0.8336. 0 / 0 AFSCs fixed. 0 / 0 AFSCs reserved. 0 / 0 alternate list scenarios respected.
    Blocking pairs: 3. Unmatched cadets: 0.
    Matched cadets: 20/20. N^Match: 20. Ineligible cadets: 0.
    ```

Another note, the "GP" model is not bound by any of the same constraints as the other models. This includes the 
value parameter constraints and the common optimization handling constraints. This is how her model was formulated and 
so it doesn't adhere to any of our "rules". This model remains here in case any future researchers want to look at her 
work and expand upon it in some capacity to work with the current state of the problem.

## Sensitivity
There are some sensitivity analysis functions within `afccp`, and this is really the next step of what to do with 
this project. I don't have many yet, but the goal is to continue to develop these capabilities in the future. 
(I'm skipping this section for now except the one function below).

```python
# Iteratively solve model activating constraints one at a time to check feasibility
instance.solve_for_constraints()
```
??? note "💻 Console Output"
    ```text
    Initializing Assignment Model Constraint Algorithm...
    Done. Solving model with no constraints active...
    Done. New solution objective value: 0.8652
    Running through 4 total constraint iterations...
    
    ------[1] AFSC R4 Objective Combined Quota------------
    Constraint 1 Active Constraints: 1 Validated: 1
    Result: SKIPPED [Measure: 3.0],  Range: (2.0, 3.0)
    Active Objective Measure Constraints: 1
    Total Failed Constraints: 0
    Current Objective Measure: 3.0 Range: 2, 3
    ---------- Objective Measure Fails:0-------------------
    
    ------[2] AFSC R2 Objective Combined Quota------------
    Constraint 2 Active Constraints: 2 Validated: 2
    Result: SOLVED [Z = 0.8587]
    Active Objective Measure Constraints: 2
    Total Failed Constraints: 0
    Current Objective Measure: 5.0 Range: 5, 9
    ---------- Objective Measure Fails:0-------------------
    
    ------[3] AFSC R3 Objective Combined Quota------------
    Constraint 3 Active Constraints: 3 Validated: 3
    Result: SOLVED [Z = 0.8364]
    Active Objective Measure Constraints: 3
    Total Failed Constraints: 0
    Current Objective Measure: 3.0 Range: 2, 3
    ---------- Objective Measure Fails:0-------------------
    
    ------[4] AFSC R1 Objective Combined Quota------------
    Constraint 4 Active Constraints: 4 Validated: 4
    Result: SKIPPED [Measure: 9.0],  Range: (8.0, 9.0)
    Active Objective Measure Constraints: 4
    Total Failed Constraints: 0
    Current Objective Measure: 9.0 Range: 8, 9
    ---------- Objective Measure Fails:0-------------------
    ```

## Recommended Model Flow

The "Solutions" section first discussed how solutions are represented in csv format as well as in the code. 
Just like the parameters, we have a csv dataframe containing the content which is then pulled into a `solutions` 
dictionary containing the various metrics of one particular solution given all the many parameters to the problem. 
The section then discusses the various solution algorithms/models that may be applied to the problem. Below is my 
recommended approach to the problem. I start by importing the data and applying the SOC rated algorithms first, and 
then exporting it back.

```python
# Import the "Random_1" instance
instance = CadetCareerProblem('Random_1')

# "Activate" a particular set of value parameters (since you can have multiple)
instance.set_value_parameters("VP")  # There could be "VP", "VP2", "VP3", etc.

# In case you change some parameter that the value parameters depend on
instance.update_value_parameters()  # AFSC quotas are a good example here

# Always make sure your data is good to go!
instance.parameter_sanity_check()

# Run the SOC algorithms!
instance.soc_rated_matching_algorithm({"soc": "rotc", "ma_printing": True})
instance.soc_rated_matching_algorithm({"soc": "usafa", "ma_printing": True})

# Export data back to csvs
instance.export_data()
```
??? note "💻 Console Output"
    ```text
    Importing 'Random_1' instance...
    Instance 'Random_1' initialized.
    Sanity checking the instance parameters...
    Done, 0 issues found.
    Solving the rated matching algorithm for ROTC cadets...
    
    Iteration 1
    Proposals: {'R2': 6, 'R4': 6}
    Matched {'R2': 4, 'R4': 2}
    Rejected {'R2': 2, 'R4': 4}
    
    Iteration 2
    Proposals: {'R2': 6, 'R4': 4}
    Matched {'R2': 4, 'R4': 2}
    Rejected {'R2': 4, 'R4': 6}
    Solving the rated matching algorithm for USAFA cadets...
    
    Iteration 1
    Proposals: {'R2': 5, 'R4': 2}
    Matched {'R2': 1, 'R4': 0}
    Rejected {'R2': 4, 'R4': 2}
    
    Iteration 2
    Proposals: {'R2': 2, 'R4': 2}
    Matched {'R2': 1, 'R4': 0}
    Rejected {'R2': 5, 'R4': 4}
    Exporting datasets ['Cadets', 'AFSCs', 'Preferences', 'Goal Programming', 'Value Parameters', 'Solutions', 'Additional', 'Base Solutions', 'Course Solutions']
    ```

The reason I have the above code split up from below is because you only need to run the rated SOC algorithms 
themselves once, and then from that point on they exist in your "Solutions.csv" file, and you can just incorporate them 
into the parameters everytime!

```python
# Import the "Random_1" instance
instance = CadetCareerProblem('Random_1')

# "Activate" a particular set of value parameters (since you can have multiple)
instance.set_value_parameters("VP")  # There could be "VP", "VP2", "VP3", etc.

# In case you change some parameter that the value parameters depend on
instance.update_value_parameters()  # AFSC quotas are a good example here

# From now on, this becomes one of your "default" functions to apply!
instance.incorporate_rated_algorithm_results()

# Always make sure your data is good to go!
instance.parameter_sanity_check()

# Run GUO!
instance.solve_guo_pyomo_model()

# Export data back to csvs
instance.export_data()
```
??? note "💻 Console Output"
    ```text
    Importing 'Random_1' instance...
    Instance 'Random_1' initialized.
    Incorporating rated algorithm results...
    Rated SOC Algorithm Results:
    USAFA Fixed Cadets: 1, USAFA Reserved Cadets: 0, ROTC Fixed Cadets: 2, ROTC Reserved Cadets: 4
    USAFA Rated Alternates: 0, ROTC Rated Alternates: 7
    Sanity checking the instance parameters...
    Done, 0 issues found.
    Building assignment problem (GUO) model...
    Done. Solving model...
    Solving GUO Model instance with solver cbc...
    Start Time: June 10 2025 10:22:40 AM.
    Model solved in 0.2 seconds at June 10 2025 10:22:40 AM. Pyomo reported objective value: 0.8364
    New Solution Evaluated.
    Measured exact VFT objective value: 0.8718.
    Global Utility Score: 0.8364. 3 / 3 AFSCs fixed. 4 / 4 AFSCs reserved. 7 / 7 alternate list scenarios respected.
    Blocking pairs: 4. Unmatched cadets: 0.
    Matched cadets: 20/20. N^Match: 20. Ineligible cadets: 0.
    
    Exporting datasets ['Cadets', 'AFSCs', 'Preferences', 'Goal Programming', 'Value Parameters', 'Solutions', 'Additional', 'Base Solutions', 'Course Solutions']
    ```

If you've followed along with your own random dataset, then you should have several solutions now in your 
"Random_1 Solutions.csv" file:

<p align="center">
  <img src="/afccp/images/user-guide/pic51.png" width="800px">
</p>

We now have the "GUO" solution too which we will use to visualize in the next tutorial.

## 📌 Summary

Tutorial 8 outlines the core **optimization models** used to solve the `CadetCareerProblem`, focusing on how 
cadet-to-AFSC assignments are generated using formal decision models. The section presents multiple optimization strategies 
that evolved from academic and operational contributions.

It covers the following models and methods:

1. **Value-Focused Thinking (VFT) Models** – Includes both Approximate and Exact models, originally developed at AFIT.

    - Highlights their differences, how they’re solved using Pyomo, and why Approximate models are often preferred in practice.
    - Explains how the `con_fail_dict` is used to manage constraint violations and guide genetic algorithms.

2. **Global Utility Optimization (GUO)** – The current default assignment model that maximizes a merged cadet-AFSC utility matrix.

    - Efficient and scalable with accurate Pyomo objective value evaluation.
    - Used with rated cadet integration and the “USSF OM” constraint when applicable.
   
3. **Original AFPC Model** – Rebuilt for historical reference, this model mimics the legacy assignment method used through 2023.
4. **Goal Programming Model** – Captures Lt. Rebecca Reynold's research contributions using “Mandatory/Desired/Permitted” logic instead of value tiers.

    - The model can be executed after translating value parameters and is unconstrained by `afccp`'s standard rule set.
   
5. **Sensitivity Analysis** – An early capability that tests the impact of activating constraints one at a time on solution feasibility.

The tutorial emphasizes how constraints, value parameters, and alternate list logic continue to play a vital role in 
each formulation. The GUO model is currently used for official cadet-AFSC classification. All models integrate with the 
solution evaluation tools covered in [Tutorial 6](../user-guide/tutorial_6.md) and the algorithm controls from 
[Tutorial 7](../user-guide/tutorial_7.md). Continue on to [Tutorial 9](../user-guide/tutorial_9.md) for more information
on the visualizations within `afccp`.