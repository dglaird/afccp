# Tutorial 7: Algorithms

Now that we've discussed how a solution is represented in the data and stored within the `CadetCareerProblem` object, 
we can talk about the methods of generating solutions (ones that are hopefully a bit better than throwing darts at the 
board with the "Random" method). The first series of solution techniques I'll describe are matching algorithms. 

## Matching Algorithms

Matching algorithms encompass the various methods of assignment by which we don't incorporate any sort of optimization 
framework. The first of which is the classic "Hospital/Residents" (HR) deferred acceptance algorithm (DAA). 

### Classic HR

We formulate the AFSC/Cadet matching problem as a Hospital/Residents problem where each entity possesses an ordinal 
preference list for members of the other. It's a simple algorithm, and the function itself is 
[`classic_hr()`](../../../afccp/reference/solutions/algorithms/#solutions.algorithms.classic_hr).

```python
# Run the "Classic HR" algorithm on the "Random_1" instance
s = instance.classic_hr({'ma_printing': True})
```
??? note "💻 Console Output"
    ```text
    Modeling this as an H/R problem and solving with DAA...

    Iteration 1
    Proposals: {'R1': 9, 'R2': 3, 'R3': 6, 'R4': 2}
    Matched {'R1': 9.0, 'R2': 3.0, 'R3': 3.0, 'R4': 2.0}
    Rejected {'R1': 0.0, 'R2': 0.0, 'R3': 3.0, 'R4': 0.0}
    
    Iteration 2
    Proposals: {'R1': 11, 'R2': 3, 'R3': 3, 'R4': 3}
    Matched {'R1': 9.0, 'R2': 3.0, 'R3': 3.0, 'R4': 3.0}
    Rejected {'R1': 2.0, 'R2': 0.0, 'R3': 3.0, 'R4': 0.0}
    
    Iteration 3
    Proposals: {'R1': 9, 'R2': 5, 'R3': 3, 'R4': 3}
    Matched {'R1': 9.0, 'R2': 5.0, 'R3': 3.0, 'R4': 3.0}
    Rejected {'R1': 2.0, 'R2': 0.0, 'R3': 3.0, 'R4': 0.0}
    New Solution Evaluated.
    Measured exact VFT objective value: 0.8777.
    Global Utility Score: 0.8158. 0 / 0 AFSCs fixed. 0 / 0 AFSCs reserved. 0 / 0 alternate list scenarios respected.
    Blocking pairs: 0. Unmatched cadets: 0.
    Matched cadets: 20/20. N^Match: 20. Ineligible cadets: 0.
    ```

The "ma_printing" parameter controls whether the algorithm should print iteration-specific information 
(it's defaulted to False). As you can see, with 20 cadets and 4 AFSCs this is a pretty short algorithm. 
With my particular randomly generated cadets, only three cadets were rejected in iteration 1 (all by R3). 
In the second iteration, two of these cadets propose to R1 and one proposes to R4. The two proposing to R1 are then 
rejected by R1 while the 1 proposing to R4 is accepted. Finally, in the third iteration those two cadets both propose
and are accepted by R2 and the algorithm concludes.

```python
# Current solution (HR)
print("Solution Name:", instance.solution['name'])
print("Solution Array:", instance.solution['afsc_array'])
```
??? note "💻 Console Output"
    ```text
    Solution Name: HR
    Solution Array: ['R1' 'R1' 'R2' 'R2' 'R1' 'R1' 'R2' 'R1' 'R1' 'R4' 'R4' 'R2' 'R3' 'R1' 'R2' 'R1' 'R3' 'R3' 'R1' 'R4']
    ```

### SOC Rated Algorithm

One very important solution technique we need to address is the specific rated algorithm we employ to pre-process 
cadets into rated career fields in the spirit of the DAA. Although the DAA in its true form does not work for all 
career fields across the Air & Space Force (due to degree requirements and popularity issues), it does work really 
well if we use it on the rated career fields in a unique way. Pilot, and the other three rated career fields to a 
lesser extent, is a very "political" career field and one that must be handled with great care. Any kind of matches 
are fair game in an optimization model, and people don't always get matched to their desired career fields in order of 
merit unless we explicitly enforce it. One huge benefit of the DAA is that it's simple and defendable. It is fair to 
all cadets by directly adhering to the career field rankings. Because we want a defendable solution (especially when 
it comes to pilot), we apply the DAA to rated cadets from both SOCs to constrain AFSCs for cadets based on what 
they would have received as a result of the "complete" DAA.

As discussed, this algorithm strips out all non-rated AFSC choices and purely looks at the rated choices, 
ordered amongst themselves. We then run the DAA, matching cadets to their rated AFSCs from each SOC honoring their 
specific rated PGL targets. If the DAA matches a cadet to a rated AFSC that was also their true first choice then that
cadet is definitively matched to that AFSC. If the cadet is matched to a rated AFSC but that AFSC was not their 
first choice, we reserve that slot for them but allow them to compete in the optimization model for their more 
desired AFSCs. To apply this methodology, we run the [`soc_rated_matching_algorithm()`](../../../afccp/reference/solutions/algorithms/#solutions.algorithms.soc_rated_matching_algorithm)
function for each SOC. We'll show this method for ROTC first:

```python
# Run the SOC rated algorithm for ROTC
s = instance.soc_rated_matching_algorithm({"soc": "rotc", "ma_printing": True})  # "s =" prevents lots of output
```
??? note "💻 Console Output"
    ```text
    Solving the rated matching algorithm for ROTC cadets...

    Iteration 1
    Proposals: {'R2': 6, 'R4': 6}
    Matched {'R2': 4, 'R4': 2}
    Rejected {'R2': 2, 'R4': 4}
    
    Iteration 2
    Proposals: {'R2': 6, 'R4': 4}
    Matched {'R2': 4, 'R4': 2}
    Rejected {'R2': 4, 'R4': 6}
    ```

As you can see from the output, we're evaluating three new solutions. I use the solutions dictionary to store my 
results from the rated algorithm for both SOCs. One solution contains the information on cadets with reserved slots 
for rated AFSCs, one solution contains the information on cadets that were definitively matched to a rated AFSC, 
and the last one simply combines the two. You can see these solutions below for ROTC:

```python
for s_name in instance.solutions:
    
    if "Rated" in s_name:
        print("Solution name: '" + s_name + "'")
        print("Solution:", instance.solutions[s_name]['afsc_array'])
```
??? note "💻 Console Output"
    ```text
    Solution name: 'Rated ROTC HR (Reserves)'
    Solution: ['*' '*' '*'  '*' '*' '*' '*' 'R2' '*' '*' '*' 'R2' 'R2' '*' '*' '*' '*' '*' 'R4' '*' ]
    Solution name: 'Rated ROTC HR (Matches)'
    Solution: ['*' '*' 'R2' '*' '*' '*' '*' '*'  '*' '*' '*' '*'  '*'  '*' '*' '*' '*' '*' '*'  'R4']
    Solution name: 'Rated ROTC HR'
    Solution: ['*' '*' 'R2' '*' '*' '*' '*' 'R2' '*' '*' '*' 'R2' 'R2' '*' '*' '*' '*' '*' 'R4' 'R4']
    ```

In my simple "Random_1" data example, we have two rated AFSCs: R2 and R4. There are 4 ROTC slots for R2 and 2 ROTC 
slots for R4. The cadet at index 2 and the cadet at index 19 both had R2 and R4, respectively, as their first choice 
AFSCs. Therefore, they are matched to those AFSCs. The other 4 cadets are reserved for these rated AFSCs.
Also in my data example, USAFA has 1 slot for R2 and doesn't have any slots for R4, so that algorithm isn't as
useful. We still need to run it to have the required components, however!

```python
print("USAFA Targets:", {p['afscs'][j]: p['usafa_quota'][j] for j in p['J']})  # None for R4!
print("ROTC Targets:", {p['afscs'][j]: p['rotc_quota'][j] for j in p['J']})  # 2 for R4!
print("")  # just adding a space

# Run the USAFA rated algorithm
s = instance.soc_rated_matching_algorithm({"soc": "usafa", "ma_printing": True})  # "s =" prevents lots of output
```
??? note "💻 Console Output"
    ```text
    USAFA Targets: {'R1': 1.0, 'R2': 1.0, 'R3': 0.0, 'R4': 0.0}
    ROTC Targets: {'R1': 7.0, 'R2': 4.0, 'R3': 2.0, 'R4': 2.0}
    
    Solving the rated matching algorithm for USAFA cadets...
    
    Iteration 1
    Proposals: {'R2': 5, 'R4': 2}
    Matched {'R2': 1, 'R4': 0}
    Rejected {'R2': 4, 'R4': 2}
    
    Iteration 2
    Proposals: {'R2': 2, 'R4': 2}
    Matched {'R2': 1, 'R4': 0}
    Rejected {'R2': 5, 'R4': 4}
    ```

We've now run these two algorithms (USAFA/ROTC) and can incorporate their information into our instance parameters. 
This will be used later on within the optimization models, but worth discussing here. In order to incorporate these 
solutions into the parameters, we need to call the 
[`incorporate_rated_results_in_parameters()`](../../../afccp/reference/solutions/handling/#solutions.handling.incorporate_rated_results_in_parameters) 
function!

```python
# Integrate the Rated algorithm solutions into "instance.parameters" 
instance.incorporate_rated_algorithm_results()
```
??? note "💻 Console Output"
    ```text
    Incorporating rated algorithm results...
    Rated SOC Algorithm Results:
    USAFA Fixed Cadets: 1, USAFA Reserved Cadets: 0, ROTC Fixed Cadets: 2, ROTC Reserved Cadets: 4
    USAFA Rated Alternates: 0, ROTC Rated Alternates: 7
    ```

This creates two new dictionaries: "J^Fixed" and "J^Reserved". They are used to constrain the available AFSCs for 
each cadet as a result of the SOC-specific rated DAA described above.

```python
# 3 cadets are fixed to AFSCs. Ex: Cadet 14 must be matched to the AFSC at index 1 (R2)
print("'J^Fixed':", instance.parameters['J^Fixed'])
```
??? note "💻 Console Output"
    ```text
    'J^Fixed': {14: 1, 2: 1, 19: 3}
    ```

"J^Fixed" is a dictionary where the keys are the cadets with "fixed" AFSCs and the values are the AFSC indices that 
they are fixed to. In a similar, but slightly different way, "J^Reserved" is a dictionary where the keys are the 
cadets with "reserved" AFSCs. For each key here, however, the value is an ordered list of the cadet's preferences up 
to and including the rated AFSC they're reserved for. This ensures that the cadet receives one of these AFSCs, since 
the rated AFSC is reserved for them.

```python
# Cadet 3 must receive either AFSC 0 (R1), 2 (R3), or 3 (R4) -> R4 is reserved for them
print("'J^Reserved':", instance.parameters['J^Reserved'])
```
??? note "💻 Console Output"
    ```text
    'J^Reserved': {7: array([0, 3, 2, 1]), 11: array([0, 1]), 12: array([2, 0, 1]), 18: array([0, 3])}
    ```

### SOC Rated Algorithm-Alternates

One thing I haven't mentioned yet is how we give even more special attention to rated AFSCs with an 
"alternate list" idea. The fixed/reserved slot idea works really well to "protect" those individuals that would get 
rated slots based purely on their rated OM, by giving them a chance to get their more preferred AFSC. 
If they don't get that more preferred AFSC, they have their reserved rated AFSC to fall back on. What this current 
methodology does not address, however, is who back-fills those reserved slots. To do this, we have to run a separate 
algorithm to determine who is on each alternate list for each SOC. Essentially, we want to know who is "next in line" 
to receive each of the rated AFSCs. These individuals just missed the cutoff to be a reserved/fixed cadet for that AFSC. 

The SOC rated matching algorithm must be run once, and then every time we execute 
[`instance.incorporate_rated_results_in_parameters()`](../../../afccp/reference/solutions/handling/#solutions.handling.incorporate_rated_results_in_parameters), 
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

Nothing really happens here as a product of the randomness in this small example. Effectively, the rated alternates'
algorithm addition provides the set of cadet-AFSC pairs that must not form blocking pairs. In the optimization models,
these cadet-AFSC pairs are provided special blocking pair constraints, enforcing the "alternate list" concept.

### ROTC Rated Board Original Algorithm

This one is not too important anymore, but it is included in the code since it helped my case to convince ROTC that 
this new rated algorithm was a good idea. I'm not going to discuss it too much, but it uses the ROTC rated OM 
dataset and the rotc rated interest dataset: Random_1 ROTC Rated Interest.csv. Their board then commenced in 
different phases and if you're really interested I suggest you read the 
[`rotc_rated_board_original()`](../../../afccp/reference/solutions/algorithms/#solutions.algorithms.rotc_rated_board_original)
function and then reach out with questions if it's still unclear!

```python
# Run the algorithm
s = instance.rotc_rated_board_original()  # "s =" prevents lots of output
```
??? note "💻 Console Output"
    ```text
    Running status quo ROTC rated algorithm...

    Phase 1 High OM & High Interest
    R2 Phase Matched: 2   --->   Total Matched: 2 / 4.0
    R4 Phase Matched: 2   --->   Total Matched: 2 / 2.0
    
    Phase 2 High OM & Med Interest
    R2 Phase Matched: 2   --->   Total Matched: 4 / 4.0
    New Solution Evaluated.
    Measured exact VFT objective value: 0.3705.
    Global Utility Score: 0.6372. 3 / 3 AFSCs fixed. 3 / 4 AFSCs reserved. 6 / 7 alternate list scenarios respected.
    Blocking pairs: 19. Unmatched cadets: 14.
    Matched cadets: 6/20. N^Match: 20. Ineligible cadets: 0.
    ```

Now that we've incorporated the rated algorithm results within our parameters, you'll notice the solution evaluation 
output reflects that we should have three cadets with fixed AFSCs and four cadets with reserved AFSCs. 
In the original rated board algorithm we just ran, one of those constraints was adequately met and the other wasn't. 
Additionally, we have adequately met 6/7 of the alternate cases, meaning there is still one rated cadet-AFSC blocking
pair which is in violation of the alternate list concept.

## Meta-heuristics
This section describes the meta-heuristics we have to solve the problem. Meta-heuristics refer to solution 
techniques that use unconventional algorithms to solve optimization models. They are typically meant for really 
complicated optimization models that are too computationally expensive to solve with a conventional "global" solver. 
Right now, the only meta-heuristic I have in the code is the genetic algorithm (GA). Genetic algorithms are great
methods of finding initial solutions that are then "evolved" into better and better solutions. This algorithm is 
currently used in conjunction with the VFT pyomo model (specifically, the "Approximate Model" I discuss in my thesis) 
to get closer to the optimal solution since I don't currently have a method of finding the global optimal solution to
the VFT "Exact Model". If these terms are confusing, please reference section 3.4.1 of my master's thesis.

### VFT Genetic Algorithm
For more information on this algorithm specifically, you can find it here: 
[`vft_genetic_algorithm()`](../../../afccp/reference/solutions/algorithms/#solutions.algorithms.vft_genetic_algorithm). 
One big note here is that I haven't found a good way of reconciling the many constraints of the optimization model(s) 
with the GA. When we were just solving the NRL process, before combining Rated and USSF for FY2024 (so just for 2023 
really), the only constraints I had placed were the AFSC objective constraints and some cadet utility constraints 
(top 10%). I will describe how these constraints are handled later on using the "con_fail_dict" (constraint fail 
dictionary). This is the reason why there are initially a lot of 0s in the initial population of solutions below 
(constraint violations result in a fitness score of 0). Since this is a small & simple example, however, we quickly 
determine feasible solutions right out the gate.

```python
# Solve the GA using the VFT objective function as the fitness function
s = instance.vft_genetic_algorithm({"initialize": False, "ga_max_time": 20})
```
??? note "💻 Console Output"
    ```text
    Running Genetic Algorithm with no initial solutions (not advised!)...
    Initial Fitness Scores [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    10% complete. Best solution value: 0.7579
    Average evaluation time for 12 solutions: 0.0143 seconds.
    Average generation time: 0.0156 seconds.
    20% complete. Best solution value: 0.7794
    Average evaluation time for 12 solutions: 0.013 seconds.
    Average generation time: 0.0141 seconds.
    30% complete. Best solution value: 0.7794
    Average evaluation time for 12 solutions: 0.0128 seconds.
    Average generation time: 0.0139 seconds.
    40% complete. Best solution value: 0.8029
    Average evaluation time for 12 solutions: 0.0125 seconds.
    Average generation time: 0.0136 seconds.
    50% complete. Best solution value: 0.8129
    Average evaluation time for 12 solutions: 0.0123 seconds.
    Average generation time: 0.0134 seconds.
    60% complete. Best solution value: 0.8129
    Average evaluation time for 12 solutions: 0.0125 seconds.
    Average generation time: 0.0134 seconds.
    70% complete. Best solution value: 0.8129
    Average evaluation time for 12 solutions: 0.0125 seconds.
    Average generation time: 0.0136 seconds.
    80% complete. Best solution value: 0.8129
    Average evaluation time for 12 solutions: 0.0125 seconds.
    Average generation time: 0.0135 seconds.
    90% complete. Best solution value: 0.8129
    Average evaluation time for 12 solutions: 0.0123 seconds.
    Average generation time: 0.0134 seconds.
    100% complete. Best solution value: 0.8129
    Average evaluation time for 12 solutions: 0.0123 seconds.
    Average generation time: 0.0134 seconds.
    End time reached in 20.01 seconds.
    New Solution Evaluated.
    Measured exact VFT objective value: 0.8129.
    Global Utility Score: 0.7814. 3 / 3 AFSCs fixed. 4 / 4 AFSCs reserved. 7 / 7 alternate list scenarios respected.
    Blocking pairs: 6. Unmatched cadets: 0.
    Matched cadets: 20/20. N^Match: 20. Ineligible cadets: 0.
    ```

### Genetic Matching Algorithm

Ian Macdonald had the idea to create a GA in order to iteratively determine the optimal capacities for each AFSC to 
find a solution that is as stable as possible. This algorithm works if we have a designated range on the number of 
cadets for each AFSC. Essentially, we need a sizeable "surplus" of cadets above the PGL targets. For sake of time, 
I won't go into too much detail here but the algorithm is shown below.

```python
# Because this is a small/easy problem, we easily find a stable solution! (GMA isn't necessary)
s = instance.genetic_matching_algorithm({'gma_printing': True})  # Turns on print statements for the GMA
```
??? note "💻 Console Output"
    ```text
    Generation 0 Fitness [0. 0. 0. 1.]
    Final capacities: [9. 9. 2. 3.]
    Modeling this as an H/R problem and solving with DAA...
    New Solution Evaluated.
    Measured exact VFT objective value: 0.8425.
    Global Utility Score: 0.7998. 3 / 3 AFSCs fixed. 4 / 4 AFSCs reserved. 7 / 7 alternate list scenarios respected.
    Blocking pairs: 0. Unmatched cadets: 0.
    Matched cadets: 20/20. N^Match: 20. Ineligible cadets: 0.
    ```

## 📌 Summary

Tutorial 7 dives into the **algorithmic backbone** of how `afccp` generates cadet-to-AFSC assignments. 
It introduces a variety of algorithmic approaches—both matching-based and meta-heuristic—that serve as the 
foundation for feasible and defensible solutions before any optimization model is solved.

The tutorial walks through:

- Classic matching algorithms like **Deferred Acceptance (DAA)** using the Hospital/Residents (HR) formulation.
- SOC-specific **Rated Matching Algorithms** that pre-process cadets into fixed or reserved rated slots to ensure fairness and defendability—particularly for pilot and other rated fields.
- How to incorporate fixed/reserved/alternate outcomes from the rated matching into the `CadetCareerProblem` parameters for use in optimization models.
- The structure and purpose of alternate lists: identifying cadets who just missed a rated assignment.
- The legacy **ROTC Rated Board algorithm** used to demonstrate improvements over prior processes.

It then transitions to **meta-heuristic methods** like:

- The `vft_genetic_algorithm()`—a genetic algorithm designed to navigate complex constraints in the VFT (Value-Focused Thinking) optimization model.
- The `genetic_matching_algorithm()`—which uses evolutionary logic to discover AFSC capacities that yield the most stable HR solution.

These methods lay the groundwork for solving more advanced optimization models under real-world constraints. 
Continue on to [Tutorial 8](../user-guide/tutorial_8.md) to learn how these results integrate into the full optimization pipeline of `afccp`.