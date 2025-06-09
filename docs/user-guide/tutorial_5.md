# Tutorial 5: Data Methods

Now that I've described in detail the data that afccp uses, let's talk about the data-manipulation methods available 
for the `CadetCareerProblem` class. For a "Random" generated problem set, it doesn't really matter what you do to 
the data which is why I created a convenient 
[`instance.fix_generated_data()`](../../../afccp/reference/main/cadetcareerproblem_generated_data_corrections/) 
function that gets your generated data looking right for you to run different models & algorithms. 
Many of the components of that function are also relevant for a real class year, but not all!

```python
# Takes the CIP codes and translates them to the qualification matrix (only works for real data)
instance.calculate_qualification_matrix()
```
!!! danger "Traceback (most recent call last)"
    ```text
        ValueError                                Traceback (most recent call last)
    Input In [116], in <cell line: 2>()
          1 # Takes the CIP codes and translates them to the qualification matrix (only works for real data)
    ----> 2 instance.calculate_qualification_matrix()
    
    File ~/Desktop/Coding Projects/afccp/afccp/core/main.py:432, in CadetCareerProblem.calculate_qualification_matrix(self, printing)
        429         qual_matrix = afccp.core.data.support.cip_to_qual_tiers(
        430             parameters["afscs"][:parameters["M"]], parameters['cip1'])
        431 else:
    --> 432     raise ValueError("Error. Need to update the degree tier qualification matrix to include tiers "
        433                      "('M1' instead of 'M' for example) but don't have CIP codes. Please incorporate this.")
        435 # Load data back into parameters
        436 parameters["qual"] = qual_matrix
    
    ValueError: Error. Need to update the degree tier qualification matrix to include tiers ('M1' instead of 'M' for example) but don't have CIP codes. Please incorporate this.
    ```

As you can see, the code above produces an error since we don't have any CIP codes in "Random_1 Cadets.csv". 
Going further, even if we did put them into our dataset it still wouldn't work since the AFSCs themselves are also 
meaningless. This function only works with real AFSC names and with CIP codes in our cadets data. As an aside, 
I've tried to incorporate a lot of error handling throughout `afccp` but it's certainly not flawless and still a 
work in progress so if you encounter something that doesn't make since please let me know and I will a) help you fix it 
and also b) include some more error handling measures so that we catch that kind of error in a manner that 
makes more sense to the analyst using `afccp`.

## Operational Data Processing (Meant just for AFPC/DSYA)

The intent of this section is primarily for the "operational lens" of this problem, since we have the easy 
method I've already described that will fix the fake data for you anyway: [`instance.fix_generated_data()`](../../../afccp/reference/main/cadetcareerproblem_generated_data_corrections/). 

Ok, your main job for processing a given class year of cadets is to construct the "Cadets.csv", "AFSCs.csv" and 
"AFSCs Preferences.csv", and the default value parameters excel file located in the support sub-folder. 
Additionally, to create the AFSC preferences for Rated career fields, you'll also need the "ROTC Rated OM.csv" and 
"USAFA Rated OM.csv" files. "AFSCs Preferences.csv" should contain the necessary rankings for Non-Rated AFSCs, 
with something as a placeholder for rated career field rankings. The two OM datasets should have the percentiles 
for their respective source of commissioning cadets that are rated eligible (they volunteered). If a cadet volunteered 
for Rated, and is eligible for at least one of the rated AFSCs, they should be in this dataset. If the cadet is not 
eligible for a specific rated AFSC, they have a "0" in that position. The important thing here is that the cadets 
can be sorted in descending order with highest ranking at the top and lowest at the bottom using those "percentile" 
values (1, 0.99, 0.98, ..., 0.01, 0, 0, etc.). This is what the code does to get the ordered list of cadets (AFSC preferences). 

```python
# Takes the two Rated OM datasets and re-calculates the AFSC rankings for Rated AFSCs for both SOCs
instance.construct_rated_preferences_from_om_by_soc()
```
??? note "💻 Console Output"
    ```text
    Integrating rated preferences from OM matrices for each SOC...
    ```

The [`construct_rated_preferences_from_om_by_soc()`](../../../afccp/reference/main/cadetcareerproblem_main_data_corrections/#afccp.main.CadetCareerProblem.construct_rated_preferences_from_om_by_soc)
creates the rated AFSC preferences in the `a_pref_matrix` by splicing together the Rated OM data from each SOC. 
Initially, the AFSC preferences for each rated AFSC should just be all ones or something like that as a placeholder. 
After this method is run, we have the true preferences from the OM data, and we will know true eligibility as well.

```python
instance.parameters['a_pref_matrix']  # Nothing will change here since we already had this merged correctly
```
??? note "💻 Console Output"
    ```text
    array([[ 6, 16,  0, 16],
           [ 9, 15,  0, 15],
           [20,  4, 13,  9],
           [17, 11, 11,  0],
           [10,  0,  0, 10],
           [13, 13, 12,  7],
           [ 7,  5,  9,  0],
           [ 1,  6,  5,  4],
           [12, 14,  0, 14],
           [19,  0,  4, 11],
           [ 4,  0,  0,  3],
           [16,  8,  0, 13],
           [11,  3,  2,  6],
           [ 3,  0, 10,  0],
           [ 2,  2,  0, 12],
           [15, 12,  7,  0],
           [14, 10,  1,  8],
           [ 8,  7,  3,  5],
           [ 5,  1,  8,  1],
           [18,  9,  6,  2]])
    ```

Once this method is run, we should now have an `a_pref_matrix` that is almost 100% accurate (the only drift should come 
from cadet preferences). Once you have the AFSC preferences file that is accurate, you can then update the 
qualification matrix using these preferences. The default qual matrix allows all cadets to be eligible for rated 
AFSCs and for the Space Force in general. It also restricts eligibility for Non-Rated AFSCs to be based on the 
cadets' degrees (CIP codes). We need to update it by restricting Rated/USSF eligibility down to volunteerism, and 
relaxing the AFOCD a bit for Non-Rated AFSCs (creating "exceptions" designated by "E"). This is accomplished through 
the [`update_qualification_matrix_from_afsc_preferences()`](../../../afccp/reference/main/cadetcareerproblem_main_data_corrections/#afccp.main.CadetCareerProblem.update_qualification_matrix_from_afsc_preferences)
method:

```python
# Update qualification matrix from AFSC preferences (treating CFM lists as "gospel" except for Rated/USSF)
instance.update_qualification_matrix_from_afsc_preferences()
```
??? note "💻 Console Output"
    ```text
    1 Making 4 cadets ineligible for 'R2' by altering their qualification to 'I2'. 
    3 Making 4 cadets ineligible for 'R4' by altering their qualification to 'I2'. 
    ```

The message above isn't actually doing anything since those cadets were already ineligible since we've run 
[instance.fix_generated_data()](../../../afccp/reference/main/cadetcareerproblem_generated_data_corrections/#afccp.main.CadetCareerProblem.fix_generated_data) earlier.

Now, we should have a qualification matrix that agrees with the AFSC preference matrix in terms of eligibility. 
If it didn't, it would warn you and tell you where the discrepancies are, so that you can correct them or ignore them 
depending on what's going on. 

Once these two matrices are rectified, it's time to focus on fixing the cadet preferences. First, we fill in any AFSCs 
that the cadets may be eligible for, but didn't preference. Since the cadets only have to place 10 choices, they may 
not select every career field for which they are eligible. We fill in the rest of the choices arbitrarily after the 
AFSCs they selected. Additionally, the cadets may have indicated which AFSCs they do NOT want. In that case, we put them
at the end of their preference lists. This occurs in the [`fill_remaining_afsc_choices()`](../../../afccp/reference/main/cadetcareerproblem_main_data_corrections/#afccp.main.CadetCareerProblem.fill_remaining_afsc_choices)
method:

```python
# Fill in remaining AFSC choices that the cadets could be eligible for, but...
instance.fill_remaining_afsc_choices()  # ...didn't select
```
!!! danger "Traceback (most recent call last)"
    ```text
        KeyError                                  Traceback (most recent call last)
        <ipython-input-213-64cad255bf8b> in <cell line: 2>()
              1 # Fill in remaining AFSC choices that the cadets could be eligible for, but...
        ----> 2 instance.fill_remaining_afsc_choices()  # ...didn't select
        
        ~/Coding Projects/afccp/afccp/main.py in fill_remaining_afsc_choices(self, printing)
            947         # Import default value parameters
            948         self.import_default_value_parameters(printing=printing, vp_defaults_filename=vp_defaults_filename)
        --> 949 
            950         # Takes the two Rated OM datasets and re-calculates the AFSC rankings for Rated AFSCs for both SOCs
            951         self.construct_rated_preferences_from_om_by_soc(printing=printing)
        
        ~/Coding Projects/afccp/afccp/data/preferences.py in fill_remaining_preferences(parameters)
            647 
            648         # If this cadet does not have any preferences, we skip them (must be an OTS candidate)
        --> 649         if len(p['cadet_preferences'][i]) == 0:
            650             continue
            651 
        
        KeyError: 'J^Bottom 2 Choices'
    ```

As you can see, there is an error when we try to run this method on fake data. This error occurs because we do not have
the 'J^Bottom 2 Choices' parameter which is one of the features produced by the survey data. This is an indexed set of
the 1 or 2 bottom choice AFSCs that the cadets could select (the LAST choice itself is in the 'J^Last Choice' parameter).
These parameters do not exist for random data, so we don't run this action as part of the `fix_generated_data()` method.

Now that we have all possible AFSCs in each cadet's preferences, we must remove AFSCs for which the cadet is not 
actually eligible. Be careful with this one as you need to make sure 
that the AFSC preferences and the qual matrix reflects accurate eligibility as this will remove cadets' choices! 
It checks all three "sources of truth" in terms of eligibility (cadet/AFSC preferences and degree qual matrix) and 
if one of them says that the cadet is ineligible for a given AFSC, this method forces all three to reflect 
ineligibility. It is a rather "nuclear" approach, so again, be careful! Let's run the 
[`remove_ineligible_choices()`](../../../afccp/reference/main/cadetcareerproblem_main_data_corrections/#afccp.main.CadetCareerProblem.remove_ineligible_choices) 
method here:

```python
# Removes ineligible cadets from all 3 matrices: degree qualifications, cadet preferences, AFSC preferences
instance.remove_ineligible_choices()  # Nothing changed since we've already done this!
```
??? note "💻 Console Output"
    ```text
    Removing ineligible cadets based on any of the three eligibility sources (c_pref_matrix, a_pref_matrix, qual)...
    0 total adjustments.
    ```

This method also re-runs the parameter additions function which will correct the preference lists themselves 
(the dictionaries for each cadet/AFSC "key" that has the sorted AFSCs/cadets list as the "value"). Just always 
remember the difference between "a_pref_matrix" and "afsc_preferences" (similarly, "c_pref_matrix" and 
"cadet_preferences"). One is a 2-dimensional numpy array and the other is a dictionary. 

From here, we want to "fill the gaps" in the matrix to create the final 1-N lists. We use the preference 
dictionaries ("afsc_preferences" and "cadet_preferences") to construct their corresponding matrices from scratch using 
the [`update_preference_matrices()`](../../../afccp/reference/main/cadetcareerproblem_main_data_corrections/#afccp.main.CadetCareerProblem.update_preference_matrices).

```python
# Take the preferences dictionaries and update the matrices from them (using cadet/AFSC indices)
instance.update_preference_matrices()  # 1, 2, 4, 6, 7 -> 1, 2, 3, 4, 5 (preference lists need to omit gaps)
```
??? note "💻 Console Output"
    ```text
    Updating cadet preference matrices from the preference dictionaries. ie. 1, 2, 4, 6, 7 -> 1, 2, 3, 4, 5 (preference lists need to omit gaps)
    ```

When we have the final cadet preference lists, we must ensure that each cadet has a 100% utility value for their first
choice AFSC. There is the possibility, and it will happen multiple times each classification cycle, that some cadets
are ineligible for this first choice AFSC. Their first choice always has a utility value of 100% on the survey, but
if they cannot be matched to it then they get a new first choice (their most preferred AFSC that they CAN be assigned to).
Therefore, whichever AFSC is their final first choice will be given a utility value of 1 in the 
[`update_first_choice_cadet_utility_to_one()`](../../../afccp/reference/main/cadetcareerproblem_main_data_corrections/#afccp.main.CadetCareerProblem.update_first_choice_cadet_utility_to_one) 
method:

```python
# Force first choice utility values to be 100%
instance.update_first_choice_cadet_utility_to_one()  # (We've already run this method so no one gets fixed)
```
??? note "💻 Console Output"
    ```text
    Updating cadet first choice utility value to 100%...
    Fixed 0 first choice cadet utility values to 100%.
    Cadets: []
    ```

Once we have the final AFSC rankings, we can construct the afsc_utility matrix (that lives in "AFSCs Utility.csv") 
that converts the 1-N preference list into linear percentiles. This occurs in the 
[`convert_afsc_preferences_to_percentiles()`](../../../afccp/reference/main/cadetcareerproblem_main_data_corrections/#afccp.main.CadetCareerProblem.convert_afsc_preferences_to_percentiles) 
method:

```python
# Convert AFSC preferences to percentiles (0 to 1)
instance.convert_afsc_preferences_to_percentiles()  # 1, 2, 3, 4, 5 -> 1, 0.8, 0.6, 0.4, 0.2
```
??? note "💻 Console Output"
    ```text
    Converting AFSC preferences (a_pref_matrix) into percentiles (afsc_utility on AFSCs Utility.csv)...
    ```

Until now, we've only been manipulating the cadet/AFSC preference matrices (information contained in 
"Cadets Preferences.csv" and "AFSC Preferences.csv"). We haven't adjusted the information from "Cadets.csv", 
since preferences are also contained there. We do this in the 
[`update_cadet_columns_from_matrices()`](../../../afccp/reference/main/cadetcareerproblem_main_data_corrections/#afccp.main.CadetCareerProblem.update_cadet_columns_from_matrices) 
method:

```python
# The "cadet columns" are located in Cadets.csv and contain the utilities/preferences in order of preference
instance.update_cadet_columns_from_matrices()  # We haven't touched "c_preferences" and "c_utilities" until now
```
??? note "💻 Console Output"
    ```text
    Updating cadet columns (Cadets.csv...c_utilities, c_preferences) from the preference matrix (c_pref_matrix)...
    ```

One other thing you'll want to do is update the utility matrices for cadets from the cadets' data. These are the 
"utility" and "cadet_utility" matrices that live in "Cadets Utility.csv" and "Cadets Utility (Final).csv", respectfully.
We update the cadet utility matrices in the 
[`update_cadet_utility_matrices_from_cadets_data()`](../../../afccp/reference/main/cadetcareerproblem_main_data_corrections/#afccp.main.CadetCareerProblem.update_cadet_utility_matrices_from_cadets_data) 
method:

```python
instance.update_cadet_utility_matrices_from_cadets_data()
```
??? note "💻 Console Output"
    ```text
    Updating cadet utility matrices ('utility' and 'cadet_utility') from the 'c_utilities' matrix
    ```

The last thing you should do is remove any cadets from the specific rated OM lists if they're no longer eligible for
at least one rated AFSC. Cadets need to be eligible for at least one rated AFSC to be considered a "rated eligible" 
cadet, otherwise the code breaks for "divide-by-zero" or "empty set" type errors. The 
[`modify_rated_cadet_lists_based_on_eligibility()`](../../../afccp/reference/main/cadetcareerproblem_main_data_corrections/#afccp.main.CadetCareerProblem.modify_rated_cadet_lists_based_on_eligibility) 
method will handle this necessary step:

```python
# Modify rated eligibility by SOC, removing cadets that are on "Rated Cadets" list...
instance.modify_rated_cadet_lists_based_on_eligibility()  # ...but not eligible for any rated AFSC
```
??? note "💻 Console Output"
    ```text
    Modifying rated eligibiity lists/matrices by SOC... 
    (Removing cadets that are on the lists but not eligible for any rated AFSC)
    ```

The methods I've just described all contribute to the main parameters ("parameters") of the problem. 
The only other thing I've alluded to is getting the default value parameters setup in the 
"support/value parameters defaults/" sub-folder. We created this file earlier with the "Random_1" instance by 
exporting the randomly generated set back to excel as defaults, so it's there for reference with the name 
"Value_Parameters_Defaults_Random_1.xlsx". The best way to do this for a real class year is to simply copy the 
previous year's default file and make adjustments as needed. I did the heavy lifting for incorporating Rated/USSF 
with 2024 and so the 2025+ files should be pretty similar, for example 
(Update: 2025/2026 files were!).

Once you create that default file, you'll just need to import your value parameters as defaults, and it will 
initialize them for your given class year from that file. This should be one of the very first things you do 
because my code relies on the value parameters extensively. 

```python
# Execute that function to get your set of value parameters
v = instance.import_default_value_parameters()  # "v =" prevents a lot of output (v is meaningless)
```
??? note "💻 Console Output"
    ```text
    Importing default value parameters...
    Imported.
    ```

For a real class year, if you followed those steps I described above then you should have all your files good to go 
and can start thinking about getting solutions! If you understand the data, and have processed it all 
correctly, then solving the model and getting a solution is as easy as hitting "go" with whatever algorithm/model 
you're using. In an effort to make this as easy as possible, I do have a method for processing the "real" data just like
the "fake" data. If you're modifying fake/generated data with no care to the efficacy of that data, you can run the
[`fix_generated_data()`](../../../afccp/reference/main/cadetcareerproblem_generated_data_corrections/) method. If, 
however, you're working with real or "realistic" data (data that you want to treat as real), then you should execute the
[`make_all_initial_real_instance_modifications()`](../../../afccp/reference/main/cadetcareerproblem_main_data_corrections/#afccp.main.CadetCareerProblem.make_all_initial_real_instance_modifications) 
method. This will run all the steps I've outlined in this tutorial in the correct order. I still believe this is an 
important tutorial because you should understand the data modifications that are occurring, and why they're necessary.

## 📌 Summary

This tutorial walks through all the core data-manipulation methods in the `CadetCareerProblem` class, especially 
useful when preparing a real-world dataset for modeling.

### Key Highlights

**Generated Data**: If you're using synthetic data, you can run [`instance.fix_generated_data()`](../../../afccp/reference/main/cadetcareerproblem_generated_data_corrections/) to automatically clean and configure everything for simulation and testing.

**Real Data**: For operational use, you’ll need to manually initialize the "Cadets.csv", "AFSCs.csv", "AFSCs Preferences.csv", 
and OM files (e.g., "ROTC Rated OM.csv"). The sequence of methods outlined ensures these files reflect realistic and 
consistent eligibility and preferences.

**Critical Processing Functions**:

  - `import_default_value_parameters()`: Initializes value parameters from Excel templates.
  - `construct_rated_preferences_from_om_by_soc()`: Integrates OM-based rated preferences by SOC.
  - `update_qualification_matrix_from_afsc_preferences()`: Aligns AFSC preferences with qualification logic.
  - `fill_remaining_afsc_choices()`: Fills in missing preferences.
  - `remove_ineligible_choices()`: Removes cadets/AFSCs that fail any eligibility check.
  - `update_preference_matrices()`: Rebuilds matrices from preference dictionaries.
  - `update_first_choice_cadet_utility_to_one()`: Ensures top choice utility is set to 1.0.
  - `convert_afsc_preferences_to_percentiles()`: Converts ranked preferences to utility percentiles.
  - `update_cadet_columns_from_matrices()`: Syncs Cadets.csv preference columns.
  - `update_cadet_utility_matrices_from_cadets_data()`: Generates utility matrices from preferences.
  - `modify_rated_cadet_lists_based_on_eligibility()`: Ensures Rated OM cadets are still eligible.

For real data, consider running [`make_all_initial_real_instance_modifications()`](../../../afccp/reference/main/cadetcareerproblem_main_data_corrections/#afccp.main.CadetCareerProblem.make_all_initial_real_instance_modifications) to streamline the full process.

> ⚠️ Note: Some functions rely on data only present in real datasets (e.g., CIP codes, OM ratings) and will throw errors if used with fake/generated data.

### Takeaway

Understand and validate your input data before modeling. `afccp` offers robust tools for cleaning, verifying, and 
transforming both synthetic and real datasets to prepare them for solution modeling.

For an initial overview of the `solutions` module, continue on to [Tutorial 6](../user-guide/tutorial_6.md).