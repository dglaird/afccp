# Air Force Cadet Career Problem

## Description

I created this project for the completion of my master's thesis. My thesis, titled "A New Approach to Career Field Matching for Air Force Cadets", developed a new methodology for assigning commissioning cadets from the United States Air Force Academy (USAFA) and several Reserve Officers' Training Corps (ROTC) detachments to their career fields. Career fields are alpha-numerically designated by Air Force Specialty Codes (AFSCs) and are referred to in this project as such. I developed an optimization model using a Value-Focused Thinking (VFT) framework to assign cadets to their AFSCs, implemented using the library "pyomo." The VFT model allows for solutions to be compared based on their overall quality, which is the weighted sum of all AFSC/cadet objective values. By specifying the weights and values of the decision maker (what they want and how much they want it) we can effectively generate solutions that meet their objective criteria. This project incorporates several Operations Research techniques that all work together to solve this assignment problem efficiently.

## Table of Contents 

- [Installation](#installation)
- [Usage](#usage)
    - [Globals](#globals)
    - [Data Handling](#data-handling)
    - [Value Parameter Handling](#value-parameter-handling)
    - [Heuristic Solvers](#heuristic-solvers)
    - [Pyomo Models](#pyomo-models)
    - [Simulation Functions](#simulation-functions)
- [Credits](#credits)
- [License](#license)

## Installation

No installation steps yet. Still a work in progress here.

## Usage 

The CadetCareerProblem class is the main class object that the user will be controlling. There are 4 main ways of initializing an instance of this problem. The first, and main way of initializing the problem, is to import a pre-existing problem instance from excel. The other three initialization methods pertain to generating fake problem instances. I have not taken the time to write a documentation file yet with all the methods of the class, but you can view them within the problem_class.py script. The methods pertain to different ways of specifying value parameters, finding solutions, measuring/comparing solutions, and more. 

### Globals
This script (globals.py) contains all the user-specific information. Please input your own specific directory path. It is very critical that all the subfolders are in the correct place and that your main directory's path is placed into the "dir_path" variable! I also included a variable called "specify_engine" because I know that some versions of pandas have to add the parameter "engine='openpyxl'" when importing pandas dataframes. By default, we include the parameter but that can be changed if the user does not have the openpyxl module installed.
    
### Data Handling
This script (data_handling.py) processes the data used in the model building. It contains the functions to import excel sheets, convert them to a dictionary of "fixed" parameters (AFSC/cadet data that cannot be changed) and process that data through different means. This sheet also contains the "measuring" functions that evaluate solutions given the fixed parameters and the "value" parameters. The value parameters, as I refer to them in the project, are the weights and value functions used in the model. Value parameters are contained in a dictionary as well. We can also export results to excel, and the function to do that is contained in this script as well. 
    
### Value Parameter Handling
This script (value_parameter_handling.py) processes the "value" parameters to the model. These are the weights and value functions used in the VFT value hierarchy. Similar to the "data_handling.py" script, this script contains the functions that load in pandas dataframes from excel and convert them to a dictionary of parameters. This script also contains functions for generating default value parameters, as well as exporting a particular instance's value parameters as defaults.
    
### Heuristic Solvers
This script (heuristic_solvers.py) contains the functions that perform heuristic techniques. There is a stable marriage heuristic and a greedy heuristic that are meant to provide additional initial solutions to the genetic algorithm (the main meta-heuristic). The genetic algorithm has many hyper-parameters that could be tuned further.
    
### Pyomo Models
This script (pyomo_models.py) contains the functions that utilize the optimization package "pyomo" (therefore, the Pyomo library is a key dependency). These are the "original model" functions, the sensitivity analysis functions, and the main new VFT model functions. 
    
### Simulation Functions
This script (simulation_functions.py) contains the functions that pertain to simulating cadet and AFSC "fixed" parameters. We can generate random sets of cadets and AFSCs with any number of each (useful for testing optimization models with different solvers), adn we can generate "perfect" sets of cadets and AFSCs (sets where the best solution meets all objectives perfectly- could be useful for illustrating the objective value and what it means- "percent of a perfect solution obtained"). Additionally, we can generate realistic sets of cadets and AFSCs using CTGAN. These functions include all of the CTGAN related procedures (the SDV package is necessary).
    
To add a screenshot, create an `assets/images` folder in your repository and upload your screenshot to it. Then, using the relative filepath, add it to your README using the following syntax:

    ```md
    ![alt text](assets/images/af_logo.jpeg)
    ```
![alt text](assets/images/af_logo.jpeg)
## Credits

List your collaborators, if any, with links to their GitHub profiles.

If you used any third-party assets that require attribution, list the creators with links to their primary web presence in this section.

If you followed tutorials, include links to those here as well.

## License

The last section of a high-quality README file is the license. This lets other developers know what they can and cannot do with your project. If you need help choosing a license, refer to [https://choosealicense.com/](https://choosealicense.com/).

---

🏆 The previous sections are the bare minimum, and your project will ultimately determine the content of this document. You might also want to consider adding the following sections.

## Badges

![badmath](https://img.shields.io/github/languages/top/lernantino/badmath)

Badges aren't necessary, per se, but they demonstrate street cred. Badges let other developers know that you know what you're doing. Check out the badges hosted by [shields.io](https://shields.io/). You may not understand what they all represent now, but you will in time.

## Features

If your project has a lot of features, list them here.

## How to Contribute

If you created an application or package and would like other developers to contribute it, you can include guidelines for how to do so. The [Contributor Covenant](https://www.contributor-covenant.org/) is an industry standard, but you can always write your own if you'd prefer.

## Tests

Go the extra mile and write tests for your application. Then provide examples on how to run them here.
