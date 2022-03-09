<div align="center">
<br/>

# Air Force Cadet Career Problem

![alt text](assets/images/af_logo.png)

</div>

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

This is not a python package, so this repository will need to be cloned. Create a virtual environment from the requirements.txt file, and then you should be good to go! The only other hurdle to jump through could be the solvers used by pyomo. There is a folder called "solvers" in this repository that has a couple of them, but you should also have them once you install the pyomo package as well. 

## Usage 

The CadetCareerProblem class is the main class object that the user will be controlling. There are 4 main ways of initializing an instance of this problem. The first, and main way of initializing the problem, is to import a pre-existing problem instance from excel. The other three initialization methods pertain to generating fake problem instances. I have not taken the time to write a documentation file yet with all the methods of the class, but you can view them within the problem_class.py script. The methods pertain to different ways of specifying value parameters, finding solutions, measuring/comparing solutions, and more. 

### Problem Class
This script contains the problem class itself: CadetCareerProblem. An object of this class is an instance of the problem. This class contains various methods that pull from several other scripts to do certain things. Importing data, exporting data, generating data, and so on are necessary features that are found within those other scripts. The weight and value parameters used by the VFT model are needed for the instance to be solved. There are different methods to solve these instances

### Data Handling
This script (data_handling.py) processes the data used in the model building. It contains the functions to import excel sheets, convert them to a dictionary of "fixed" parameters (AFSC/cadet data that cannot be changed) and process that data through different means. This sheet also contains the "measuring" functions that evaluate solutions given the fixed parameters and the "value" parameters. The value parameters, as I refer to them in the project, are the weights and value functions used in the model. Value parameters are contained in a dictionary as well. We can also export results to excel, and the function to do that is contained in this script as well. 
    
### Value Parameter Handling
This script (value_parameter_handling.py) processes the "value" parameters to the model. These are the weights and value functions used in the VFT value hierarchy. Similar to the "data_handling.py" script, this script contains the functions that load in pandas dataframes from excel and convert them to a dictionary of parameters. This script also contains functions for generating default value parameters, as well as exporting a particular instance's value parameters as defaults.
    
### Heuristic Solvers
This script (heuristic_solvers.py) contains the functions that perform heuristic techniques. There is a stable marriage heuristic and a greedy heuristic that are meant to provide additional initial solutions to the genetic algorithm (the main meta-heuristic). The genetic algorithm has many hyper-parameters that could be tuned further.
    
### Pyomo Models
This script (pyomo_models.py) contains the functions that utilize the optimization package "pyomo" (therefore, the Pyomo library is a key dependency). These are the "original model" functions, the sensitivity analysis functions, and the main new VFT model functions. 
    
### Simulation Functions
This script (simulation_functions.py) contains the functions that pertain to simulating cadet and AFSC "fixed" parameters. We can generate random sets of cadets and AFSCs with any number of each (useful for testing optimization models with different solvers), and we can generate "perfect" sets of cadets and AFSCs (sets where the best solution meets all objectives perfectly- could be useful for illustrating the objective value and what it means- "percent of a perfect solution obtained"). Additionally, we can generate realistic sets of cadets and AFSCs using CTGAN. These functions include all of the CTGAN related procedures (the SDV package is necessary).

## Credits

This project would not have been possible without the various python libraries used. In addition to the widely used libraries: pandas, numpy, matplotlib, etc. I will also mention a couple less popular ones that have been very useful to this project. The SDV library allowed me to generate realistic cadet/AFSC problem instances using their conditional-tabular generative adversarial network (CTGAN) module. Traditionally used for generating data for a machine learning model, CTGAN was able to generate problem instances for my optimization model to solve: which was a very valuable tool. I formulated the optimization model using the library Pyomo, another great python package. Pyomo is an easy-to-use optimization library that can solve many different kinds of optimization problems using a variety of commercial and open-source solvers. 

SDV: [https://github.com/sdv-dev/SDV](https://github.com/sdv-dev/SDV)
CTGAN: [https://github.com/sdv-dev/CTGAN](https://github.com/sdv-dev/CTGAN)
Pyomo: [https://github.com/Pyomo/pyomo](https://github.com/Pyomo/pyomo)

## License

The last section of a high-quality README file is the license. This lets other developers know what they can and cannot do with your project. If you need help choosing a license, refer to [https://choosealicense.com/](https://choosealicense.com/).

