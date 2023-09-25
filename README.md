<div align="center">
<br/>

# Air Force Cadet Career Problem

![alt text](assets/images/af_logo.png)

</div>

## Description

I created this project for the completion of my master's thesis and continue to contribute to it for my current role as a Force Development Analyst at the Air Force Personnel Center. My thesis, titled "A New Approach to Career Field Matching for Air Force Cadets", developed a new methodology for assigning commissioning cadets from the United States Air Force Academy (USAFA) and from Reserve Officers' Training Corps (ROTC) detachments to their career fields. Career fields are alpha-numerically designated by Air Force Specialty Codes (AFSCs) and are referred to in this project as such. I developed an optimization model using a Value-Focused Thinking (VFT) framework to assign cadets to their AFSCs, implemented using the library "pyomo." The VFT model allows for solutions to be compared based on their overall quality, which is the weighted sum of all AFSC/cadet objective values. By specifying the weights and values of the decision maker (what they want and how much they want it) we can effectively generate solutions that meet their objective criteria. This project incorporates several Operations Research techniques that all work together to solve this assignment problem efficiently. In addition to the VFT model, for which this module was designed for, 'afccp' also includes several other models that may be applied to solve this problem. These include a couple of generalised assignment problem models, matching algorithms, meta-heuristics, and a goal programming linear program.

## Table of Contents 

- [Installation](#installation)
- [Usage](#usage)
    - [Main](#main)
    - [Data](#data)
        - [Parameters](#parameters)
        - [Value Parameters](#value-parameters)
    - [Solutions](#solutions)
        - [Algorithms](#algorithms)
        - [Optimization](#optimization)
        - [Handling](#handling)
    - [Visualizations](#visualizations)
- [Credits](#credits)
- [License](#license)

## Installation

While "afccp" certainly can be installed from github using setup.py, it is recommended that this project be cloned since it is a work in progress. Create a virtual environment from the requirements.txt file, and then you should be good to go! The only other hurdle to jump through could be the solvers used by pyomo. The "solvers" folder will be created once you import the module, and you can manually add the solver executable files (baron, bonmin, couenne, cbc, etc.) there. There are other methods of getting solvers, but the method that has worked for me has been downloading the zip folder of coin-or solvers from the ampl website here: [https://portal.ampl.com/user/ampl/download/list](https://portal.ampl.com/user/ampl/download/list). Here you can download the compressed folder of several coin-or solvers by selecting the appropriate "coin" option based on your operating system. Once finished, drag those executables into your solvers folder in your working directory with afccp.

## Usage 

The CadetCareerProblem class is the main object that the user will be controlling. There are two main ways of initializing an instance of this class. The first, and main way of initializing the problem, is to import a pre-existing problem instance. The other way is to generate fake data (which you can export and then import later as well). I have not taken the time to write a documentation file yet with all the methods of the class, but you can view them within the main.py script. There is, however, a tutorial available in the "examples" folder titled "afccp_tutorial.ipynb". It is a jupyter notebook that presents many of the various methods available to the CadetCareerProblem class.

### Main
This script contains the problem class itself: CadetCareerProblem. An object of this class is an instance of the AFSC matching problem. This class contains various methods that pull from several other scripts to do many things. Importing data, exporting data, generating data, and so on are necessary features that are found within those other scripts. The weight and value parameters used by the VFT model (and several other optimization models) are needed for the instance to be solved as well. For now, the best way of seeing what you have available is to look at "main.py" or view the example jupyter notebook included in this project.

### Data
The "data" module includes several .py scripts that all handle the various data components of the problem. These are all the parameters that are used to solve the various models. I have two main dictionaries of parameters that are both attributes of CadetCareerProblem: "parameters" and "value_parameters". 

#### Parameters
The first, "parameters", includes all of the components of the problem that the analyst has no control over. These are the actual cadets and AFSCs that we're solving for as well as all of their various components. For cadets, these include source of commissioning, merit, preferences, degree qualifications, and so on. Similarly for AFSCs, we have their names, degree tier information, quotas, etc. I import all of these parameters in "processing.py" using pandas and quickly convert them into numpy arrays that live in the parameters dictionary. 

#### Value Parameters
The "value_parameters" are handled in a very similar manner in "values.py". In contrast to the "parameters" dictionary, these are all of the parameters that the analyst does have control over. These are the constraints, weights, value functions, etc. all used to solve some of the optimization models (the main focus being the VFT model). 

### Solutions
The "solutions" modules includes a few .py scripts that each do very specific things involving finding and evaluating solution assignments of cadets to their AFSCs.

#### Algorithms
The "algorithms.py" script contains the matching algorithms and genetic algorithms used for this problem. So far, "matching_algorithm_1" is the only matching algorithm that we have and it is the simple hospital & residents algorithm applied to this problem where cadets are akin to residents and likewise with AFSCs/hospitals. This is largely inspired by the work of Capt. Ian MacDonald for his master's thesis. There will soon be more of his algorithms here as well. There are two genetic algorithms: one used for the VFT model and the other recently developed to find the optimal capacities for a deferred acceptance algorithm to solve to reduce blocking pairs as much as possible while meeting minimum quotas (PGL targets). This GA was Ian MacDonald's idea that I (Griffen Laird) coded up and included here!

#### Optimization
The "optimization.py" script contains the various optimization models used for "afccp". There is the "assignment problem" model that contains both the original problem formulation as well as a newer version using a new "global utility" matrix based on the combination of cadet utility (given by the cadets) and AFSC utility, normalized preferences provided by the career field managers (CFMs). These CFM preferences are new parameters to this problem that have been developed in 2023 by the Office of Labor and Economic Analysis (OLEA) at the Air Force Academy in Colorado Springs. There is, of course, also the VFT model on this script. A goal programming model, developed by *former* Lt. Rebecca Reynolds (*now* Capt. Rebecca Eisemann), is included here as well. In order to call this module there is a method of CadetCareerProblem to translate my value_parameters into her parameters ("gp_parameters") as depicted in her thesis.

#### Handling
The "handling.py" script holds many functions all meant to evaluate or process the various solutions to this problem. Calculating metrics is the main idea whether that be the VFT objective function and/or fitness function, the number of blocking pairs, AFSC normalized scores, etc. 

### Visualizations
The "visualizations" module contains many scripts that are meant to depict the various components of this problem in various ways. There are many flavors of AFSC charts in "charts.py" alongside other kinds of figures all build using matplotlib. There are also a few scripts meant for creating dashboards that largely haven't been explored further since they were created originally many months ago! My personal pride and joy here is in "animation.py". The "CadetBoardFigure" object creates the matching algorithm & solution depiction animation. This was a fun project where I depict cadets as dots on large "board"-like graph. This animation is meant to show the different matching algorithms that are included here and compare their results to other methods.

## Credits

This project would not have been possible without the various python libraries used. In addition to the widely used libraries: pandas, numpy, matplotlib, etc. I will also mention a couple less popular ones that have been very useful to this project. The SDV library allowed me to generate realistic cadet/AFSC problem instances using their conditional-tabular generative adversarial network (CTGAN) module. Traditionally used for generating data for a machine learning model, CTGAN was able to generate problem instances for my optimization model to solve: which was a very valuable tool. I formulated the optimization model using the library Pyomo, another great python package. Pyomo is an easy-to-use optimization library that can solve many different kinds of optimization problems using a variety of commercial and open-source solvers. 

SDV: [https://github.com/sdv-dev/SDV](https://github.com/sdv-dev/SDV)

CTGAN: [https://github.com/sdv-dev/CTGAN](https://github.com/sdv-dev/CTGAN)

Pyomo: [https://github.com/Pyomo/pyomo](https://github.com/Pyomo/pyomo)

