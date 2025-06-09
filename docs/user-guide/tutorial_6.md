# Tutorial 6: Solutions Overview

As you've seen, `afccp` requires a substantial amount of data in order to work. All the various parameters, v
alue parameters, hyper-parameters, and other "controls" are necessary to determine the best solution assignment of 
cadets to AFSCs. This is the whole point of afccp: obtain the right solution! This section of the tutorial contains 
the information needed for you to calculate different solutions through various algorithms and models.

## `solutions` Module

The `solutions` module of afccp.core contains the scripts and functions that handle the determination of a solution 
assignment of cadets to AFSCs. Each script tackles solutions in a different way. At a high-level, the module is set up 
as follows: