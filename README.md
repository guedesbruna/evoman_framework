# Evolutionary Computing 
This repository contains the implementation of Evolutionary Algorithms for the task of video game playing using a python framework called EvoMan.
Each task has two EA methods, however the first task uses Individual Evolution mode with independent enemies (games), and the second uses groups of enemies per game to develop a generalist agent.


## Task 1: Specialist Agent

This code contains different functions for training two instances of GA10 (one with 1-step mutation and another one with n-step mutation) on 8 possible different enemies. 

- final_EA1.py, function that runs EA1 for a specified enemy and saves results of the runs and best individuals based on individual gain
- final_EA2.py, function that runs EA2 for a specified enemy and saves results of the runs and best individuals based on individual gain
- best_individual.py, function that runs the best individual of each run for a specific enemy, given enemy number and EA number
- plotting_final.py, function that plots line and boxplots to compare the 2 EAs when passed an enemy as a parameter, also calculates T-Tests or Kolmogorov-Smirnov Tests after checking for normality 

## Task 2: Generalist Agent

- T2_EA1.py, function that runs EA1 for a specified enemy and saves results of the runs and best individuals based on individual gain
- T2_EA2.py, function that runs EA2 for a specified enemy and saves results of the runs and best individuals based on individual gain
- T2_best_individual.py, function that runs the best individual of each run for a specific enemy, given enemy number and EA number
- T2_plotting.py, function that plots line and boxplots to compare the 2 EAs when passed an enemy as a parameter, also calculates T-Tests or Kolmogorov-Smirnov Tests after checking for normality 


## Disclaimer
The base controllers and functions, together with some parts of the code, are based on/inspired by the work of Karine Miras et al. (2016).

