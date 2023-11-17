# Machine Learning versus Physicians

This repository contains the code and resources for our project titled: "Machine Learning versus Physicians". The project compares machine learning models and clinical evaluation for mortality prediction in patients with sepsis.

This work is based on the paper:

"A comparison of machine learning models versus clinical evaluation for mortality prediction in patients with sepsis"  
Published in PLoS One. 2021; 16(1): e0245157.  
[DOI: 10.1371/journal.pone.0245157](https://doi.org/10.1371/journal.pone.0245157)

## Repository Structure

**algorithms/**  
Contains various versions of the algorithm and the associated results. 

**data/**  
Holds all the data files used in the project, including laboratory data from the emergency department, clinical and baseline data from the ED sepsis study, and data from the questionnaires from the internists.

**docs/**  
Includes all the necessary documentation for this project, as well as documentation from external packages and/or software systems. 

**figures/**  
Stores all the figures produced by the different experiments conducted in our study.

**models/**  
Contains the specific files with pre-processed data that serve as "data models" (i.e., not as an algorithm, but e.g., the lab or lab + clinical datasets). _TODO: consider renaming this to 'datasets'_

**notebooks/**  
Holds all Jupyter notebooks used during the project. These were mainly involved during code design and simple experiments. 

**src/**  
Contains all the source code from the project.