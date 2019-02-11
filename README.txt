### CS 7641 Assignment 1 - Supervised Learning

Code and data can be found at: https://github.com/isw4/ML01-Supervised_Learning

## Directories
src/        Contains the source code
data/       Contains the raw and cleaned csv data files, along with some description of each set
graphs/     Contains graphs that are output by the various experiments


## Setup

1)  Make sure to have Conda installed

2)  Install the conda environment:
    ~~~
    conda env create -f environment.yml
    ~~~

3)  Activate the environment:
    If using Windows, open the Anaconda prompt and enter:
    ~~~
    activate ml_hw1
    ~~~

    If using Mac or Linux, open the terminal and enter:
    ~~~
    source activate ml_hw1
    ~~~


## Running the datasets
The 'save' argument at the end of some calls is optional. With the 'save' argument, it will save
the graphs to '../graphs/wine/' or '../graphs/pet/'.

-   Cleaning the raw data and outputting it to a csv file:
    ~~~
    python wine.py clean_and_output
    python pet.py clean_and_output
    ~~~

-   Getting some graphs and descriptive stats of the data:
    ~~~
    python wine.py explore_features [save]
    python pet.py explore_features [save]
    ~~~

-   Tuning hyperparameters and getting some graphs:
    ~~~
    python wine.py clean_and_output [save]
    python pet.py clean_and_output [save]
    ~~~

-   Finding the learning rate of the models and getting more graphs:
    ~~~
    python wine.py clean_and_output [save]
    python pet.py clean_and_output [save]
    ~~~