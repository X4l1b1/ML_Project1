# Higgs Boson detection using various regression techniques

This project aims to present various regression techniques to detect Higgs Boson particles on a dataset provided by the CERN.

## Resources

* You can download the test and training dataset on [Aicrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019)
* You can find a description of each features on the [CERN](http://opendata.cern.ch/record/328) website

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

In order to run the project, you need to have the following dependencies on your computer:

```
Python v3.5 or higher
Numpy v1.17 or higher

```

### Install

* Simply clone or download the current project from github
* Download the test and train datasets [here](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019)
* Put the downloaded datasets in the `data` folder

### Run our best model

Once the dataset is downloaded simply go to the `script` folder with a terminal and type:
```
python run.py
```

## Important Files

- The `report.pdf`(report.pdf) file contains our final report on this project, all the steps from the exploratory analysis to the final model are described.
- The `project1_description.pdf`(project1_description.pdf) contains the full description of the project
- The `scripts`(scripts) folder contains all our code:
  - `run.py`(scripts/run.py) : Script that produces exactly the same .csv predictions for our best submission to the competition system.
  - `implementations.py`(scripts/implementations.py) : Implementation of our six baseline regression methods.
  - `tools.py`(scripts/tools.py) : Toolbox with everything we need for features engineering and Hyper-Parameters' Tuning
  - `plot.py`(scripts/plot.py) : Some visualization functions to run in jupyter notebook
  - `project1.ipynb`(scripts/project1.ipynb) : a jupyter notebook containing some of our exploratory data analysis and tests we made in our project.

## Results

Our best model did **83.1%** of correct predictions on the test dataset.


## Authors

* **Arthur Passuello** - [X4l1b1](https://github.com/X4l1b1)
* **François Quellec** - [Fanfou02](https://github.com/Fanfou02)
* **Julien Muster** - [Jmuster](https://github.com/Jmuster)

## License

This project is licensed under the MIT License
