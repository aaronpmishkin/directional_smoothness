# Optimal Sets and Solution Paths of ReLU Networks 

Code to replicate experiments in the paper 
[Directional Smoothness and Gradient Methods: Convergence and Adaptivity](https://openreview.net/forum?id=m9WZrEXWl5) by Aaron 
Mishkin*, Ahmed Khaled*, Yuanhao Wang, Aaron Defazio, and Robert M. Gower.

*Equation Contribution

### Requirements

Python 3.8 or newer.

### Setup

Clone the repository using

```
git clone https://github.com/aaronpmishkin/directional_smoothness.git
```

We provide a script for easy setup on Unix systems. Run the `setup.sh` file with

```
./setup.sh
```

This will:

1. Create a virtual environment in `.venv` and install the project dependencies.
2. Install our modified version of [`stepback`](https://github.com/fabian-sp/step-back) in development mode. 
This library contains infrastructure for running our experiments.
3. Create the `data`, `figures`, `tables`, and `results`  directories.

After running `setup.sh`, you need to activate the virtualenv using

```
source .venv/bin/activate
```

### Replications

The experiments are run via a command-line interface.
First, make sure that the virtual environment is active.
Running `which python` in bash will show you where the active Python binaries are; 
this will point to a file in `directional_smoothness/.venv/bin` if the virtual 
environment is active.
Then, execute one of the files in the `scripts/` directory. 
Before running the experiments, you must compute the optimal values and 
smoothness constants for each of the datasets with which we experiment. 
You can do this by running,
```
python scripts/compute_optimal_value_smoothness.py
```
Then, run, 
```
python scripts/run.py -E uci_logreg
```
to run the logistic regression experiments.
Finally, execute,
```
python scripts/distance_to_opt.py
```
to compute the initial distances to the global optimum for each optimizer.
This is necessary to plot the theoretical bounds in Figure 1.
Now you can generate the plots by running,
```
python scripts/make_figure_1.py
python scripts/make_figure_2.py
```

Note that the UCI datasets must be manually retrieved from 
[here](http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz).


### Citation

Please cite our paper if you make use of our code or figures from our paper. 

```
@article{mishkin2024directional,
  author       = {Aaron Mishkin and
                  Ahmed Khaled and
                  Yuanhao Wang and
                  Aaron Defazio and
                  Robert M. Gower},
  title        = {Directional Smoothness and Gradient Methods: Convergence and Adaptivity},
  journal      = {CoRR},
  volume       = {abs/2403.04081},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2403.04081},
  doi          = {10.48550/ARXIV.2403.04081},
  eprinttype    = {arXiv},
  eprint       = {2403.04081},
  timestamp    = {Thu, 11 Apr 2024 16:45:45 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2403-04081.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

Looking for the poster for this paper?
See [directional smoothness poster](https://github.com/aaronpmishkin/directional_smoothness_poster).

### Bugs or Other Issues

Please open an issue if you experience any bugs or have trouble replicating the experiments.

