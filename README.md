# An Application of Deep Reinforcement Learning to Algorithmic Trading

This codebase builds upon the original implementation available at https://github.com/ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading.

It supports and extends the experimental framework presented in the following research paper:

> Thibaut Théate and Damien Ernst.  
> *An Application of Deep Reinforcement Learning to Algorithmic Trading* (2020).  
> https://arxiv.org/abs/2004.06627

As explained in the report, the modified, improved, and fixed files are: `TDQN.py`, `dataDownloader.py`, `main.py`, `tradingEnv.py`, and `tradingPerformance.py`.

Additionally, the file `TActorCritic.py` implements a new algorithm (reusing the `TDQN.py` as template) that was not included in the original report.


## Dependencies

The dependencies are listed in the text file "requirements.txt":

## Usage

To run the code, execute the `main.py` file.

### Selecting an Algorithm 
To run a different algorithm, modify line 29 in `main.py` by changing the value of `default='TActorCritic'` to the algorithm you want to run. The two available options are:
- `'TDQN'`
- `'TActorCritic'`

### Changing Company or Index
To run the model on a different stock or index, change the value of `default='S&P 500'` to the name of the desired stock or index (e.g., `'Apple'`).

- Note: The full list of available options can be found in the dictionaries: `companies`, `indices`, and `stocks` in the top of the file `tradingSimulator.py`.

### Changing Dates
If you want to change the date range, modify the values in the `dates.py` file.
