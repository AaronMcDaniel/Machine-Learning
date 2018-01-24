For this assignment I Picked 2 binary classification problems, HTRU2 and DOTA2

HTRU2 is designed to train machine learning algorithms to identify Pulsars, a rare Neutron star. They rotate very fast and are objects of interest to most astronomers. Each instance in this dataset can either be a Pulsar(1) or not(0), with 95% of instances not being pulsars and 5% being Pulsars. Like I said, they are rare.
More information on the dataset can be found at this link:
https://archive.ics.uci.edu/ml/datasets/HTRU2


DOTA2 is a dataset of match information form the game called Dota2. The information known is that which is avaiable before the match starts (i.e. characters and game mode). Each game can be won by the first team(1) or the second team(-1) with about 53% being positive matches and 47% being negative matches. The original dataset includes server information, but I removed this because it seemed irrelevent.
More information on the dataset can be found at this link:
https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results

The data files can be found in the "data" folder as .csv files.

TO RUN PROGRAMS:
This requires:
python 2.7 (don't think python 3 works, sorry)
scikit library
NumPy >= 1.8
SciPy >= 0.13.3

scikit can be aquired using the following command in linux:
pip install -U scikit-learn

There are 10 .py files, each one corresponding to a comination of machine learning algorithm to dataset. Their names describe them pretty well, but the naming convention is: "Algorithm_Dataset_TrainNTest.py"
Simply run the scripts to train and test each algorithm on a specific dataset.

They all have different variables that can be changed to customize the run, but the most important thing is that each file has a bool called "multipleTests" where if it is true it will run multiple times, generating the data used to make the graphs. If it is false it will only run once. In both cases files will go into the corresonding algorithms folder for further inspection. If the file is labeled "analysis" then it is from a multipl run, if it is labeled "results" then it is from a single run.

Just a heads up, the multiple runs can last a very long time. They can take from 30 mins to 17 hours, so just try to do a single run first to see what you are getting into. Sorry about that :/
The log messages on the terminal are very helpful to judge the progress of the program though.

AWKNOLEDGEMENTS:
I originally stole the data extraction part from this repo by fucusy:
https://github.com/fucusy/ml-decision-tree
But much of it has diverged form his code. Still a big thanks to fucusy for pointing me in the right direction.
