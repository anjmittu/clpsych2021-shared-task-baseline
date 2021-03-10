# Suicide Risk Model
This project contains the code for a suicide risk model based on a person's tweets.

## Overview
### Directory Structure
```
- risk_model
    - baseline_model.py         :- Runs the baseline model and creates a results output
    - evaluation.py             :- Evaluations the results from the model
    - print_data_statistics.py  :- Prints some stats about the data
    - tokenize_data.py          :- Preforms the preprocessing on the data.
```

## How to run
### Set up
These steps should only be run once.

1. Copy this directory to your home directory
```
$ cp -r ~/teamdata/clpsych2021-shared-task-baseline ~/clpsych2021-shared-task-baseline
```

2. Copy the nltk data directory to your home directory
```
$ cp -r ~/teamdata/nltk_data ~/nltk_data
```

3. Move into the copied directory
```
cd ~/clpsych2021-shared-task-baseline
```

4. Python 3.6+ is required. Install the needed libraries
```
module load python3.9.1
pip install -r requirements.txt --user
python -m nltk.downloader words
```

### Getting the Data
The data can be found in the ~/teamdata/ directory.  This directory is read only, so you can read the original data 
from here, but any tokenized data or results should be save to a directory in your shared team folder or home directory.
There are three datasets: the practice dataset, the OurDataHelps.org data with tweets 6 months (182 days) before a user's 
attempt, and the OurDataHelps.org data with tweets 1 month (30 days) before a user's attempt.  More information about
the data can be found in the shared task description.

The practice data set includes test data, however, the OurDataHelps.org data does not.  In order to run these scripts
with the ODH, you should create your own validation set.  The validation set can be created after tokenizing the data.
For example after running `tokenize_data.py` with the odh_shared_task_data_30, you can run the code below to create the 
validation set.

```
import json
import math
import os

train_file = os.path.expanduser("~/odh_shared_task_data_30_results/train_tokenized.jsonl")
test_file = os.path.expanduser("~/odh_shared_task_data_30_results/test_tokenized.jsonl")

train_size = .85
users = []
with open(train_file, "r") as f:
    for json_obj in f:
        users.append(json.loads(json_obj))

amount_of_users = len(users)
val_start_ind = math.floor(train_size*amount_of_users)

if os.path.exists(train_file):
    os.remove(train_file)
if os.path.exists(test_file):
    os.remove(test_file)

with open(train_file, "w") as f:
    for u in users[:val_start_ind]:
        json.dump(u, f)
        f.write("\n")
with open(test_file, "w") as f:
    for u in users[val_start_ind:]:
        json.dump(u, f)
        f.write("\n")
```

### Using Slurm
All of the baseline scripts should be run using slurm and not run using python on the head node.
There are example slurm scripts for each of the python scripts.  See below for more information.  Before running the
slurm scripts, there is one instances of <HOME_DIR> in each script which need to be replaced with your home directory.
To find the path of your home directory run:
```
$cd ~
$pwd
```
Copy the output from this commnad, open the slurm script and replace "<HOME_DIR>" with the output.

You should also be sure that all directories which the slurm script references already exist (log directories, output
directories).

### Running
All of the script are run with python and should be run in the order listed below.

#### tokenize_data.py
This script will preprocess the data.  The following steps are done:
- URLs are removed from the tweets and tweets are made to be lowercase
- The tweets are tokenized using twikenizer
- User mentions and emojis are removed from the tweets
- Hashtags are split into separate words using three methods.  The first to work is used:
    - Split by camel case
    - Split on underscores
    - Smallest split into real words
- Stop words are removed

##### Slurm examples
See the slurm scripts for examples of running 'tokenize_data.py`. For example, with the practice dataset, the slurm 
script is configured to read the input data from `$HOME/teamdata/practice_data` and write to 
`$HOME/practice_data_results`.  In the input directory, the script expects there to be a file named `train.jsonl` 
and an optional file named `test.jsonl`.  The script will create the files`train_tokenized.jsonl` and a file named 
`test_tokenized.jsonl` (if a test file is avalible in the input) in the output directory. 
There are three slurm scripts associated with this script:
- `tokenize_data_practice.slurm`
- `tokenize_data_182.slurm`
- `tokenize_data_30.slurm`

These can be run like, where ## is replaced with your team number:
```
$ sbatch -comment team## tokenize_data_practice.slurm
```

##### Expected output
```
Tokenizing training data
Tokenizing test data
Done
```

#### baseline_model.py
This script will create the baseline model from the data and will output a results file.  The
results file is a tsv file with the following form: 
```
<USER_ID> \t <LABEL> \t <SCORE>
```
Where `USER_ID` is the ID field from the source file, `LABEL` is either "1" for suicide 
or "0" for control, and `SCORE` is a real-valued score output score from your system, where
larger numbers indicate the `SUICIDE` class and lower numbers indicate the `CONTROL` class.

The baseline model is a bag of words model.  It uses count vectors with unigrams and bigrams and 
Logistic Regression for classification. 

##### Slurm examples
See the slurm scripts for examples of running 'baseline_model.py`.   For example, with the practice dataset, the 
slurm script is configured to read the input data from `$HOME/practice_data_results` and write to 
`$HOME/practice_data_results`.  In the input directory, the script expects there to be a file named 
`train_tokenized.jsonl` and one named `test_tokenized.jsonl`. The script will create the files 
`results.tsv` in the output directory.
There are three slurm scripts associated with this script:
- `run_baseline_practice.slurm`
- `run_baseline_182.slurm`
- `run_baseline_30.slurm`

These can be run like, where ## is replaced with your team number:
```
$ sbatch -comment team## run_baseline_practice.slurm
```


##### Expected output
No output

#### evaluation.py
This script will read the results created by the baseline_model.py and output a score.  The 
script will output `<F1>, <F2>, <True Positive Rate>, <False Alarm Rate>, <AUC>`.

##### Slurm examples
See the slurm scripts for examples of running 'evaluation.py`.  This script takes two files as input. For the practice
dataset the files are: a results file located here: `$HOME/practice_data_results/results.tsv` 
and a file with the test truths here: `$HOME/teamdata/practice_data/test.jsonl`.  The output is sent to stdout.
There are three slurm scripts associated with this script:
- `evaluate_practice.slurm`
- `evaluate_182.slurm`
- `evaluate_30.slurm`

These can be run like, where ## is replaced with your team number:
```
$ sbatch -comment team## evaluate_practice.slurm
```

##### Expected output
The expected output for the practice dataset is:
```
0.689655172413793, 0.6369426751592356, 0.6060606060606061, 0.15151515151515152, 0.7704315886134068
```
