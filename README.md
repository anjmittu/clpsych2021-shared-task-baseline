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

1. First create a new conda environment.
```
conda create -n shared_task_baseline python
```

2. Activate the environment
```
source activate shared_task_baseline
```

3. Install the needed libraries
```
pip install -r requirements.txt
python -m nltk.downloader words
```

4. Set the environment variable (The `set_env.sh` should be updated to point to the correct location)
```
source set_env.sh
```

### Running
All of the script are run with python

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

Below is an example calling this script when the data resides in a folder `practice_data`:
```
python risk_model/tokenize_data.py --input practice_data --output practice_data
```

#### baseline_model.py
This script will create the baseline model from the data and will output a results file.  The
results file is a tsv file with the following form: `<user id>\t<predicted label>\t<pos probability>`.
The baseline model is a bag of words model.  It uses count vectors with unigrams and bigrams and 
Logistic Regression for classification. 

Below is an example calling this script when the data resides in a folder `practice_data`:
```
python risk_model/baseline_model.py --input practice_data --output practice_data
```

#### evaluation.py
This script will read the results created by the baseline_model.py and output a score.  The 
script will output `<f1 score>, <true positive rate>, <false alarm rate>, <auc>`.

Below is an example calling this script when the data resides in a folder `practice_data`:
```
python risk_model/evaluation.py --results practice_data/results.tsv --truth practice_data/test_truths.jsonl
```