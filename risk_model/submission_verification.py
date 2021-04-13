import argparse
import csv
from sklearn.metrics import f1_score, fbeta_score, confusion_matrix, roc_curve, auc

"""
This will verification that the submission file is formatted correctly
"""


def main(results_file_path):
    problem = False
    predictions = []
    probs = []

    with open(results_file_path) as rf:
        results = csv.reader(rf, delimiter="\t")
        for pred in results:
            if pred[1] != "0" and pred[1] != "1":
                problem = True
                print("{}: prediction should be either 0 or 1; prediction given {}".format(pred[0], pred[1]))
            if float(pred[2]) < 0 or float(pred[2]) > 1:
                problem = True
                print("{}: probability should be between 0 and 1; probability given {}".format(pred[0], pred[2]))
            predictions.append(pred[1])
            probs.append(float(pred[2]))

    truths = ["0"] * len(predictions)

    if not problem:
        f1 = f1_score(truths, predictions, average='binary', pos_label="1")
        f2 = fbeta_score(truths, predictions, beta=2, average='binary', pos_label="1")

        tn, fp, fn, tp = confusion_matrix(truths, predictions).ravel()
        
        if tp == 0 and fn == 0:
            print("There are no problems with the submission")
        else:
            print("There was a problem with scoring")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verifies the submission file is formatted correctly.')
    parser.add_argument('--results', help='path to the results file')

    args = parser.parse_args()
    main(args.results)
