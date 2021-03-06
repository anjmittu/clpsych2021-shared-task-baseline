import argparse
import csv
from sklearn.metrics import f1_score, fbeta_score, confusion_matrix, roc_curve, auc

"""
This will verify that the submission file is formatted correctly
"""


def main(results_file_path):
    problem = False
    predictions = []
    probs = []

    with open(results_file_path) as rf:
        results = csv.reader(rf, delimiter="\t")
        for pred in results:
            try:
                pred[2] = float(pred[2])
                if pred[1] != "0" and pred[1] != "1":
                    problem = True
                    print("{}: prediction should be either 0 or 1; prediction given {}".format(pred[0], pred[1]))
                predictions.append(pred[1])
                probs.append(pred[2])
            except ValueError:
                problem = True
                print("{}: probability can not be made a float: {}".format(pred[0], pred[2]))
            except IndexError:
                problem = True
                print("{}: Missing values in result row: {}".format(pred[0], pred))

    truths = ["0"] * len(predictions)

    if not problem:
        f1 = f1_score(truths, predictions, average='binary', pos_label="1")
        f2 = fbeta_score(truths, predictions, beta=2, average='binary', pos_label="1")

        tn, fp, fn, tp = confusion_matrix(truths, predictions).ravel()
        
        fpr, tpr, thresholds = roc_curve(truths, probs, pos_label="1")
        auc_score = auc(fpr, tpr)
        
        if tp == 0 and fn == 0:
            print("There are no problems with the submission")
        else:
            print("There was a problem with scoring")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verifies the submission file is formatted correctly')
    parser.add_argument('--results', help='path to the results file')

    args = parser.parse_args()
    main(args.results)
