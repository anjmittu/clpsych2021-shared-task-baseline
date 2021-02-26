import json
import argparse
import csv
from sklearn.metrics import f1_score, fbeta_score, confusion_matrix, roc_curve, auc

"""
This will calculate statistics on predictions from the model
"""


def main(results_file_path, truth_file_path, pos_label_name):
    truth_values = {}
    with open(truth_file_path) as f:
        for json_obj in f:
            data = json.loads(json_obj)
            truth_values[data["id"]] = str(data["label"])

    predictions = []
    probs = []
    truths = []
    with open(results_file_path) as rf:
        results = csv.reader(rf, delimiter="\t")
        for pred in results:
            predictions.append(pred[1])
            probs.append(float(pred[2]))
            truths.append(truth_values[pred[0]])

    f1 = f1_score(truths, predictions, average='binary', pos_label=pos_label_name)
    f2 = fbeta_score(truths, predictions, beta=2, average='binary', pos_label=pos_label_name)

    tn, fp, fn, tp = confusion_matrix(truths, predictions).ravel()
    true_positives = tp / (tp + fn)
    false_alarms = fp / (fp + tn)

    fpr, tpr, thresholds = roc_curve(truths, probs, pos_label=pos_label_name)
    auc_score = auc(fpr, tpr)
    print("{}, {}, {}, {}, {}".format(f1, f2, true_positives, false_alarms, auc_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prints statistics about the predictions.')
    parser.add_argument('--results', help='path to the results file')
    parser.add_argument('--truth', help='path to the truth file')
    parser.add_argument('--pos', help='the name of the pos label')

    args = parser.parse_args()
    main(args.results, args.truth, args.pos)
