import sklearn.metrics
import numpy as np

def print_metrics(scores):
    print("Computed mean values of")
    print("Accuracy : %.4f" % np.mean(scores['test_accuracy']))
    print("Recall : %.4f" % np.mean(scores['test_recall']))
    print("Precision: %.4f" % np.mean(scores['test_precision']))
    print("F1 : %.4f" % np.mean(scores['test_f1']))
    print()
    # accuracy - chance of correct result (true positives + true negatives) / all
    # precision - positive predictive value (true positives / (false positives + true positives) )
    # recall - sensitivity (positives retrieved / all positives) - all positives = false negatives + true positives
    # f1 - measure of predictive performance
