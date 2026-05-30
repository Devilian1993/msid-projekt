import sklearn.metrics


def print_metrics(y_test, y_pred):
    print("Accuracy:", "%.4f" % sklearn.metrics.accuracy_score(y_test, y_pred))
    print("Precision:", "%.4f" % sklearn.metrics.precision_score(y_test, y_pred))
    print("Recall:", "%.4f" % sklearn.metrics.recall_score(y_test, y_pred))
    print("F1 Score:", "%.4f" % sklearn.metrics.f1_score(y_test, y_pred))
    # accuracy - chance of correct result (true positives + true negatives) / all
    # precision - positive predictive value (true positives / (false positives + true positives) )
    # recall - sensitivity (positives retrieved / all positives) - all positives = false negatives + true positives
    # f1 - measure of predictive performance