from src.data_prep import DiabetesData, get_raw_data

import sklearn


def run_decision_tree(data: DiabetesData):
    decision_tree = sklearn.tree.DecisionTreeClassifier(random_state=42)
    decision_tree.fit(data.x_train, data.y_train)
    predictions = decision_tree.predict(data.x_test)

    print("Decision Tree")
    print("Accuracy:", "%.6f" % sklearn.metrics.accuracy_score(data.y_test, predictions))
    print("Precision:", "%.6f" % sklearn.metrics.precision_score(data.y_test, predictions))
    print("Recall:", "%.6f" % sklearn.metrics.recall_score(data.y_test, predictions))
    print("F1 Score:", "%.6f" % sklearn.metrics.f1_score(data.y_test, predictions))
    # accuracy - chance of correct result (true positives + true negatives) / all
    # precision - positive predictive value (true positives / (false positives + true positives) )
    # recall - sensitivity (true positives / real positives) - real positives = true negatives + true positives
    # f1 - measure of predictive performance

if __name__ == '__main__':
    data: DiabetesData = get_raw_data()
    run_decision_tree(data)