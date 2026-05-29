from src.data_prep import DiabetesData, get_raw_data, clean_df_null

import sklearn

from utils import print_metrics


def run_decision_tree(data: DiabetesData):
    decision_tree = sklearn.tree.DecisionTreeClassifier(random_state=42)
    decision_tree.fit(data.x_train, data.y_train)
    y_pred = decision_tree.predict(data.x_test)

    print("Decision Tree")
    print_metrics(data.y_test, y_pred)

if __name__ == '__main__':
    data: DiabetesData = clean_df_null(get_raw_data())
    run_decision_tree(data)