from src.data_prep import DiabetesData, get_raw_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def run_random_forest(data: DiabetesData):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data.x_train, data.y_train)

    y_pred = model.predict(data.x_test)
    print("Random forest")
    print("Accuracy:", "%.6f" % accuracy_score(data.y_test, y_pred))
    print("Precision:", "%.6f" % precision_score(data.y_test, y_pred))
    print("Recall:", "%.6f" % recall_score(data.y_test, y_pred))
    print("F1 Score:", "%.6f" % f1_score(data.y_test, y_pred))


if __name__ == '__main__':
    data = get_raw_data()
    run_random_forest(data)
