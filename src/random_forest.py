from src.data_prep import DiabetesData, get_raw_data, clean_df_null
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from utils import print_metrics


def run_random_forest(data: DiabetesData):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data.x_train, data.y_train)

    y_pred = model.predict(data.x_test)
    print("Random forest")
    print_metrics(data.y_test, y_pred)


if __name__ == '__main__':
    data = clean_df_null(get_raw_data())
    run_random_forest(data)
