from src.data_prep import DiabetesData, get_raw_data, clean_df_null
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from utils import print_metrics


def scale_data(data: DiabetesData):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(data.x_train)
    x_test_scaled = scaler.transform(data.x_test)

    return DiabetesData(x_train_scaled, x_test_scaled, data.y_train, data.y_test)

def run_logistic_regression(data: DiabetesData):
    scaled_data = scale_data(data)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(scaled_data.x_train, scaled_data.y_train)

    y_pred = model.predict(scaled_data.x_test)
    print("Logistic regression")
    print_metrics(data.y_test, y_pred)

if __name__ == '__main__':
    data = clean_df_null(get_raw_data())
    run_logistic_regression(data)