from src.data_prep import DiabetesData, get_raw_data, clean_df_null
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.utils import print_metrics


def run_logistic_regression(data: DiabetesData):
    from sklearn.model_selection import cross_validate, StratifiedKFold
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            C=1,
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(estimator=model, X=data.x, y=data.y, cv=cv, scoring={
        'accuracy' : 'accuracy',
        'recall' : 'recall',
        'precision' : 'precision',
        'f1' : 'f1',
    }, n_jobs=-1)

    print("Logistic Regression")
    print_metrics(scores)

if __name__ == '__main__':
    data = clean_df_null(get_raw_data())
    run_logistic_regression(data)