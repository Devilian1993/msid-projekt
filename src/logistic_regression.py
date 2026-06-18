from src.data_prep import DiabetesData, get_raw_data, clean_df_null
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

from src.utils import print_metrics


def run_logistic_regression(data: DiabetesData):

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=10000,
            random_state=42,
            class_weight='balanced',
            C=0.1,
            solver='saga',
            l1_ratio=1
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

def run_logistic_regression_gridsearch(data: DiabetesData):
    base_model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ))
    ])


    param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__l1_ratio': [1, 0, 0.5],
        'model__solver': ['liblinear', 'saga']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'recall': 'recall',
        'precision': 'precision',
        'f1': 'f1'
    }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit='f1',
        n_jobs=-1
    )

    grid_search.fit(data.x, data.y)

    print("\nLogistic Regression - GridSearch")
    print(f"Parametry: {grid_search.best_params_}\n")

    best_idx = grid_search.best_index_
    cv_res = grid_search.cv_results_

    print("Computed mean values of")
    print(f"Accuracy : {cv_res['mean_test_accuracy'][best_idx]:.4f}")
    print(f"Recall   : {cv_res['mean_test_recall'][best_idx]:.4f}")
    print(f"Precision: {cv_res['mean_test_precision'][best_idx]:.4f}")
    print(f"F1       : {cv_res['mean_test_f1'][best_idx]:.4f}")

if __name__ == '__main__':
    data = clean_df_null(get_raw_data())
    # run_logistic_regression_gridsearch(data)
    run_logistic_regression(data)