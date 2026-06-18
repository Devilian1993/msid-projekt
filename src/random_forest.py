from sklearn.model_selection import GridSearchCV

from src.data_prep import DiabetesData, get_raw_data, clean_df_null
from sklearn.ensemble import RandomForestClassifier

from utils import print_metrics


def run_random_forest(data: DiabetesData):
    from sklearn.model_selection import cross_validate, StratifiedKFold

    model = RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=4,
        max_features='sqrt',
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(estimator=model, X=data.x, y=data.y, cv=cv, scoring={
        'accuracy' : 'accuracy',
        'recall' : 'recall',
        'precision' : 'precision',
        'f1' : 'f1',
    }, n_jobs=-1)

    print("Random Forest")
    print_metrics(scores)

def run_random_forest_gridsearch(data: DiabetesData):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier

    base_model = RandomForestClassifier(
        random_state=42,
        class_weight='balanced'
    )

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5]
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

    print("\nRandom Forest - GridSearch")
    print(f"Parameters: {grid_search.best_params_}\n")

    best_idx = grid_search.best_index_
    cv_res = grid_search.cv_results_

    print("Computed mean values of")
    print(f"Accuracy : {cv_res['mean_test_accuracy'][best_idx]:.4f}")
    print(f"Recall   : {cv_res['mean_test_recall'][best_idx]:.4f}")
    print(f"Precision: {cv_res['mean_test_precision'][best_idx]:.4f}")
    print(f"F1       : {cv_res['mean_test_f1'][best_idx]:.4f}")


if __name__ == '__main__':
    data = clean_df_null(get_raw_data())
    # run_random_forest_gridsearch(data)
    run_random_forest(data)
