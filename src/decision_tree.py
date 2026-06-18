from src.data_prep import DiabetesData, get_raw_data, clean_df_null

import sklearn
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from src.utils import print_metrics


def run_decision_tree(data: DiabetesData):

    model = DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced',
        max_depth=7,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features='sqrt',
        criterion='entropy',
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(estimator=model, X=data.x, y=data.y, cv=cv, scoring={
        'accuracy' : 'accuracy',
        'recall' : 'recall',
        'precision' : 'precision',
        'f1' : 'f1',
    }, n_jobs=-1)
    print("Decision Tree")
    print_metrics(scores)


def run_decision_tree_gridsearch(data: DiabetesData):
    from sklearn.tree import DecisionTreeClassifier

    base_model = DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced'
    )

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': [None, 'sqrt', 'log2']
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
    print("Decision Tree - GridSearch results")
    print(f"Parameters: {grid_search.best_params_}\n")

    best_idx = grid_search.best_index_
    cv_res = grid_search.cv_results_

    print("Computed mean values of")
    print(f"Accuracy : {cv_res['mean_test_accuracy'][best_idx]:.4f}")
    print(f"Recall   : {cv_res['mean_test_recall'][best_idx]:.4f}")
    print(f"Precision: {cv_res['mean_test_precision'][best_idx]:.4f}")
    print(f"F1       : {cv_res['mean_test_f1'][best_idx]:.4f}")

if __name__ == '__main__':
    data: DiabetesData = clean_df_null(get_raw_data())
    # run_decision_tree_gridsearch(data)
    run_decision_tree(data)