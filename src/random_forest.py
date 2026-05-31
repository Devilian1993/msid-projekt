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
        max_features='log2',
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


if __name__ == '__main__':
    data = clean_df_null(get_raw_data())
    run_random_forest(data)
