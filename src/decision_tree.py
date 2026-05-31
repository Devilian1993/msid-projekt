from src.data_prep import DiabetesData, get_raw_data, clean_df_null

import sklearn
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.utils import print_metrics


def run_decision_tree(data: DiabetesData):

    model = sklearn.tree.DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced',
        max_depth=5,
        min_samples_split=8,
        min_samples_leaf=2,
        criterion='gini',
        ccp_alpha=0.0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(estimator=model, X=data.x, y=data.y, cv=cv, scoring='f1', n_jobs=-1)

    print("Decision Tree")
    print_metrics(scores)

if __name__ == '__main__':
    data: DiabetesData = clean_df_null(get_raw_data())
    run_decision_tree(data)