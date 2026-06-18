import numpy as np
import skfuzzy as fuzz
import sklearn
from skfuzzy import control as ctrl
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve

from data_prep import clean_df_null, get_raw_data, DiabetesData
from utils import print_metrics


def setup_rules(method: str):
    glucose = ctrl.Antecedent(np.linspace(30, 270, 500), 'glucose')
    bmi = ctrl.Antecedent(np.linspace(15, 70, 500), 'bmi')
    age = ctrl.Antecedent(np.linspace(1, 100, 100), 'age')
    pedigree = ctrl.Antecedent(np.linspace(0.0, 2.5, 100), 'pedigree')
    diabetes = ctrl.Consequent(np.linspace(0, 100, 200), 'diabetes', defuzzify_method=method)

    glucose['low'] = fuzz.trapmf(glucose.universe, [0.0, 0.0, 65.0, 90.0])
    glucose['medium'] = fuzz.trapmf(glucose.universe, [80.0, 95.0, 115.0, 135.0])
    glucose['high'] = fuzz.trimf(glucose.universe, [125.0, 150.0, 175.0])
    glucose['alarming'] = fuzz.trapmf(glucose.universe, [165.0, 190.0, 270.0, 270.0])

    bmi['underweight'] = fuzz.trapmf(bmi.universe, [0.0, 0.0, 17.0, 20.0])
    bmi['normal'] = fuzz.trimf(bmi.universe, [18.5, 23.0, 27.5])
    bmi['overweight'] = fuzz.trimf(bmi.universe, [26.0, 31.0, 36.0])
    bmi['obese'] = fuzz.trapmf(bmi.universe, [34.0, 45.0, 70.0, 70.0])

    age['young'] = fuzz.trapmf(age.universe, [0.0, 0.0, 25.0, 35.0])
    age['adult'] = fuzz.trimf(age.universe, [30.0, 50.0, 65.0])
    age['elderly'] = fuzz.trapmf(age.universe, [60.0, 75.0, 100.0, 100.0])

    pedigree['low'] = fuzz.trapmf(pedigree.universe, [0.0, 0.0, 0.3, 0.5])
    pedigree['medium'] = fuzz.trimf(pedigree.universe, [0.4, 0.6, 0.8])
    pedigree['high'] = fuzz.trapmf(pedigree.universe, [0.7, 1.0, 2.5, 2.5])

    diabetes['low'] = fuzz.trapmf(diabetes.universe, [0, 0, 30, 50])
    diabetes['medium'] = fuzz.trimf(diabetes.universe, [35, 50, 65])
    diabetes['high'] = fuzz.trimf(diabetes.universe, [55, 75, 90])
    diabetes['certain'] = fuzz.trapmf(diabetes.universe, [85, 95, 100, 100])

    # 3. Baza reguł
    rules = [
        ctrl.Rule(glucose['low'], diabetes['low']),
        ctrl.Rule(glucose['medium'] & bmi['underweight'], diabetes['low']),
        ctrl.Rule(glucose['medium'] & bmi['normal'], diabetes['low']),
        ctrl.Rule(glucose['medium'] & bmi['overweight'], diabetes['medium']),
        ctrl.Rule(glucose['medium'] & bmi['obese'], diabetes['medium']),
        ctrl.Rule(glucose['high'] & bmi['underweight'], diabetes['medium']),
        ctrl.Rule(glucose['high'] & bmi['normal'], diabetes['medium']),
        ctrl.Rule(glucose['high'] & bmi['overweight'], diabetes['high']),
        ctrl.Rule(glucose['high'] & bmi['obese'], diabetes['high']),
        ctrl.Rule(glucose['high'] & age['elderly'] & bmi['obese'], diabetes['certain']),
        ctrl.Rule(glucose['alarming'], diabetes['high']),
        ctrl.Rule(glucose['alarming'] & age['young'], diabetes['medium']),
        ctrl.Rule(glucose['alarming'] & bmi['obese'], diabetes['certain']),
        ctrl.Rule(glucose['alarming'] & bmi['overweight'] & age['elderly'], diabetes['certain']),
        ctrl.Rule(bmi['obese'] & age['elderly'] & glucose['medium'], diabetes['high']),

        ctrl.Rule(age['adult'] & pedigree['high'], diabetes['high']),
        ctrl.Rule(age['adult'] & pedigree['low'] & bmi['normal'], diabetes['low']),
        ctrl.Rule(age['adult'] & pedigree['medium'] & glucose['high'], diabetes['high']),
        ctrl.Rule(age['adult'] & bmi['obese'] & pedigree['medium'], diabetes['medium']),
        ctrl.Rule(bmi['obese'] & pedigree['high'], diabetes['high'])
    ]

    rules_recall = [
        ctrl.Rule(glucose['low'], diabetes['low']),

        ctrl.Rule(glucose['medium'] & bmi['normal'] & pedigree['low'], diabetes['low']),
        ctrl.Rule(glucose['medium'] & bmi['underweight'], diabetes['low']),

        ctrl.Rule(glucose['medium'] & bmi['overweight'], diabetes['medium']),
        ctrl.Rule(age['adult'] & pedigree['medium'] & bmi['normal'], diabetes['medium']),
        ctrl.Rule(glucose['medium'] & bmi['obese'], diabetes['high']),
        ctrl.Rule(glucose['high'] & bmi['underweight'], diabetes['high']),
        ctrl.Rule(glucose['high'] & bmi['normal'], diabetes['high']),
        ctrl.Rule(glucose['alarming'] & age['young'], diabetes['high']),
        ctrl.Rule(age['adult'] & bmi['obese'] & pedigree['medium'], diabetes['high']),

        ctrl.Rule(glucose['high'] & bmi['overweight'], diabetes['high']),
        ctrl.Rule(glucose['alarming'], diabetes['high']),
        ctrl.Rule(age['adult'] & pedigree['high'], diabetes['high']),
        ctrl.Rule(bmi['obese'] & pedigree['high'], diabetes['high']),
        ctrl.Rule(bmi['obese'] & age['elderly'] & glucose['medium'], diabetes['high']),
        ctrl.Rule(age['adult'] & pedigree['medium'] & glucose['high'], diabetes['high']),

        ctrl.Rule(glucose['high'] & bmi['obese'], diabetes['certain']),
        ctrl.Rule(glucose['high'] & age['elderly'] & bmi['obese'], diabetes['certain']),
        ctrl.Rule(glucose['alarming'] & bmi['obese'], diabetes['certain']),
        ctrl.Rule(glucose['alarming'] & bmi['overweight'] & age['elderly'], diabetes['certain']),
        ctrl.Rule(glucose['medium'] & bmi['normal'] & pedigree['medium'], diabetes['medium']),
        ctrl.Rule(glucose['medium'] & bmi['normal'] & pedigree['high'], diabetes['medium']),
    ]

    # swap with rules_recall to maximalize recall - much more liberal rules
    diabetes_ctrl = ctrl.ControlSystem(rules_recall)
    return ctrl.ControlSystemSimulation(diabetes_ctrl)


def get_fuzzy_predictions(data_x, diabetes_system):
    raw_scores = []
    for index, row in data_x.iterrows():
        diabetes_system.input['glucose'] = row['Glucose']
        diabetes_system.input['bmi'] = row['BMI']
        diabetes_system.input['age'] = row['Age']
        diabetes_system.input['pedigree'] = row['DiabetesPedigreeFunction']
        diabetes_system.compute()
        raw_scores.append(diabetes_system.output['diabetes'])
    return raw_scores


def run_fuzzy_logic(data: DiabetesData, method: str):
    print(f"Fuzzy (method: {method})")
    diabetes_system = setup_rules(method)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = {'test_f1': [], 'test_accuracy': [], 'test_precision': [], 'test_recall': []}

    for train_idx, test_idx in cv.split(data.x, data.y):
        x_train, y_train = data.x.iloc[train_idx], data.y.iloc[train_idx]
        x_test, y_test = data.x.iloc[test_idx], data.y.iloc[test_idx]

        raw_scores_train = get_fuzzy_predictions(x_train, diabetes_system)
        precisions, recalls, thresholds = precision_recall_curve(y_train, raw_scores_train)

        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        best_threshold = thresholds[np.argmax(f1_scores)]

        raw_scores_test = get_fuzzy_predictions(x_test, diabetes_system)
        predictions_round = [1 if score >= best_threshold else 0 for score in raw_scores_test]

        scores['test_f1'].append(sklearn.metrics.f1_score(y_test, predictions_round))
        scores['test_accuracy'].append(sklearn.metrics.accuracy_score(y_test, predictions_round))
        scores['test_precision'].append(sklearn.metrics.precision_score(y_test, predictions_round))
        scores['test_recall'].append(sklearn.metrics.recall_score(y_test, predictions_round))

    print_metrics(scores)

if __name__ == '__main__':
    data = clean_df_null(get_raw_data())

    METHOD = 'centroid'
    run_fuzzy_logic(data, METHOD)
