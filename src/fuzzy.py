import numpy as np
import skfuzzy as fuzz
import sklearn
from skfuzzy import control as ctrl
from sklearn.model_selection import StratifiedKFold

from data_prep import clean_df_null, get_raw_data
from utils import print_metrics


def setup_rules():
    glucose = ctrl.Antecedent(np.linspace(30, 270, 500), 'glucose')  # above 300 can be fatal,
    bmi = ctrl.Antecedent(np.linspace(15, 70, 500), 'bmi')  # not sure about the ranges
    age = ctrl.Antecedent(np.linspace(1, 100, 100), 'age')  # may cover more variables later on
    diabetes = ctrl.Consequent(np.linspace(0, 100, 200), 'diabetes',
                               defuzzify_method='centroid')

    # inputs
    glucose['low'] = fuzz.trapmf(glucose.universe, [0.0, 0.0, 65.0, 90.0])
    glucose['medium'] = fuzz.trapmf(glucose.universe, [80.0, 95.0, 115.0, 135.0])
    glucose['high'] = fuzz.trimf(glucose.universe, [125.0, 150.0, 175.0])
    glucose['alarming'] = fuzz.trapmf(glucose.universe, [165.0, 190.0, 270.0, 270.0])

    bmi['underweight'] = fuzz.trapmf(bmi.universe, [0.0, 0.0, 17.0, 20.0])
    bmi['normal'] = fuzz.trimf(bmi.universe, [18.5, 23.0, 27.5])  # as in normal weight
    bmi['overweight'] = fuzz.trimf(bmi.universe, [26.0, 31.0, 36.0])
    bmi['obese'] = fuzz.trapmf(bmi.universe, [34.0, 45.0, 70.0, 70.0])

    age['young'] = fuzz.trapmf(age.universe, [0.0, 0.0, 25.0, 35.0])
    age['adult'] = fuzz.trimf(age.universe, [30.0, 50.0, 65.0])
    age['elderly'] = fuzz.trapmf(age.universe, [60.0, 75.0, 100.0, 100.0])

    # output, chances for diabetes
    diabetes['low'] = fuzz.trapmf(diabetes.universe, [0, 0, 30, 50])
    diabetes['medium'] = fuzz.trimf(diabetes.universe, [35, 50, 65])
    diabetes['high'] = fuzz.trimf(diabetes.universe, [55, 75, 90])
    diabetes['certain'] = fuzz.trapmf(diabetes.universe, [85, 95, 100, 100])

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
        ctrl.Rule(bmi['overweight'] & age['elderly'] & glucose['medium'], diabetes['medium'])
    ]

    diabetes_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(diabetes_ctrl)


if __name__ == '__main__':
    diabetes_system = setup_rules()
    data = clean_df_null(get_raw_data())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_f1 = []
    test_accuracy = []
    test_precision = []
    test_recall = []
    scores = {'test_f1': [],
              'test_accuracy': [],
              'test_precision': [],
              'test_recall': []}

    for train_idx, test_idx in cv.split(data.x, data.y):
        x_fold = data.x.iloc[test_idx]
        y_fold = data.y.iloc[test_idx]

        predictions_round = []
        for index, row in x_fold.iterrows():
            diabetes_system.input['glucose'] = row['Glucose']
            diabetes_system.input['bmi'] = row['BMI']
            diabetes_system.input['age'] = row['Age']
            diabetes_system.compute()
            predictions_round.append(0 if diabetes_system.output['diabetes'] < 55.0 else 1)

        scores['test_f1'].append(sklearn.metrics.f1_score(y_fold, predictions_round))
        scores['test_accuracy'].append(sklearn.metrics.accuracy_score(y_fold, predictions_round))
        scores['test_precision'].append(sklearn.metrics.precision_score(y_fold, predictions_round))
        scores['test_recall'].append(sklearn.metrics.recall_score(y_fold, predictions_round))

    print("Fuzzy")
    print_metrics(scores)
