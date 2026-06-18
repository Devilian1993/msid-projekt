from time import perf_counter

import decision_tree
import fuzzy
import logistic_regression
import random_forest
from data_prep import DiabetesData, get_raw_data, clean_df_null, clean_df_median, split_data

df_raw = get_raw_data()
METHOD = 'lom'

# 3 datasets - raw data (some missing values), clean data (no missing values), median data (missing filled with median)
nulls_data = split_data(df_raw.copy())
clean_data = clean_df_null(df_raw.copy())
median_data = clean_df_median(df_raw.copy())

# testing for following:
# accuracy - chance of correct result (true positives + true negatives) / all
# precision - positive predictive value (true positives / (false positives + true positives) )
# recall - sensitivity (positives retrieved / all positives) - all positives = false negatives + true positives
# f1 - measure of predictive performance


# 1. testing the raw dataset
print("Testing the raw dataset!\n")
fuzzy.run_fuzzy_logic(nulls_data, METHOD)
# logistic_regression.run_logistic_regression(nulls_data) # logistic regression falls short, as it does not accept missing values
decision_tree.run_decision_tree(nulls_data)
random_forest.run_random_forest(nulls_data)
print("Testing finished!")

# observed:
# logistic regression fails to run with missing values
# random forest performs the best, even though fuzzy logic performs with better precision
# fuzzy logic is quite accurate, but recall is the worst of the three
# decision tree with the lowest precision, f1 score still a bit higher than fuzzy logic


# 2. testing the clean dataset
print("Testing the clean dataset!\n")
fuzzy.run_fuzzy_logic(clean_data, METHOD)
logistic_regression.run_logistic_regression(clean_data)
decision_tree.run_decision_tree(clean_data)
random_forest.run_random_forest(clean_data)
print("Testing finished!")

# observed:
# overall scores higher than on the first dataset, missing values do affect the outcomes
# fuzzy logic performs a bit better, high accuracy but low precision once again
# logistic regression can run on the clean dataset, provides similar accuracy to fuzzy logic, but with better recall
# decision tree with the highest recall and the lowest precision out of all three, overall placing itself in the middle according to f1 score
# random fores once again with the highest f1 score, best accuracy, good recall and best precision (still not that high)


# 3. testing the median dataset
print("Testing the median dataset!\n")
fuzzy.run_fuzzy_logic(median_data, METHOD)
logistic_regression.run_logistic_regression(median_data)
decision_tree.run_decision_tree(median_data)
random_forest.run_random_forest(median_data)
print("Testing finished!")

# observed:
# fuzzy logic performs worse, most likely due to overfitting and bias in the data (median here is much different from overall median)
# logistic regression once again fails to compute the values
# decision tree providing the worst performance so far with the lowest precision observed across all tests
# random forest still providing the best performance

# overall observations:
# logistic regression often fails to compute - the data has to be especially cleaned and prepared in order to run this method
# random forest provides very good results in all test cases - comparably the best of provided methods
# decision tree falls short compared to random forest, comparable to fuzzy logic
# fuzzy logic, even with just a few attributes (instead of all possible like in the other methods) performs fairly well,
# provides very good accuracy across all tests (>0.74), but a bit poorer on other markers