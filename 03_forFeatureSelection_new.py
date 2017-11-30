import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

import weka.core.jvm as jvm
import weka.core.converters as converters
from weka.attribute_selection import ASSearch
from weka.attribute_selection import ASEvaluation
from weka.attribute_selection import AttributeSelection

jvm.start()

col = ['id', \
       'ps_ind_01', \
       'ps_ind_02_cat', \
       'ps_ind_03', \
       'ps_ind_04_cat', \
       'ps_ind_05_cat', \
       'ps_ind_06_bin', \
       'ps_ind_07_bin', \
       'ps_ind_08_bin', \
       'ps_ind_09_bin', \
       'ps_ind_10_bin', \
       'ps_ind_11_bin', \
       'ps_ind_12_bin', \
       'ps_ind_13_bin', \
       'ps_ind_14', \
       'ps_ind_15', \
       'ps_ind_16_bin', \
       'ps_ind_17_bin', \
       'ps_ind_18_bin', \
       'ps_reg_01', \
       'ps_reg_02', \
       'ps_reg_03', \
       'ps_car_01_cat', \
       'ps_car_02_cat', \
       'ps_car_03_cat', \
       'ps_car_04_cat', \
       'ps_car_05_cat', \
       'ps_car_06_cat', \
       'ps_car_07_cat', \
       'ps_car_08_cat', \
       'ps_car_09_cat', \
       'ps_car_10_cat',\
       'ps_car_11_cat', \
       'ps_car_11', \
       'ps_car_12', \
       'ps_car_13', \
       'ps_car_14', \
       'ps_car_15', \
       'ps_calc_01', \
       'ps_calc_02', \
       'ps_calc_03', \
       'ps_calc_04', \
       'ps_calc_05', \
       'ps_calc_06', \
       'ps_calc_07', \
       'ps_calc_08', \
       'ps_calc_09', \
       'ps_calc_10', \
       'ps_calc_11', \
       'ps_calc_12', \
       'ps_calc_13', \
       'ps_calc_14', \
       'ps_calc_15_bin', \
       'ps_calc_16_bin', \
       'ps_calc_17_bin', \
       'ps_calc_18_bin', \
       'ps_calc_19_bin', \
       'ps_calc_20_bin']

col_dlt = ['id', \
           'ps_ind_01', \
           'ps_ind_03', \
           'ps_ind_06_bin', \
           'ps_ind_07_bin', \
           'ps_ind_08_bin', \
           'ps_ind_09_bin', \
           'ps_ind_10_bin', \
           'ps_ind_11_bin', \
           'ps_ind_12_bin', \
           'ps_ind_13_bin', \
           'ps_ind_14', \
           'ps_ind_15', \
           'ps_ind_16_bin', \
           'ps_ind_17_bin', \
           'ps_ind_18_bin', \
           'ps_reg_01', \
           'ps_reg_02', \
           'ps_car_04_cat', \
           'ps_car_06_cat', \
           'ps_car_08_cat', \
           'ps_car_10_cat',\
           'ps_car_11_cat', \
           'ps_car_13', \
           'ps_car_15', \
           'ps_calc_01', \
           'ps_calc_02', \
           'ps_calc_03', \
           'ps_calc_04', \
           'ps_calc_05', \
           'ps_calc_06', \
           'ps_calc_07', \
           'ps_calc_08', \
           'ps_calc_09', \
           'ps_calc_10', \
           'ps_calc_11', \
           'ps_calc_12', \
           'ps_calc_13', \
           'ps_calc_14', \
           'ps_calc_15_bin', \
           'ps_calc_16_bin', \
           'ps_calc_17_bin', \
           'ps_calc_18_bin', \
           'ps_calc_19_bin', \
           'ps_calc_20_bin']

#################################################################################
# Missing Data Handling 1
# Select Features for Missing Data Handling 1
# ps_car_13
# ps_ind_17_bin
# ps_reg_02
# ps_ind_06_bin
# ps_reg_01
# ps_ind_10_bin
# ps_ind_07_bin
# ps_ind_12_bin
# ps_ind_13_bin
# ps_ind_16_bin
# ps_ind_11_bin
# ps_car_08_cat
# ps_car_15
# ps_calc_02
# ps_calc_20_bin
# ps_calc_01
# ps_calc_03
# ps_calc_13
# ps_calc_19_bin
# ps_car_10_cat
# ps_calc_06
# ps_calc_05
# ps_calc_15_bin
# ps_calc_16_bin
# ps_calc_09
# ps_calc_17_bin
# ps_calc_18_bin
# ps_calc_04
# ps_calc_14
# ps_calc_08

train_1 = pd.read_csv('train_forMissing1.csv')
x_train_1 = train_1.drop(['target'], axis=1)
x_train_1 = x_train_1.values
y_train_1 = train_1['target']

# Random Over Sampling with Replacement
for i in range(5):
    ros = RandomOverSampler(ratio='minority')
    folds = StratifiedKFold(n_splits = 10, shuffle=True)
    for train_index, test_index in folds.split(x_train_1, y_train_1):
        x_train_tmp, x_test_tmp = x_train_1[train_index], x_train_1[test_index]
        y_train_tmp, y_test_tmp = y_train_1[train_index], y_train_1[test_index]
        x_ros, y_ros = ros.fit_sample(x_test_tmp, y_test_tmp)
        x_tmp = pd.DataFrame(x_ros, columns=col_dlt)
        y_tmp = pd.DataFrame(y_ros, columns=['target'])
        t_tmp_file = pd.concat([x_tmp, y_tmp], axis=1)
        t_tmp_file.to_csv('tmp.csv', index=False)
        data = converters.load_any_file("tmp.csv")
        data.class_is_last()
        print(data.num_instances)
        search = ASSearch(classname="weka.attributeSelection.BestFirst")
        evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval")
        attsel = AttributeSelection()
        attsel.search(search)
        attsel.evaluator(evaluator)
        attsel.select_attributes(data)
        print("number of attributes: " + str(attsel.number_attributes_selected))
        print("attributes: " + str(attsel.selected_attributes))
        print("result string:\n" + attsel.results_string)
        print(np.sum(y_ros))
        print(y_ros.shape[0])

# Synthetic Minority Over-Sampling Technique
for i in range(5):
    smt = SMOTE(ratio='minority')
    folds = StratifiedKFold(n_splits = 10, shuffle=True)
    for train_index, test_index in folds.split(x_train_1, y_train_1):
        x_train_tmp, x_test_tmp = x_train_1[train_index], x_train_1[test_index]
        y_train_tmp, y_test_tmp = y_train_1[train_index], y_train_1[test_index]
        x_ros, y_ros = smt.fit_sample(x_test_tmp, y_test_tmp)
        x_tmp = pd.DataFrame(x_ros, columns=col_dlt)
        y_tmp = pd.DataFrame(y_ros, columns=['target'])
        t_tmp_file = pd.concat([x_tmp, y_tmp], axis=1)
        t_tmp_file.to_csv('tmp.csv', index=False)
        data = converters.load_any_file("tmp.csv")
        data.class_is_last()
        print(data.num_instances)
        search = ASSearch(classname="weka.attributeSelection.BestFirst")
        evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval")
        attsel = AttributeSelection()
        attsel.search(search)
        attsel.evaluator(evaluator)
        attsel.select_attributes(data)
        print("number of attributes: " + str(attsel.number_attributes_selected))
        print("attributes: " + str(attsel.selected_attributes))
        print("result string:\n" + attsel.results_string)
        print(np.sum(y_ros))
        print(y_ros.shape[0])
#################################################################################

#################################################################################
# Missing Data Handling 2
# Select Features for Missing Data Handling 2
# ps_ind_05_cat
# ps_ind_17_bin
# ps_car_13
# ps_car_12
# ps_car_07_cat
# ps_reg_01
# ps_ind_06_bin
# ps_reg_02
# ps_ind_07_bin
# ps_reg_03
# ps_ind_10_bin
# ps_ind_16_bin
# ps_car_02_cat
# ps_car_08_cat
# ps_ind_12_bin
# ps_car_15
# ps_ind_13_bin
# ps_ind_11_bin
# ps_ind_14
# ps_calc_01
# ps_car_10_cat
# ps_calc_02
# ps_calc_03
# ps_calc_08
# ps_calc_18_bin
# ps_calc_20_bin
# ps_calc_13
# ps_calc_06
# ps_calc_14
# ps_calc_17_bin

train_2 = pd.read_csv('train_forMissing2.csv')
x_train_2 = train_2.drop(['target'], axis=1)
x_train_2 = x_train_2.values
y_train_2 = train_2['target']

# Random Over Sampling with Replacement
for i in range(5):
    ros = RandomOverSampler(ratio='minority')
    folds = StratifiedKFold(n_splits = 10, shuffle=True)
    for train_index, test_index in folds.split(x_train_2, y_train_2):
        x_train_tmp, x_test_tmp = x_train_2[train_index], x_train_2[test_index]
        y_train_tmp, y_test_tmp = y_train_2[train_index], y_train_2[test_index]
        x_ros, y_ros = ros.fit_sample(x_test_tmp, y_test_tmp)
        x_tmp = pd.DataFrame(x_ros, columns=col)
        y_tmp = pd.DataFrame(y_ros, columns=['target'])
        t_tmp_file = pd.concat([x_tmp, y_tmp], axis=1)
        t_tmp_file.to_csv('tmp.csv', index=False)
        data = converters.load_any_file("tmp.csv")
        data.class_is_last()
        print(data.num_instances)
        search = ASSearch(classname="weka.attributeSelection.BestFirst")
        evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval")
        attsel = AttributeSelection()
        attsel.search(search)
        attsel.evaluator(evaluator)
        attsel.select_attributes(data)
        print("number of attributes: " + str(attsel.number_attributes_selected))
        print("attributes: " + str(attsel.selected_attributes))
        print("result string:\n" + attsel.results_string)
        print(np.sum(y_ros))
        print(y_ros.shape[0])

# Synthetic Minority Over-Sampling Technique
for i in range(5):
    smt = SMOTE(ratio='minority')
    folds = StratifiedKFold(n_splits = 10, shuffle=True)
    for train_index, test_index in folds.split(x_train_2, y_train_2):
        x_train_tmp, x_test_tmp = x_train_2[train_index], x_train_2[test_index]
        y_train_tmp, y_test_tmp = y_train_2[train_index], y_train_2[test_index]
        x_ros, y_ros = smt.fit_sample(x_test_tmp, y_test_tmp)
        x_tmp = pd.DataFrame(x_ros, columns=col)
        y_tmp = pd.DataFrame(y_ros, columns=['target'])
        t_tmp_file = pd.concat([x_tmp, y_tmp], axis=1)
        t_tmp_file.to_csv('tmp.csv', index=False)
        data = converters.load_any_file("tmp.csv")
        data.class_is_last()
        print(data.num_instances)
        search = ASSearch(classname="weka.attributeSelection.BestFirst")
        evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval")
        attsel = AttributeSelection()
        attsel.search(search)
        attsel.evaluator(evaluator)
        attsel.select_attributes(data)
        print("number of attributes: " + str(attsel.number_attributes_selected))
        print("attributes: " + str(attsel.selected_attributes))
        print("result string:\n" + attsel.results_string)
        print(np.sum(y_ros))
        print(y_ros.shape[0])
#################################################################################

jvm.stop()
