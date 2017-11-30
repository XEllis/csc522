import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier

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

def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0.0
    gini = 0.0
    delta = 0.0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = eval_gini(labels, preds)
    return [('gini', gini_score)]

#################################################################################
# Missing Data Handling 1
# Selected Features for Missing Data Handling 1
train_features_1 = [ 'ps_car_13', \
                     'ps_ind_17_bin', \
                     'ps_reg_02', \
                     'ps_ind_06_bin', \
                     'ps_reg_01', \
                     'ps_ind_10_bin', \
                     'ps_ind_07_bin', \
                     'ps_ind_12_bin', \
                     'ps_ind_13_bin', \
                     'ps_ind_16_bin', \
                     'ps_ind_11_bin', \
                     'ps_car_08_cat', \
                     'ps_car_15', \
                     'ps_calc_02', \
                     'ps_calc_20_bin', \
                     'ps_calc_01', \
                     'ps_calc_03', \
                     'ps_calc_13', \
                     'ps_calc_19_bin', \
                     'ps_car_10_cat', \
                     'ps_calc_06', \
                     'ps_calc_05', \
                     'ps_calc_15_bin', \
                     'ps_calc_16_bin', \
                     'ps_calc_09', \
                     'ps_calc_17_bin', \
                     'ps_calc_18_bin', \
                     'ps_calc_04', \
                     'ps_calc_14', \
                     'ps_calc_08']

train_1 = pd.read_csv('train_forMissing1.csv')
x_train_1 = train_1.drop(['target'], axis=1)
x_train_1 = x_train_1.values
y_train_1 = train_1['target']
test_1 = pd.read_csv('test_forMissing1.csv')
ID_label = test_1['id']
x_test_1 = test_1[train_features_1]
test_pred_1 = np.zeros(len(x_test_1)) #xgb
test_pred_2 = np.zeros(len(x_test_1)) #bg
test_pred_3 = np.zeros(len(x_test_1)) #lr
test_pred_xgb = np.zeros(len(x_test_1))
test_pred_lr = np.zeros(len(x_test_1))
test_pred_bg = np.zeros(len(x_test_1))
predictions_1 = np.zeros(len(x_train_1))
predictions_2 = np.zeros(len(x_train_1))
predictions_3 = np.zeros(len(x_train_1))

# smote
for i in range(1):
    ros = SMOTE(ratio='minority')
    count = 0
    folds = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2017)
    for train_index, test_index in folds.split(x_train_1, y_train_1):
        x_train_tmp, x_test_tmp = x_train_1[train_index], x_train_1[test_index]
        y_train_tmp, y_test_tmp = y_train_1[train_index], y_train_1[test_index]
        x_ros, y_ros = ros.fit_sample(x_test_tmp, y_test_tmp)
        
        x_ros = pd.DataFrame(x_ros, columns=col_dlt)
        x_train_tmp = pd.DataFrame(x_train_tmp, columns=col_dlt)
        x_train_1 = pd.DataFrame(x_train_1, columns=col_dlt)
        
        x_ros = x_ros[train_features_1]
        x_train_tmp = x_train_tmp[train_features_1]
        x_train_1 = x_train_1[train_features_1]
        
        print(np.sum(y_ros))
        print(x_ros.shape[0])
        count = count + 1
        print('fold: ')
        print(count)
        clf_1 = XGBClassifier( objective="binary:logistic", \
                               n_estimators=50, \
                               learning_rate=0.1, \
                               min_child_weight = 25, \
                               max_depth=4, \
                               subsample = 0.8, \
                               colsample_bytree = 0.8, \
                               gamma = 1, \
                               reg_alpha = 0, \
                               reg_lambda = 1)
        clf_1.fit(x_ros, y_ros, \
                  eval_set=[(x_ros, y_ros), (x_train_tmp, y_train_tmp)], \
                  eval_metric=gini_xgb)
        predictions_1 += clf_1.predict_proba(x_train_1)[:, 1] / 5
        test_pred_1 += clf_1.predict_proba(x_test_1)[:, 1] / 5
        results = eval_gini(y_train_1, predictions_1)
        print(results)
        base_clf = DecisionTreeClassifier(max_depth=4, min_samples_split=25)
        clf_2 = BaggingClassifier(base_estimator=base_clf, n_estimators=50)
        clf_2.fit(x_ros, y_ros)
        predictions_2 += clf_2.predict_proba(x_train_1)[:, 1] / 5
        test_pred_2 += clf_2.predict_proba(x_test_1)[:, 1] / 5
        results = eval_gini(y_train_1, predictions_2)
        print(results)
        clf_3 = LogisticRegression(C=1, class_weight='balanced')
        clf_3.fit(x_ros, y_ros)
        predictions_3 += clf_3.predict_proba(x_train_1)[:, 1] / 5
        test_pred_3 += clf_3.predict_proba(x_test_1)[:, 1] / 5
        results = eval_gini(y_train_1, predictions_3)
        print(results)
        x_train_1 = train_1.drop(['target'], axis=1)
        x_train_1 = x_train_1.values
    test_pred_xgb += test_pred_1
    test_pred_bg += test_pred_2
    test_pred_lr += test_pred_3
    predictions_1 = np.zeros(len(x_train_1))
    predictions_2 = np.zeros(len(x_train_1))
    predictions_3 = np.zeros(len(x_train_1))
    test_pred_1 = np.zeros(len(x_test_1))
    test_pred_2 = np.zeros(len(x_test_1))
    test_pred_3 = np.zeros(len(x_test_1))

xgb_sub = pd.DataFrame({'id': ID_label, 'target': test_pred_xgb})
xgb_sub.to_csv('xgb_e12.csv', index=False, float_format="%.9f")
lr_sub = pd.DataFrame({'id': ID_label, 'target': test_pred_lr})
lr_sub.to_csv('lr_e12.csv', index=False, float_format="%.9f")
bg_sub = pd.DataFrame({'id': ID_label, 'target': test_pred_bg})
bg_sub.to_csv('bg_e12.csv', index=False, float_format="%.9f")
