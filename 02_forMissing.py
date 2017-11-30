import pandas as pd
import numpy as np
from scipy import stats

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

train = pd.read_csv('train.csv')
x_test = pd.read_csv('test.csv')
train = train.replace(-1, np.NaN)
x_test = x_test.replace(-1, np.NaN)

x_train = train.drop(['target'], axis=1)
y_train = train['target']
ytmp = pd.DataFrame(y_train, columns=['target'])

# Handling Missing Values:
# Method: (1) Elementing Features That Have Missing Values

x_train_1 = train.drop(['target', \
                        'ps_ind_02_cat', \
                        'ps_ind_04_cat', \
                        'ps_ind_05_cat', \
                        'ps_reg_03', \
                        'ps_car_01_cat', \
                        'ps_car_02_cat', \
                        'ps_car_03_cat', \
                        'ps_car_05_cat', \
                        'ps_car_07_cat', \
                        'ps_car_09_cat', \
                        'ps_car_11', \
                        'ps_car_12', \
                        'ps_car_14'], axis=1)

x_test_1 = x_test.drop(['ps_ind_02_cat', \
                        'ps_ind_04_cat', \
                        'ps_ind_05_cat', \
                        'ps_reg_03', \
                        'ps_car_01_cat', \
                        'ps_car_02_cat', \
                        'ps_car_03_cat', \
                        'ps_car_05_cat', \
                        'ps_car_07_cat', \
                        'ps_car_09_cat', \
                        'ps_car_11', \
                        'ps_car_12', \
                        'ps_car_14'], axis=1)

xtmp_1 = pd.DataFrame(x_train_1, columns=col_dlt)
train_1_file = pd.concat([xtmp_1, ytmp], axis=1)
train_1_file.to_csv('train_forMissing1.csv', index=False)

test_1_file = pd.DataFrame(x_test_1, columns=col_dlt)
test_1_file.to_csv('test_forMissing1.csv', index=False)

# Handling Missing Values:
# Method: (2) Mean or Mode imputation

m1, c1 = stats.mode(x_train['ps_ind_02_cat'], nan_policy='omit')
m2, c2 = stats.mode(x_train['ps_ind_04_cat'], nan_policy='omit')
m3, c3 = stats.mode(x_train['ps_ind_05_cat'], nan_policy='omit')
m4 = np.nanmean(x_train['ps_reg_03'])
m5, c5 = stats.mode(x_train['ps_car_01_cat'], nan_policy='omit')
m6, c6 = stats.mode(x_train['ps_car_02_cat'], nan_policy='omit')
m7, c7 = stats.mode(x_train['ps_car_03_cat'], nan_policy='omit')
m8, c8 = stats.mode(x_train['ps_car_05_cat'], nan_policy='omit')
m9, c9 = stats.mode(x_train['ps_car_07_cat'], nan_policy='omit')
m10, c10 = stats.mode(x_train['ps_car_09_cat'], nan_policy='omit')
m11 = np.nanmedian(x_train['ps_car_11'])
m12 = np.nanmean(x_train['ps_car_12'])
m13 = np.nanmean(x_train['ps_car_14'])

x_train['ps_ind_02_cat'] = x_train['ps_ind_02_cat'].fillna(m1[0])
x_train['ps_ind_04_cat'] = x_train['ps_ind_04_cat'].fillna(m2[0])
x_train['ps_ind_05_cat'] = x_train['ps_ind_05_cat'].fillna(m3[0])
x_train['ps_reg_03'] = x_train['ps_reg_03'].fillna(m4)
x_train['ps_car_01_cat'] = x_train['ps_car_01_cat'].fillna(m5[0])
x_train['ps_car_02_cat'] = x_train['ps_car_02_cat'].fillna(m6[0])
x_train['ps_car_03_cat'] = x_train['ps_car_03_cat'].fillna(m7[0])
x_train['ps_car_05_cat'] = x_train['ps_car_05_cat'].fillna(m8[0])
x_train['ps_car_07_cat'] = x_train['ps_car_07_cat'].fillna(m9[0])
x_train['ps_car_09_cat'] = x_train['ps_car_09_cat'].fillna(m10[0])
x_train['ps_car_11'] = x_train['ps_car_11'].fillna(m11)
x_train['ps_car_12'] = x_train['ps_car_12'].fillna(m12)
x_train['ps_car_14'] = x_train['ps_car_14'].fillna(m13)

x_test['ps_ind_02_cat'] = x_test['ps_ind_02_cat'].fillna(m1[0])
x_test['ps_ind_04_cat'] = x_test['ps_ind_04_cat'].fillna(m2[0])
x_test['ps_ind_05_cat'] = x_test['ps_ind_05_cat'].fillna(m3[0])
x_test['ps_reg_03'] = x_test['ps_reg_03'].fillna(m4)
x_test['ps_car_01_cat'] = x_test['ps_car_01_cat'].fillna(m5[0])
x_test['ps_car_02_cat'] = x_test['ps_car_02_cat'].fillna(m6[0])
x_test['ps_car_03_cat'] = x_test['ps_car_03_cat'].fillna(m7[0])
x_test['ps_car_05_cat'] = x_test['ps_car_05_cat'].fillna(m8[0])
x_test['ps_car_07_cat'] = x_test['ps_car_07_cat'].fillna(m9[0])
x_test['ps_car_09_cat'] = x_test['ps_car_09_cat'].fillna(m10[0])
x_test['ps_car_11'] = x_test['ps_car_11'].fillna(m11)
x_test['ps_car_12'] = x_test['ps_car_12'].fillna(m12)
x_test['ps_car_14'] = x_test['ps_car_14'].fillna(m13)

xtmp_2 = pd.DataFrame(x_train, columns=col)
train_2_file = pd.concat([xtmp_2, ytmp], axis=1)
train_2_file.to_csv('train_forMissing2.csv', index=False)

test_2_file = pd.DataFrame(x_test, columns=col)
test_2_file.to_csv('test_forMissing2.csv', index=False)
