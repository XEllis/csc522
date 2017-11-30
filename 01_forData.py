import pandas as pd
import numpy as np
from scipy import stats

train = pd.read_csv('train.csv')
x_test = pd.read_csv('test.csv')
train = train.replace(-1, np.NaN)
x_test = x_test.replace(-1, np.NaN)

x_train = train.drop(['target'], axis=1)
y_train = train['target']

rows_train = x_train.shape[0]
columns_train = x_train.shape[1]
rows_test = x_test.shape[0]
columns_test = x_test.shape[1]

print("(1) Training Data Set")
print("Number of Instances: \t%d" % (rows_train)) # 595212
print("Number of Features: \t%d" % (columns_train)) # 58
print("Number of Classes: \t%d" % (np.unique(y_train).size)) # 2
print("Number of Claims: \t%d" % (np.sum(y_train))) # 21694
print("Number of No Claims: \t%d" % (rows_train - np.sum(y_train))) # 573518

print(stats.mode(x_train['ps_ind_02_cat'], nan_policy='omit'))
print(stats.mode(x_train['ps_ind_04_cat'], nan_policy='omit'))
print(stats.mode(x_train['ps_ind_05_cat'], nan_policy='omit'))
print(np.nanmedian(x_train['ps_reg_03']))
print(np.nanmean(x_train['ps_reg_03']))
print(stats.mode(x_train['ps_car_01_cat'], nan_policy='omit'))
print(stats.mode(x_train['ps_car_02_cat'], nan_policy='omit'))
print(stats.mode(x_train['ps_car_03_cat'], nan_policy='omit'))
print(stats.mode(x_train['ps_car_05_cat'], nan_policy='omit'))
print(stats.mode(x_train['ps_car_07_cat'], nan_policy='omit'))
print(stats.mode(x_train['ps_car_09_cat'], nan_policy='omit'))
print(np.nanmedian(x_train['ps_car_11']))
print(np.nanmean(x_train['ps_car_11']))
print(np.nanmedian(x_train['ps_car_12']))
print(np.nanmean(x_train['ps_car_12']))
print(np.nanmedian(x_train['ps_car_14']))
print(np.nanmean(x_train['ps_car_14']))

print(np.unique(x_train['id']).size) # Unique ID

print(np.unique(x_train['ps_ind_02_cat'][~np.isnan(x_train['ps_ind_02_cat'])]).size) # 4
print(np.unique(x_train['ps_ind_04_cat'][~np.isnan(x_train['ps_ind_04_cat'])]).size) # 2
print(np.unique(x_train['ps_ind_05_cat'][~np.isnan(x_train['ps_ind_05_cat'])]).size) # 7
print(np.unique(x_train['ps_car_01_cat'][~np.isnan(x_train['ps_car_01_cat'])]).size) # 12
print(np.unique(x_train['ps_car_02_cat'][~np.isnan(x_train['ps_car_02_cat'])]).size) # 2
print(np.unique(x_train['ps_car_03_cat'][~np.isnan(x_train['ps_car_03_cat'])]).size) # 2
print(np.unique(x_train['ps_car_05_cat'][~np.isnan(x_train['ps_car_05_cat'])]).size) # 2
print(np.unique(x_train['ps_car_07_cat'][~np.isnan(x_train['ps_car_07_cat'])]).size) # 2
print(np.unique(x_train['ps_car_09_cat'][~np.isnan(x_train['ps_car_09_cat'])]).size) # 5

print(train.isnull().sum())
print(train.info())

print("(2) Testing Data Set")
print("Number of Instances: \t%d" % (rows_test)) # 892816
print("Number of Features: \t%d" % (columns_test)) # 58

print(np.unique(x_test['id']).size) # Unique ID

print(np.unique(x_test['ps_ind_02_cat'][~np.isnan(x_test['ps_ind_02_cat'])]).size) # 4
print(np.unique(x_test['ps_ind_04_cat'][~np.isnan(x_test['ps_ind_04_cat'])]).size) # 2
print(np.unique(x_test['ps_ind_05_cat'][~np.isnan(x_test['ps_ind_05_cat'])]).size) # 7
print(np.unique(x_test['ps_car_01_cat'][~np.isnan(x_test['ps_car_01_cat'])]).size) # 12
print(np.unique(x_test['ps_car_02_cat'][~np.isnan(x_test['ps_car_02_cat'])]).size) # 2
print(np.unique(x_test['ps_car_03_cat'][~np.isnan(x_test['ps_car_03_cat'])]).size) # 2
print(np.unique(x_test['ps_car_05_cat'][~np.isnan(x_test['ps_car_05_cat'])]).size) # 2
print(np.unique(x_test['ps_car_07_cat'][~np.isnan(x_test['ps_car_07_cat'])]).size) # 2
print(np.unique(x_test['ps_car_09_cat'][~np.isnan(x_test['ps_car_09_cat'])]).size) # 5

print(x_test.isnull().sum())
print(x_test.info())

print(stats.mode(x_test['ps_ind_02_cat'], nan_policy='omit'))
print(stats.mode(x_test['ps_ind_04_cat'], nan_policy='omit'))
print(stats.mode(x_test['ps_ind_05_cat'], nan_policy='omit'))
print(np.nanmedian(x_test['ps_reg_03']))
print(np.nanmean(x_test['ps_reg_03']))
print(stats.mode(x_test['ps_car_01_cat'], nan_policy='omit'))
print(stats.mode(x_test['ps_car_02_cat'], nan_policy='omit'))
print(stats.mode(x_test['ps_car_03_cat'], nan_policy='omit'))
print(stats.mode(x_test['ps_car_05_cat'], nan_policy='omit'))
print(stats.mode(x_test['ps_car_07_cat'], nan_policy='omit'))
print(stats.mode(x_test['ps_car_09_cat'], nan_policy='omit'))
print(np.nanmedian(x_test['ps_car_11']))
print(np.nanmean(x_test['ps_car_11']))
print(np.nanmedian(x_test['ps_car_12']))
print(np.nanmean(x_test['ps_car_12']))
print(np.nanmedian(x_test['ps_car_14']))
print(np.nanmean(x_test['ps_car_14']))
