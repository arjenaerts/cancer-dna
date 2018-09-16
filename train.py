import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

"""
Script that trains two classifiers on the DNA data using Logistic Regression. 
Requires the output of filter.py as input.
Output is two graphs and table with regression coefficients 
"""

solver = 'newton-cg'
regul_coeffs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]


######################################################################################
# with call columns

data = pd.read_csv('filtered_data_with_call.csv', index_col=0)
index = data.index
columns = data.columns

X = data.drop('class', axis=1).values
y = data['class'].values

# split into train_test and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44) # 80 / 20 split
scaler = StandardScaler().fit(X_train) 
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X = scaler.transform(X)

# create naive estimate
y_naive_test = np.zeros(shape=(len(y_test), 3)) 
class_counts = np.unique(y_train, return_counts=True)[1]
train_len = len(y_train)
y_naive_test[:, 0] += class_counts[0] / train_len # b'HER2+' class
y_naive_test[:, 1] += class_counts[1] / train_len # b'HR+' class
y_naive_test[:, 2] += class_counts[2] / train_len # b'TN' class
test_naive_error = log_loss(y_test, y_naive_test, 1e-15)

# cross-validation
train_errors = pd.Series(index=regul_coeffs)
test_errors = pd.Series(index=regul_coeffs)
test_naive_errors = pd.Series(index=regul_coeffs)

for regul_coeff in regul_coeffs:
    logistic_regression = LogisticRegression(C=regul_coeff, multi_class='multinomial', penalty='l2', solver=solver)
    logistic_regression.fit(X_train, y_train)
    y_pred_train = logistic_regression.predict_proba(X_train)
    y_pred_test = logistic_regression.predict_proba(X_test)
    
    train_errors[regul_coeff] = log_loss(y_train, y_pred_train, 1e-15)
    test_errors[regul_coeff] = log_loss(y_test, y_pred_test, 1e-15)
    test_naive_errors[regul_coeff] = test_naive_error

# plot
plt.figure()
plt.plot(train_errors)
plt.plot(test_errors)
plt.plot(test_naive_errors)
plt.xlabel('regularization coefficients (log-scale)')
plt.ylabel('log loss (log-scale)')
plt.xscale('log')
plt.yscale('log')
plt.legend(('train_error', 'test_error', 'naive_test_error'))
plt.savefig('errors_call.png')


######################################################################################
# without call columns

data = pd.read_csv('filtered_data.csv', index_col=0)
index = data.index
columns = data.columns

X = data.drop('class', axis=1).values
y = data['class'].values

# split into train_test and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44) # 80 / 20 split
scaler = StandardScaler().fit(X_train) 
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X = scaler.transform(X)

# create naive estimate
y_naive_test = np.zeros(shape=(len(y_test), 3)) 
class_counts = np.unique(y_train, return_counts=True)[1]
train_len = len(y_train)
y_naive_test[:, 0] += class_counts[0] / train_len # b'HER2+' class
y_naive_test[:, 1] += class_counts[1] / train_len # b'HR+' class
y_naive_test[:, 2] += class_counts[2] / train_len # b'TN' class
test_naive_error = log_loss(y_test, y_naive_test, 1e-15)

# cross-validation
train_errors = pd.Series(index=regul_coeffs)
test_errors = pd.Series(index=regul_coeffs)
test_naive_errors = pd.Series(index=regul_coeffs)

for regul_coeff in regul_coeffs:
    logistic_regression = LogisticRegression(C=regul_coeff, multi_class='multinomial', penalty='l2', solver=solver)
    logistic_regression.fit(X_train, y_train)
    y_pred_train = logistic_regression.predict_proba(X_train)
    y_pred_test = logistic_regression.predict_proba(X_test)
    
    train_errors[regul_coeff] = log_loss(y_train, y_pred_train, 1e-15)
    test_errors[regul_coeff] = log_loss(y_test, y_pred_test, 1e-15)
    test_naive_errors[regul_coeff] = test_naive_error

# plot
plt.figure()
plt.plot(train_errors)
plt.plot(test_errors)
plt.plot(test_naive_errors)
plt.xlabel('regularization coefficients (log-scale)')
plt.ylabel('log loss (log-scale)')
plt.xscale('log')
plt.yscale('log')
plt.legend(('train_error', 'test_error', 'naive_test_error'))
plt.savefig('errors.png')

######################################################################################
# continue without call columns (based on graph inspection)

# cross-validation on zoomed in domain
regul_coeffs_zoom = list(np.arange(0.0001, 0.01, 0.0001))
test_errors = pd.Series(index=regul_coeffs_zoom)
regr_coeffs = dict.fromkeys(regul_coeffs_zoom, 0)

for regul_coeff in regul_coeffs_zoom:
    logistic_regression = LogisticRegression(C=regul_coeff, multi_class='multinomial', penalty='l2', solver=solver) 
    logistic_regression.fit(X_train, y_train)
    y_pred_test = logistic_regression.predict_proba(X_test)
    print(regul_coeff)
    test_errors[regul_coeff] = log_loss(y_test, y_pred_test, 1e-15)
    regr_coeffs[regul_coeff] = logistic_regression.coef_
    print('coeff: ', regul_coeff)
    print('')

opt_reg_coeff = test_errors.idxmin()   

# print optimal regularization coefficient and associated test error
print(opt_reg_coeff)
print(test_errors[opt_reg_coeff])

# write estimation output to csv file
table = pd.DataFrame(regr_coeffs[opt_reg_coeff], index=['bHER2+', 'bHR+', 'bTN'], columns=columns[0:20])
table.to_csv('regression_output.csv')