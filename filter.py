import numpy as np
import pandas as pd
import mifs
import time

"""
Script that filters the DNA data using Joint Mutual Information. 
Requires the output of load_explore_clean.py as input.
Output (dataframe with only the filtered columns) is used by train.py.
"""

data = pd.read_csv('cleaned_data.csv', index_col=0)
index = data.index
columns = data.columns

# load X and y
X = data.drop('class', axis=1).values
y = data['class'].values

# define MI_FS feature selection method
feat_selector = mifs.MutualInformationFeatureSelector(n_features=20)

# find all relevant features and time this
tic = time.clock()
feat_selector.fit(X, y)
toc = time.clock()
print(toc - tic)

# check selected features
print(feat_selector.support_)

# check ranking of features
print(feat_selector.ranking_)

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)

# write to file
y = y.reshape(y.shape[0],1)
column_selection = np.append(feat_selector.support_, True)
data_filtered = pd.DataFrame(np.concatenate((X_filtered,y), axis=1), index=index, columns=columns[column_selection])
print(data_filtered.head())
data_filtered.to_csv('filtered_data_2.csv')