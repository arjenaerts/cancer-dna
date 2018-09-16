import pandas as pd
from sklearn import preprocessing
from scipy.io.arff import loadarff 

"""
Script that loads, explores and cleans the DNA data. 
Output is used by filter.py.
"""

# load data
raw_data = loadarff('BreastCancerAll.original.arff')
data = pd.DataFrame(raw_data[0])

# inspect data
print(data.head(n=100))

# check for missing data
print(data.isnull().values.any())

# compute number of observations per class
print(data.groupby('class').count())

# inspect data types
print(data.dtypes)

# inspect normalization for first reg sequence
print(data['1_chrom1_reg2927-43870_probloss'].values + data['2_chrom1_reg2927-43870_probnorm'].values + data['3_chrom1_reg2927-43870_probgain'].values)

# convert class variable type into int labels
data['class'] = data['class'].astype(str)
le = preprocessing.LabelEncoder()
le.fit(data['class'])
data['class'] = le.transform(data['class'])

# convert non-float featurs' type into int labels
call_cols = data.filter(regex='call').columns
for col in call_cols:
    data[col] = data[col].astype(str)
    le = preprocessing.LabelEncoder()
    le.fit(data[col])
    data[col] = le.transform(data[col])

# col_list = data.filter(regex='probnorm').columns
# reg_list = col_list.str.split('_').str[1] + '_' + reg_list_cols.str.split('_').str[2]
# for reg in reg_list:
#     cols = new_data.columns[new_data.columns.str.contains(reg)]
#     norm = new_data[cols].sum(axis=1)
#     new_data[cols] = new_data[cols].div(norm, axis=0)

# write to file
data.to_csv('cleaned_data.csv')