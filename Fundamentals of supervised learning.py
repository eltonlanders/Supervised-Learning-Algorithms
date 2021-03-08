# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 08:27:48 2021

@author: elton
"""

import pandas as pd
import numpy as np

#Loading and Summarizing the Titanic Dataset
df = pd.read_csv(r'C:/Users/elton/Documents/Books/Packt workshops/The Supervised Learning Workshop/The-Supervised-Learning-Workshop-master/Chapter01/Datasets/titanic.csv')

#Indexing and Selecting Data
df['Age']
df.Age #when no spaces in col name
df[['Name', 'Parch', 'Sex']]
df.iloc[0] #first row
df.iloc[[0, 1, 2]] #first 3 rows
columns = df.columns 
df[columns[1:4]]
len(df)
df.iloc[2]['Fare'] #row-centric method
df.iloc[2].Fare
df['Fare'][2] #column centric method
df.Fare[2]

#Advanced Indexing and Selection
child_passengers = df[df.Age < 21][['Name', 'Age']]
len(child_passengers)
young_adult_passengers = df.loc[(df.Age > 21) & (df.Age < 30)]
first_or_third_class = df[(df.Pclass == 3) | (df.Pclass == 1)]
not_first_or_third_class = df[~((df.Pclass == 3) | (df.Pclass == 1))]
del df['Unnamed: 0']

described_table = df.describe()
described_table_2 = df.describe(include='all')


# Splitting, Applying, and Combining Data Sources
embarked_group = df.groupby('Embarked')
len(embarked_group)

embarked_group.groups #it's a dictionary where keys are groups
#The values are the rows or indexes of the entries that belong to that group
df.iloc[1]
for name, group in embarked_group:
    print(name, group.Age.mean())
embarked_group.agg(np.mean)

def first_val(x):
    return x.values[0]
embarked_group.agg(first_val)


# Creating Lambda Functions
embarked_group = df.groupby('Embarked')
embarked_group.agg(lambda x: x.values[0])
first_mean_std = embarked_group.agg([lambda x: x.values[0], np.mean, np.std])

embarked_group.agg({ #the agg method with a dictionary of different columns
    'Fare': np.sum,
    'Age': lambda x: x.values[0]
    })

age_embarked_group = df.groupby(['Sex', 'Embarked'])
age_embarked_group.groups


# Managing Missing Data
len(df)
df.dropna()
len(df.dropna())

df.aggregate(lambda x: x.isna().sum())

df_valid = df.loc[(~df.Embarked.isna()) & (~df.Fare.isna())]
df_valid['Age'] = df_valid['Age'].fillna(df_valid.Age.mean()) #mean-imputing

df_valid.loc[df.Pclass == 1, 'Age'].mean() #imputing age by avg class age
df_valid.loc[df.Pclass == 2, 'Age'].mean()
df_valid.loc[df.Pclass == 3, 'Age'].mean()
for name, grp in df_valid.groupby(['Pclass', 'Sex']):
    print('%i' % name[0], name[1], '%0.2f' % grp['Age'].mean())

mean_ages = df_valid.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))
df_valid.loc[:, 'Age'] = mean_ages


# Class imbalance
len(df.loc[df.Survived == 1])
len(df.loc[df.Survived == 0])

#copying the first row to the end of the DataFrame
df_oversample = df.append(df.iloc[0])


# Implementing Pandas Functions
df = pd.read_csv(r'C:/Users/elton/Documents/Books/Packt workshops/The Supervised Learning Workshop/The-Supervised-Learning-Workshop-master/Chapter01/Datasets/titanic.csv')
df_described = df.describe(include='all')
df.drop('Unnamed: 0',axis=1, inplace=True)

df.mean()
df.std()
df.min()
df.max()

df.quantile(0.33)
df.quantile(0.66)
df.quantile(0.99)

class_groups = df.groupby(['Pclass'])
for name, index in class_groups:
    print(f'Class: {name}: {len(index)}')

third_class = df.loc[df.Pclass == 3]
age_max = third_class.loc[(third_class.Age == third_class.Age.max())]

fare_max = df.Fare.max()
age_max = df.Age.max()
df.agg({
 'Fare': lambda x: x / fare_max,
 'Age': lambda x: x / age_max,
}).head()

missing = df.loc[df['Fare'].isna() == True]
df_nan_fare = df.loc[(df.Fare.isna())]

embarked_class_groups = df.groupby(['Embarked', 'Pclass'])
indices = embarked_class_groups.groups[(df_nan_fare.Embarked.
values[0], df_nan_fare.Pclass.values[0])]
mean_fare = df.iloc[indices].Fare.mean()
df.loc[(df.index == 1043), 'Fare'] = mean_fare
df.iloc[1043]



"""
Notes:
1. Understand the source and type of the data, the means by which it is
   collected, and any errors potentially resulting from the collection process.
2. Any function that can take a list or a similar iterable and compute a single
   value as a result can be used with agg.
3. If you have clean data, in sufficient quantity, with a good correlation 
   between the input data type and the desired output, then the specifics regarding
   the type and details of the selected supervised learning model become significantly
   less important in achieving a good result.
4. Treating class imbalance: Oversample the under-represented class by randomly
   copying samples from the under-represented class in the dataset to boost the
   number of samples.
5. Dealing with low sample size: Transfer learning.
"""