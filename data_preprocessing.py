import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl

#显示所有列
pd.set_option('display.max_columns', None)

data_train = pd.read_csv("data/train.csv")
data_train = data_train[np.isnan(data_train.Age)==False]
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

data_train = pd.concat([data_train, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
data_train.drop(columns=["Pclass","Sex","Embarked","Cabin","Name","Ticket"], inplace=True)
print(data_train)