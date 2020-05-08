import pandas as pd
import numpy as np

def  get_data(file_name):
    #显示所有列
    pd.set_option('display.max_columns', None)

    data_pre = pd.read_csv("data/" + file_name)
    data_pre = data_pre[np.isnan(data_pre.Age)==False]
    dummies_Embarked = pd.get_dummies(data_pre['Embarked'], prefix= 'Embarked')
    dummies_Sex = pd.get_dummies(data_pre['Sex'], prefix= 'Sex')
    dummies_Pclass = pd.get_dummies(data_pre['Pclass'], prefix= 'Pclass')

    data_pre = pd.concat([data_pre, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    data_pre.drop(columns=["PassengerId","Pclass","Sex","Embarked","Cabin","Name","Ticket"], inplace=True)

    return data_pre