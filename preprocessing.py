import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing

def  get_data(file_name):
    #显示所有列
    pd.set_option('display.max_columns', None)

    data_train = pd.read_csv("data/" + file_name)
    data_train = data_train[np.isnan(data_train.Age)==False]
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

    data_train = pd.concat([data_train, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    #数据归一化
    scaler = preprocessing.StandardScaler()
    data_train['Age_scaled'] = scaler.fit_transform(data_train['Age'].values.reshape(-1,1))
    data_train['Fare_scaled'] = scaler.fit_transform(data_train['Fare'].values.reshape(-1,1))

    data_train.drop(columns=["PassengerId","Pclass","Age","Fare","Sex","Embarked","Cabin","Name","Ticket"], inplace=True)

    return data_train