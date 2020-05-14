import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def get_data(file_name):
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # # 显示所有行
    # pd.set_option('display.max_rows', None)

    data_pre = pd.read_csv("data/" + file_name)
    data_pre.loc[data_pre.Fare.isnull(),"Fare"] = np.mean(data_pre.Fare)
    data_pre = set_missing_age(data_pre)
    for i in range(len(data_pre)):
        if data_pre.loc[i,"Age"] <= 8.38:
            data_pre.loc[i, "Age"] = 0
        elif data_pre.loc[i,"Age"] > 8.38 and data_pre.loc[i,"Age"] <= 16.34:
            data_pre.loc[i, "Age"] = 1
        elif data_pre.loc[i, "Age"] > 16.34and data_pre.loc[i, "Age"] <= 24.29:
            data_pre.loc[i, "Age"] = 2
        elif data_pre.loc[i, "Age"] > 24.29 and data_pre.loc[i, "Age"] <= 32.25:
            data_pre.loc[i, "Age"] = 3
        elif data_pre.loc[i, "Age"] > 32.25 and data_pre.loc[i, "Age"] <= 40.21:
            data_pre.loc[i, "Age"] = 4
        elif data_pre.loc[i, "Age"] > 40.21 and data_pre.loc[i, "Age"] <= 48.17:
            data_pre.loc[i, "Age"] = 5
        elif data_pre.loc[i, "Age"] > 48.17 and data_pre.loc[i, "Age"] <= 56.13:
            data_pre.loc[i, "Age"] = 6
        elif data_pre.loc[i, "Age"] > 56.13 and data_pre.loc[i, "Age"] <= 64.28:
            data_pre.loc[i, "Age"] = 7
        elif data_pre.loc[i, "Age"] > 64.28:
            data_pre.loc[i, "Age"] = 8

    data_pre.Cabin.fillna("0", inplace=True)
    data_pre.loc[data_pre.Cabin.str[0] == 'A', 'Cabin'] = 1
    data_pre.loc[data_pre.Cabin.str[0] == 'B', 'Cabin'] = 2
    data_pre.loc[data_pre.Cabin.str[0] == 'C', 'Cabin'] = 3
    data_pre.loc[data_pre.Cabin.str[0] == 'D', 'Cabin'] = 4
    data_pre.loc[data_pre.Cabin.str[0] == 'E', 'Cabin'] = 5
    data_pre.loc[data_pre.Cabin.str[0] == 'F', 'Cabin'] = 6
    data_pre.loc[data_pre.Cabin.str[0] == 'G', 'Cabin'] = 7
    data_pre.loc[data_pre.Cabin.str[0] == 'T', 'Cabin'] = 8
    data_pre.loc[data_pre.Fare <= 51.23,"Fare"] = 51.0
    data_pre.loc[(data_pre.Fare <= 102.47) & (data_pre.Fare > 51.23),"Fare"] = 102.0
    data_pre.loc[data_pre.Fare > 102.48 ,"Fare"] = 150.0
    data_pre.loc[data_pre.Name.str.contains("Mrs"), "Name"] = "1"
    data_pre.loc[data_pre.Name.str.contains("Mr"), "Name"] = "0"
    data_pre.loc[data_pre.Name.str.contains("Miss"), "Name"] = "2"
    data_pre.loc[data_pre.Name.str.contains("Master"), "Name"] = "3"
    data_pre.loc[(data_pre.Name!= "0") & (data_pre.Name!= "1") & (data_pre.Name!= "2") & (data_pre.Name!= "3"), "Name"] = "4"
    dummies_Name = pd.get_dummies(data_pre["Name"], prefix="Name")
    dummies_Cabin = pd.get_dummies(data_pre["Cabin"],prefix="Cabin")
    dummies_Embarked = pd.get_dummies(data_pre["Embarked"], prefix= "Embarked")
    dummies_Sex = pd.get_dummies(data_pre["Sex"], prefix= "Sex")
    dummies_Pclass = pd.get_dummies(data_pre["Pclass"], prefix= "Pclass")
    dummies_Age = pd.get_dummies(data_pre["Age"], prefix="Age")
    dummies_Fare = pd.get_dummies(data_pre["Fare"], prefix="Fare")

    data_pre = pd.concat([data_pre, dummies_Embarked, dummies_Sex, dummies_Pclass, dummies_Age, dummies_Cabin, dummies_Fare, dummies_Name], axis=1)
    data_pre.drop(columns=["Pclass","Sex","Embarked","Cabin","Name","Ticket","Age", "Fare"], inplace=True)

    return data_pre

def set_missing_age(data):
    data_age = data[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = data_age[data_age.Age.notnull()].values
    unknown_age = data_age[data_age.Age.isnull()].values

    train_X = known_age[:,1:]
    train_y = known_age[:,0]

    predict_X = unknown_age[:,1:]
    rfr = RandomForestRegressor(random_state=42)

    rfr.fit(train_X, train_y)

    predict_y = rfr.predict(predict_X)

    data.loc[data.Age.isnull(),"Age"] = predict_y

    return data