import pandas as pd
import numpy as np

def  get_data(file_name):
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    # pd.set_option('display.max_rows', None)

    data_pre = pd.read_csv("data/" + file_name)
    data_pre.drop(index=(data_pre[data_pre.Age.isnull()].index), inplace=True)
    # Drop之后要重新建立索引，不然遍历会有问题
    data_pre.reset_index(drop=True, inplace=True)
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

    data_pre.loc[data_pre.Cabin.notnull(),"Cabin"] = "Yes"
    data_pre.loc[data_pre.Cabin.isnull(),"Cabin"] = "No"
    data_pre.loc[data_pre.Fare <= 51.23,"Fare"] = 51.0
    data_pre.loc[(data_pre.Fare <= 102.47) & (data_pre.Fare > 51.23),"Fare"] = 102.0
    data_pre.loc[data_pre.Fare > 102.48 ,"Fare"] = 150.0
    dummies_Cabin = pd.get_dummies(data_pre["Cabin"],prefix="Cabin")
    dummies_Embarked = pd.get_dummies(data_pre["Embarked"], prefix= "Embarked")
    dummies_Sex = pd.get_dummies(data_pre["Sex"], prefix= "Sex")
    dummies_Pclass = pd.get_dummies(data_pre["Pclass"], prefix= "Pclass")
    dummies_Age = pd.get_dummies(data_pre["Age"], prefix="Age")
    dummies_Fare = pd.get_dummies(data_pre["Fare"], prefix="Fare")

    data_pre = pd.concat([data_pre, dummies_Embarked, dummies_Sex, dummies_Pclass, dummies_Age, dummies_Cabin, dummies_Fare], axis=1)
    data_pre.drop(columns=["PassengerId","Pclass","Sex","Embarked","Cabin","Name","Ticket","Age", "Fare"], inplace=True)

    return data_pre