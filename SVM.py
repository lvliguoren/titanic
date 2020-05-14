from preprocessing import get_data
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 绘制精度和召回率
def plot_precision_recall_vs_threshold(model, X, y):
    y_scores = cross_val_predict(model, X, y, cv=5, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
    plt.show()


# 绘制学习曲线
def plot_learning_curve(model, X, y):
    train_sizes = np.linspace(0.1, 1.0, endpoint=True, dtype='float')
    thresholds,train_scores,test_scores = learning_curve(model,X,y,train_sizes=train_sizes,scoring="accuracy")
    train_scores_mean = np.mean(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.plot(thresholds,train_scores_mean,"b--", label="Train Accuracy")
    plt.plot(thresholds,test_scores_mean,"g-", label="Test Accuracy")
    plt.xlabel("Sample Nums")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()

train_df = get_data("train.csv")
train_f = train_df.filter(regex='Survived|SibSp|Parch|Sex_.*|Pclass_.*|Age_.*|Cabin_.*|Name_.*')
train_np = train_f.values
train_X = train_np[:,1:]
train_y = train_np[:,0]

test_df = get_data("test.csv")
test_f = test_df.filter(regex='SibSp|Parch|Sex_.*|Pclass_.*|Age_.*|Cabin_.*|Name_.*')
test_X = test_f.values

rbf_kernel_svm_clf = Pipeline([
    # ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma="auto",C=10))
])

# plot_learning_curve(rbf_kernel_svm_clf, train_X, train_y)
# print(cross_val_score(rbf_kernel_svm_clf, train_X, train_y, cv=10))
# plot_precision_recall_vs_threshold(rbf_kernel_svm_clf, train_X, train_y)

# 对测试集做出预测
rbf_kernel_svm_clf.fit(train_X, train_y)
predictions = rbf_kernel_svm_clf.predict(test_X)
result = pd.DataFrame({'PassengerId':test_df['PassengerId'].values, 'Survived':predictions})
result.to_csv("data/logistic_regression_predictions.csv", index=False)