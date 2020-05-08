from preprocessing import get_data
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  cross_val_predict
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])

train_df = get_data("train.csv")
train_f = train_df.filter(regex='Survived|Age|SibSp|Parch|Fare|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_f.values
train_X = train_np[:,1:]
train_y = train_np[:,0]

rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=10))
])
# 得到预测分数
y_scores = cross_val_predict(rbf_kernel_svm_clf, train_X, train_y, cv=5, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(train_y, y_scores)

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()




