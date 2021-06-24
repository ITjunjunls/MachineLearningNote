import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split#分割训练集和测试集
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.preprocessing import LabelEncoder
#读取数据集，并进行清洗，处理
def process_data():
    #读取数据
    data = pd.read_csv("../data/bank-full.csv",delimiter=";")
    #删除空值
    data = data.dropna()
    #获取keys
    keys = data.keys()
    keys = keys.drop(["day","month","pdays"])
    #将文本数据转换为数字类别
    data = data[keys].apply(LabelEncoder().fit_transform)
    X = data.iloc[:,:-1]#特征集
    y = data.iloc[:,-1]#标签集
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)#分割训练集和测试集
    return x_train,x_test,y_train,y_test

from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_curve
import matplotlib.pyplot as plt
#评价指标
def score(y_test,y_pred):
    print("准确率：",accuracy_score(y_test,y_pred))
    print("精确率：",precision_score(y_test,y_pred))
    print("召回率：",recall_score(y_test,y_pred))

    #ROC曲线
    fpr,tpr,thresholds = roc_curve(y_test,y_pred)
    fig = plt.figure(figsize=(10,6))
    plt.xlim(0,1)#设定x轴的范围
    plt.ylim(0,1)
    plt.xlabel("False Postive Rate")
    plt.ylabel("True Postive Rate")
    plt.plot(fpr,tpr,color="red")
    plt.show()


def GauNB(x_train,x_test,y_train,y_test):
    #建立贝叶斯分类器模型
    bayes = GaussianNB().fit(x_train,y_train)
    y_pred = bayes.predict(x_test)#预测值
    #评价指标
    score(y_test,y_pred)

#多项式朴素贝叶斯（特征值不能为负）
def MulNB(x_train,x_test,y_train,y_test):
    mul = MultinomialNB().fit(x_train,y_train)
    y_pred = mul.predict(x_test)
    #评价指标
    score(y_test, y_pred)

if __name__ == "__main__":
    #读取数据并处理数据
    x_train,x_test,y_train,y_test=process_data()
    #GauNB(x_train,x_test,y_train,y_test)
    MulNB(x_train,x_test,y_train,y_test)


