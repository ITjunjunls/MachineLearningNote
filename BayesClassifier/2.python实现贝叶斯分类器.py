import pandas as pd
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
    x_train = data.iloc[:,:-1]
    y_train = data.iloc[:,-1]
    x_test = data.iloc[40000:,:-1]
    y_test = data.iloc[40000:,-1]
    return x_train,x_test,y_train,y_test

#计算正负样本的所有概率（条件概率，正负样本的概率）
def cal_p(train_x,train_y):
    all_dict = {"yes":{},"no":{}}
    data_yes = train_x.loc[train_y==1]#获取所有正样本数据
    data_no = train_x.loc[train_y==0]#获取所有负样本数据
    all_dict["yes"]["p"] = len(data_yes)/len(train_x)#获取正样本的概率
    all_dict["no"]["p"] = len(data_no)/len(train_x)#获取负样本的概率
    for key in data_yes.keys():
        #计算正样本的条件概率
        all_dict["yes"][key] = data_yes.groupby(by=key).size()/len(data_yes)
        #计算负样本的条件概率
        all_dict["no"][key] = data_no.groupby(by=key).size()/len(data_no)
    return all_dict

#测试数据
def bayes_test(test_x,all_dict):
    keys = test_x.keys()
    pred_y=[]
    #计算条件概率的值
    for i in range(test_x.shape[0]):
        p_yes = 1
        p_no = 1
        for j in range(test_x.shape[1]):
            if test_x.iloc[i,j] not in all_dict["yes"][keys[j]]:#在正样本概率集合中，找不到该分类
                p_yes = 0
            else:
                p_yes *= all_dict["yes"][keys[j]][test_x.iloc[i,j]]

            if test_x.iloc[i,j] not in all_dict["no"][keys[j]]:#在负样本概率集合中，找不到该分类
                p_no = 0
            else:
                p_no *= all_dict["no"][keys[j]][test_x.iloc[i,j]]
        p_yes *= all_dict["yes"]["p"]
        p_no  *= all_dict["no"]["p"]
        #判断p_yes和p_no的大小
        if p_yes >= p_no:
            pred_y.append(1)
        else:
            pred_y.append(0)
    return pred_y

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

if __name__ == "__main__":
    x_train,x_test,y_train,y_test=process_data()#读取数据
    all_dict = cal_p(x_train,y_train)#计算正负样本的所有概率（条件概率，正负样本的概率）
    pred_y = bayes_test(x_test,all_dict)#测试
    score(y_test,pred_y)#准确性评价指标
