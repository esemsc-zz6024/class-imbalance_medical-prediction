# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score,accuracy_score,roc_auc_score,precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler


import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek

import warnings
warnings.filterwarnings("ignore")

def main():
    p1 = r'C:/Users/12769/Desktop/datasets/AKI_str_test.csv'
    p2 = r'C:/Users/12769/Desktop/datasets/AKI_str_train.csv'
    p3 = r'C:/Users/12769/Desktop/datasets/AKI_str_val.csv'
    df = pd.read_csv(p1)
    df2 = pd.read_csv(p2)
    df3 = pd.read_csv(p3)
    df = pd.concat([df,df2],axis=0)
    df = pd.concat([df,df3],axis=0)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

   #归一化处理
    transfer=MinMaxScaler(feature_range=(0,1))
    X=transfer.fit_transform(X)

    logistic(X,y)

def logistic(X,y):
    k = 5
    kf = KFold(n_splits=k, shuffle=True)
    precs = []  # 精准度
    recalls = []  # 召回率
    accurs = []  # 准确率
    f1s = []  # f1值
    aucs = []  # auc
    for train, test in kf.split(X, y):
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]

        # #随机欠采样法
        # rus=RandomUnderSampler()
        # x_train_b, y_train_b=rus.fit_resample(X_train,y_train)

        # #聚类质心法
        # cc=ClusterCentroids()
        # x_train_b, y_train_b=cc.fit_resample(X_train,y_train)

        # # #ENN
        # enn=EditedNearestNeighbours()
        # x_train_b, y_train_b=enn.fit_resample(X_train,y_train)


        # #合成少数类过采样法
        # smt = SMOTE()
        # x_train_b, y_train_b = smt.fit_resample(X_train, y_train)

        # #SVMSMOTE
        # smt2=SVMSMOTE()
        # x_train_b,y_train_b=smt2.fit_resample(X_train,y_train)

        # #BorderSMOTE
        # smt1=BorderlineSMOTE()
        # x_train_b,y_train_b=smt1.fit_resample(X_train,y_train)

        # #adasyn
        # ada=ADASYN()
        # x_train_b, y_train_b=ada.fit_resample(X_train,y_train)



        # #SMOTE结合Tomek
        # smtk=SMOTETomek()
        # x_train_b, y_train_b=smtk.fit_resample(X_train, y_train)

        model = LogisticRegression()
        model.fit(X_train ,y_train)
        #     y_train_pred = model.predict(x_train_b)
        y_test_pred = model.predict(X_test)
        probas=model.predict_proba(X_test)
        #     print(y_test_pred[:10],y_test[:10])
        score = model.score(X_test, y_test)
        print('测试集精度:', score)
        test_prec = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_accur = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)
        precs.append(test_prec)
        recalls.append(test_recall)
        accurs.append(test_accur)
        f1s.append(test_f1)
        aucs.append(test_auc)
        print(confusion_matrix(y_test, y_test_pred))
        print('准确率:', test_accur)
        print('精准度:', test_prec)
        print('召回率:', test_recall)
        print('f1值:', test_f1)
        print('auc', test_auc)
        print('==================')
        prec, recall, thresholds = precision_recall_curve(y_test, probas[:, 1])
        plt.plot(recall, prec)
        plt.title('Precision Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()


    print('5折平均精准度:', np.mean(precs))
    print('5折平均召回率:', np.mean(recalls))
    print('5折平均准确率:', np.mean(accurs))
    print('5折平均f1值:  ', np.mean(f1s))
    print('5折平均AUC值: ', np.mean(aucs))




if __name__ == '__main__':
    main()

