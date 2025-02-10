import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score,accuracy_score,roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
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

    # 归一化处理
    transfer = MinMaxScaler(feature_range=(0, 1))
    X = transfer.fit_transform(X)



    GBDT(X,y)

def GBDT(X,y):
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

        # #随机欠采样法  max_depth=7
        # rus=RandomUnderSampler()
        # x_train_b, y_train_b=rus.fit_resample(X_train,y_train)


        # #ENN  max_depth=7
        # enn=EditedNearestNeighbours()
        # x_train_b, y_train_b=enn.fit_resample(X_train,y_train)

        # #TomekLinks
        # t1=TomekLinks()
        # x_train_b, y_train_b=t1.fit_resample(X_train,y_train)


        # #合成少数类过采样法 max_depth=1
        # smt = SMOTE()
        # x_train_b, y_train_b = smt.fit_resample(X_train, y_train)

        # #BorderSMOTE max_depth=1
        # smt1=BorderlineSMOTE()
        # x_train_b,y_train_b=smt1.fit_resample(X_train,y_train)

        # #SVMSMOTE max_depth=1
        # smt2=SVMSMOTE()
        # x_train_b,y_train_b=smt2.fit_resample(X_train,y_train)

        # #adasyn max_depth=1
        # ada=ADASYN()
        # x_train_b, y_train_b=ada.fit_resample(X_train,y_train)

        #SMOTE结合Tomek max_depth=1
        smtk=SMOTETomek()
        x_train_b, y_train_b=smtk.fit_resample(X_train, y_train)

        model = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=1,
                               min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                              objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
        # param={
        #     'n_estimators':range(30,90,10),
        # }
        # gs=GridSearchCV(model,param,cv=3,scoring='roc_auc')
        # gs.fit(X_train,y_train)
        # print('最好参数:{},最好分数:{}'.format(gs.best_params_,gs.best_score_))
        model.fit(x_train_b, y_train_b)
        #     y_train_pred = model.predict(x_train_b)
        y_test_pred = model.predict(X_test)
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
    print('5折平均精准度:', np.mean(precs))
    print('5折平均召回率:', np.mean(recalls))
    print('5折平均准确率:', np.mean(accurs))
    print('5折平均f1值:  ', np.mean(f1s))
    print('5折平均AUC值: ', np.mean(aucs))



if __name__ == '__main__':
    main()
