import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

def confusion_matrix_scorer(clf, X, y):
     y_pred = clf.predict(X)
     cm = confusion_matrix(y, y_pred)
     return cm[0,0], cm[0,1], cm[1,0], cm[1,1]

def plot_conf_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

def exec_time(algo, X_val, cnt=1000):
    min = 1000
    max = 0
    total = 0
    for i in range(cnt):
        start = time.time()
        algo.predict(X_val)
        end = time.time()
        ex_tm = end - start
        if ex_tm > max:
            max = ex_tm
        if ex_tm < min:
            min = ex_tm
        total = total + ex_tm
    med = total / cnt
    return med, max, min


def algor(algo, X_train, y_train, X_test, y_test, name, std, col, features=None):
    algo.fit(X_train, y_train)

    acc_train = algo.score(X_train, y_train)

    tn, fp, fn, tp = confusion_matrix_scorer(algo, X_test, y_test)
    acc = (tp + tn) / (tp + tn + fn + fp)
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    if not std:
        X_val = X_test.iloc[[0]]
        X_val = [X_val.values[0]]
    else:
        X_val = [X_test[0,:]]
    med, max, min = exec_time(algo, X_val)

    data = [name, std, acc_train, acc, prec, recall, tn, fp, fn, tp, med, max, min]
    if features:
        for feature in features:
            data.append(feature)

    df = pd.DataFrame([data], columns=col)
    return df

def test_several(X_train, y_train, X_test, y_test, X_train_std, X_test_std):
    column_names = ['Algorithm', 'Standarised', 'AccTrain', 'Acc', 'Prec', 'Recall', 'TN', 'FP', 'FN', 'TP', 'TimeMed', 'TimeMax', 'TimeMin']
    alg_rs = pd.DataFrame(columns=column_names)
    
    logreg = LogisticRegression(class_weight="balanced")
    alg_rs = alg_rs.append(algor(logreg, X_train, y_train, X_test, y_test, 'LogisticRegression', False, column_names))
    alg_rs = alg_rs.append(algor(logreg, X_train_std, y_train, X_test_std, y_test, 'LogisticRegression', True, column_names))

    dt = DecisionTreeClassifier(class_weight="balanced").fit(X_train, y_train)
    alg_rs = alg_rs.append(algor(dt, X_train, y_train, X_test, y_test, 'DecisionTree', False, column_names))
    alg_rs = alg_rs.append(algor(dt, X_train_std, y_train, X_test_std, y_test, 'DecisionTree', True, column_names))

    rdf = RandomForestClassifier(class_weight="balanced").fit(X_train, y_train)
    alg_rs = alg_rs.append(algor(rdf, X_train, y_train, X_test, y_test, 'RandomForest', False, column_names))
    alg_rs = alg_rs.append(algor(rdf, X_train_std, y_train, X_test_std, y_test, 'RandomForest', True, column_names))

    svm = SVC(class_weight="balanced")
    alg_rs = alg_rs.append(algor(svm, X_train, y_train, X_test, y_test, 'SVC', False, column_names))
    alg_rs = alg_rs.append(algor(svm, X_train_std, y_train, X_test_std, y_test, 'SVC', True, column_names))
    
    gnb = GaussianNB()
    alg_rs = alg_rs.append(algor(gnb, X_train, y_train, X_test, y_test, 'NaiveBayes', False, column_names))
    alg_rs = alg_rs.append(algor(gnb, X_train_std, y_train, X_test_std, y_test, 'NaiveBayes', True, column_names))

    knn = KNeighborsClassifier()
    alg_rs = alg_rs.append(algor(knn, X_train, y_train, X_test, y_test, 'KNN', False, column_names))
    alg_rs = alg_rs.append(algor(knn, X_train_std, y_train, X_test_std, y_test, 'KNN', True, column_names))

    lda = LinearDiscriminantAnalysis()
    alg_rs = alg_rs.append(algor(lda, X_train, y_train, X_test, y_test, 'LDA', False, column_names))
    alg_rs = alg_rs.append(algor(lda, X_train_std, y_train, X_test_std, y_test, 'LDA', True, column_names))
    
    nn = MLPClassifier(solver = 'lbfgs', max_iter = 2000, random_state=1)
    alg_rs = alg_rs.append(algor(nn, X_train, y_train, X_test, y_test, 'NN-MLP', False, column_names))
    alg_rs = alg_rs.append(algor(nn, X_train_std, y_train, X_test_std, y_test, 'NN-MLP', True, column_names))

    return alg_rs



def test_dec_tree(X_train, y_train, X_test, y_test, X_train_std, X_test_std):
    column_names = ['Algorithm', 'Standarised', 'AccTrain', 'Acc', 'Prec', 'Recall', 'TN', 'FP', 'FN', 'TP', 'TimeMed', 'TimeMax', 'TimeMin', 'max_leaf_node']
    alg_rs = pd.DataFrame(columns=column_names)

    for leafs in range(2,50):
        dt = DecisionTreeClassifier(class_weight="balanced", max_leaf_nodes=leafs).fit(X_train, y_train)
        alg_rs = alg_rs.append(algor(dt, X_train, y_train, X_test, y_test, 'DecisionTree', False, column_names, [leafs]))
        alg_rs = alg_rs.append(algor(dt, X_train_std, y_train, X_test_std, y_test, 'DecisionTree', True, column_names, [leafs]))
    
    for leafs in range(2,50):
        rdf = RandomForestClassifier(class_weight="balanced").fit(X_train, y_train)
        alg_rs = alg_rs.append(algor(rdf, X_train, y_train, X_test, y_test, 'RandomForest', False, column_names, [leafs]))
        alg_rs = alg_rs.append(algor(rdf, X_train_std, y_train, X_test_std, y_test, 'RandomForest', True, column_names, [leafs]))
    
    return alg_rs

def test_MLP(X_train, y_train, X_test, y_test, X_train_std, X_test_std):
    column_names = ['Algorithm', 'Standarised', 'AccTrain', 'Acc', 'Prec', 'Recall', 'TN', 'FP', 'FN', 'TP', 'TimeMed', 'TimeMax', 'TimeMin', 'alpha', 'hidden_layer_sizes']
    alg_rs = pd.DataFrame(columns=column_names)
    
    for alp in 10.0 ** -np.arange(1, 10):
        for hid_lay in [10, 20, 50, 70, 100, 150, 200]:
            nn = MLPClassifier(hidden_layer_sizes= (hid_lay, ), solver = 'lbfgs', alpha = alp, max_iter = 2000, random_state=1)
            alg_rs = alg_rs.append(algor(nn, X_train_std, y_train, X_test_std, y_test, 'NN-MLP', True, column_names, [alp, hid_lay]))
    return alg_rs


def test_algorithms(X, y, func):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    nan_elems = y_test.isnull()
    y_test = y_test[~nan_elems]
    X_test = X_test[~nan_elems]
    y_train = y_train.astype(bool)
    y_test = y_test.astype(bool)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    result = func(X_train, y_train, X_test, y_test, X_train_std, X_test_std)
 
    return result


csv_path = '../data/ACC.csv'
df_acc = pd.read_csv(csv_path)
df_acc = df_acc.drop([ 'ID', 'scenario', 'seed', 'Controller', 'numHumanCars', 'BeaconIntervalPlat', 'BeaconIntervalHuman', 'PacketSize', 'waitTime', 'DENMInterval'], axis=1)
df_acc = df_acc.dropna()
df_acc = df_acc.drop_duplicates()

csv_path = '../data/CACC_PLOEG.csv'
df = pd.read_csv(csv_path)
index_cacc = df.index[df['Controller'] != "CACC"].tolist()
df_cacc = df.drop(index_cacc)
index_ploeg = df.index[df['Controller'] != "PLOEG"].tolist()
df_ploeg = df.drop(index_ploeg)

df_cacc_ploeg = df.drop([ 'ID', 'scenario', 'seed'], axis=1)
df_cacc_ploeg = df_cacc_ploeg.dropna()
df_cacc_ploeg = df_cacc_ploeg.drop_duplicates()

df_cacc = df_cacc.drop([ 'ID', 'scenario', 'seed', 'Controller'], axis=1)
df_cacc = df_cacc.dropna()
df_cacc = df_cacc.drop_duplicates()

df_ploeg = df_ploeg.drop([ 'ID', 'scenario', 'seed', 'Controller'], axis=1)
df_ploeg = df_ploeg.dropna()
df_ploeg = df_ploeg.drop_duplicates()

f_names_acc = ['numPlatCars', 'spdLeader', 'DecelerationRate', 'Ctr_Param(Ploegh||Spacing)']
X_acc = df_acc[f_names_acc]
y_acc = df_acc['Collision']

f_names = ['numPlatCars', 'numHumanCars', 'spdLeader', 'BeaconIntervalPlat', 'BeaconIntervalHuman', 'DENMInterval', 'DecelerationRate', 'PacketSize', 'Ctr_Param(Ploegh||Spacing)', 'waitTime']
X_cacc = df_cacc[f_names]
y_cacc = df_cacc['Collision']

X_ploeg = df_ploeg[f_names]
y_ploeg = df_ploeg['Collision']

f_names_all = ['Controller', 'numPlatCars', 'numHumanCars', 'spdLeader', 'BeaconIntervalPlat', 'BeaconIntervalHuman', 'DENMInterval', 'DecelerationRate', 'PacketSize', 'Ctr_Param(Ploegh||Spacing)', 'waitTime']
X_cacc_ploeg = df_cacc_ploeg[f_names_all]
y_cacc_ploeg = df_cacc_ploeg['Collision']


rs_acc = test_algorithms(X_acc, y_acc, test_several)
rs_acc.to_excel('../results/resultsACC.xlsx')

rs_cacc = test_algorithms(X_cacc, y_cacc, test_several)
rs_cacc.to_excel('../results/resultsCACC.xlsx')

rs_ploeg = test_algorithms(X_ploeg, y_ploeg, test_several)
rs_ploeg.to_excel('../results/resultsPLOEG.xlsx')

#rs_cacc_ploeg = test_algorithms(X_cacc_ploeg, y_cacc_ploeg, test_several)
#rs_cacc_ploeg.to_excel('../results/resultsCACC_PLOEG.xlsx')


rs_acc = test_algorithms(X_acc, y_acc, test_dec_tree)
rs_acc.to_excel('../results/resultsACC_dt.xlsx')

rs_cacc = test_algorithms(X_cacc, y_cacc, test_dec_tree)
rs_cacc.to_excel('../results/resultsCACC_dt.xlsx')

rs_ploeg = test_algorithms(X_ploeg, y_ploeg, test_dec_tree)
rs_ploeg.to_excel('../results/resultsPLOEG_dt.xlsx')

#rs_cacc_ploeg = test_algorithms(X_cacc_ploeg, y_cacc_ploeg, test_dec_tree)
#rs_cacc_ploeg.to_excel('../results/resultsCACC_PLOEG_dt.xlsx')


rs_acc = test_algorithms(X_acc, y_acc, test_MLP)
rs_acc.to_excel('../results/resultsACC_nn.xlsx')

rs_cacc = test_algorithms(X_cacc, y_cacc, test_MLP)
rs_cacc.to_excel('../results/resultsCACC_nn.xlsx')

rs_ploeg = test_algorithms(X_ploeg, y_ploeg, test_MLP)
rs_ploeg.to_excel('../results/resultsPLOEG_nn.xlsx')

#rs_cacc_ploeg = test_algorithms(X_cacc_ploeg, y_cacc_ploeg, test_MLP)
#rs_cacc_ploeg.to_excel('../results/resultsCACC_PLOEG_nn.xlsx')