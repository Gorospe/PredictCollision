import pandas as pd
import ClassificatorUtils as cls
csv_path = '../data/ACC.csv'
df_acc = pd.read_csv(csv_path)
df_acc = df_acc.drop([ 'ID', 'scenario', 'seed', 'Controller', 'numHumanCars', 'BeaconIntervalPlat', 'BeaconIntervalHuman', 'PacketSize', 'waitTime', 'DENMInterval'], axis=1)
df_acc = df_acc.dropna()
df_acc = df_acc.drop_duplicates()

csv_path = '../data/CACC_truncated.xlsx'
df_cacc = pd.read_excel(csv_path)
df_cacc = df_cacc.drop([ 'ID', 'scenario', 'seed', 'Controller'], axis=1)
df_cacc = df_cacc.dropna()
df_cacc = df_cacc.drop_duplicates()

csv_path = '../data/PLOEG_truncated.xlsx'
df_ploeg = pd.read_excel(csv_path)
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


rs_acc = cls.test_algorithms(X_acc, y_acc, cls.test_several)
rs_acc.to_excel('../results/resultsACC.xlsx')

rs_cacc = cls.test_algorithms(X_cacc, y_cacc, cls.test_several)
rs_cacc.to_excel('../results/resultsCACC_truncated.xlsx')

rs_ploeg = cls.test_algorithms(X_ploeg, y_ploeg, cls.test_several)
rs_ploeg.to_excel('../results/resultsPLOEG_truncated.xlsx')

#rs_cacc_ploeg = test_algorithms(X_cacc_ploeg, y_cacc_ploeg, test_several)
#rs_cacc_ploeg.to_excel('../results/resultsCACC_PLOEG.xlsx')


rs_acc = cls.test_algorithms(X_acc, y_acc, cls.test_dec_tree)
rs_acc.to_excel('../results/resultsACC_dt.xlsx')

rs_cacc = cls.test_algorithms(X_cacc, y_cacc, cls.test_dec_tree)
rs_cacc.to_excel('../results/resultsCACC_dt_truncated.xlsx')

rs_ploeg = cls.test_algorithms(X_ploeg, y_ploeg, cls.test_dec_tree)
rs_ploeg.to_excel('../results/resultsPLOEG_dt_truncated.xlsx')

#rs_cacc_ploeg = test_algorithms(X_cacc_ploeg, y_cacc_ploeg, test_dec_tree)
#rs_cacc_ploeg.to_excel('../results/resultsCACC_PLOEG_dt.xlsx')


rs_acc = cls.test_algorithms(X_acc, y_acc, cls.test_MLP)
rs_acc.to_excel('../results/resultsACC_nn.xlsx')

rs_cacc = cls.test_algorithms(X_cacc, y_cacc, cls.test_MLP)
rs_cacc.to_excel('../results/resultsCACC_nn_truncated.xlsx')

rs_ploeg = cls.test_algorithms(X_ploeg, y_ploeg, cls.test_MLP)
rs_ploeg.to_excel('../results/resultsPLOEG_nn_truncated.xlsx')

#rs_cacc_ploeg = test_algorithms(X_cacc_ploeg, y_cacc_ploeg, test_MLP)
#rs_cacc_ploeg.to_excel('../results/resultsCACC_PLOEG_nn.xlsx')
