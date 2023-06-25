from os import system
system("clear")

import numpy as np
from sklearn.model_selection import KFold
from sklearn import svm, ensemble, tree
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

#load data
df = pd.read_csv('Dataset_ML Project.csv', sep=',', header='infer').dropna()

#normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.iloc[:,1:4]) #exclude date and y values

#apply PCA
pca = PCA()
X_new = pca.fit(X_scaled)

#explained variance ratio
variance_ratio = pca.explained_variance_ratio_
total_variance = pca.explained_variance_ratio_.sum() * 100

print("Total PCs:", len(variance_ratio))
print("Total Variance:", total_variance)
print(" ")

#create table
table_EVR = [
    ['Principal Component', 'Explained Variance Ratio'],
    ['PC1', round(variance_ratio[0], 4)],
    ['PC2', round(variance_ratio[1], 4)],
    ['PC3', round(variance_ratio[2], 4)]
]

#print table 
print(tabulate(table_EVR, headers='firstrow'))
print("")

plt.figure(figsize=(8, 6))
plt.bar(['PC1', 'PC2', 'PC3'], variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio for Principal Components')
plt.show()

#since most of the variance in the data explained by PC1 and PC2, let's only take these two into accounts for further analysis
pca_new = PCA(n_components=2)
X_reduced = pca_new.fit_transform(X_scaled)

#create machine learning objects
LR = linear_model.LogisticRegression(random_state=42)
SVM = svm.SVC(random_state=42)
RFC = ensemble.RandomForestClassifier(random_state=42)
DTC = tree.DecisionTreeClassifier(random_state=42)

#convert continuous y values to multiple classes
def class_MC(data):
    conditions = [
        (data['close'] > data['open']),  #closing price higher than opening price: Positive
        (data['close'] == data['open']), #closing price equals to opening price: Neutral
        (data['close'] < data['open'])   #closing price less than opening price: Negative
    ]
    classes = [2, 1, 0]   #Positive, Neutral and Negative
    y2_multiclass = np.select(conditions, classes)
    return pd.Series(y2_multiclass, copy=False)
y_MC = class_MC(df)

#try 5 folds of cross-validation

#create lists to store the scores for each model
scores_LR_5 = []
scores_SVM_5 = []
scores_RFC_5 = []
scores_DTC_5 = []

#perform K-fold cross-validation
for fold, (train_index, test_index) in enumerate(KFold(n_splits=5).split(X_reduced)):

    #split data into training & test sets
    X_train_5, X_test_5 = X_reduced[train_index], X_reduced[test_index]
    y_train_5, y_test_5 = y_MC[train_index], y_MC[test_index]

    #train models
    SVM.fit(X_train_5, y_train_5)
    LR.fit(X_train_5, y_train_5)
    RFC.fit(X_train_5, y_train_5)
    DTC.fit(X_train_5, y_train_5)

    #evaluate the models
    score_LR_5 = LR.score(X_test_5, y_test_5)
    score_SVM_5 = SVM.score(X_test_5, y_test_5)
    score_RFC_5 = RFC.score(X_test_5, y_test_5)
    score_DTC_5 = DTC.score(X_test_5, y_test_5)

    #append the scores to the respective lists
    scores_LR_5.append(score_LR_5)
    scores_SVM_5.append(score_SVM_5)
    scores_RFC_5.append(score_RFC_5)
    scores_DTC_5.append(score_DTC_5)

#create a list of the scores for each model
all_scores_5 = [scores_LR_5, scores_SVM_5, scores_RFC_5, scores_DTC_5]

#create a boxplot
fig, ax = plt.subplots()
bp = ax.boxplot(all_scores_5)
ax.set_xticklabels(['LR', 'SVM', 'RFC', 'DTC'])
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Cross-Validation Scores (K=5)')
plt.show()

#calculate the statistics for each model
stats_5 = []
for scores in all_scores_5:
    stats = {
        'min': np.min(scores),
        '25%': np.percentile(scores, 25),
        'median': np.median(scores),
        '75%': np.percentile(scores, 75),
        'max': np.max(scores),
    }
    stats_5.append(stats)

#create a table to display the statistics
table_stats_5 = {
    'Model': ['LR', 'SVM', 'RFC', 'DTC'],
    'Min': [stats['min'] for stats in stats_5],
    'Q1': [stats['25%'] for stats in stats_5],
    'Median': [stats['median'] for stats in stats_5],
    'Q3': [stats['75%'] for stats in stats_5],
    'Max': [stats['max'] for stats in stats_5],
}

#print the table
print("Boxplot Statistics (K=5):")
print("Model  |  Min  |  Q1  |  Median  |  Q3  |  Max")
for i in range(len(table_stats_5['Model'])):
    row = [table_stats_5['Model'][i]]
    row.extend([table_stats_5[column][i] for column in ['Min', 'Q1', 'Median', 'Q3', 'Max']])
    print("{:6s} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(*row))
print(" ")

#try 10 folds of cross-validation

#create lists to store the scores for each model
scores_LR_10 = []
scores_SVM_10 = []
scores_RFC_10 = []
scores_DTC_10 = []

#perform K-fold cross-validation
for fold, (train_index, test_index) in enumerate(KFold(n_splits=10).split(X_reduced)):
    
    #split data into training & test sets
    X_train_10, X_test_10 = X_reduced[train_index], X_reduced[test_index]
    y_train_10, y_test_10 = y_MC[train_index], y_MC[test_index]

    #train models
    SVM.fit(X_train_10, y_train_10)
    LR.fit(X_train_10, y_train_10)
    RFC.fit(X_train_10, y_train_10)
    DTC.fit(X_train_10, y_train_10)

    #evaluate the models
    score_LR_10 = LR.score(X_test_10, y_test_10)
    score_SVM_10 = SVM.score(X_test_10, y_test_10)
    score_RFC_10 = RFC.score(X_test_10, y_test_10)
    score_DTC_10 = DTC.score(X_test_10, y_test_10)

    #append the scores to the respective lists
    scores_LR_10.append(score_LR_10)
    scores_SVM_10.append(score_SVM_10)
    scores_RFC_10.append(score_RFC_10)
    scores_DTC_10.append(score_DTC_10)

#create a list of the scores for each model
all_scores_10 = [scores_LR_10, scores_SVM_10, scores_RFC_10, scores_DTC_10]

#create a boxplot
fig, ax = plt.subplots()
bp = ax.boxplot(all_scores_10)
ax.set_xticklabels(['LR', 'SVM', 'RFC', 'DTC'])
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Cross-Validation Scores (K=10)')
plt.show()

#calculate the statistics for each model
stats_10 = []
for scores in all_scores_10:
    stats = {
        'min': np.min(scores),
        'Q1': np.percentile(scores, 25),
        'median': np.median(scores),
        'Q3': np.percentile(scores, 75),
        'max': np.max(scores),
    }
    stats_10.append(stats)

#create a table to display the statistics
table_stats_10 = {
    'Model': ['LR', 'SVM', 'RFC', 'DTC'],
    'Min': [stats['min'] for stats in stats_10],
    'Q1': [stats['Q1'] for stats in stats_10],
    'Median': [stats['median'] for stats in stats_10],
    'Q3': [stats['Q3'] for stats in stats_10],
    'Max': [stats['max'] for stats in stats_10],
}

#print the table
print("Boxplot Statistics (K=10):")
print("Model  |  Min  |  Q1  |  Median  |  Q3  |  Max")
for i in range(len(table_stats_10['Model'])):
    row = [table_stats_10['Model'][i]]
    row.extend([table_stats_10[column][i] for column in ['Min', 'Q1', 'Median', 'Q3', 'Max']])
    print("{:6s} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(*row))
