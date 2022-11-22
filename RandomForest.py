# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:13:34 2022

@author: Admin
"""

import numpy as np
#import matplotlib as plt
import pandas as pd
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn import preprocessing
from matplotlib import pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report, confusion_matrix
data=pd.read_excel(r"D:\stat_LU\stat.xlsx")

#Data exploration
print(data.shape)
print(list(data.columns))
data.head()
data.tail()
data=data.drop(['Unnamed: 0'], axis=1)

texture=pd.read_excel(r"D:\stat_LU\Texture2010_12_11.xlsx")
data['Homogeneity']=texture['Homogeneity']


data['mm']=(data.Max-data.Mean)/data.Mean
data['coeffvar']=data.Std/data.Mean

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data=data.dropna() #Delete cells with nodata

data['mmnorm']=(data.mm-np.mean(data.mm))/np.std(data.mm)
data['coeffvarnorm']=(data.coeffvar-np.mean(data.coeffvar))/np.std(data.coeffvar)
data['B_Beetle'].value_counts()

sns.countplot(x = 'B_Beetle', data = data, palette = 'hls')
plt.show() 

count_no_bb = len(data[data['B_Beetle']==0])
count_bb = len(data[data['B_Beetle']==1])
pct_of_no_bb = count_no_bb/(count_no_bb+count_bb)
print("percentage of no damage is", pct_of_no_bb*100)
pct_of_bb = count_bb/(count_no_bb+count_bb)
print("percentage of damage", pct_of_bb*100)
 
 #Balance Dependent variable (binary)
Healthy= data[data.B_Beetle==0]
Damaged= data[data.B_Beetle==1]
Healthy_balanced = resample(Healthy,
                            replace=False,
                            n_samples=19208,
                            random_state=123)
BB_Balancedrf= pd.concat([Healthy_balanced, Damaged])
BB_Balancedrf.value_counts()


max_mean_std= ['Homogeneity','B_Beetle']
x=BB_Balancedrf[max_mean_std]
#x.replace([np.inf, -np.inf], np.nan, inplace=True)
#x=x.dropna()
y=x.loc[:,"B_Beetle"]
x=x.drop(['B_Beetle'], axis=1)




from sklearn.model_selection import train_test_split
# i.e. 70 % training dataset and 30 % test datasets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100, max_depth=13) 
 
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
 
# performing predictions on the test dataset
y_pred = clf.predict(X_test)
 
# metrics are used to find accuracy or error
from sklearn import metrics 
print()
 
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
names=['Homogeneity']
# feature_imp = pd.Series(clf.feature_importances_, index = names).sort_values(ascending = False)


# cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.imshow(cnf_matrix)
# ax.grid(False)
# ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
# ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
# ax.set_ylim(1.5, -0.5)
# for i in range(2):
#     for j in range(2):
#         ax.text(j, i, cnf_matrix[i, j], ha='center', va='center', color='blue')
# plt.show()

# class_names=[0,1] # name  of classes
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)
# # create heatmap
# sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
# ax.xaxis.set_label_position("top")
# plt.tight_layout()
# plt.title('Confusion matrix', y=1.1)
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')

from sklearn.inspection import PartialDependenceDisplay

common_params = {
    "subsample": 50,
    "n_jobs": 2,
    "grid_resolution": 20,
    "random_state": 0,
}

print("Computing partial dependence plots...")

display = PartialDependenceDisplay.from_estimator(
    clf,
    X_train,
    features=['Homogeneity'],
    kind="both",
    **common_params,
)

display.figure_.suptitle(
    "PDP of Forest Structure from Sweden Lidar data 2010-2012 and Bark Beetle occurance\n"
    "Random Forest Accuracy: 0.5167895878524946"
)
display.figure_.subplots_adjust(hspace=0.4)

from sklearn.inspection import permutation_importance
#names=['Max','Mean', 'mm_log', 'coeffvar_log', 'Homogeneity']

result = permutation_importance(
    clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
forest_importances = pd.Series(result.importances_mean, index=names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()