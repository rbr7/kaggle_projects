
# loading libraries

import numpy as np 
import pandas as pd 

import os
# print(os.listdir("../input"))



# load dataset


dataset = pd.read_csv('../input/train_2v.csv')



# preprocess

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Imputer
labelEncoder = LabelEncoder()
x = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1].values
x.smoking_status.fillna(x.smoking_status.dropna().max(), inplace=True)
work_type = pd.get_dummies(x.work_type)
smoking_status = pd.get_dummies(x.smoking_status)
train = x.drop(['work_type', 'smoking_status'], axis =1)
train= train.join(pd.DataFrame(work_type, index=train.index))
train = train.join(pd.DataFrame(smoking_status, index=train.index))
x = train.values
x[:,0] = labelEncoder.fit_transform(x[:,0])
x[:,4] = labelEncoder.fit_transform(x[:,4])
x[:,5] = labelEncoder.fit_transform(x[:,6])
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:,7:8])
x[:,7:8] = imputer.transform(x[:,7:8])



# train

x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.2)
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)


count = 0
for i in range(len(x_test)):
    pre = clf.predict([x_test[i]])
    if pre == y_test[i]:
        count += 1
print(count/len(x_test))



# test

test_dataset = pd.read_csv('../input/test_2v.csv')
x_test = test_dataset.iloc[:,1:]
x_test.smoking_status.fillna(x_test.smoking_status.dropna().max(), inplace=True)
test_work_type = pd.get_dummies(x_test.work_type)
test_smoking_status = pd.get_dummies(x_test.smoking_status)
test = x_test.drop(['work_type', 'smoking_status'], axis =1)
test = test.join(pd.DataFrame(test_work_type, index=test.index))
test = test.join(pd.DataFrame(test_smoking_status, index=test.index))
x_test = test.values
x_test[:,0] = labelEncoder.fit_transform(x_test[:,0])
x_test[:,4] = labelEncoder.fit_transform(x_test[:,4])
x_test[:,5] = labelEncoder.fit_transform(x_test[:,6])
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x_test[:,7:8])
x_test[:,7:8] = imputer.transform(x_test[:,7:8])


# results


file = open('submission.csv','w')
file.write('Prediction'+'\n')
for i in x_test:
    prediction = clf.predict([i])
    file.write(str(prediction).replace("[","").replace("]","") + '\n')
file.close()

