import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score

# read the file into a dataframe
X = pd.read_csv('data/trainremake.csv')

Y = pd.read_csv('data/testremake.csv')

X['Type'] = 0 #Create a flag for Train and Test Data set
Y['Type'] = 1

fullData = pd.concat([X,Y],axis=0) #Combined both Train and Test Data set
#######################
# basic data cleaning #
#######################

#### Decide which categorical variables you want to use in model
# Make an array of it
var = []
for col_name in fullData.columns:
    if fullData[col_name].dtypes == 'object':
        unique_cat = len(fullData[col_name].unique())
        var.append(col_name)
        # print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))

print var
# Create a list of features to dummy
todummy_list = ['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'CentralAir', 'Condition1', 'Condition2', 'Electrical', 'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd', 'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC', 'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig', 'LotShape', 'MSZoning', 'MasVnrType', 'MiscFeature', 'Neighborhood', 'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'Street', 'Utilities']


# Function to dummy all the categorical variables used for modeling
def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df


fullData = dummy_df(fullData, todummy_list)


#### Handling missing data

# How much of our data is missing?
# print X.isnull().sum().sort_values(ascending=False).head()

# print X.columns

#Imputer in sklearn.preprocessing
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(fullData)
fullData = pd.DataFrame(data=imp.transform(fullData) , columns=fullData.columns)

# Now check again to see if you still have missing data
print fullData.isnull().sum().sort_values(ascending=False).head()

#### Remove and detect outliers
# Removed four data entries


###############################
# trying out different models #
###############################

rs = 1
ests = [ linear_model.LinearRegression(), linear_model.Ridge(),
        linear_model.Lasso(), linear_model.ElasticNet(),
        linear_model.BayesianRidge(), linear_model.OrthogonalMatchingPursuit() ]
ests_labels = np.array(['Linear', 'Ridge', 'Lasso', 'ElasticNet', 'BayesRidge', 'OMP'])
errvals = np.array([])
train=fullData[fullData['Type']==0]
test=fullData[fullData['Type']==1]


X_train, X_test, y_train, y_test = train_test_split(train.drop(['SalePrice'], axis=1),
                                                    train.SalePrice, test_size=0.2, random_state=20)


for e in ests:
    e.fit(X_train, y_train)
    this_err = metrics.median_absolute_error(y_test, e.predict(X_test))
    #print "got error %0.2f" % this_err
    errvals = np.append(errvals, this_err)

print errvals

pos = np.arange(errvals.shape[0])
srt = np.argsort(errvals)
plt.figure(figsize=(7,5))
plt.bar(pos, errvals[srt], align='center')
plt.xticks(pos, ests_labels[srt])
plt.xlabel('Estimator')
plt.ylabel('Median Absolute Error')
# plt.show()


##################
# model ensemble #
##################

n_est = 800

tuned_parameters = {
    "n_estimators": [ n_est ],
    "max_depth" : [ 8 ],
    "learning_rate": [ 0.001 ],
    "min_samples_split" : [ 3 ],
    "loss" : [ 'ls', 'lad' ]
}


gbr = ensemble.GradientBoostingRegressor()
clf = GridSearchCV(gbr, cv=3, param_grid=tuned_parameters,
        scoring='neg_median_absolute_error')
preds = clf.fit(X_train, y_train)
best = clf.best_estimator_


# plot error for each round of boosting
test_score = np.zeros(n_est, dtype=np.float64)

train_score = best.train_score_
for i, y_pred in enumerate(best.staged_predict(X_test)):
    test_score[i] = best.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(np.arange(n_est), train_score, 'darkblue', label='Training Set Error')
plt.plot(np.arange(n_est), test_score, 'red', label='Test Set Error')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Least Absolute Deviation')
plt.show()


feature_importance = clf.best_estimator_.feature_importances_
print "feature_importance"
print feature_importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)

print "sorted_idx"
print sorted_idx

pos = np.arange(sorted_idx.shape[0]) + 2
pvals = feature_importance[sorted_idx]
print "pvals"
print pvals

pcols = X_train.columns[sorted_idx]
print "pcols"
print pcols

plt.figure(figsize=(32,48))
plt.barh(pos, pvals, align='center')
plt.yticks(pos, pcols)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(test.drop(['SalePrice'], axis=1),
                                                    test.SalePrice, test_size=0, random_state=0)
# X_train = X_train[colnames_selected]
Y = pd.read_csv('data/testremake.csv')
import csv
with open('output.csv', 'w') as outcsv: 
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(['Id', 'SalePrice'])
    dataPrediction = clf.predict(X_train)
    for i in xrange(0, len(Y)):
        dataId = Y['Id'][i]
        writer.writerow([dataId, dataPrediction[i]])
