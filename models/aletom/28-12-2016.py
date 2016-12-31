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
from math import sqrt
from sklearn import cross_validation

# read the file into a dataframe
train = pd.read_csv('data/trainremake.csv')

test = pd.read_csv('data/testremake.csv')

train['Type'] = 0 #Create a flag for Train and Test Data set
test['Type'] = 1

# Handle missing values for features where median/mean or most common value doesn't make sense

# Alley : data description says NA means "no alley access"
train.loc[:, "Alley"] = train.loc[:, "Alley"].fillna("None")
# BedroomAbvGr : NA most likely means 0
train.loc[:, "BedroomAbvGr"] = train.loc[:, "BedroomAbvGr"].fillna(0)
# BsmtQual etc : data description says NA for basement features is "no basement"
train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")
train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")
train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")
train.loc[:, "BsmtFullBath"] = train.loc[:, "BsmtFullBath"].fillna(0)
train.loc[:, "BsmtHalfBath"] = train.loc[:, "BsmtHalfBath"].fillna(0)
train.loc[:, "BsmtUnfSF"] = train.loc[:, "BsmtUnfSF"].fillna(0)
# CentralAir : NA most likely means No
train.loc[:, "CentralAir"] = train.loc[:, "CentralAir"].fillna("N")
# Condition : NA most likely means Normal
train.loc[:, "Condition1"] = train.loc[:, "Condition1"].fillna("Norm")
train.loc[:, "Condition2"] = train.loc[:, "Condition2"].fillna("Norm")
# EnclosedPorch : NA most likely means no enclosed porch
train.loc[:, "EnclosedPorch"] = train.loc[:, "EnclosedPorch"].fillna(0)
# External stuff : NA most likely means average
train.loc[:, "ExterCond"] = train.loc[:, "ExterCond"].fillna("TA")
train.loc[:, "ExterQual"] = train.loc[:, "ExterQual"].fillna("TA")
# Fence : data description says NA means "no fence"
train.loc[:, "Fence"] = train.loc[:, "Fence"].fillna("No")
# FireplaceQu : data description says NA means "no fireplace"
train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")
train.loc[:, "Fireplaces"] = train.loc[:, "Fireplaces"].fillna(0)
# Functional : data description says NA means typical
train.loc[:, "Functional"] = train.loc[:, "Functional"].fillna("Typ")
# GarageType etc : data description says NA for garage features is "no garage"
train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")
train.loc[:, "GarageArea"] = train.loc[:, "GarageArea"].fillna(0)
train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)
# HalfBath : NA most likely means no half baths above grade
train.loc[:, "HalfBath"] = train.loc[:, "HalfBath"].fillna(0)
# HeatingQC : NA most likely means typical
train.loc[:, "HeatingQC"] = train.loc[:, "HeatingQC"].fillna("TA")
# KitchenAbvGr : NA most likely means 0
train.loc[:, "KitchenAbvGr"] = train.loc[:, "KitchenAbvGr"].fillna(0)
# KitchenQual : NA most likely means typical
train.loc[:, "KitchenQual"] = train.loc[:, "KitchenQual"].fillna("TA")
# LotFrontage : NA most likely means no lot frontage
train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna(0)
# LotShape : NA most likely means regular
train.loc[:, "LotShape"] = train.loc[:, "LotShape"].fillna("Reg")
# MasVnrType : NA most likely means no veneer
train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")
train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)
# MiscFeature : data description says NA means "no misc feature"
train.loc[:, "MiscFeature"] = train.loc[:, "MiscFeature"].fillna("No")
train.loc[:, "MiscVal"] = train.loc[:, "MiscVal"].fillna(0)
# OpenPorchSF : NA most likely means no open porch
train.loc[:, "OpenPorchSF"] = train.loc[:, "OpenPorchSF"].fillna(0)
# PavedDrive : NA most likely means not paved
train.loc[:, "PavedDrive"] = train.loc[:, "PavedDrive"].fillna("N")
# PoolQC : data description says NA means "no pool"
train.loc[:, "PoolQC"] = train.loc[:, "PoolQC"].fillna("No")
train.loc[:, "PoolArea"] = train.loc[:, "PoolArea"].fillna(0)
# SaleCondition : NA most likely means normal sale
train.loc[:, "SaleCondition"] = train.loc[:, "SaleCondition"].fillna("Normal")
# ScreenPorch : NA most likely means no screen porch
train.loc[:, "ScreenPorch"] = train.loc[:, "ScreenPorch"].fillna(0)
# TotRmsAbvGrd : NA most likely means 0
train.loc[:, "TotRmsAbvGrd"] = train.loc[:, "TotRmsAbvGrd"].fillna(0)
# Utilities : NA most likely means all public utilities
train.loc[:, "Utilities"] = train.loc[:, "Utilities"].fillna("AllPub")
# WoodDeckSF : NA most likely means no wood deck
train.loc[:, "WoodDeckSF"] = train.loc[:, "WoodDeckSF"].fillna(0)



#######################
# basic data cleaning #
#######################

#### Decide which categorical variables you want to use in model
# Make an array of it
fullData = pd.concat([train,test],axis=0) #Combined both Train and Test Data set
var = []
for col_name in fullData.columns:
    if fullData[col_name].dtypes == 'object':
        unique_cat = len(fullData[col_name].unique())
        var.append(col_name)
        print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))

# print var
# Create a list of features to dummy


todummy_list = ['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'CentralAir', 'Condition1', 'Condition2', 'Electrical', 'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd', 'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC', 'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig', 'LotShape', 'MSZoning', 'MasVnrType', 'MiscFeature', 'Neighborhood', 'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'Street', 'Utilities','StreetName']

# How much of our data is missing?
print fullData.isnull().sum().sort_values(ascending=False).head(len(fullData))
# Function to dummy all the categorical variables used for modeling
def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df


fullData = dummy_df(fullData, todummy_list)


#### Handling missing data

from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

fullData = DataFrameImputer().fit_transform(fullData)
# How much of our data is missing?
print fullData.isnull().sum().sort_values(ascending=False).head(len(fullData))

# Use PolynomialFeatures in sklearn.preprocessing to create two-way interactions for all features
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

def add_interactions(df):
    # Get feature names
    combos = list(combinations(list(df.columns), 2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]
    
    # Find interactions
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames
    
    # Remove interaction terms with all 0 values            
    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
    df = df.drop(df.columns[noint_indicies], axis=1)
    
    return df

fullData = add_interactions(fullData)
# Now check again to see if you still have missing data
# print fullData.isnull().sum().sort_values(ascending=False).head()

#### Remove and detect outliers
# Removed four data entries


###############################
# trying out different models #
###############################
ests = [ linear_model.LinearRegression(fit_intercept=True), linear_model.Ridge(),
        linear_model.Lasso(), linear_model.ElasticNet(),
        linear_model.BayesianRidge(), linear_model.OrthogonalMatchingPursuit() ]
ests_labels = np.array(['Linear', 'Ridge', 'Lasso', 'ElasticNet', 'BayesRidge', 'OMP'])

train=fullData[fullData['Type']==0]
test=fullData[fullData['Type']==1]
X_train, X_test, y_train, y_test = train_test_split(train.drop(['SalePrice'], axis=1),
                                                    train.SalePrice, test_size=0.2, random_state=1)


errvals = np.array([])
r2vals = np.array([])
accvals = np.array([])
for e in ests:
	e.fit(X_train, y_train)
	rms = metrics.median_absolute_error(y_test, e.predict(X_test))
	r2 = metrics.r2_score(y_test, e.predict(X_test))
	acc = e.score(X_test, y_test)
	print("MAD: %.4f" % rms)
	print("R2: %.4f" % r2)
	print("Accuracy: %.4f" % acc)
	accvals = np.append(accvals,acc)
	errvals = np.append(errvals, rms)
	r2vals = np.append(r2vals, r2)

# print errvals

# pos = np.arange(errvals.shape[0])
# srt = np.argsort(errvals)
# plt.figure(figsize=(7,5))
# plt.bar(pos, errvals[srt], align='center')
# plt.xticks(pos, ests_labels[srt])
# plt.xlabel('Estimator')
# plt.ylabel('Median Absolute Error')
# plt.show()



##################
# model ensemble #
##################

tuned_parameters = {'learning_rate': [0.1, 0.01, 0.001],
'max_depth': [4, 6, 8],
'min_samples_leaf': [3,5,8],
'max_features' : [10, 15, 20],
'loss' : ['ls']
}
n_est = 1500

gbr = ensemble.GradientBoostingRegressor(n_estimators=n_est)
clf = GridSearchCV(gbr, tuned_parameters, scoring='neg_median_absolute_error', n_jobs=4).fit(X_train, y_train)
print('Best hyperparameters: %r' % clf.best_params_)
gbr.set_params(** clf.best_params_)
gbr.fit(X_train, y_train)
rms = metrics.median_absolute_error(y_test, gbr.predict(X_test))
r2 = metrics.r2_score(y_test, gbr.predict(X_test))
acc = gbr.score(X_test, y_test)                           
print("RMS: %.4f" % rms)
print("R2: %.4f" % r2)
print("Accuracy: %.4f" % acc)

# plot error for each round of boosting
test_score = np.zeros(n_est, dtype=np.float64)
best = clf.best_estimator_
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

# feature_importance = clf.best_estimator_.feature_importances_
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# sorted_idx = np.argsort(feature_importance)

# pos = np.arange(sorted_idx.shape[0]) + 2
# pvals = feature_importance[sorted_idx]


# pcols = X_train.columns[sorted_idx]

# plt.figure(figsize=(32,48))
# plt.barh(pos, pvals, align='center')
# plt.yticks(pos, pcols)
# plt.xlabel('Relative Importance')
# plt.title('Variable Importance')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(test.drop(['SalePrice'], axis=1),
                                                    test.SalePrice, test_size=0, random_state=0)
# X_train = X_train[colnames_selected]
Y = pd.read_csv('data/testremake.csv')
import csv
with open('output.csv', 'w') as outcsv: 
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(['Id', 'SalePrice'])
    dataPrediction = gbr.predict(X_train)
    for i in xrange(0, len(Y)):
        dataId = Y['Id'][i]
        writer.writerow([dataId, dataPrediction[i]])
