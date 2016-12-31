# Import data and take a look
import numpy as np
import pandas as pd

train = pd.read_csv('../data/trainremake.csv', na_values=['#NAME?'])
test = pd.read_csv('../data/testremake.csv', na_values=['#NAME?'])

train['Type'] = 0 #Create a flag for Train and Test Data set
test['Type'] = 1

fullData = pd.concat([train,test],axis=0) #Combined both Train and Test Data set

var = []
# Decide which categorical variables you want to use in model
for col_name in fullData.columns:
    if fullData[col_name].dtypes == 'object':
        unique_cat = len(fullData[col_name].unique())
        var.append(col_name)
        # print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))
print var

# Create a list of features to dummy
todummy_list = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

# Function to dummy all the categorical variables used for modeling
def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df

fullData = dummy_df(fullData, todummy_list)

#Imputer in sklearn.preprocessing
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(fullData)
fullData = pd.DataFrame(data=imp.transform(fullData) , columns=fullData.columns)

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

    # Use train_test_split in sklearn.cross_validation to split data into train and test sets
from sklearn.cross_validation import train_test_split

train=fullData[fullData['Type']==0]
test=fullData[fullData['Type']==1]

X_train, X_test, y_train, y_test = train_test_split(train.drop(['SalePrice'], axis=1),
                                                    train.SalePrice, test_size=0.2, random_state=20)


# Such a large set of features can cause overfitting and also slow computing
# Use feature selection to select the most important features
import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(k=20)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [fullData.columns[i] for i in indices_selected]

X_train = X_train[colnames_selected]
X_test = X_test[colnames_selected]

# print(colnames_selected)

# Function to build model and find model performance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score



clf = LinearRegression(fit_intercept=True)
clf.fit(X_train, y_train)



X_train, X_test, y_train, y_test = train_test_split(test.drop(['SalePrice'], axis=1),
                                                    test.SalePrice, test_size=0, random_state=0)
X_train = X_train[colnames_selected]
X_test = X_test[colnames_selected]
# X_train = X_train[colnames_selected]
Y = pd.read_csv('../data/testremake.csv')
import csv
with open('output.csv', 'w') as outcsv: 
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(['Id', 'SalePrice'])
    dataPrediction = clf.predict(X_train)
    for i in xrange(0, len(Y)):
        dataId = Y['Id'][i]
        writer.writerow([dataId, dataPrediction[i]])


