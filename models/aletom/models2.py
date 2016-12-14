# Import data and take a look
import numpy as np
import pandas as pd

df = pd.read_csv('data/train.csv', na_values=['#NAME?'])
tf = pd.read_csv('data/test.csv', na_values=['#NAME?'])
# Assign X as a DataFrame of features and y as a Series of the outcome variable
k = tf.SalePrice
X = df.drop('SalePrice', 1)
t = tf.drop('SalePrice', 1)
y = df.SalePrice


# Decide which categorical variables you want to use in model
for col_name in X.columns:
    if X[col_name].dtypes == 'object':
        unique_cat = len(X[col_name].unique())
        # print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))
        
# Create a list of features to dummy
todummy_list = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

# Function to dummy all the categorical variables used for modeling
def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df

X = dummy_df(X, todummy_list)
t = dummy_df(t, todummy_list)

# Impute missing values using Imputer in sklearn.preprocessing
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(X)
X = pd.DataFrame(data=imp.transform(X) , columns=X.columns)

imps = Imputer(missing_values='NaN', strategy='median', axis=0)
imps.fit(t)
t = pd.DataFrame(data=imps.transform(t) , columns=t.columns)

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

    
X = add_interactions(X)
t = add_interactions(t)
    # Use train_test_split in sklearn.cross_validation to split data into train and test sets
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=1)


# Such a large set of features can cause overfitting and also slow computing
# Use feature selection to select the most important features
import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(k=20)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]

# print(colnames_selected)

# Function to build model and find model performance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score



clf = LinearRegression(fit_intercept=True)
clf.fit(X_train_selected, y_train)
predictionData =  X_test_selected
# print(clf.predict(predictionData))




df_unprocessed = t

# Remove non-numeric columns so model does not throw an error
for col_name in df_unprocessed.columns:
    if df_unprocessed[col_name].dtypes not in ['int32','int64','float32','float64']:
        df_unprocessed = df_unprocessed.drop(col_name, 1)

# Split into features and outcomes
X_unprocessed = df_unprocessed
y_unprocessed = k


# Split unprocessed data into train and test set
# Build model and assess performance
X_train_unprocessed, X_test_unprocessed, y_train, y_test = train_test_split(
    X_unprocessed, y_unprocessed, train_size=1, random_state=1)


predictionData =  X_test_unprocessed[colnames_selected]
import csv
with open('output.csv', 'w') as outcsv: 
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(['Id', 'SalePrice'])
    dataPrediction = clf.predict(predictionData)
    print dataPrediction
    for i in xrange(0, len(predictionData)):
        dataId = t['Id'][i]
        if dataPrediction[i] < 0:
            writer.writerow([dataId, 0])
        else:
            writer.writerow([dataId, dataPrediction[i]])

