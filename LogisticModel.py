# Importing the model class 
from sklearn.linear_model import LogisticRegression

# Cross validation 
from sklearn.model_selection import RepeatedKFold

# Precision and recall scores
from sklearn.metrics import recall_score, precision_score

# Data wrangling 
import pandas as pd 

# Saving the model 
import pickle

# Configuration reading 
import yaml 

# Reading the configurations
conf = yaml.load(open("conf.yml"), Loader=yaml.FullLoader)

# Reading data 
d = pd.read_pickle('clean-data/clean-data.pkl')

# Creating the X and Y matrices 
X = d.drop('TenYearCHD', axis=1)
y = d['TenYearCHD']

# Initiating the cross validation spliter 
rkf = RepeatedKFold(
    n_splits=conf.get('kfolds', 5), 
    n_repeats=conf.get('repeats', 2)
    )

# Loading the hyper parameters
hp = conf.get("LogisticRegressionHP")

# Placeholders
recalls, precisions = [], []

# Iterating over the dataset 
for train_index, test_index in rkf.split(X):
    # Getting the train and test splits 
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Initiating the model 
    model = LogisticRegression(**hp)

    # Training the model 
    model.fit(X_train, y_train)

    # Predicting 
    yhat = model.predict(X_test)

    # Getting the recall and precision
    recall, precision = recall_score(y_test, yhat), precision_score(y_test, yhat)

    # Appending to list 
    recalls.append(recall)
    precisions.append(precision)

# Creating the results data frame 
valFrame = pd.DataFrame({'recall': recalls, 'precision': precisions})

# Printing the descriptive stats 
print(valFrame.describe())

# Initiating the model 
model = LogisticRegression(**hp)

# Fitting the model 
model.fit(X, y)

# Saving the feature names to model object
model.feature_names = X.columns.tolist()

# Getting the model name from the configurations
model_name = conf.get("model_name", "misc")

# Saving the model object to a pickle file 
pickle.dump(model, open(f'models/model_{model_name}.sav', 'wb'))