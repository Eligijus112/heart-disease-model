# Importing the model class 
from sklearn.linear_model import LogisticRegression

# Cross validation 
from sklearn.model_selection import RepeatedKFold

# Precision and recall scores
from sklearn.metrics import recall_score, precision_score, roc_auc_score

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

# Placeholder 
results = pd.DataFrame({})

for c in hp.get('C'):
    # Defining dict for model creation
    hpinstance = {
        'solver': hp.get('solver'),
        'max_iter': hp.get('max_iter'),
        'C': c
    }

    # Placeholders
    recalls, precisions, aucs = [], [], []

    # Iterating over the dataset 
    for train_index, test_index in rkf.split(X):
        # Getting the train and test splits 
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Initiating the model 
        model = LogisticRegression(**hpinstance)

        # Training the model 
        model.fit(X_train, y_train)

        # Predicting 
        yhat = model.predict(X_test)

        # Getting the probabilities 
        yhatp = [x[1] for x in model.predict_proba(X_test)]

        # Getting the recall and precision
        recall, precision, auc = recall_score(y_test, yhat), precision_score(y_test, yhat), roc_auc_score(y_test, yhatp)

        # Appending to list 
        recalls.append(recall)
        precisions.append(precision)
        aucs.append(auc)

    # Creating the results data frame 
    valFrame = pd.DataFrame({
        'recall': recalls, 
        'precision': precisions,
        'auc': aucs
        })

    # Calculating the F1 score 
    valFrame['f1'] = (2 * valFrame['recall'] * valFrame['precision']) / (valFrame['recall'] + valFrame['precision'])

    # Calculating GINI 
    valFrame['gini'] = 2 * valFrame['auc'] - 1

    # Printing the descriptive stats 
    print(valFrame.describe())

    # Saving some hyper parameters
    valFrame['C'] = c
    
    # Saving the results 
    results = pd.concat([results, valFrame])

# Aggregating and constructing the final best hyper parameter dict 
agg = results.groupby('C', as_index=False)['gini'].mean()
agg.sort_values('gini', ascending=False, inplace=True)

# Printing out the agg results
print(agg)

hpbest = {
        'solver': hp.get('solver'),
        'max_iter': hp.get('max_iter'),
        'C': agg.head(1)['C'].values[0]
    }

# Initiating the model
model = LogisticRegression(**hpbest)

# Fitting the model 
model.fit(X, y)

# Saving the feature names to model object
model.feature_names = X.columns.tolist()

# Creating the coefficient frame to print out 
coefFrame = pd.DataFrame({
    'feature': X.columns.tolist(),
    'coef': model.coef_[0]
})
print(coefFrame)

# Saving the categorical features
model.categorical_features = conf.get("categorical_features")

# Saving the numerical features 
model.numeric_features = conf.get('numeric_features')

# Getting the model name from the configurations
model_name = conf.get("model_name", "misc")

# Saving the model object to a pickle file 
pickle.dump(model, open(f'models/model_{model_name}.sav', 'wb'))