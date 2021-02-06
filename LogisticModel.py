# Importing the model class 
from sklearn.linear_model import LogisticRegression

# Data wrangling 
import pandas as pd 

# Saving the model 
import pickle

# Configuration reading 
import yaml 

# Reading the configurations
conf = yaml.load(open("conf.yml"), Loader=yaml.FullLoader)

# Reading data 
d = pd.read_csv('framingham_heart_disease.csv')

# Defining the feature list 
features = conf.get("features")

# Creating the X and Y matrices 
X = d[features].copy()
Y = d['TenYearCHD']

# Filling possible missing values 
X.fillna(X.mean(), inplace=True)

# Initiating the model 
model = LogisticRegression()

# Fitting the model 
model.fit(X, Y)

# Saving the feature names to model object
model.feature_names = X.columns.tolist()

# Getting the model name from the configurations
model_name = conf.get("model_name", "misc")

# Saving the model object to a pickle file 
pickle.dump(model, open(f'models/model_{model_name}.sav', 'wb'))