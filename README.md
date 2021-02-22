# heart-disease-model
A project to suplement an ML API with a model. The API can be found via: https://github.com/Eligijus112/flask-application 

# Data source 

More about the heart disease data can be found via: https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset 

# Creating the virtual env

```
virtualenv heart
```

```
source heart/bin/activate
```

```
pip install -r requirements.txt
```

# Dependant variable 

TenYearCHD - whether or not a person developed Coronary Heart Disease (CHD).

# Feature list for the independant variables 

male 

age 

education 

currentSmoker 

cigsPerDay 

BPMeds

prevalentStroke 

prevalentHyp 

diabetes 

totChol 

sysBP

diaBP 

BMI 

heartRate 

glucose

# Amount of data 

The data has 4238 observations.

# Running the pipeline 

## Configuration file 

The configurations should be in **conf.yml** file. 

## Cleaning the data 

The data is cleaned using the command: 

```
python dataPreproc.py 
```

The cleaned data is stored in the directory clean-data/ .

## Creating the model 

To create the model run:

```
python LogisticModel.py 
```