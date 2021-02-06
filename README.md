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

# Running the pipeline 

## Configuration file 

The configurations should be in **conf.yml** file. 

## Creating the model 

To create the model run:

```
python LogisticModel.py 
```