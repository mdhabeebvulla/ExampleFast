from fastapi import FastAPI
import pandas as pd
import numpy as np
import re
import json
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline  #creates a machine learning pipeline
from sklearn.naive_bayes import MultinomialNB  #to apply Naive Bayes Classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  #accuracy metrics
from sklearn.feature_extraction.text import CountVectorizer #for BoW
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
data = pd.read_csv('Entry_cleaned.csv')
class Testdata(BaseModel):
    endpoint:str
def preprocess(endpoint: str) -> str:
    r = re.sub(r'{[^)]*}', '', endpoint)
    r = r.replace('/', ' ').replace('-', ' ')
    terms = r.split()
    if 'api' in terms: terms.remove('api')
    if 'v1' in terms: terms.remove('v1')
    if 'v2' in terms: terms.remove('v2')

    return ' '.join(terms)
data['path'] = data['path'].apply(preprocess)
data.to_csv('newentry.csv')
x = data.path.values.reshape(-1,1)
y = data.response.values
import imblearn

from imblearn.under_sampling import RandomUnderSampler

ros = RandomUnderSampler(random_state=42)


x_ros, y_ros = ros.fit_resample(x, y)
df =pd.DataFrame({'path':x_ros.flatten(),'response':y_ros.flatten()})
print(data.shape)
print(df.shape) 
X_train,X_test,y_train,y_test = train_test_split(df['path'],df['response'],random_state=42)
cv = CountVectorizer()
nb_pipeline = Pipeline([('vect', cv), ('clf', MultinomialNB())])
nb_pipeline.fit(X_train, y_train)
y_predn = nb_pipeline.predict(X_test)
print(accuracy_score(y_predn,y_test))
print(confusion_matrix(y_predn,y_test))
print(classification_report(y_predn,y_test))
report = classification_report(y_test, y_predn, output_dict=True)
with open('classification_report.json', 'w') as j:
    json.dump(report, j)
app = FastAPI()
@app.post("/api/predict")
def predict_authorization(endp:Testdata):
    endpoint = endp.endpoint
    y_predt = nb_pipeline.predict([endpoint])
    yn = int(y_predt[0])
    return {'prediction': yn}
