import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

df = pd.read_csv("https://docs.google.com/spreadsheets/d/1HjH89Ryw3NzT62sedvTC54_DqNkvRgTZn4XXTN4sZuM/export?format=csv&id=1HjH89Ryw3NzT62sedvTC54_DqNkvRgTZn4XXTN4sZuM&gid=1781495660")

df.isnull().sum()

severity_columns = df.filter(like='Severity_').columns
df['Severity_None'].replace({1:'None',0:'No'},inplace =True)
df['Severity_Mild'].replace({1:'Mild',0:'No'},inplace =True)
df['Severity_Moderate'].replace({1:'Moderate',0:'No'},inplace =True)
df['Severity_Severe'].replace({1:'Severe',0:'No'},inplace =True)

df['Condition']=df[severity_columns].values.tolist()

def remove(lista):
    lista = set(lista) 
    lista.discard("No")
    final = ''.join(lista)
    return final

df['Condition'] = df['Condition'].apply(remove)

df.drop(severity_columns,axis=1,inplace=True)

X= df.drop(['Condition'],axis=1)
y= df['Condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf=RandomForestClassifier(n_estimators=1000, criterion='gini')
clf.fit(X_train,y_train)

@app.route('/', methods=['GET'])
def main():
    return "Test Test Test"

@app.route('/predict/<v1>/<v2>/<v3>/<v4>/<v5>/<v6>/<v7>/<v8>/<v9>', methods=['POST', "GET"])
def scrapeData(v1, v2, v3, v4, v5, v6, v7, v8, v9):
    y_pred=clf.predict([[v1, v2, v3, v4, v5, v6, v7, v8, v9]])
    return y_pred

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)
