import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("https://docs.google.com/spreadsheets/d/1HjH89Ryw3NzT62sedvTC54_DqNkvRgTZn4XXTN4sZuM/export?format=csv&id=1HjH89Ryw3NzT62sedvTC54_DqNkvRgTZn4XXTN4sZuM&gid=1781495660")

df.isnull().sum()

# df.drop("Country",axis=1,inplace=True)

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

# age_columns = df.filter(like='Age_').columns
# gender_columns = df.filter(like='Gender_').columns
# contact_columns = df.filter(like='Contact_').columns

# No_risk_age = df.groupby(['Severity_None'])[age_columns].sum()
# No_risk_gender = df.groupby(['Severity_None'])[gender_columns].sum()
# No_risk_contact = df.groupby(['Severity_None'])[contact_columns].sum()

# Low_risk_age = df.groupby(['Severity_Mild'])[age_columns].sum()
# Low_risk_gender = df.groupby(['Severity_Mild'])[gender_columns].sum()
# Low_risk_contact = df.groupby(['Severity_Mild'])[contact_columns].sum()

# Moderate_risk_age = df.groupby(['Severity_Moderate'])[age_columns].sum()
# Moderate_risk_gender = df.groupby(['Severity_Moderate'])[gender_columns].sum()
# Moderate_risk_contact = df.groupby(['Severity_Moderate'])[contact_columns].sum()

# Severe_risk_age = df.groupby(['Severity_Severe'])[age_columns].sum()
# Severe_risk_gender = df.groupby(['Severity_Severe'])[gender_columns].sum()
# Severe_risk_contact = df.groupby(['Severity_Severe'])[contact_columns].sum()

df.drop(severity_columns,axis=1,inplace=True)

X= df.drop(['Condition'],axis=1)
y= df['Condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf=RandomForestClassifier(n_estimators=1000, criterion='gini')
clf.fit(X_train,y_train)
while True:
    
    print(clf.score(X_train, y_train))
    y_pred=clf.predict([[1,1,1,0,1,0,1,1,1]])
    print(y_pred)
    print(df.shape)
