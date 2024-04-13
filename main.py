import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("diabetes.csv")
#print(df.tail(30))

def float_to_numeric(df,columns):
    for i in columns:
        df[i] = df[i].str.replace(',', '.').astype(float)
    return df

float_cols=["chol_hdl_ratio","bmi","waist_hip_ratio"]
df=float_to_numeric(df,float_cols)
df.head(15)

le=LabelEncoder()
df['gender']=le.fit_transform(df['gender'])
df.diabetes=df.diabetes.replace({"No diabetes":0,"Diabetes":1})# no diabetes =0 diabetes =1
#print(df.tail)

#print(df.info())

y=df.diabetes.values
X=df.drop(columns=["diabetes","patient_number"])
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=17,shuffle=True,stratify=y)

lr = LogisticRegression(solver='liblinear', random_state=42)
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)

#print("Accuracy Score :",accuracy_score(test_y,predictions))
#print("Classification Report \n",classification_report(test_y,predictions))

test_X.shape

test_X.iloc[16:17]

a=list(float(i) for i in input().split())
a1=np.array(a)
a1.reshape(1,-1)
a11=(pd.DataFrame(a1)).T
new_pred=lr.predict(a11)
if(new_pred==[0]):
  print('You Probably dont have diabetes')
else:
  print("BYE-BYE🗣️🗿")