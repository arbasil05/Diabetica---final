from django.shortcuts import render
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
import math
from math import floor
from .models import Patient
from django.core.mail import send_mail
from django.conf import settings
def home(request):
    return render(request,'home.html')

def choice(request):
    return render(request,'choice.html')

def forms(request):
    return render(request,'form.html')

def about(request):
    return render(request,'about.html')

def contact(request):
    return render(request,'Contact.html')

# def send_email(request):
#     if request.method == 'POST':
#         message = request.POST['message']
#         name = request.POST['name']
#         email = request.POST['email']
#         phno = request.POST['phno']

#         message = f"Name: {name}\nEmail: {email}\nPhone Number: {phno}\nQuery: {message}"
#         send_mail(
#             'Feedback | Help',
#             message,
#             settings.EMAIL_HOST_USER,
#             ['aafreenb716@gmail.com'],
#             fail_silently = False)
#     return render(request,'Home.html')

#fix this function vizz 🦖🦖🦖
def demo(request):
    # loading the dataset
    #basil's
    data = pd.read_csv("D:/cn 2.o/diabetes.csv")

    #vizz's
    #data=pd.read_csv('C:/basil/hehe/Diabetica---final/diabetes.csv')
   
    # Preprocessing the data
    data['chol_hdl_ratio'] = data['chol_hdl_ratio'].str.replace(',','.').astype(float)
    data['bmi'] = data['bmi'].str.replace(',','.').astype(float)
    data['waist_hip_ratio'] = data['waist_hip_ratio'].str.replace(',','.').astype(float)
    data['gender'] = data['gender'].replace({'male':1,'female':0})

    # Encode the target variable 'diabetes' using LabelEncoder
    le = LabelEncoder()
    data['diabetes'] = le.fit_transform(data['diabetes'])

    # Test and train data
    y = data['diabetes'].values
    X = data.drop(columns=['diabetes','patient_number'])
    train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=48)

    # Training the model
    lr = LogisticRegression(solver='liblinear',random_state=42)
    lr.fit(train_X,train_y)
    predictions = lr.predict(test_X)
    
    # Randomly choosing the row
    max_rows=[]
    indexes = list(data.index[data['diabetes'] == 1][:60])
    indexes2 = list(data.index[data['diabetes'] == 0][:60])
    max_rows=indexes.copy()
    max_rows.extend(indexes2)
    random_index = random.choice(max_rows)
    random_row = data.iloc[random_index]
    #print(random_row)
    a = random_row.iloc[1:-1]
    list_a = a.tolist()
    print(list_a[0])
    chol_valu = list_a[0]
    glu = list_a[1]
    HDL = list_a[2]
    age = list_a[4]
    BMI = list_a[8]
    gender = list_a[6]
    if (gender == 0):
        gender = "Female"
    else:
        gender = "Male"
    height = list_a[7]
    weight = list_a[8]
    
    ideal_chol = ideal_cholesterol(chol_valu,age)
    ideal_hdl = ideal_hdl_cholestrol(HDL,gender,age)
    ideal_glu = ideal_glucose_level(age,gender,glu)
    ideal_bmis = ideal_bmi(BMI)

    a1=np.array(a)
    a1.reshape(1,-1)
    a11=(pd.DataFrame(a1)).T
    print(a11)
    
    # Prepare the selected row for prediction
    random_row_values = random_row.drop(['diabetes','patient_number']).values.reshape(1,-1)
    new_pred = lr.predict(a11)

    # Decode the prediction
    no = "You probably don't have Diabetes."
    yes = "You probably have diabetes."
    if(new_pred==[0]):
        final = no
    else:
        final = yes

    print(final)
    return render(request,'demo.html',{'Final_Answer':final,
                                       'chol_valu':int(chol_valu),
                                       'glu':int(glu),
                                       'HDL':int(HDL),
                                       'BMI':BMI,
                                       'Age':int(age),
                                       'Gender' : gender,
                                       'height':int(height),
                                       'weight':int(weight),
                                       'Cholesterol_Analysis':ideal_chol,
                                       'HDL_Analysis':ideal_hdl,
                                       'Glucose_Analysis':ideal_glu,
                                       'BMI_Analysis' : ideal_bmis

                                       })

def diabetes(request):
    if request.method == 'POST':
        chol_level = int(request.POST.get("cholestrol"))
        glu_lvl = int(request.POST.get("glucose"))
        hdl_glu = int(request.POST.get("hdlchol"))
        age = int(request.POST.get("age"))
        gender = request.POST.get("gender")
        height_cms = int(request.POST.get("height"))
        weight_kgs = int(request.POST.get("weight"))
        sys_bp = int(request.POST.get("sysbp"))
        dia_bp = int(request.POST.get("diabp"))
        waist = int(request.POST.get("waist"))
        hip = int(request.POST.get("hip"))

        #orm:
        patient = Patient.objects.create(
            chol_level=chol_level,
            glu_lvl=glu_lvl,
            hdl_glu=hdl_glu,
            age=age,
            gender=gender,
            height_cms=height_cms,
            weight_kgs=weight_kgs,
            sys_bp=sys_bp,
            dia_bp=dia_bp,
            waist=waist,
            hip=hip
        )
        patient.save()

        
        
        # finding hdl_ratio:
        chol_hdl_ratio = round((chol_level) / (hdl_glu),1)
        #hip_waist_ratio:
        weight_pounds = int(round((weight_kgs * 2.2)))
        height_meter = height_cms / 100
        hip_waist_ratio = round((waist/hip),2)
        # print("Height in cm: ",height_cms)
        # print("Height in m: ",height_meter)

        #cms to inches:
        height_inches = int(round(height_cms/2.54))


        #gender_to_num;
        new_gen = gender.lower()
        if(new_gen=='female'):
            new_gen = 0
        elif (new_gen=='male'):
            new_gen = 1

        
        #bmi:
        bmi = weight_kgs / (height_meter * height_meter)
        bmi=round(bmi,1)
        #print(bmi)       
        # print(chol_hdl_ratio)
        #print(new_gen)
        # print("Cholestrol: ",chol_level)
        # print("Glucose level: ",glu_lvl)
        # print("HDL glucose level: ",hdl_glu)
        # print("HDL_Cholestrol_ratio: ",chol_hdl_ratio)
        # print("Age: ",age)
        # print("Gender:",new_gen)
        # print("Height in inches: ",height_inches)
        # print("Weight: ",weight_pounds)
        # print("bmi: ",bmi)
        # print("systolic blood pressure: ",sys_bp)
        # print("Diastolic Blood pressure: ",dia_bp)
        # print("Waist: ",waist)
        # print("Hip: ",hip)
        # print("Hip_Waist_Ratio: ",hip_waist_ratio)
        
        # print("Cholestrol: ",type(chol_level))
        # print("Glucose level: ",type(glu_lvl))
        # print("HDL glucose level: ",type(hdl_glu))
        # print("HDL_Cholestrol_ratio: ",type(chol_hdl_ratio))
        # print("Age: ",type(age))
        # print("Gender:",type(new_gen))
        # print("Height in inches: ",type(height_inches))
        # print("Weight: ",type(weight_pounds))
        # print("bmi: ",type(bmi))
        # print("systolic blood pressure: ",type(sys_bp))
        # print("Diastolic Blood pressure: ",type(dia_bp))
        # print("Waist: ",type(waist))
        # print("Hip: ",type(hip))
        # print("Hip_Waist_Ratio: ",type(hip_waist_ratio))
        list_1 = [chol_level,glu_lvl,hdl_glu,chol_hdl_ratio,age,new_gen,height_inches,weight_pounds,bmi,sys_bp,dia_bp,waist,hip,hip_waist_ratio]
        # print(list_1)
        # for i in list_1:
        #     print(type(i))
        df=pd.read_csv('C:/Users/admin/Desktop/basilfinalcnfinal/Diabetica---final/diabetes.csv')
        #print(df.head())
        def float_to_numeric(df,columns):
            for i in columns:
                df[i] = df[i].str.replace(',', '.').astype(float)
            return df
        float_cols = ["chol_hdl_ratio", "bmi", "waist_hip_ratio"]
        df = float_to_numeric(df, float_cols)        
        #print(df.head())

        le=LabelEncoder()
        df['gender']=le.fit_transform(df['gender'])
        df.diabetes=df.diabetes.replace({"No diabetes":0,"Diabetes":1})

        y=df.diabetes.values
        X=df.drop(columns=["diabetes","patient_number"])
        train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=38)

        lr = LogisticRegression(solver='liblinear',random_state=42)
        lr.fit(train_X, train_y)
        predictions = lr.predict(test_X)
        accuracy=accuracy_score(test_y,predictions)
        print("acc:",accuracy)
        #test_X.shape
        #test_X.iloc[16:17]
        a= list_1
        a1=np.array(a)
        a1.reshape(1,-1)
        a11=(pd.DataFrame(a1)).T
        new_pred=lr.predict(a11)
        no = "You don't have Diabetes"
        yes = "You probably have diabetes"
        if(new_pred==[0]):
            final = no
        else:
            final = yes

        return render(request,'Result.html',{'Final_answer':final,
                                             'chol_output':ideal_cholesterol(chol_level,age),
                                             'hdl_output':ideal_hdl_cholestrol(hdl_glu,gender,age),
                                             'glu_output':ideal_glucose_level(age,gender,glu_lvl),
                                             'bmi_output':ideal_bmi(bmi)})
        
        
    return render(request, 'home.html')

def ideal_cholesterol(ch_lvl, age):
    ideal_cholesterol_level = ""

    if age < 20:  # Teens and children
        if ch_lvl < 170:
            ideal_cholesterol_level = "Your cholesterol level is ideal for teens and children"
        else:
            ideal_cholesterol_level = "Your cholesterol level is not ideal for teens and children"
    elif 20 <= age < 30:  # Young adults
        if ch_lvl < 190:
            ideal_cholesterol_level = "Your cholesterol level is ideal for young adults"
        else:
            ideal_cholesterol_level = "Your cholesterol level is not ideal for young adults"
    elif 30 <= age < 60:  # Adults
        if ch_lvl < 200:
            ideal_cholesterol_level = "Your cholesterol level is ideal for adults"
        else:
            ideal_cholesterol_level = "Your cholesterol level is not ideal for adults"
    else:  
        if ch_lvl < 220:
            ideal_cholesterol_level = "Your cholesterol level is ideal for seniors"
        else:
            ideal_cholesterol_level = "Your cholesterol level is not ideal for seniors"

    return ideal_cholesterol_level

def ideal_hdl_cholestrol(hdl,gender,age):
    if gender.lower() == 'male':
        if age < 20 and hdl >= 45:
            return "The HDL cholesterol level is ideal for males under 20 years old."
        elif age >= 20 and hdl >= 40:
            return "The HDL cholesterol level is ideal for adult males."
        else:
            return "The HDL cholesterol level is not ideal."
    elif gender.lower() == 'female':
        if age < 20 and hdl >= 49:
            return "The HDL cholesterol level is ideal for females under 20 years old."
        elif age >= 20 and hdl >= 50:
            return "The HDL cholesterol level is ideal for adult females."
        else:
            return "The HDL cholesterol level is not ideal."
    

def ideal_glucose_level(age, gender, glucose_level):
    if gender.lower() == 'male':
        if age < 20:
            if 70 <= glucose_level <= 100:
                return "Glucose level is ideal."
            else:
                return "Glucose level is not ideal."
        else:
            if 70 <= glucose_level <= 99:
                return "Glucose level is ideal."
            else:
                return "Glucose level is not ideal."
    elif gender.lower() == 'female':
        if age < 20:
            if 70 <= glucose_level <= 100:
                return "Glucose level is ideal."
            else:
                return "Glucose level is not ideal."
        else:
            if 70 <= glucose_level <= 99:
                return "Glucose level is ideal."
            else:
                return "Glucose level is not ideal."
            
def ideal_bmi(bmi):
    if bmi<18.5:
        return f"Your BMI is {bmi} and you are underweight"
    elif bmi>=18.5 and bmi<=24.9:
        return f"Your BMI is {bmi} and you are in ideal state"
    elif bmi>=25 and bmi <=29.9:
        return f"Your BMI is {bmi} and you are overweight"
    elif bmi>30:
        return f"Your BMI is {bmi} and you have obesity"
        