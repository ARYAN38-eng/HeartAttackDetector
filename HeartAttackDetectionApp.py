import streamlit as st 
import pandas as pd
import numpy as np
import pickle 
ohe=pickle.load(open('heartattackohe.pkl','rb'))
rfc=pickle.load(open('heartattackrf.pkl','rb'))
df=pickle.load(open('heartattackdf.pkl','rb'))
sc=pickle.load(open('heartattackSC.pkl','rb'))
st.title("Heart Attack Detection")
st.sidebar.image("iStock-1156928054.jpg", caption="Heart Attack", use_column_width=True)
description = """
A heart attack occurs when the flow of blood to the heart is severely reduced or blocked. The blockage is usually due to a buildup of fat, cholesterol and other substances in the heart (coronary) arteries. The fatty, cholesterol-containing deposits are called plaques. The process of plaque buildup is called atherosclerosis.

Sometimes, a plaque can rupture and form a clot that blocks blood flow. A lack of blood flow can damage or destroy part of the heart muscle.
"""
st.sidebar.markdown(description)
age=st.selectbox('Select Age',df['age'].unique())
user=st.selectbox("Enter your Gender",['Male','Female'])
if user=='Male':
    sex=1
else:
    sex=0  
chest_pain=st.selectbox("Enter Chest Pain Type",df['chest_pain'].unique())
restingBP=st.number_input("Enter Resting Blood Pressure")
serum_cholesterol=st.number_input("Enter Serum Cholesterol")	
user1=st.selectbox("Enter Fasting Blood Sugar",['Less than equal to 120','More than 120'])
if user1=='Less than equal to 120':
    fasting_sugar=0
else:
    fasting_sugar=1	
user2=st.selectbox('Resting Electrocardiographic',['Normal','Abnormality','Hypertrophy'])
if user2=='Normal':
    resting_electrocardiographic=0
elif user2=='Abnormality':
    resting_electrocardiographic=1
else:
    resting_electrocardiographic=2
max_heartrate=st.number_input("Enter Maximum Heart Rate")
user3=st.selectbox("Exercise Induced Angina",['NO',"YES"])
if user3=='NO':
    exercise_induced_angina=0
else:
    exercise_induced_angina=1
oldpeak=st.number_input("oldpeak(ST Depression)")
slopofpeakexerciseSTsegment=st.selectbox("slop of peak exercise ST segment",df['slop of peak exercise ST segment'].unique())
NMVCF=st.selectbox("NMVCF",df['NMVCF'].unique())
thalassemia=st.selectbox("thalassemia",df['thalassemia'].unique())



if st.button("Predict"):
    new_df=pd.DataFrame([{
    
    'age':age,
    'sex':sex,
    'chest_pain':chest_pain,
    'restingBP':restingBP,
    'serum_cholesterol':serum_cholesterol,
    'fasting_sugar':fasting_sugar,
    'resting_electrocardiographic':resting_electrocardiographic,
    'max_heartrate':max_heartrate,
    'exercise_induced_angina':exercise_induced_angina,
    'oldpeak(ST Depression)':oldpeak,
    'slop of peak exercise ST segment':slopofpeakexerciseSTsegment,
    'NMVCF':NMVCF,
    'thalassemia':thalassemia,
   }
    
    ])
    columns_to_include=df.select_dtypes('object').columns
    encoded__cat_df=ohe.transform(new_df[columns_to_include])
    encoded_df=pd.DataFrame(encoded__cat_df,columns=ohe.get_feature_names_out(columns_to_include))
    new_df.reset_index(drop=True,inplace=True)
    encoded_df=pd.concat([new_df,encoded_df],axis=1)
    encoded_df.drop(columns=columns_to_include,inplace=True)
    scaled_df=sc.transform(encoded_df)
    for_scaled_cols=encoded_df.columns
    scaled_df=pd.DataFrame(scaled_df,columns=for_scaled_cols)
    prediction=rfc.predict(scaled_df)
    if prediction[0] ==0:
        st.write("No Heart Attack Detected")
    else:
        st.write("Heart Attack Detected")
        
