import streamlit as st
import pandas as pd
from pickle import load

model = load(open('diabetes.pkl','rb'))

## Paramaters for prediction
# 1. Pregnancies
# 2. Glucose
# 3. Blood Pressure
# 4. Skin Thickness
# 5. Insulin
# 6. BMI
# 7. Diabetes predigree function
# 8. Age

st.title('Diabetes Prediction App')
st.subheader('This app will help doctors automate the process of carrying out repetitive tasks of analysing patients for diabetes')
st.write('### Please enter the patient\'s information for examination')
col1, col2, col3, col4 = st.beta_columns(4)

pregnancy   = col1.number_input('Number of times pregnant')
insulin     = col1.number_input('Insulin Level')

glucose     = col2.number_input('Glucose Level')
bmi         = col2.number_input('BMI')

blood_pressure  = col3.number_input('Blood Pressure')
dpf             = col3.number_input('Diabetes Predigree Function')

skin            = col4.number_input('Skin Thickness')
age             = col4.number_input('Age')

check           = st.button('Check Patient')

# A function to predict the 
def predict(params = [pregnancy, insulin, glucose, bmi, blood_pressure, dpf, skin, age]):
    prediction = model.predict(params)
    return prediction

if check:
    prediction = predict([[pregnancy, insulin, glucose, bmi, blood_pressure, dpf, skin, age]])
    if prediction >= 1:
        st.error('This patient is Diabetic :(')
    else:
        st.success('Hurray! This patient is Healthy!')
        st.balloons()