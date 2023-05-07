import streamlit as st
import pickle

# Loading trained models
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('lr_model.pkl', 'rb'))

# Adding image to sidebar
st.sidebar.image('dsef.png.jpg', width=300)
st.sidebar.write("Prepared by: Aissa BERROUHOU")
st.sidebar.write("supervised by: Mr tali")
st.sidebar.image('aissa fb.jpg', use_column_width=True, output_format='PNG', width=250, clamp=False, channels='RGB')

# Prediction function
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction

# Home page of the application
def homepage():
    st.title('Diabetes Prediction')
    st.write('Welcome to our diabetes prediction application. Please enter the information below to know the probability of diabetes.')

    # User input data
    pregnancies = st.slider('Number of Pregnancies', 0, 17, 3)
    glucose = st.slider('Glucose', 0, 199, 117)
    blood_pressure = st.slider('Blood Pressure', 0, 122, 72)
    skin_thickness = st.slider('Skin Thickness', 0, 99, 23)
    insulin = st.slider('Insulin', 0.0, 846.0, 30.0)
    bmi = st.slider('Body Mass Index (BMI)', 0.0, 67.1, 32.0)
    dpf = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    age = st.slider('Age', 21, 81, 29)

    # Prediction button
    if st.button('Predict'):
        result = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
        st.write('The probability of having diabetes is:', round(result[0], 2))

# Application entry point
if __name__ == '__main__':
    homepage()
