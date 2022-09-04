import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.image("http://www.ehtp.ac.ma/images/lo.png")

st.markdown(f'<h1 style="color:#773723;text-align: center;font-size:48px;">{"KAGGLE CLOUD COMPUTING"}</h1>', unsafe_allow_html=True)
st.markdown(f'<h1 style="color:#da9954;text-align: center;font-size:36px;">{"DIABETES Prediction App"}</h1>', unsafe_allow_html=True)
st.markdown(f'<h1 style="color:#557caf;font-size:24px;">{"PROF : MR SAAD EDDINE AHALLI"}</h1>', unsafe_allow_html=True)


option = st.selectbox(
     'How would you like to use the prediction model?',
     ('','input parameters directly', 'Load a file of data'))


if option=='input parameters directly':
     st.balloons()



elif option=='Load a file of data':
    uploaded_file = st.file_uploader("Choose a file to load")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

    model_loan=pickle.load(open("Model.pkl", "rb"))

    if st.button('Predict'):
        prediction = model_loan.predict(df)
        prediction_proba = model_loan.predict_proba(df)*100
        df["Prediction"] = prediction
        st.balloons()
        st.write(df)
        st.write(prediction_proba)



