import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os

from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment

with st.sidebar:
    st.image("https://miro.medium.com/v2/resize:fit:775/0*rZecOAy_WVr16810")
    st.title("AutoML")
    st.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling and PyCaret.",icon="ℹ️")
    choice = st.radio("Navigation", ["Upload Data", "Profile Data", "Train Data", "Download Model"])

data =None
#st.title("Welcome to AutoML!")
if os.path.exists("sourceddata.csv"):
    df = pd.read_csv("sourceddata.csv", index_col=None)
    data = True

if choice == "Upload Data":
    st.title("Upload your Data for Modelling!")
    file = st.file_uploader("Upload your Dataset Here!")
    if file:
        df = pd.read_csv(file, index_col=None)
        data = True
        df.to_csv("sourceddata.csv", index=None)
        st.dataframe(df, use_container_width=True)
        

if choice == "Profile Data":
    st.title("Automated Exploratory Data Analysis")
    if data:
        profile_report = ProfileReport(df, minimal=True)
        st_profile_report(profile_report)
    else:
        st.info("Upload Data to start!!!",icon="🚨")

if choice == "Train Data":
    st.title("Machine Learning - Train your Model!")
    if data:
        target = st.selectbox("Select Your Target Column", df.columns,len(df.columns)-1)
        if st.button("Train model"):
            if target in df.select_dtypes(include=['object']).columns.tolist():
                pycar = ClassificationExperiment() #using classification
                type = "Classification"
            else:
                pycar = RegressionExperiment()#using regression
                type = "Regression"


            st.info(f"The selected prediction column is: {target}, therefore it is a {type} problem. Below is the ML settings:")
            
            pycar.setup(df, target=target) 
            st.dataframe(pycar.pull(), use_container_width=True)
            st.info(f"Various {type} models and their performances on the dataset:")
            best_model = pycar.compare_models()
            st.dataframe(pycar.pull())
            st.info(f"Best performing model: {best_model}")
            pycar.save_model(best_model, "best_model")
        
    else:
        st.info("Upload Data to start!!!",icon="🚨")

if choice == "Download Model":
    with open("best_model.pkl", "rb") as f:
        st.download_button("Download the best model", f, "trained_model.pkl")
    
