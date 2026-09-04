import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib 

kmeans = joblib.load("kmeans_model.pkl") 
scaler = joblib.load("scaler.pkl") 

st.title("Customer Segmmentation App") 
st.write("Enter customer details to predict the segment.") 

age = st.number_input("Age", min_value= 18, max_value = 100, value = 35)
income = st.number_input("Income",min_value= 0, max_value = 200000, value = 50000)
total_spending = st.number_input("Total_Spending (sum of purchases)", min_value = 0, max_value =5000, value = 1000)
num_web_purchase = st.number_input("Number of Web Purchases", min_value= 0, max_value= 100, value = 10) 
num_store_purchase = st.number_input("Number of Stores Purchases", min_value= 0, max_value= 100, value = 10)
num_web_visit = st.number_input("Number of Web Visit per Month", min_value= 0, max_value= 50, value = 3) 
recency = st.number_input("Recency (days since last purchase)", min_value= 0, max_value= 365, value = 30) 

input_data = pd.DataFrame({
    "Age" : [age] ,
    "Income": [income] ,
    "Total_Spending": [total_spending] ,
    "NumWebPurchases": [num_web_purchase] ,
    "NumStorePurchases": [num_store_purchase] ,
    "NumWebVisitsMonth": [num_web_visit] ,
    "Recency" : [recency]
})

input_scaled = scaler.transform(input_data) 

if st.button("Predict Segment"):

    cluster = kmeans.predict(input_scaled) [0]

    st.success(f"Predicted Segment: Cluster {cluster}")

    st.write("""""
            Cluster 0: High budget, web visiters
            Cluster 1: High spending
            Cluster2: Web visitors 
            """) 