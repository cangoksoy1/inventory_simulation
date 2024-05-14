# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11e-RcWntViT9vJeT1Z5JpKYpz5VvDgYC
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def simulate_inventory(duration, demand_mean, demand_std, policy, distribution, service_level, s=None, Q=None):
    # ... (your existing simulation code) ...
    return results_df, service_level_achieved

def send_email(file_path, to_email):
    from_email = "your_email@example.com"
    password = "your_password"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = "Inventory Simulation Results"
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(open(file_path, "rb").read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename= "inventorycontrol.csv"')
    msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, password)
    server.sendmail(from_email, to_email, msg.as_string())
    server.quit()

st.title("Inventory Simulation")

duration = st.number_input("Duration (days)", value=30)
demand_mean = st.number_input("Demand Mean", value=50)
demand_std = st.number_input("Demand Std Dev", value=10)
policy = st.selectbox("Policy", ["s,Q"])
distribution = st.selectbox("Demand Distribution", ["Normal", "Poisson"])
service_level = st.slider("Service Level", min_value=0.0, max_value=1.0, value=0.95)
s = st.number_input("Reorder Point (s)", value=20)
Q = st.number_input("Order Quantity (Q)", value=40)
email = st.text_input("Email")

if st.button("Run Simulation"):
    results_df, service_level_achieved = simulate_inventory(duration, demand_mean, demand_std, policy, distribution, service_level, s, Q)
    st.dataframe(results_df)
    st.write(f"Achieved Service Level: {service_level_achieved:.2f}")

    file_path = "inventorycontrol.csv"
    results_df.to_csv(file_path, index=False)
    send_email(file_path, email)
    st.success(f"Results saved to {file_path} and emailed to {email}")