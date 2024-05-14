import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def generate_demand(distribution, duration, mean, std_dev):
    if distribution == "Normal":
        return np.random.normal(loc=mean, scale=std_dev, size=duration)
    elif distribution == "Poisson":
        return np.random.poisson(lam=mean, size=duration)
    elif distribution == "Uniform":
        return np.random.uniform(low=mean - std_dev, high=mean + std_dev, size=duration)

def calculate_safety_stock(mean, std_dev, service_level):
    z = norm.ppf(service_level)
    return z * std_dev

def simulate_inventory(policy, duration, demand, s, Q, S, R, service_level_target, std_dev):
    d_mu = 5
    d_std = 1
    lead_times = np.maximum(1, np.random.normal(loc=d_mu, scale=d_std, size=duration).astype(int))

    safety_stock = calculate_safety_stock(mean=np.mean(demand), std_dev=std_dev, service_level=service_level_target)

    inventory_levels = np.zeros(duration, dtype=int)
    orders = np.zeros(duration, dtype=int)
    in_transit = np.zeros(duration, dtype=int)
    shortages = np.zeros(duration, dtype=int)
    on_hand = np.zeros(duration, dtype=int)

    inventory_levels[0] = S if 'S' in policy else 0

    for t in range(1, duration):
        on_hand[t] = max(0, inventory_levels[t-1] - demand[t-1])
        shortages[t] = max(0, demand[t-1] - inventory_levels[t-1])

        if t >= lead_times[t]:
            inventory_levels[t] = on_hand[t] + in_transit[t - lead_times[t]]
        else:
            inventory_levels[t] = on_hand[t]

        if policy == 's,Q' and inventory_levels[t] < s:
            orders[t] = Q
            if t + lead_times[t] < duration:
                in_transit[t + lead_times[t]] += Q
        elif policy == 's,S' and inventory_levels[t] < s:
            order_quantity = S - inventory_levels[t]
            orders[t] = order_quantity
            if t + lead_times[t] < duration:
                in_transit[t + lead_times[t]] += order_quantity

        inventory_levels[t] = max(0, inventory_levels[t])

    service_level_achieved = (1 - np.sum(shortages) / np.sum(demand)) * 100
    return inventory_levels, orders, in_transit, shortages, on_hand, service_level_achieved

def send_email(file_path, to_email):
    from_email = "goksoyy@mef.edu.tr"
    password = "euf4svrk"

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

# Initial inputs
duration = st.number_input("Duration (days)", value=30)
demand_mean = st.number_input("Demand Mean", value=50)
demand_std = st.number_input("Demand Std Dev", value=10)
policy = st.selectbox("Policy", ["s,Q", "R,s,Q", "s,S", "R,s,S"])
distribution = st.selectbox("Demand Distribution", ["Normal", "Poisson", "Uniform"])
service_level = st.slider("Service Level", min_value=0.0, max_value=1.0, value=0.95)

# Show additional inputs based on the selected policy
if policy == "s,Q":
    s = st.number_input("Reorder Point (s)", value=20)
    Q = st.number_input("Order Quantity (Q)", value=40)
elif policy == "R,s,Q":
    R = st.number_input("Review Period (R)", value=10)
    s = st.number_input("Reorder Point (s)", value=20)
    Q = st.number_input("Order Quantity (Q)", value=40)
elif policy == "s,S":
    s = st.number_input("Reorder Point (s)", value=20)
    S = st.number_input("Order-up-to Level (S)", value=80)
elif policy == "R,s,S":
    R = st.number_input("Review Period (R)", value=10)
    s = st.number_input("Reorder Point (s)", value=20)
    S = st.number_input("Order-up-to Level (S)", value=80)

email = st.text_input("Email")

if st.button("Run Simulation"):
    demand = generate_demand(distribution, duration, demand_mean, demand_std)
    
    if policy == "s,Q":
        inventory_levels, orders, in_transit, shortages, on_hand, service_level_achieved = simulate_inventory(
            policy, duration, demand, s, Q, None, None, service_level, demand_std)
    elif policy == "R,s,Q":
        inventory_levels, orders, in_transit, shortages, on_hand, service_level_achieved = simulate_inventory(
            policy, duration, demand, s, Q, None, R, service_level, demand_std)
    elif policy == "s,S":
        inventory_levels, orders, in_transit, shortages, on_hand, service_level_achieved = simulate_inventory(
            policy, duration, demand, s, None, S, None, service_level, demand_std)
    elif policy == "R,s,S":
        inventory_levels, orders, in_transit, shortages, on_hand, service_level_achieved = simulate_inventory(
            policy, duration, demand, s, None, S, R, service_level, demand_std)
    
    results_df = pd.DataFrame({
        "Time": range(duration),
        "Inventory Level": inventory_levels,
        "Orders Placed": orders,
        "In Transit": in_transit,
        "Shortages": shortages,
        "On Hand": on_hand
    })
    
    st.dataframe(results_df)
    st.write(f"Achieved Service Level: {service_level_achieved:.2f}%")

    file_path = "inventorycontrol.csv"
    results_df.to_csv(file_path, index=False)
    send_email(file_path, email)
    st.success(f"Results saved to {file_path} and emailed to {email}")

