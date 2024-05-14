import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Define demand generation based on distribution choice
def generate_demand(distribution, duration, mean, std_dev):
    if distribution == "Normal":
        return np.random.normal(loc=mean, scale=std_dev, size=duration)
    elif distribution == "Poisson":
        return np.random.poisson(lam=mean, size=duration)
    elif distribution == "Uniform":
        return np.random.uniform(low=mean - std_dev, high=mean + std_dev, size=duration)

# Calculate safety stock
def calculate_safety_stock(mean, std_dev, service_level):
    z = norm.ppf(service_level)
    return z * std_dev

# Define a simple inventory policy simulation with stochastic lead times
def simulate_inventory(policy, duration, demand, s, Q, S, R, service_level_target, std_dev):
    # Use the teacher's approach for stochastic lead times
    d_mu = 5  # Mean demand
    d_std = 1  # Standard deviation of demand
    lead_times = np.maximum(1, np.random.normal(loc=d_mu, scale=d_std, size=duration).astype(int))

    safety_stock = calculate_safety_stock(mean=np.mean(demand), std_dev=std_dev, service_level=service_level_target)

    inventory_levels = np.zeros(duration)
    orders = np.zeros(duration)
    in_transit = np.zeros(duration)
    shortages = np.zeros(duration)
    on_hand = np.zeros(duration)

    # Initial inventory level
    inventory_levels[0] = S if 'S' in policy else 0

    for t in range(1, duration):
        # Update on-hand inventory and shortages
        on_hand[t] = max(0, inventory_levels[t-1] - demand[t-1])
        shortages[t] = max(0, demand[t-1] - inventory_levels[t-1])

        # Check for arrival of orders
        if t >= lead_times[t]:
            inventory_levels[t] = on_hand[t] + in_transit[t - lead_times[t]]
        else:
            inventory_levels[t] = on_hand[t]

        # Place orders based on the selected policy
        if policy == 's,Q' and inventory_levels[t] < s:
            orders[t] = Q
            if t + lead_times[t] < duration:
                in_transit[t + lead_times[t]] += Q
        elif policy == 's,S' and inventory_levels[t] < s:
            order_quantity = S - inventory_levels[t]
            orders[t] = order_quantity
            if t + lead_times[t] < duration:
                in_transit[t + lead_times[t]] += order_quantity

        inventory_levels[t] = max(0, inventory_levels[t])  # Ensure no negative inventory

    service_level_achieved = (1 - np.sum(shortages) / np.sum(demand)) * 100
    return inventory_levels.astype(int), orders.astype(int), in_transit.astype(int), shortages.astype(int), on_hand.astype(int), service_level_achieved

def send_email(file_path, to_email):
    from_email = "facilityreport1@gmail.com"
    password = "cancan2002"

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

# Widgets for input parameters
duration = st.number_input("Duration (days)", value=30)
mean_demand = st.number_input("Demand Mean:", value=50)
std_dev = st.number_input("Demand Std Dev:", value=10)
policy = st.selectbox("Policy:", ["s,Q", "R,s,Q", "s,S", "R,s,S"])
distribution = st.selectbox("Demand Distribution:", ["Normal", "Poisson", "Uniform"])
service_level = st.slider('Service Level:', 0.80, 1.00, 0.95)
email = st.text_input("Email")

further_calc_button = st.button("Further Calculation")

if further_calc_button:
    if policy == "s,Q":
        s = st.number_input("Reorder Point (s):", value=20)
        Q = st.number_input("Order Quantity (Q):", value=40)
        R = None
        S = None
    elif policy == "R,s,Q":
        s = st.number_input("Reorder Point (s):", value=20)
        Q = st.number_input("Order Quantity (Q):", value=40)
        R = st.number_input("Review Period (R):", value=10)
        S = None
    elif policy == "s,S":
        s = st.number_input("Reorder Point (s):", value=20)
        S = st.number_input("Order-up-to Level (S):", value=100)
        Q = None
        R = None
    elif policy == "R,s,S":
        s = st.number_input("Reorder Point (s):", value=20)
        S = st.number_input("Order-up-to Level (S):", value=100)
        R = st.number_input("Review Period (R):", value=10)
        Q = None

    if st.button("Run Simulation"):
        demand = generate_demand(distribution, duration, mean_demand, std_dev)
        inventory_levels, orders, in_transit, shortages, on_hand, service_level_achieved = simulate_inventory(
            policy, duration, demand, s, Q, S, R, service_level, std_dev)

        # Plotting results
        fig, ax = plt.subplots()
        ax.plot(inventory_levels, label='Inventory Level')
        ax.plot(orders, label='Orders Placed', linestyle='--')
        ax.plot(on_hand, label='On Hand Inventory', linestyle='--')
        ax.plot(shortages, label='Shortages', linestyle='-.')
        ax.set_title(f'Inventory Simulation with Policy: {policy}')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Units')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Writing results to CSV
        results_df = pd.DataFrame({
            'Time': range(duration),
            'Inventory Level': inventory_levels,
            'Orders Placed': orders,
            'In Transit': in_transit,
            'Shortages': shortages,
            'On Hand': on_hand
        })
        file_path = 'inventorycontrol.csv'
        results_df.to_csv(file_path, index=False)
        st.success(f"Results saved to {file_path}")
        st.write(f"Achieved Service Level: {service_level_achieved:.2f}%")
        st.download_button('Download CSV', data=results_df.to_csv(index=False), file_name=file_path, mime='text/csv')

        if st.button("Send Report to my Mail"):
            send_email(file_path, email)
            st.success(f"Results emailed to {email}")


