import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define demand generation based on distribution choice
def generate_demand(distribution, duration, mean, std_dev):
    if distribution == "Normal":
        return np.maximum(np.random.normal(loc=mean, scale=std_dev, size=duration), 0).astype(int)
    elif distribution == "Poisson":
        return np.random.poisson(lam=mean, size=duration)
    elif distribution == "Uniform":
        return np.random.uniform(low=mean - std_dev, high=mean + std_dev, size=duration).astype(int)

# Calculate safety stock
def calculate_safety_stock(mean, std_dev, service_level):
    z = norm.ppf(service_level)
    return z * std_dev

# Define a simple inventory policy simulation with stochastic lead times
def simulate_inventory(policy, duration, demand, s, Q, S, R, service_level_target, std_dev):
    L, alpha = 1, 0.95
    d_mu = 5  # Mean demand
    d_std = 1  # Standard deviation of demand
    R = R if R else 4  # Review period if not provided

    z = norm.ppf(alpha)
    x_std = d_std * np.sqrt(L + R)
    Ss = np.round(x_std * z).astype(int)
    Q = d_mu * R
    Cs = Q / 2
    Is = d_mu * L
    S = Ss + Q + Is

    hand = np.zeros(duration, dtype=int)
    transit = np.zeros((duration, L + 1), dtype=int)

    # Initialize hand and transit based on the initial demand
    hand[0] = S - demand[0]
    transit[1, -1] = demand[0]

    stock_out_period = np.full(duration, False, dtype=bool)
    stock_out_cycle = []

    for t in range(1, duration):
        # Update hand inventory and shortages
        hand[t] = max(0, hand[t - 1] - demand[t - 1])
        if hand[t] < demand[t - 1]:
            stock_out_period[t] = True
            stock_out_cycle.append(t)

        # Check for arrival of orders
        if t >= L:
            hand[t] += transit[t - L, -1]

        # Place orders based on the selected policy
        if policy == 's,Q' and hand[t] < s:
            order_qty = Q
        elif policy == 's,S' and hand[t] < s:
            order_qty = S - hand[t]
        elif policy == 'R,s,Q' and t % R == 0 and hand[t] < s:
            order_qty = Q
        elif policy == 'R,s,S' and t % R == 0 and hand[t] < s:
            order_qty = S - hand[t]
        else:
            order_qty = 0

        if order_qty > 0 and t + L < duration:
            transit[t + L, -1] += order_qty

    service_level_achieved = (1 - np.sum(stock_out_period) / duration) * 100

    # Cycle Service Level and Period Service Level
    cycle_service_level = 1 - (len(stock_out_cycle) / duration)
    period_service_level = 1 - (np.sum(stock_out_period) / duration)

    return hand.astype(int), transit.astype(int), stock_out_period.astype(int), service_level_achieved, cycle_service_level, period_service_level

st.title("Inventory Simulation")

# Initialize session state
if 'show_parameters' not in st.session_state:
    st.session_state.show_parameters = False

# Widgets for input parameters
duration = st.number_input("Duration (days)", value=30)
mean_demand = st.number_input("Demand Mean:", value=50)
std_dev = st.number_input("Demand Std Dev:", value=10)
policy = st.selectbox("Policy:", ["s,Q", "R,s,Q", "s,S", "R,s,S"])
distribution = st.selectbox("Demand Distribution:", ["Normal", "Poisson", "Uniform"])
service_level = st.slider('Service Level:', 0.80, 1.00, 0.95)

if st.button("Further Calculation"):
    st.session_state.show_parameters = True

if st.session_state.show_parameters:
    if policy == "s,Q":
        s = st.number_input("Reorder Point (s):", value=20)
        Q = st.number_input("Order Quantity (Q):", value=40)
        R = None
        S = None
    elif policy == "R,s,Q":
        R = st.number_input("Review Period (R):", value=10)
        s = st.number_input("Reorder Point (s):", value=20)
        Q = st.number_input("Order Quantity (Q):", value=40)
        S = None
    elif policy == "s,S":
        s = st.number_input("Reorder Point (s):", value=20)
        S = st.number_input("Order-up-to Level (S):", value=100)
        R = None
        Q = None
    elif policy == "R,s,S":
        R = st.number_input("Review Period (R):", value=10)
        s = st.number_input("Reorder Point (s):", value=20)
        S = st.number_input("Order-up-to Level (S):", value=100)
        Q = None

    if st.button("Run Simulation"):
        demand = generate_demand(distribution, duration, mean_demand, std_dev)
        hand, transit, stock_out_period, service_level_achieved, cycle_service_level, period_service_level = simulate_inventory(
            policy, duration, demand, s, Q, S, R, service_level, std_dev)

        # Plotting results
        fig, ax = plt.subplots()
        ax.plot(hand, label='Hand Inventory')
        ax.plot(transit[:, -1], label='In Transit', linestyle='--')
        ax.set_title(f'Inventory Simulation with Policy: {policy}')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Units')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Writing results to CSV
        results_df = pd.DataFrame({
            'Time': range(duration),
            'Hand Inventory': hand,
            'In Transit': transit[:, -1],
            'Stock Out Period': stock_out_period
        })
        file_path = 'inventorycontrol.csv'
        results_df.to_csv(file_path, index=False)
        st.success(f"Results saved to {file_path}")
        st.write(f"Achieved Service Level: {service_level_achieved:.2f}%")
        st.write(f"Cycle Service Level: {cycle_service_level:.2f}")
        st.write(f"Period Service Level: {period_service_level:.2f}")
        st.download_button('Download CSV', data=results_df.to_csv(index=False), file_name=file_path, mime='text/csv')

