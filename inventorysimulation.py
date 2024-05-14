import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

def simulate_inventory(policy, duration, demand_mean, demand_std, distribution, service_level_target, s=None, Q=None, S=None, R=None):
    np.random.seed(0)
    days = np.arange(1, duration + 1)

    if distribution == "Normal":
        demand = np.random.normal(demand_mean, demand_std, duration)
    elif distribution == "Poisson":
        demand = np.random.poisson(demand_mean, duration)
    elif distribution == "Uniform":
        demand = np.random.uniform(demand_mean - demand_std, demand_mean + demand_std, duration)

    d_mu = 5  # Mean demand
    d_std = 1  # Standard deviation of demand
    lead_times = np.maximum(1, np.random.normal(loc=d_mu, scale=d_std, size=duration).astype(int))

    safety_stock = norm.ppf(service_level_target) * demand_std

    inventory_levels = np.zeros(duration)
    orders = np.zeros(duration)
    in_transit = np.zeros(duration)
    shortages = np.zeros(duration)
    on_hand = np.zeros(duration)

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

    results_df = pd.DataFrame({
        "Time": days,
        "Inventory Level": inventory_levels.astype(int),
        "Orders Placed": orders.astype(int),
        "In Transit": in_transit.astype(int),
        "Shortages": shortages.astype(int),
        "On Hand": on_hand.astype(int)
    })

    return results_df, service_level_achieved

st.title("Inventory Simulation")

duration = st.number_input("Duration (days)", value=30)
demand_mean = st.number_input("Demand Mean", value=50)
demand_std = st.number_input("Demand Std Dev", value=10)
policy = st.selectbox("Policy", ["s,Q"])
distribution = st.selectbox("Demand Distribution", ["Normal", "Poisson", "Uniform"])
service_level = st.slider("Service Level", min_value=0.0, max_value=1.0, value=0.95)
s = st.number_input("Reorder Point (s)", value=20)
Q = st.number_input("Order Quantity (Q)", value=40)

if st.button("Run Simulation"):
    results_df, service_level_achieved = simulate_inventory(policy, duration, demand_mean, demand_std, distribution, service_level, s, Q)
    st.dataframe(results_df)
    st.write(f"Achieved Service Level: {service_level_achieved:.2f}%")

    st.download_button(
        label="Download Report as CSV",
        data=results_df.to_csv(index=False),
        file_name='inventory_control_report.csv',
        mime='text/csv',
    )

    st.write("## Inventory Levels Over Time")
    plt.figure(figsize=(10, 5))
    plt.plot(results_df["Time"], results_df["Inventory Level"], label='Inventory Level')
    plt.plot(results_df["Time"], results_df["Orders Placed"], label='Orders Placed', linestyle='--')
    plt.plot(results_df["Time"], results_df["On Hand"], label='On Hand Inventory', linestyle='--')
    plt.plot(results_df["Time"], results_df["Shortages"], label='Shortages', linestyle='-.')
    plt.title(f'Inventory Simulation with Policy: {policy}')
    plt.xlabel('Time (days)')
    plt.ylabel('Units')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

