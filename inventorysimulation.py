import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Background image setup
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background: url("https://i.imgur.com/kox6xPx.png");
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

if st.button('Press Me', key='press_me_button', on_click=lambda: st.session_state.update(button_clicked=True)):
    st.session_state.button_clicked = True

if st.session_state.button_clicked:
    st.markdown(
        """
        <style>
        .modal-background {
            background-color: rgba(0, 0, 0, 0.8);
            width: 100%;
            height: 50%;
            position: fixed;
            top: 25%;
            left: 0;
            z-index: 1;
        }
        .modal-content {
            background-color: rgba(0,0,0,0.8);
            color: white;
            padding: 20px;
            border-radius: 10px;
            width: 80%;
            margin: 0 auto;
            margin-top: 100px;
            z-index: 2;
            position: relative;
        }
        </style>
        <div class="modal-background"></div>
        <div class="modal-content">
            <div id="inventory-management" style="display: block;">
        """, unsafe_allow_html=True
    )

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

        # Calculate Cycle Service Level and Period Service Level
        stock_out_period = np.full(duration, False, dtype=bool)
        stock_out_cycle = []

        for t in range(1, duration):
            if orders[t] > 0 and shortages[t] > 0:
                stock_out_cycle.append(True)
            else:
                stock_out_cycle.append(False)
            if shortages[t] > 0:
                stock_out_period[t] = True

        SL_alpha = 1 - sum(stock_out_cycle) / len(stock_out_cycle)
        SL_period = 1 - sum(stock_out_period) / duration

        return inventory_levels.astype(int), orders.astype(int), in_transit.astype(int), shortages.astype(int), on_hand.astype(int), service_level_achieved, SL_alpha, SL_period

    st.title("Inventory Management")

    # Initialize session state
    if 'show_parameters' not in st.session_state:
        st.session_state.show_parameters = [False, False]

    # Widgets for input parameters
    col1, col2 = st.columns(2)

    with col1:
        st.header("Policy 1")
        duration1 = st.number_input("Duration (days)", value=30, key="duration1")
        mean_demand1 = st.number_input("Demand Mean:", value=50, key="mean_demand1")
        std_dev1 = st.number_input("Demand Std Dev:", value=10, key="std_dev1")
        policy1 = st.selectbox("Policy:", ["s,Q", "R,s,Q", "s,S", "R,s,S"], key="policy1")
        distribution1 = st.selectbox("Demand Distribution:", ["Normal", "Poisson", "Uniform"], key="distribution1")
        
        if st.button("Further Calculation for Policy 1"):
            st.session_state.show_parameters[0] = True

        if st.session_state.show_parameters[0]:
            if policy1 == "s,Q":
                s1 = st.number_input("Reorder Point (s):", value=20, key="s1")
                Q1 = st.number_input("Order Quantity (Q):", value=40, key="Q1")
                R1 = None
                S1 = None
            elif policy1 == "R,s,Q":
                R1 = st.number_input("Review Period (R):", value=10, key="R1")
                s1 = st.number_input("Reorder Point (s):", value=20, key="s1")
                Q1 = st.number_input("Order Quantity (Q):", value=40, key="Q1")
                S1 = None
            elif policy1 == "s,S":
                s1 = st.number_input("Reorder Point (s):", value=20, key="s1")
                S1 = st.number_input("Order-up-to Level (S):", value=100, key="S1")
                R1 = None
                Q1 = None
            elif policy1 == "R,s,S":
                R1 = st.number_input("Review Period (R):", value=10, key="R1")
                s1 = st.number_input("Reorder Point (s):", value=20, key="s1")
                S1 = st.number_input("Order-up-to Level (S):", value=100, key="S1")
                Q1 = None

    with col2:
        st.header("Policy 2")
        duration2 = st.number_input("Duration (days)", value=30, key="duration2")
        mean_demand2 = st.number_input("Demand Mean:", value=50, key="mean_demand2")
        std_dev2 = st.number_input("Demand Std Dev:", value=10, key="std_dev2")
        policy2 = st.selectbox("Policy:", ["s,Q", "R,s,Q", "s,S", "R,s,S"], key="policy2")
        distribution2 = st.selectbox("Demand Distribution:", ["Normal", "Poisson", "Uniform"], key="distribution2")
        
        if st.button("Further Calculation for Policy 2"):
            st.session_state.show_parameters[1] = True

        if st.session_state.show_parameters[1]:
            if policy2 == "s,Q":
                s2 = st.number_input("Reorder Point (s):", value=20, key="s2")
                Q2 = st.number_input("Order Quantity (Q):", value=40, key="Q2")
                R2 = None
                S2 = None
            elif policy2 == "R,s,Q":
                R2 = st.number_input("Review Period (R):", value=10, key="R2")
                s2 = st.number_input("Reorder Point (s):", value=20, key="s2")
                Q2 = st.number_input("Order Quantity (Q):", value=40, key="Q2")
                S2 = None
            elif policy2 == "s,S":
                s2 = st.number_input("Reorder Point (s):", value=20, key="s2")
                S2 = st.number_input("Order-up-to Level (S):", value=100, key="S2")
                R2 = None
                Q2 = None
            elif policy2 == "R,s,S":
                R2 = st.number_input("Review Period (R):", value=10, key="R2")
                s2 = st.number_input("Reorder Point (s):", value=20, key="s2")
                S2 = st.number_input("Order-up-to Level (S):", value=100, key="S2")
                Q2 = None

    service_level = st.slider('Service Level:', 0.80, 1.00, 0.95)

    if st.button("Run Simulation"):
        demand1 = generate_demand(distribution1, duration1, mean_demand1, std_dev1)
        demand2 = generate_demand(distribution2, duration2, mean_demand2, std_dev2)
        
        inventory_levels1, orders1, in_transit1, shortages1, on_hand1, service_level_achieved1, SL_alpha1, SL_period1 = simulate_inventory(
            policy1, duration1, demand1, s1, Q1, S1, R1, service_level, std_dev1)

        inventory_levels2, orders2, in_transit2, shortages2, on_hand2, service_level_achieved2, SL_alpha2, SL_period2 = simulate_inventory(
            policy2, duration2, demand2, s2, Q2, S2, R2, service_level, std_dev2)

        # Plotting results
        fig, ax = plt.subplots()
        ax.plot(inventory_levels1, label=f'Inventory Level (Policy 1: {policy1})')
        ax.plot(orders1, label=f'Orders Placed (Policy 1: {policy1})', linestyle='--')
        ax.plot(on_hand1, label=f'On Hand Inventory (Policy 1: {policy1})', linestyle='--')
        ax.plot(shortages1, label=f'Shortages (Policy 1: {policy1})', linestyle='-.')

        ax.plot(inventory_levels2, label=f'Inventory Level (Policy 2: {policy2})')
        ax.plot(orders2, label=f'Orders Placed (Policy 2: {policy2})', linestyle='--')
        ax.plot(on_hand2, label=f'On Hand Inventory (Policy 2: {policy2})', linestyle='--')
        ax.plot(shortages2, label=f'Shortages (Policy 2: {policy2})', linestyle='-.')

        ax.set_title(f'Inventory Simulation Comparison')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Units')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Writing results to CSV
        results_df1 = pd.DataFrame({
            'Time': range(duration1),
            'Inventory Level': inventory_levels1,
            'Orders Placed': orders1,
            'In Transit': in_transit1,
            'Shortages': shortages1,
            'On Hand': on_hand1
        })

        results_df2 = pd.DataFrame({
            'Time': range(duration2),
            'Inventory Level': inventory_levels2,
            'Orders Placed': orders2,
            'In Transit': in_transit2,
            'Shortages': shortages2,
            'On Hand': on_hand2
        })

        # Ensure sheet names are valid by removing any special characters
        valid_policy1 = ''.join(e for e in policy1 if e.isalnum())
        valid_policy2 = ''.join(e for e in policy2 if e.isalnum())

        file_path = 'inventorycontrol_comparison.xlsx'
        with pd.ExcelWriter(file_path) as writer:
            results_df1.to_excel(writer, sheet_name=f'Policy1_{valid_policy1}', index=False)
            results_df2.to_excel(writer, sheet_name=f'Policy2_{valid_policy2}', index=False)

        st.success(f"Results saved to {file_path}")
        st.write(f"Service Level for Policy 1: {service_level_achieved1:.2f}% (Cycle: {SL_alpha1:.2f}, Period: {SL_period1:.2f})")
        st.write(f"Service Level for Policy 2: {service_level_achieved2:.2f}% (Cycle: {SL_alpha2:.2f}, Period: {SL_period2:.2f})")
        st.download_button('Download Comparison Report', data=open(file_path, 'rb').read(), file_name=file_path, mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown(
        """
        <style>
        #press-me-button {
            position: absolute;
            top: 300px;
            left: 50%;
            transform: translateX(-50%);
            background-color: #000000;
            color: white;
            font-size: 24px;
            padding: 15px 30px;
            border: none;
            cursor: pointer;
        }
        </style>
        <div>
            <button id="press-me-button">Press Me</button>
        </div>
        """, unsafe_allow_html=True
    )

    # JavaScript to handle button click
    js_code = """
    <script>
    document.getElementById('press-me-button').onclick = function() {
        const streamlit = window.parent;
        streamlit.postMessage({isOpen: true}, '*');
        document.getElementById('press-me-button').style.display = 'none';
    }
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)
