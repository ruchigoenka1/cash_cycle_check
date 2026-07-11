import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go

st.set_page_config(page_title="Inventory & Cash Flow Simulation", layout="wide")

st.title("📦 Inventory & Cash Flow Simulation")
st.markdown("This dashboard simulates daily inventory operations and precisely tracks the **Cash Utilized** (Working Capital) based on credit terms to calculate the true cost of capital.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Demand & Lead Time")
avg_demand = st.sidebar.number_input("Average Daily Demand", min_value=1.0, value=50.0)
variation = st.sidebar.number_input("Demand Variation (Std Dev)", min_value=0.0, value=10.0)
lead_time = st.sidebar.number_input("Lead Time (Days)", min_value=1, value=5)

st.sidebar.header("2. Financial & Credit Terms")
unit_value = st.sidebar.number_input("Value of Product (Unit Cost $)", min_value=0.1, value=100.0)
physical_holding_cost = st.sidebar.number_input("Physical Holding Cost/Unit/Year ($)", min_value=0.0, value=10.0, help="Warehousing, insurance, etc. Excluding capital cost.")
cost_of_capital_pct = st.sidebar.number_input("Cost of Capital (Annual %)", min_value=0.0, value=12.0, help="Interest rate on utilized cash.") / 100.0
ordering_cost = st.sidebar.number_input("Ordering Cost per Order ($)", min_value=1.0, value=250.0)
credit_rx = st.sidebar.number_input("Credit Time from Supplier (Days)", min_value=0, value=30)
credit_given = st.sidebar.number_input("Credit Time given to Buyer (Days)", min_value=0, value=15)

st.sidebar.header("3. Service & Ordering Parameters")
service_level = st.sidebar.slider("Target Service Level (%)", min_value=50.0, max_value=99.99, value=95.0, step=0.1)

# --- CALCULATIONS FOR RECOMMENDED ROP & EOQ ---
z_score = stats.norm.ppf(service_level / 100.0)
safety_stock = z_score * variation * np.sqrt(lead_time)
recommended_rop = (avg_demand * lead_time) + safety_stock

# For EOQ, base Holding Cost = Physical + Base Capital Cost
base_capital_cost_per_unit = unit_value * cost_of_capital_pct
eoq_holding_cost = physical_holding_cost + base_capital_cost_per_unit
annual_demand = avg_demand * 365
eoq = np.sqrt((2 * annual_demand * ordering_cost) / eoq_holding_cost)

st.sidebar.markdown("---")
st.sidebar.subheader("Recommended Values")
st.sidebar.info(f"**Calculated EOQ:** {int(eoq)} units\n\n**Recommended ROP:** {int(recommended_rop)} units\n\n**Safety Stock:** {int(safety_stock)} units")

# --- USER INPUT FOR ROP & ORDER QUANTITY ---
rop_input = st.sidebar.number_input("Actual Reorder Point (ROP)", min_value=0, value=int(recommended_rop))
order_qty = st.sidebar.number_input("Order Quantity", min_value=1, value=int(eoq))

# --- SIMULATION LOGIC ---
days = 365
np.random.seed(42)
daily_demands = np.maximum(0, np.random.normal(avg_demand, variation, days))

# Initialize variables
inventory = rop_input + order_qty
days_until_delivery = 0
order_pending = False
orders_placed = 0
stockout_days = 0
total_demand_sim = 0
fulfilled_demand = 0

# Financial tracking schedules (padded to handle future payments beyond 365 days)
max_buffer = max(credit_rx, credit_given) + lead_time + 1
ap_schedule = np.zeros(days + max_buffer)
ar_schedule = np.zeros(days + max_buffer)
current_ap = 0.0 # Accounts Payable (Money we owe supplier)
current_ar = 0.0 # Accounts Receivable (Money buyers owe us)

# History lists for charts and tables
history = []

for day in range(days):
    # Process cash movements due today
    current_ap -= ap_schedule[day]
    current_ar -= ar_schedule[day]
    
    # 1. Receive pending orders
    if order_pending and days_until_delivery == 0:
        inventory += order_qty
        order_pending = False
        # We owe this to supplier in 'credit_rx' days
        current_ap += order_qty * unit_value
        ap_schedule[day + credit_rx] += order_qty * unit_value
    
    # 2. Process today's demand
    demand_today = daily_demands[day]
    total_demand_sim += demand_today
    
    if inventory >= demand_today:
        sold = demand_today
    else:
        sold = inventory
        stockout_days += 1
        
    inventory -= sold
    fulfilled_demand += sold
    
    # We collect money for sold goods in 'credit_given' days
    # (Calculated at cost value to measure exact capital tied up)
    current_ar += sold * unit_value
    ar_schedule[day + credit_given] += sold * unit_value
        
    # 3. Place new orders
    placed_today = 0
    if inventory <= rop_input and not order_pending:
        order_pending = True
        days_until_delivery = lead_time
        orders_placed += 1
        placed_today = 1
        
    # 4. Advance delivery timer
    if order_pending and placed_today == 0:
        days_until_delivery -= 1
        
    # 5. Financial Daily Calculations
    inventory_value = inventory * unit_value
    
    # Capital Utilized = (Cash tied up in Inventory) + (Cash tied up in AR) - (Cash financed by AP)
    capital_utilized = inventory_value + current_ar - current_ap
    
    # Calculate daily costs
    daily_phys_cost = inventory * (physical_holding_cost / 365.0)
    daily_cap_cost = max(0, capital_utilized) * (cost_of_capital_pct / 365.0)
    
    history.append({
        "Day": day + 1,
        "Demand": round(demand_today, 1),
        "Sales": round(sold, 1),
        "Inventory Units": round(inventory, 1),
        "Inventory Value ($)": round(inventory_value, 2),
        "Accounts Receivable ($)": round(current_ar, 2),
        "Accounts Payable ($)": round(current_ap, 2),
        "Capital Utilized ($)": round(capital_utilized, 2),
        "Daily Phys. Holding Cost ($)": round(daily_phys_cost, 2),
        "Daily Capital Cost ($)": round(daily_cap_cost, 2)
    })

# Convert to DataFrame
df = pd.DataFrame(history)

# --- KPI CALCULATIONS ---
avg_inv = df["Inventory Units"].mean()
min_inv = df["Inventory Units"].min()
max_inv = df["Inventory Units"].max()
fill_rate = (fulfilled_demand / total_demand_sim) * 100 if total_demand_sim > 0 else 0

total_phys_holding_cost = df["Daily Phys. Holding Cost ($)"].sum()
total_capital_cost = df["Daily Capital Cost ($)"].sum()
total_ordering_cost = orders_placed * ordering_cost
total_inventory_cost = total_phys_holding_cost + total_capital_cost + total_ordering_cost

# --- DASHBOARD METRICS ---
st.header("KPIs & Results (365 Days)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Fill Rate", value=f"{fill_rate:.2f}%")
    st.metric(label="Stock Out Days", value=f"{stockout_days}")
with col2:
    st.metric(label="Average Inventory", value=f"{int(avg_inv)} units")
    st.metric(label="Min / Max Inventory", value=f"{int(min_inv)} / {int(max_inv)}")
with col3:
    st.metric(label="Physical Holding Cost", value=f"${total_phys_holding_cost:,.0f}")
    st.metric(label="Cost of Capital", value=f"${total_capital_cost:,.0f}")
with col4:
    st.metric(label="Ordering Cost", value=f"${total_ordering_cost:,.0f}")
    st.metric(label="Total Inventory Cost", value=f"${total_inventory_cost:,.0f}")

st.markdown("---")

# --- CHARTS ---
tab1, tab2 = st.tabs(["📉 Inventory Movement", "💵 Cash / Capital Movement"])

with tab1:
    fig_inv = go.Figure()
    fig_inv.add_trace(go.Scatter(x=df["Day"], y=df["Inventory Units"], mode='lines', name='Inventory Level', line=dict(color='blue')))
    fig_inv.add_trace(go.Scatter(x=[1, days], y=[rop_input, rop_input], mode='lines', name='Reorder Point (ROP)', line=dict(color='red', dash='dash')))
    fig_inv.add_trace(go.Scatter(x=[1, days], y=[safety_stock, safety_stock], mode='lines', name='Safety Stock', line=dict(color='orange', dash='dash')))
    fig_inv.update_layout(xaxis_title="Days", yaxis_title="Units in Stock", hovermode="x unified", height=500)
    st.plotly_chart(fig_inv, use_container_width=True)

with tab2:
    fig_cash = go.Figure()
    # Positive Capital Contributors (Inventory + AR)
    fig_cash.add_trace(go.Scatter(x=df["Day"], y=df["Inventory Value ($)"], mode='lines', name='Inventory Value', stackgroup='one', fillcolor='lightblue', line=dict(width=0)))
    fig_cash.add_trace(go.Scatter(x=df["Day"], y=df["Accounts Receivable ($)"], mode='lines', name='Accounts Receivable (Buyers owe us)', stackgroup='one', fillcolor='lightgreen', line=dict(width=0)))
    
    # Negative Capital Contributors (AP)
    fig_cash.add_trace(go.Scatter(x=df["Day"], y=-df["Accounts Payable ($)"], mode='lines', name='Accounts Payable (We owe supplier)', stackgroup='two', fillcolor='salmon', line=dict(width=0)))
    
    # Net Capital Utilized Line
    fig_cash.add_trace(go.Scatter(x=df["Day"], y=df["Capital Utilized ($)"], mode='lines', name='Net Capital Utilized', line=dict(color='purple', width=3)))
    
    fig_cash.update_layout(
        title="Where is the Cash? (Capital Utilized Breakdown)",
        xaxis_title="Days",
        yaxis_title="Dollars ($)",
        hovermode="x unified",
        height=500,
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
    )
    st.plotly_chart(fig_cash, use_container_width=True)

# --- DAILY DATA TABLE ---
st.markdown("---")
st.subheader("📋 Daily Simulation Data")
st.markdown("Use this table to audit the day-by-day flow of inventory and capital.")
st.dataframe(df, use_container_width=True, height=300)
