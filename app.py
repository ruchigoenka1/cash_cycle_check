import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

st.set_page_config(page_title="Inventory Simulation App", layout="wide")

st.title("📦 Inventory Management & Simulation")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Demand & Lead Time")
avg_demand = st.sidebar.number_input("Average Daily Demand", min_value=1.0, value=50.0)
variation = st.sidebar.number_input("Demand Variation (Std Dev)", min_value=0.0, value=10.0)
lead_time = st.sidebar.number_input("Lead Time (Days)", min_value=1, value=5)

st.sidebar.header("2. Financial & Credit Terms")
unit_value = st.sidebar.number_input("Value of Product (Unit Cost $)", min_value=0.1, value=100.0)
holding_cost = st.sidebar.number_input("Annual Holding Cost per Unit ($)", min_value=0.1, value=20.0)
ordering_cost = st.sidebar.number_input("Ordering Cost per Order ($)", min_value=1.0, value=250.0)
credit_rx = st.sidebar.number_input("Credit Time from Supplier (Days)", min_value=0, value=30)
credit_given = st.sidebar.number_input("Credit Time given to Buyer (Days)", min_value=0, value=15)

st.sidebar.header("3. Service & Ordering Parameters")
service_level = st.sidebar.slider("Target Service Level (%)", min_value=50.0, max_value=99.99, value=95.0, step=0.1)

# --- CALCULATIONS FOR RECOMMENDED ROP & EOQ ---
# Calculate Z-score based on service level
z_score = stats.norm.ppf(service_level / 100.0)

# Safety Stock = Z * std_dev * sqrt(Lead Time)
safety_stock = z_score * variation * np.sqrt(lead_time)

# Recommended ROP = (Avg Demand * Lead Time) + Safety Stock
recommended_rop = (avg_demand * lead_time) + safety_stock

# Annual Demand
annual_demand = avg_demand * 365

# EOQ Formula
eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)

st.sidebar.markdown("---")
st.sidebar.subheader("Recommended Values")
st.sidebar.info(f"**Calculated EOQ:** {int(eoq)} units\n\n**Recommended ROP:** {int(recommended_rop)} units\n\n**Safety Stock:** {int(safety_stock)} units")

# --- USER INPUT FOR ROP & ORDER QUANTITY ---
rop_input = st.sidebar.number_input("Actual Reorder Point (ROP)", min_value=0, value=int(recommended_rop))
order_qty = st.sidebar.number_input("Order Quantity", min_value=1, value=int(eoq))

# --- SIMULATION LOGIC ---
st.header("Simulation Results (365 Days)")

# Initialize simulation variables
days = 365
current_inventory = rop_input + order_qty
inventory_history = []
stockout_days = 0
total_demand_sim = 0
fulfilled_demand = 0
orders_placed = 0

days_until_delivery = 0
order_pending = False

# Run daily simulation
np.random.seed(42) # For reproducible results
daily_demands = np.maximum(0, np.random.normal(avg_demand, variation, days))

for day in range(days):
    # 1. Receive pending orders
    if order_pending and days_until_delivery == 0:
        current_inventory += order_qty
        order_pending = False
    
    # 2. Process today's demand
    demand_today = daily_demands[day]
    total_demand_sim += demand_today
    
    if current_inventory >= demand_today:
        fulfilled_demand += demand_today
        current_inventory -= demand_today
    else:
        fulfilled_demand += current_inventory
        current_inventory = 0
        stockout_days += 1
        
    # 3. Check if we need to place an order
    if current_inventory <= rop_input and not order_pending:
        order_pending = True
        days_until_delivery = lead_time
        orders_placed += 1
        
    # 4. Advance delivery timer
    if order_pending:
        days_until_delivery -= 1
        
    inventory_history.append(current_inventory)

# --- KPI CALCULATIONS ---
avg_inv = np.mean(inventory_history)
min_inv = np.min(inventory_history)
max_inv = np.max(inventory_history)
fill_rate = (fulfilled_demand / total_demand_sim) * 100 if total_demand_sim > 0 else 0

total_holding_cost = avg_inv * holding_cost
total_ordering_cost = orders_placed * ordering_cost
total_inventory_cost = total_holding_cost + total_ordering_cost

# --- DASHBOARD METRICS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Fill Rate", value=f"{fill_rate:.2f}%", help="Percentage of demand met by available stock")
    st.metric(label="Total Inv Cost", value=f"${total_inventory_cost:,.2f}")
with col2:
    st.metric(label="Stock Out Days", value=f"{stockout_days}", help="Number of days inventory hit 0")
    st.metric(label="Holding Cost", value=f"${total_holding_cost:,.2f}")
with col3:
    st.metric(label="Average Inventory", value=f"{int(avg_inv)} units")
    st.metric(label="Ordering Cost", value=f"${total_ordering_cost:,.2f}")
with col4:
    st.metric(label="Min / Max Inventory", value=f"{int(min_inv)} / {int(max_inv)}")
    st.metric(label="Orders Placed (Year)", value=f"{orders_placed}")

# --- INVENTORY CHART ---
st.subheader("Inventory Level Over Time")
fig = go.Figure()

# Plot actual inventory
fig.add_trace(go.Scatter(
    x=list(range(days)), 
    y=inventory_history,
    mode='lines',
    name='Inventory Level',
    line=dict(color='blue')
))

# Plot ROP line
fig.add_trace(go.Scatter(
    x=[0, days], 
    y=[rop_input, rop_input],
    mode='lines',
    name='Reorder Point (ROP)',
    line=dict(color='red', dash='dash')
))

# Plot Safety Stock line
fig.add_trace(go.Scatter(
    x=[0, days], 
    y=[safety_stock, safety_stock],
    mode='lines',
    name='Safety Stock (Minimum Level)',
    line=dict(color='orange', dash='dash')
))

fig.update_layout(
    xaxis_title="Days",
    yaxis_title="Units in Stock",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=30, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# --- NOTES ---
st.markdown("---")
st.caption(f"**Note on Credit Terms:** You entered Supplier Credit Time ({credit_rx} days) and Buyer Credit Time ({credit_given} days). While these do not affect physical inventory stockouts or standard EOQ, they are critical for calculating your **Cash Conversion Cycle** and working capital requirements in broader financial models.")
