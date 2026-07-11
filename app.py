import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go

st.set_page_config(page_title="Inventory & Cash Flow Simulation", layout="wide")

st.title("📦 Inventory & Cash Flow Simulation")
st.markdown("This dashboard simulates daily operations, tracks the true **Cost of Capital**, and uses vectorized array operations for high-speed Monte Carlo simulations.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Demand & Lead Time")
avg_demand = st.sidebar.number_input("Average Daily Demand", min_value=1.0, value=50.0)
variation = st.sidebar.number_input("Demand Variation (Std Dev)", min_value=0.0, value=0.0)
lead_time = st.sidebar.number_input("Lead Time (Days)", min_value=1, value=60)

st.sidebar.header("2. Financial & Credit Terms")
opening_capital = st.sidebar.number_input("Initial Capital Balance ($)", min_value=0.0, value=100000.0, step=5000.0)
unit_value = st.sidebar.number_input("Value of Product (Unit Cost $)", min_value=0.1, value=100.0)
physical_holding_cost = st.sidebar.number_input("Physical Holding Cost/Unit/Year ($)", min_value=0.0, value=10.0)
cost_of_capital_pct = st.sidebar.number_input("Cost of Capital (Annual %)", min_value=0.0, value=12.0) / 100.0
ordering_cost = st.sidebar.number_input("Ordering Cost per Order ($)", min_value=1.0, value=250.0)
credit_rx = st.sidebar.number_input("Credit Time from Supplier (Days)", min_value=0, value=30, help="0 = Immediate payment.")
credit_given = st.sidebar.number_input("Credit Time given to Buyer (Days)", min_value=0, value=15, help="0 = Immediate payment.")

st.sidebar.header("3. Service & Ordering Parameters")
service_level = st.sidebar.slider("Target Service Level (%)", min_value=50.0, max_value=99.99, value=95.0, step=0.1)

st.sidebar.header("4. Simulation Settings")
sim_days = st.sidebar.number_input("Simulation Duration (Days)", min_value=30, value=365, step=30)
warmup_days = st.sidebar.number_input("Warm-up Period to Exclude (Days)", min_value=0, max_value=sim_days-1, value=0)

# --- CALCULATIONS FOR RECOMMENDED ROP & EOQ ---
z_score = stats.norm.ppf(service_level / 100.0)
safety_stock = z_score * variation * np.sqrt(lead_time)
recommended_rop = (avg_demand * lead_time) + safety_stock

# Base Holding Cost = Physical + Base Capital Cost
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

# --- VECTORIZED SIMULATION LOGIC ---
np.random.seed(42)
daily_demands = np.maximum(0, np.random.normal(avg_demand, variation, sim_days))

# Pre-allocate arrays for extreme speed
max_buffer = max(credit_rx, credit_given, lead_time) + 1
deliveries = np.zeros(sim_days + max_buffer)
ap_events = np.zeros(sim_days + max_buffer)
ar_events = np.zeros(sim_days + max_buffer)

# State trackers for the loop
inventory = rop_input + order_qty
on_order = 0

# History arrays
inventory_arr = np.zeros(sim_days)
inv_pos_arr = np.zeros(sim_days)
sales_arr = np.zeros(sim_days)
orders_placed_arr = np.zeros(sim_days)

# We only loop to handle the path-dependent nature of ROP triggering
for day in range(sim_days):
    # 1. Receive Deliveries
    arrived = deliveries[day]
    inventory += arrived
    on_order -= arrived
    
    # 2. Process Demand
    demand = daily_demands[day]
    sold = min(inventory, demand)
    inventory -= sold
    sales_arr[day] = sold
    
    # Financial Event Logging (AR)
    ar_events[day] += sold * unit_value
    ar_events[day + credit_given] -= sold * unit_value
    
    # 3. Check Inventory Position & Reorder
    inv_pos = inventory + on_order
    placed_today = 0
    
    while inv_pos <= rop_input:
        deliveries[day + lead_time] += order_qty
        on_order += order_qty
        inv_pos += order_qty
        placed_today += 1
        
        # Financial Event Logging (AP)
        ap_events[day] += order_qty * unit_value
        ap_events[day + credit_rx] -= order_qty * unit_value

    # Log daily states
    inventory_arr[day] = inventory
    inv_pos_arr[day] = inv_pos
    orders_placed_arr[day] = placed_today

# --- VECTORIZED FINANCIAL PROCESSING ---
# Create base DataFrame
df = pd.DataFrame({
    "Day": np.arange(1, sim_days + 1),
    "Demand": np.round(daily_demands, 1),
    "Sales": np.round(sales_arr, 1),
    "Inventory Units": np.round(inventory_arr, 1),
    "Orders Placed": orders_placed_arr,
    "Inventory Position": np.round(inv_pos_arr, 1)
})

# Vectorized rolling cumulative sums for financials
df["Inventory Value ($)"] = df["Inventory Units"] * unit_value
df["Accounts Receivable ($)"] = np.cumsum(ar_events)[:sim_days]
df["Accounts Payable ($)"] = np.cumsum(ap_events)[:sim_days]

# Vectorized Daily Capital Required & Costs
df["Net Capital Required ($)"] = df["Inventory Value ($)"] + df["Accounts Receivable ($)"] - df["Accounts Payable ($)"] - opening_capital
df["Daily Phys. Holding Cost ($)"] = df["Inventory Units"] * (physical_holding_cost / 365.0)
df["Daily Capital Cost ($)"] = np.maximum(0, df["Net Capital Required ($)"]) * (cost_of_capital_pct / 365.0)

# Slice DataFrame for Warm-up Period
df_kpi = df.iloc[warmup_days:].copy()

# --- KPI CALCULATIONS ---
avg_inv = df_kpi["Inventory Units"].mean()
min_inv = df_kpi["Inventory Units"].min()
max_inv = df_kpi["Inventory Units"].max()

max_inv_pos = df_kpi["Inventory Position"].max()
min_inv_pos = df_kpi["Inventory Position"].min()

max_inv_val = df_kpi["Inventory Value ($)"].max()
min_inv_val = df_kpi["Inventory Value ($)"].min()

max_ap = df_kpi["Accounts Payable ($)"].max()
min_ap = df_kpi["Accounts Payable ($)"].min()

max_ar = df_kpi["Accounts Receivable ($)"].max()
min_ar = df_kpi["Accounts Receivable ($)"].min()

total_demand_kpi = df_kpi["Demand"].sum()
total_sales_kpi = df_kpi["Sales"].sum()
fill_rate = (total_sales_kpi / total_demand_kpi) * 100 if total_demand_kpi > 0 else 0
stockout_days = len(df_kpi[df_kpi["Demand"] > df_kpi["Sales"]])

total_phys_holding_cost = df_kpi["Daily Phys. Holding Cost ($)"].sum()
total_capital_cost = df_kpi["Daily Capital Cost ($)"].sum()
total_orders_kpi = df_kpi["Orders Placed"].sum()
total_ordering_cost = total_orders_kpi * ordering_cost
total_inventory_cost = total_phys_holding_cost + total_capital_cost + total_ordering_cost

# --- DASHBOARD METRICS ---
st.header(f"KPIs & Results (Days {warmup_days + 1} to {sim_days})")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Fill Rate", value=f"{fill_rate:.2f}%")
    st.metric(label="Stock Out Days", value=f"{stockout_days}")
with col2:
    st.metric(label="Average Inventory", value=f"{int(avg_inv)} units")
    st.metric(label="Total Inventory Cost", value=f"${total_inventory_cost:,.0f}")
with col3:
    st.metric(label="Physical Holding Cost", value=f"${total_phys_holding_cost:,.0f}")
    st.metric(label="Cost of Capital", value=f"${total_capital_cost:,.0f}")
with col4:
    st.metric(label="Ordering Cost", value=f"${total_ordering_cost:,.0f}")
    st.metric(label="Total Orders Placed", value=f"{total_orders_kpi}")

# --- COLLAPSIBLE MIN/MAX SECTION ---
with st.expander("📊 View Min/Max Financial & Inventory Details"):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("**Inventory (Units)**")
        st.markdown(f"**Max:** {int(max_inv):,}")
        st.markdown(f"**Min:** {int(min_inv):,}")
    with c2:
        st.markdown("**Inventory Position**")
        st.markdown(f"**Max:** {int(max_inv_pos):,}")
        st.markdown(f"**Min:** {int(min_inv_pos):,}")
    with c3:
        st.markdown("**Inventory Value ($)**")
        st.markdown(f"**Max:** ${max_inv_val:,.2f}")
        st.markdown(f"**Min:** ${min_inv_val:,.2f}")
    with c4:
        st.markdown("**Accounts Payable ($)**")
        st.markdown(f"**Max:** ${max_ap:,.2f}")
        st.markdown(f"**Min:** ${min_ap:,.2f}")
    with c5:
        st.markdown("**Accounts Receivable ($)**")
        st.markdown(f"**Max:** ${max_ar:,.2f}")
        st.markdown(f"**Min:** ${min_ar:,.2f}")

st.markdown("---")

# --- CHARTS ---
tab1, tab2 = st.tabs(["📉 Inventory Movement", "💵 Cash / Capital Movement"])

with tab1:
    fig_inv = go.Figure()
    # Physical Inventory
    fig_inv.add_trace(go.Scatter(x=df_kpi["Day"], y=df_kpi["Inventory Units"], mode='lines', name='Inventory Level (On Hand)', line=dict(color='#1f77b4', width=2)))
    # Inventory Position
    fig_inv.add_trace(go.Scatter(x=df_kpi["Day"], y=df_kpi["Inventory Position"], mode='lines', name='Inventory Position (On Hand + On Order)', line=dict(color='#9467bd', dash='dot', width=2)))
    
    fig_inv.add_trace(go.Scatter(x=[df_kpi["Day"].min(), df_kpi["Day"].max()], y=[rop_input, rop_input], mode='lines', name='Reorder Point (ROP)', line=dict(color='#d62728', dash='dash')))
    fig_inv.add_trace(go.Scatter(x=[df_kpi["Day"].min(), df_kpi["Day"].max()], y=[safety_stock, safety_stock], mode='lines', name='Safety Stock', line=dict(color='#ff7f0e', dash='dash')))
    fig_inv.update_layout(xaxis_title="Days", yaxis_title="Units", hovermode="x unified", height=500)
    st.plotly_chart(fig_inv, use_container_width=True)

with tab2:
    fig_cash = go.Figure()
    
    # Positive Capital Contributors (Stack 1)
    if opening_capital > 0:
        fig_cash.add_trace(go.Scatter(x=df_kpi["Day"], y=[opening_capital]*len(df_kpi), mode='lines', name='Opening Capital Buffer', stackgroup='one', fillcolor='#ff7f0e', line=dict(width=0)))
    
    fig_cash.add_trace(go.Scatter(x=df_kpi["Day"], y=df_kpi["Inventory Value ($)"], mode='lines', name='Inventory Value', stackgroup='one', fillcolor='#1f77b4', line=dict(width=0)))
    fig_cash.add_trace(go.Scatter(x=df_kpi["Day"], y=df_kpi["Accounts Receivable ($)"], mode='lines', name='Accounts Receivable', stackgroup='one', fillcolor='#2ca02c', line=dict(width=0)))
    
    # Negative Capital Contributors (Stack 2)
    fig_cash.add_trace(go.Scatter(x=df_kpi["Day"], y=-df_kpi["Accounts Payable ($)"], mode='lines', name='Accounts Payable', stackgroup='two', fillcolor='#d62728', line=dict(width=0)))
    
    # Net Capital Required Line
    fig_cash.add_trace(go.Scatter(x=df_kpi["Day"], y=df_kpi["Net Capital Required ($)"], mode='lines', name='Net Capital Required (Borrowing)', line=dict(color='#00e5ff', width=3)))
    
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
st.subheader("📋 Steady-State Daily Data")
st.markdown("Auditing the day-by-day flow of inventory and capital (Warm-up period excluded).")
st.dataframe(df_kpi, use_container_width=True, height=300)
