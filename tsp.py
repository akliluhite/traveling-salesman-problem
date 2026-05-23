import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# App Configuration
st.set_page_config(page_title="Ultimate TSP Simulation Dashboard", layout="wide")

st.title("🧬 The Ultimate Traveling Salesman Dashboard")
st.write("Compare computational complexity, accuracy benchmarks, and run real-time flight route simulations across Europe.")

# Expanded Master Database of Cities
MASTER_CITIES = {
    'Paris (France)': (48.8566, 2.3522),
    'Brussels (Belgium)': (50.8503, 4.3517),
    'Amsterdam (Netherlands)': (52.3676, 4.9041),
    'Berlin (Germany)': (52.5200, 13.4050),
    'Prague (Czechia)': (50.0755, 14.4378),
    'Rome (Italy)': (41.9028, 12.4964),
    'Madrid (Spain)': (40.4168, -3.7038),
    'Vienna (Austria)': (48.2082, 16.3738),
    'Bern (Switzerland)': (46.9480, 7.4474),
    'London (UK)': (51.5074, -0.1278),
    'Lisbon (Portugal)': (38.7223, -9.1393),
    'Dublin (Ireland)': (53.3498, -6.2603),
    'Copenhagen (Denmark)': (55.6761, 12.5683),
    'Warsaw (Poland)': (52.2297, 21.0122)
}

# Core Math Helpers
def distance(c1, c2, city_dict):
    """Calculates geodesic distance using the Haversine formula."""
    lat1, lon1 = np.radians(city_dict[c1])
    lat2, lon2 = np.radians(city_dict[c2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 6371.0 * (2 * np.arcsin(np.sqrt(a)))

def route_distance(r, city_dict):
    """Calculates total closed loop sequence distance."""
    if not r or len(r) < 2:
        return 0.0
    return sum(distance(r[i], r[i+1], city_dict) for i in range(len(r)-1))

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if "sim_running" not in st.session_state:
    st.session_state.sim_running = False
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
if "previous_algo" not in st.session_state:
    st.session_state.previous_algo = ""

# -------------------------------------------------------------
# SIDEBAR: DYNAMIC CITY SELECTION
# -------------------------------------------------------------
with st.sidebar:
    st.header("🗺️ Network Workspace")
    selected_cities = st.multiselect(
        "Select Cities to Include in Route:",
        options=list(MASTER_CITIES.keys()),
        default=list(MASTER_CITIES.keys())[:7]
    )
    
    if len(selected_cities) < 3:
        st.error("Please select at least 3 cities to map a closed-loop route.")
        st.stop()
        
    start_city = st.selectbox("Select Starting Hub:", options=selected_cities)

ACTIVE_CITIES = {k: MASTER_CITIES[k] for k in selected_cities}
cities_list = list(ACTIVE_CITIES.keys())
other_cities = [c for c in cities_list if c != start_city]

# -------------------------------------------------------------
# TSP SOLVER CORE IMPLEMENTATIONS
# -------------------------------------------------------------
def solve_greedy():
    route = [start_city]
    unvisited = other_cities.copy()
    while unvisited:
        closest = min(unvisited, key=lambda c: distance(route[-1], c, ACTIVE_CITIES))
        route.append(closest)
        unvisited.remove(closest)
    route.append(start_city)
    return route

def solve_dynamic():
    n = len(cities_list)
    if n > 15:
        return None
        
    mapping = {city: i for i, city in enumerate(cities_list)}
    inv_mapping = {i: city for i, city in enumerate(cities_list)}
    start_idx = mapping[start_city]
    
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = distance(inv_mapping[i], inv_mapping[j], ACTIVE_CITIES)
            
    memo = {}
    
    def hk_solve(mask, u):
        if mask == (1 << n) - 1:
            return dist_matrix[u][start_idx], [start_idx]
        if (mask, u) in memo:
            return memo[(mask, u)]
            
        ans = float('inf')
        best_path = []
        
        for v in range(n):
            if not (mask & (1 << v)):
                cost, path = hk_solve(mask | (1 << v), v)
                total_cost = dist_matrix[u][v] + cost
                if total_cost < ans:
                    ans = total_cost
                    best_path = [v] + path
                    
        memo[(mask, u)] = (ans, best_path)
        return memo[(mask, u)]
        
    _, path_indices = hk_solve(1 << start_idx, start_idx)
    full_path = [start_city] + [inv_mapping[i] for i in path_indices]
    return full_path

def solve_backtracking():
    n = len(cities_list)
    if n > 10:
        return None
        
    tracker = {"best_dist": float('inf'), "best_path": []}
    
    def backtrack(curr_city, visited, current_path, current_dist):
        if current_dist >= tracker["best_dist"]:
            return
            
        if len(visited) == n:
            final_dist = current_dist + distance(curr_city, start_city, ACTIVE_CITIES)
            if final_dist < tracker["best_dist"]:
                tracker["best_dist"] = final_dist
                tracker["best_path"] = list(current_path) + [start_city]
            return
            
        for nxt_city in other_cities:
            if nxt_city not in visited:
                visited.add(nxt_city)
                leg_dist = distance(curr_city, nxt_city, ACTIVE_CITIES)
                backtrack(nxt_city, visited, current_path + [nxt_city], current_dist + leg_dist)
                visited.remove(nxt_city)
                
    backtrack(start_city, {start_city}, [start_city], 0.0)
    return tracker["best_path"]

# -------------------------------------------------------------
# LAYOUT PARTITIONING & CONTROLS
# -------------------------------------------------------------
col_control, col_display = st.columns()

with col_control:
    st.subheader("⚙️ Control Settings")
    algo = st.selectbox("Choose Active Evaluation Strategy:", 
                        ["Greedy (Fast / Heuristic)", 
                         "Dynamic Programming (Held-Karp Exact)", 
                         "Backtracking (Exhaustive Search Exact)"])
    
    if algo != st.session_state.previous_algo:
        st.session_state.sim_running = False
        st.session_state.current_step = 1
        st.session_state.previous_algo = algo
    
    run_valid = True
    if algo == "Dynamic Programming (Held-Karp Exact)" and len(selected_cities) > 15:
        st.error("⚠️ Dynamic Programming is restricted to 15 cities maximum.")
        run_valid = False
    elif algo == "Backtracking (Exhaustive Search Exact)" and len(selected_cities) > 10:
        st.error("⚠️ Backtracking is locked down above 10 cities.")
        run_valid = False
        
    st.markdown("---")
    st.subheader("🎬 Simulation Control")
    speed = st.slider("Step intervals (seconds):", min_value=0.05, max_value=2.0, value=0.3, step=0.05)
    
    def start_simulation():
        st.session_state.sim_running = True
        st.session_state.current_step = 1

    trigger_sim = st.button("▶️ Launch Live Simulation", disabled=not run_valid, on_click=start_simulation)

static_route = []
total_static_dist = 0.0
execution_time_ms = 0.0

if run_valid:
    start_time = time.perf_counter()
    if algo == "Greedy (Fast / Heuristic)":
        static_route = solve_greedy()
    elif algo == "Dynamic Programming (Held-Karp Exact)":
        static_route = solve_dynamic()
    else:
        static_route = solve_backtracking()
    execution_time_ms = (time.perf_counter() - start_time) * 1000
    total_static_dist = route_distance(static_route, ACTIVE_CITIES)

# -------------------------------------------------------------
# GRAPHICS AND MAIN RENDERING
# -------------------------------------------------------------
with col_display:
    st.subheader("📊 Algorithmic Benchmarks")
    m_col1, m_col2 = st.columns(2)
    
    if run_valid and static_route:
        m_col1.metric(label="Calculated Loop Distance", value=f"{total_static_dist:.2f} km")
        m_col2.metric(label="Compute Processing Velocity", value=f"{execution_time_ms:.2f} ms")
    else:
        m_col1.metric(label="Calculated Loop Distance", value="N/A")
        m_col2.metric(label="Compute Processing Velocity", value="N/A")

    # -------------------------------------------------------------
    # LIVE SIMULATION & MAP VISUALIZATION ENGINE
    # -------------------------------------------------------------
    st.subheader("🗺️ Interactive Route Vector Mapping")
    
    if st.session_state.sim_running and static_route:
        if st.session_state.current_step < len(static_route):
            display_route = static_route[:st.session_state.current_step + 1]
            st.session_state.current_step += 1
            time.sleep(speed)
            st.rerun()
        else:
            st.session_state.sim_running = False
            display_route = static_route
    else:
        display_route = static_route

    if run_valid and display_route:
        # FIXED: Correctly index tuple indices to extract explicit flat float lists
        lats = [ACTIVE_CITIES[city][0] for city in display_route]
        lons = [ACTIVE_CITIES[city][1] for city in display_route]
        names = [f"{idx+1}. {city}" for idx, city in enumerate(display_route)]
        
        fig = go.Figure()
        
        # Plot route paths
        fig.add_trace(go.Scattermap(
            lat=lats,
            lon=lons,
            mode='lines+markers',
            marker=dict(size=11, color='#FF4B4B'),
            line=dict(width=3, color='#FF4B4B'),
            text=names,
            hoverinfo='text',
            name="TSP Route"
        ))
        
        # Emphasize origin hub location
        start_lat, start_lon = ACTIVE_CITIES[start_city]
        fig.add_trace(go.Scattermap(
            lat=[start_lat],
            lon=[start_lon],
            mode='markers',
