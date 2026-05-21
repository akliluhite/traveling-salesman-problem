import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import itertools
import time
import random

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
    lat1, lon1 = np.radians(city_dict[c1])
    lat2, lon2 = np.radians(city_dict[c2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 6371.0 * (2 * np.arcsin(np.sqrt(a)))

def route_distance(r, city_dict):
    return sum(distance(r[i], r[i+1], city_dict) for i in range(len(r)-1))

# -------------------------------------------------------------
# FEATURE 1: DYNAMIC CITY SELECTION & FILTERING
# -------------------------------------------------------------
with st.sidebar:
    st.header("🗺️ Network Workspace")
    selected_cities = st.multiselect(
        "Select Cities to Include in Route:",
        options=list(MASTER_CITIES.keys()),
        default=list(MASTER_CITIES.keys())[:10]
    )
    
    # Validation safety check
    if len(selected_cities) < 3:
        st.error("Please select at least 3 cities to map a closed-loop route.")
        st.stop()
        
    start_city = st.selectbox("Select Starting Hub:", options=selected_cities)

# Filter active working set
ACTIVE_CITIES = {k: MASTER_CITIES[k] for k in selected_cities}
cities_list = list(ACTIVE_CITIES.keys())
other_cities = [c for c in cities_list if c != start_city]

# TSP Solver Core Implementations
def solve_nearest_neighbor():
    route = [start_city]
    unvisited = other_cities.copy()
    while unvisited:
        closest = min(unvisited, key=lambda c: distance(route[-1], c, ACTIVE_CITIES))
        route.append(closest)
        unvisited.remove(closest)
    route.append(start_city)
    return route

def solve_genetic(pop_size, generations, mutation_rate):
    population = [[start_city] + random.sample(other_cities, len(other_cities)) + [start_city] for _ in range(pop_size)]
    for _ in range(generations):
        population = sorted(population, key=lambda r: route_distance(r, ACTIVE_CITIES))
        survivors = population[:max(2, int(pop_size * 0.2))]
        next_pop = survivors.copy()
        while len(next_pop) < pop_size:
            parent = random.choice(survivors)
            child = parent.copy()
            if random.random() < mutation_rate and len(cities_list) > 2:
                idx1, idx2 = random.sample(range(1, len(cities_list)), 2)
                child[idx1], child[idx2] = child[idx2], child[idx1]
            next_pop.append(child)
        population = next_pop
    return sorted(population, key=lambda r: route_distance(r, ACTIVE_CITIES))[0]

def solve_brute_force():
    best_dist = float('inf')
    best_route = []
    for p in itertools.permutations(other_cities):
        r = [start_city] + list(p) + [start_city]
        d = route_distance(r, ACTIVE_CITIES)
        if d < best_dist:
            best_dist, best_route = d, r
    return best_route

# Layout Partitioning
col_control, col_display = st.columns([1, 2])

with col_control:
    st.subheader("⚙️ Control Settings")
    algo = st.selectbox("Choose Active Evaluation Strategy:", 
                        ["Nearest Neighbor (Fast / Greedy)", 
                         "Genetic Algorithm (Smart Balanced)", 
                         "Brute Force (Perfect / Heavy CPU)"])
    
    # -------------------------------------------------------------
    # FEATURE 2: GENETIC ALGORITHM HYPERPARAMETER CONTROLS
    # -------------------------------------------------------------
    pop_size, gen_count, mut_rate = 50, 100, 0.3
    if algo == "Genetic Algorithm (Smart Balanced)":
        st.markdown("**🧬 Genetic Algorithm Tuners**")
        pop_size = st.slider("Population Size:", 10, 200, 50, step=10)
        gen_count = st.slider("Generations Loop:", 10, 500, 100, step=10)
        mut_rate = st.slider("Mutation Probability:", 0.0, 1.0, 0.3, step=0.05)
    
    # Brute Force Safety Lockout Warning
    if algo == "Brute Force (Perfect / Heavy CPU)" and len(selected_cities) > 10:
        st.warning(f"⚠️ Warning: Evaluating {len(selected_cities)} cities creates {np.math.factorial(len(selected_cities)-1):,} possible paths. Your browser might crash.")
        
    st.markdown("---")
    st.subheader("🎬 Simulation Control")
    speed = st.slider("Step intervals (seconds):", min_value=0.05, max_value=2.0, value=0.3, step=0.05)
    trigger_sim = st.button("▶️ Launch Live Simulation")

# Run Current Selected Strategy Solver
start_time = time.perf_counter()
if algo == "Nearest Neighbor (Fast / Greedy)":
    static_route = solve_nearest_neighbor()
elif algo == "Genetic Algorithm (Smart Balanced)":
    static_route = solve_genetic(pop_size, gen_count, mut_rate)
else:
    static_route = solve_brute_force()
execution_time_ms = (time.perf_counter() - start_time) * 1000
total_static_dist = route_distance(static_route, ACTIVE_CITIES)

# Display Workspaces
with col_display:
    st.subheader("📊 Algorithmic Benchmarks")
    m_col1, m_col2 = st.columns(2)
    m_col1.metric(label="Calculated Loop Distance", value=f"{total_static_dist:.2f} km")
    m_col2.metric(label="Compute Processing Velocity", value=f"{execution_time_ms:.2f} ms")
    
    map_placeholder = st.empty()
    status_placeholder = st.empty()

    # -------------------------------------------------------------
    # FEATURE 3 & 4: CORRECTED ROUTE ANIMATION LOOP
    # -------------------------------------------------------------
    if not trigger_sim:
        map_data = [{"Order": idx + 1, "City": city, "Latitude": ACTIVE_CITIES[city][0], "Longitude": ACTIVE_CITIES[city][1]} for idx, city in enumerate(static_route)]
        df = pd.DataFrame(map_data)
        fig = px.line_mapbox(df, lat="Latitude", lon="Longitude", hover_name="City", zoom=3, height=500)
        fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        map_placeholder.plotly_chart(fig, use_container_width=True)
        status_placeholder.success(f"Static path compiled via {algo}. Press 'Launch Live Simulation' to animate this precise flight pattern.")
    else:
        # Step-by-step rendering of the algorithm's actual solution path
        for step in range(1, len(static_route) + 1):
            current_sub_route = static_route[:step]
            frame_data = [{"Order": idx + 1, "City": c, "Latitude": ACTIVE_CITIES[c][0], "Longitude": ACTIVE_CITIES[c][1]} for idx, c in enumerate(current_sub_route)]
            df_frame = pd.DataFrame(frame_data)
            
            fig_frame = px.line_mapbox(df_frame, lat="Latitude", lon="Longitude", hover_name="City", zoom=3, height=500)
            fig_frame.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
            
            if step < len(static_route):
                status_placeholder.info(f"✈️ Route leg dispatched: **{current_sub_route[-1]}**")
            else:
                status_placeholder.success(f"🏁 Circuit finalized! Returned back to hub root: **{start_city}**")
                
            map_placeholder.plotly_chart(fig_frame, use_container_width=True)
            time.sleep(speed)

    # -------------------------------------------------------------
    # FEATURE 5: BACKGROUND COMPARISON MATRIX
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("🏁 Real-time Strategy Performance Comparison Matrix")
    
    # Run algorithms for benchmark profiling matrix
    t0 = time.perf_counter()
    r_nn = solve_nearest_neighbor()
    d_nn = route_distance(r_nn, ACTIVE_CITIES)
    ms_nn = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    r_ga = solve_genetic(pop_size, gen_count, mut_rate)
    d_ga = route_distance(r_ga, ACTIVE_CITIES)
    ms_ga = (time.perf_counter() - t0) * 1000
    
    if len(selected_cities) <= 10:
        t0 = time.perf_counter()
        r_bf = solve_brute_force()
        d_bf = route_distance(r_bf, ACTIVE_CITIES)
        ms_bf = (time.perf_counter() - t0) * 1000
    else:
        d_bf, ms_bf = "Skipped (Too Slow)", "Timeout"

    comparison_data = {
        "Strategy Engine": ["Nearest Neighbor", "Genetic Algorithm", "Brute Force"],
        "Calculated Loop Distance": [f"{d_nn:.2f} km", f"{d_ga:.2f} km", f"{d_bf:.2f} km" if isinstance(d_bf, float) else d_bf],
        "Execution Overhead Time": [f"{ms_nn:.2f} ms", f"{ms_ga:.2f} ms", f"{ms_bf:.2f} ms" if isinstance(ms_bf, float) else ms_bf]
    }
    st.table(pd.DataFrame(comparison_data))
