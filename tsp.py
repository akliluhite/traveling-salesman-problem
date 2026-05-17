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

# 10 Capital Cities across Europe representing 10 countries
CITIES = {
    'Paris (France)': (48.8566, 2.3522),
    'Brussels (Belgium)': (50.8503, 4.3517),
    'Amsterdam (Netherlands)': (52.3676, 4.9041),
    'Berlin (Germany)': (52.5200, 13.4050),
    'Prague (Czechia)': (50.0755, 14.4378),
    'Rome (Italy)': (41.9028, 12.4964),
    'Madrid (Spain)': (40.4168, -3.7038),
    'Vienna (Austria)': (48.2082, 16.3738),
    'Bern (Switzerland)': (46.9480, 7.4474),
    'London (UK)': (51.5074, -0.1278)
}

def distance(c1, c2):
    """Calculates geodesic distance using the Haversine formula."""
    lat1, lon1 = np.radians(CITIES[c1])
    lat2, lon2 = np.radians(CITIES[c2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 6371.0 * (2 * np.arcsin(np.sqrt(a)))

def route_distance(r):
    """Calculates total closed loop sequence distance."""
    return sum(distance(r[i], r[i+1]) for i in range(len(r)-1))

# Layout Partitioning
col_control, col_display = st.columns([1, 2])

with col_control:
    st.subheader("⚙️ Control Settings")
    algo = st.selectbox("Choose Evaluation Strategy:", 
                        ["Nearest Neighbor (Fast / Greedy)", 
                         "Genetic Algorithm (Smart Balanced)", 
                         "Brute Force (Perfect / Heavy CPU)"])
    
    st.markdown("---")
    st.subheader("🎬 Simulation Control")
    st.write("Animate the physical route construction step-by-step.")
    speed = st.slider("Step intervals (seconds):", min_value=0.1, max_value=2.0, value=0.4, step=0.1)
    trigger_sim = st.button("▶️ Launch Live Simulation")

cities_list = list(CITIES.keys())
other_cities = [c for c in cities_list if c != 'Paris (France)']

# -------------------------------------------------------------
# STANDALONE COMPUTATION RUN (Runs immediately on load/toggle)
# -------------------------------------------------------------
start_time = time.perf_counter()
static_route = ['Paris (France)']

if algo == "Nearest Neighbor (Fast / Greedy)":
    unvisited = other_cities.copy()
    while unvisited:
        closest = min(unvisited, key=lambda c: distance(static_route[-1], c))
        static_route.append(closest)
        unvisited.remove(closest)
    static_route.append('Paris (France)')

elif algo == "Genetic Algorithm (Smart Balanced)":
    pop_size, generations = 50, 100
    population = [['Paris (France)'] + random.sample(other_cities, len(other_cities)) + ['Paris (France)'] for _ in range(pop_size)]
    
    for _ in range(generations):
        population = sorted(population, key=route_distance)
        survivors = population[:10]
        next_pop = survivors.copy()
        while len(next_pop) < pop_size:
            parent = random.choice(survivors)
            child = parent.copy()
            idx1, idx2 = random.sample(range(1, len(cities_list)), 2)
            child[idx1], child[idx2] = child[idx2], child[idx1]
            next_pop.append(child)
        population = next_pop
    static_route = sorted(population, key=route_distance)[0]

else:
    best_dist = float('inf')
    # Processes 9! permutations safely
    for p in itertools.permutations(other_cities):
        r = ['Paris (France)'] + list(p) + ['Paris (France)']
        d = route_distance(r)
        if d < best_dist:
            best_dist, static_route = d, r

execution_time_ms = (time.perf_counter() - start_time) * 1000
total_static_dist = route_distance(static_route)

# -------------------------------------------------------------
# GRAPHICS AND MAIN RENDERING
# -------------------------------------------------------------
with col_display:
    st.subheader("📊 Algorithmic Benchmarks")
    m_col1, m_col2 = st.columns(2)
    m_col1.metric(label="Calculated Loop Distance", value=f"{total_static_dist:.2f} km")
    m_col2.metric(label="Compute Processing Velocity", value=f"{execution_time_ms:.2f} ms")
    
    map_placeholder = st.empty()
    status_placeholder = st.empty()

    # If the user has NOT pressed simulate, render the completed path profile
    if not trigger_sim:
        map_data = [{"Order": idx + 1, "City": city, "Latitude": CITIES[city][0], "Longitude": CITIES[city][1]} for idx, city in enumerate(static_route)]
        df = pd.DataFrame(map_data)
        fig = px.line_mapbox(df, lat="Latitude", lon="Longitude", hover_name="City", zoom=3, height=550)
        fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        map_placeholder.plotly_chart(fig, use_container_width=True)
        status_placeholder.success(f"Static path compiled. Select 'Launch Live Simulation' to animate tracking cycles.")

    # Live Incremental Simulation Engine
    else:
        sim_route = ['Paris (France)']
        sim_unvisited = other_cities.copy()
        running_dist = 0.0
        
        while sim_unvisited:
            curr = sim_route[-1]
            nxt = min(sim_unvisited, key=lambda c: distance(curr, c))
            leg = distance(curr, nxt)
            
            running_dist += leg
            sim_route.append(nxt)
            sim_unvisited.remove(nxt)
            
            # Real-time frame building
            frame_data = [{"Order": idx + 1, "City": c, "Latitude": CITIES[c][0], "Longitude": CITIES[c][1]} for idx, c in enumerate(sim_route)]
            df_frame = pd.DataFrame(frame_data)
            
            fig_frame = px.line_mapbox(df_frame, lat="Latitude", lon="Longitude", hover_name="City", zoom=3, height=550)
            fig_frame.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
            
            status_placeholder.info(f"✈️ Dispatched from **{curr}** ➡️ Next: **{nxt}** (+{leg:.1f} km). Accumulated: {running_dist:.1f} km")
            map_placeholder.plotly_chart(fig_frame, use_container_width=True)
            time.sleep(speed)
            
        # Complete full circle back home
        final_leg = distance(sim_route[-1], 'Paris (France)')
        running_dist += final_leg
        sim_route.append('Paris (France)')
        
        final_data = [{"Order": idx + 1, "City": c, "Latitude": CITIES[c][0], "Longitude": CITIES[c][1]} for idx, c in enumerate(sim_route)]
        df_final = pd.DataFrame(final_data)
        fig_final = px.line_mapbox(df_final, lat="Latitude", lon="Longitude", hover_name="City", zoom=3, height=550)
        fig_final.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        
        status_placeholder.success(f"🏁 Route Finished! Circuit tracking successfully closed in Paris. Grand Total Distance: **{running_dist:.2f} km**")
        map_placeholder.plotly_chart(fig_final, use_container_width=True)
