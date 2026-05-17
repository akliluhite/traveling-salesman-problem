import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import itertools
import time
import random

st.set_page_config(page_title="Ultimate TSP Benchmark", layout="wide")
st.title("🧬 Traveling Salesman Problem: Genetic vs. Heuristic vs. Exact")
st.write("Benchmarking routing algorithms across **10 European Destinations** starting from Paris.")

# 10 Capital Cities across Europe
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
    lat1, lon1 = np.radians(CITIES[c1])
    lat2, lon2 = np.radians(CITIES[c2])
    return 6371.0 * (2 * np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2)))

def route_distance(r):
    return sum(distance(r[i], r[i+1]) for i in range(len(r)-1))

algo = st.selectbox("Select Strategy:", ["Nearest Neighbor (Fast / Greedy)", "Genetic Algorithm (Smart Balanced)", "Brute Force (Perfect / Heavy CPU)"])

cities_list = list(CITIES.keys())
other_cities = [c for c in cities_list if c != 'Paris (France)']
route = ['Paris (France)']

# Start Timer
start_time = time.perf_counter()

if algo == "Nearest Neighbor (Fast / Greedy)":
    unvisited = other_cities.copy()
    while unvisited:
        closest = min(unvisited, key=lambda c: distance(route[-1], c))
        route.append(closest)
        unvisited.remove(closest)
    route.append('Paris (France)')

elif algo == "Genetic Algorithm (Smart Balanced)":
    # Genetic Algorithm Parameters
    pop_size, generations = 50, 100
    population = [['Paris (France)'] + random.sample(other_cities, len(other_cities)) + ['Paris (France)'] for _ in range(pop_size)]
    
    for _ in range(generations):
        population = sorted(population, key=route_distance)
        survivors = population[:10]  # Select top 10 routes
        next_pop = survivors.copy()
        
        while len(next_pop) < pop_size:
            parent = random.choice(survivors)
            # Create a mutation by swapping two random cities
            child = parent.copy()
            idx1, idx2 = random.sample(range(1, len(cities_list)), 2)
            child[idx1], child[idx2] = child[idx2], child[idx1]
            next_pop.append(child)
        population = next_pop
    route = sorted(population, key=route_distance)[0]

else:
    # WARNING: 9! = 362,880 combinations. This will take a moment!
    best_dist = float('inf')
    for p in itertools.permutations(other_cities):
        r = ['Paris (France)'] + list(p) + ['Paris (France)']
        d = route_distance(r)
        if d < best_dist:
            best_dist, route = d, r

# Stop Timer
execution_time_ms = (time.perf_counter() - start_time) * 1000
total_dist = route_distance(route)

# Metrics UI
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Calculated Loop Distance", value=f"{total_dist:.2f} km")
with col2:
    st.metric(label="Calculation Time", value=f"{execution_time_ms:.2f} ms")

# Map Processing
map_data = [{"Order": idx + 1, "City": city, "Latitude": CITIES[city][0], "Longitude": CITIES[city][1]} for idx, city in enumerate(route)]
df = pd.DataFrame(map_data)

fig = px.line_mapbox(df, lat="Latitude", lon="Longitude", hover_name="City", zoom=3, height=600)
fig.update_layout(mapbox_style="carto-positron")
st.plotly_chart(fig, use_container_width=True)
