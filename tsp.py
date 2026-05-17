import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import random

st.set_page_config(page_title="TSP Live Simulation", layout="wide")
st.title("🎬 Traveling Salesman Live Simulation")
st.write("Watch how the algorithm builds the path step-by-step across Europe.")

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

# Simulation Speed Controller
speed = st.slider("Simulation Speed (seconds per step):", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

if st.button("▶️ Start Live Simulation"):
    cities_list = list(CITIES.keys())
    route = ['Paris (France)']
    unvisited = [c for c in cities_list if c != 'Paris (France)']
    
    # Create an empty placeholder container to update the map dynamically
    map_placeholder = st.empty()
    status_placeholder = st.empty()
    
    total_dist = 0.0
    
    while unvisited:
        current_city = route[-1]
        closest_city = min(unvisited, key=lambda c: distance(current_city, c))
        leg_dist = distance(current_city, closest_city)
        
        # Update path data
        total_dist += leg_dist
        route.append(closest_city)
        unvisited.remove(closest_city)
        
        # Format map rendering data frame for the current step
        map_data = [{"Order": idx + 1, "City": c, "Latitude": CITIES[c], "Longitude": CITIES[c]} for idx, c in enumerate(route)]
        df = pd.DataFrame(map_data)
        
        # Draw map at current frame
        fig = px.line_mapbox(df, lat="Latitude", lon="Longitude", hover_name="City", zoom=3, height=500)
        fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        
        status_placeholder.info(f"✈️ Traveling from **{current_city}** to **{closest_city}** ({leg_dist:.1f} km). Total: {total_dist:.1f} km")
        map_placeholder.plotly_chart(fig, use_container_width=True)
        
        time.sleep(speed)  # Wait according to slider value before next step
        
    # Final leg back to starting city
    final_leg = distance(route[-1], 'Paris (France)')
    total_dist += final_leg
    route.append('Paris (France)')
    
    map_data = [{"Order": idx + 1, "City": c, "Latitude": CITIES[c], "Longitude": CITIES[c]} for idx, c in enumerate(route)]
    df = pd.DataFrame(map_data)
    fig = px.line_mapbox(df, lat="Latitude", lon="Longitude", hover_name="City", zoom=3, height=500)
    fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
    
    status_placeholder.success(f"🏁 Route Complete! Returned to Paris. Grand Total Distance: **{total_dist:.2f} km**")
    map_placeholder.plotly_chart(fig, use_container_width=True)
