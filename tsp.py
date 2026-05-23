# -------------------------------------------------------------
# MAP + LIVE SIMULATION
# -------------------------------------------------------------
st.markdown("---")
st.subheader("🗺️ Live Route Map")

if route:

    # Placeholder for live updating map
    map_placeholder = st.empty()

    # ---------------------------------------------------------
    # LIVE SIMULATION
    # ---------------------------------------------------------
    if st.session_state.sim_running:

        for step in range(2, len(route) + 1):

            partial_route = route[:step]

            latitudes = []
            longitudes = []
            labels = []

            for city in partial_route:

                lat, lon = ACTIVE_CITIES[city]

                latitudes.append(lat)
                longitudes.append(lon)
                labels.append(city)

            fig = go.Figure()

            # Animated route
            fig.add_trace(
                go.Scattermapbox(
                    lat=latitudes,
                    lon=longitudes,
                    mode="lines+markers",
                    marker=dict(
                        size=14
                    ),
                    line=dict(
                        width=5
                    ),
                    text=labels,
                    hoverinfo="text"
                )
            )

            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox_zoom=3.4,
                mapbox_center={
                    "lat": 50,
                    "lon": 10
                },
                height=700,
                margin={
                    "r": 0,
                    "t": 0,
                    "l": 0,
                    "b": 0
                }
            )

            # Update map live
            map_placeholder.plotly_chart(
                fig,
                use_container_width=True
            )

            time.sleep(speed)

        st.session_state.sim_running = False

    # ---------------------------------------------------------
    # STATIC FINAL MAP
    # ---------------------------------------------------------
    else:

        latitudes = []
        longitudes = []
        labels = []

        for city in route:

            lat, lon = ACTIVE_CITIES[city]

            latitudes.append(lat)
            longitudes.append(lon)
            labels.append(city)

        fig = go.Figure()

        fig.add_trace(
            go.Scattermapbox(
                lat=latitudes,
                lon=longitudes,
                mode="lines+markers",
                marker=dict(
                    size=14
                ),
                line=dict(
                    width=5
                ),
                text=labels,
                hoverinfo="text"
            )
        )

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=3.4,
            mapbox_center={
                "lat": 50,
                "lon": 10
            },
            height=700,
            margin={
                "r": 0,
                "t": 0,
                "l": 0,
                "b": 0
            }
        )

        map_placeholder.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------------------------------------------------
    # ROUTE TABLE
    # ---------------------------------------------------------
    st.markdown("### 📍 Final Route")

    route_table = pd.DataFrame({
        "Step": list(range(1, len(route) + 1)),
        "City": route
    })

    st.dataframe(
        route_table,
        use_container_width=True
    )

else:
    st.warning("No route generated.")
