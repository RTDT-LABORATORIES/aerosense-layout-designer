"""
Aerosense Layout Designer - Streamlit App
Interactive tool for designing sensor layouts on blade geometries
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import tempfile
import os

# Import helper functions
import layout_helpers as helpers


def main():
    st.set_page_config(
        page_title="Aerosense Layout Designer",
        layout="wide"
    )
    
    st.title("Aerosense Layout Designer")
    st.markdown("Upload a blade geometry file and configure Aerosense nodes to visualize the layout.")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Blade Geometry File (.dat)",
        type=["dat", "txt"],
        help="Upload an airfoil coordinate file"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dat') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            wing_shape_dir = tmp_file.name
        
        # Wing dimensions
        st.sidebar.subheader("Wing Dimensions")
        chord = st.sidebar.number_input(
            "Chord (meters)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1
        )
        
        # Aerosense nodes configuration
        st.sidebar.subheader("Aerosense Nodes")
        num_patches = st.sidebar.number_input(
            "Number of Nodes",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )
        
        target_real_positions = []
        target_sides = []
        isens0_baros = []
        
        for i in range(num_patches):
            st.sidebar.markdown(f"**Node {i+1}**")
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                pos = st.number_input(
                    "Position",
                    min_value=0.0,
                    max_value=1.0,
                    value=[0.05, 0.1, 0.55][i] if i < 3 else 0.5,
                    step=0.05,
                    key=f"pos_{i}"
                )
                target_real_positions.append(pos)
            
            with col2:
                side = st.selectbox(
                    "Side",
                    ["pressure", "suction"],
                    index=0 if i == 0 else 1,
                    key=f"side_{i}"
                )
                target_sides.append(side)
            
            isens = st.sidebar.number_input(
                "Reference Sensor Index",
                min_value=0,
                max_value=23,
                value=12,
                key=f"isens_{i}"
            )
            isens0_baros.append(isens)
        
        # Fixed sensor specifications
        st.sidebar.subheader("Sensor Specifications")
        nbaros = 24
        sensor_spacing = 14.0
        sensor_thickness = 4.0
        limit_before = 36.0
        limit_after = 42.0
        
        st.sidebar.info(f"""
        - Sensors per node: {nbaros}
        - Sensor spacing: {sensor_spacing} mm
        - Sensor thickness: {sensor_thickness} mm
        """)
        
        # Process button
        if st.sidebar.button("Generate Layout", type="primary"):
            with st.spinner("Processing blade geometry..."):
                try:
                    # Create wing geometry
                    zwing, lwing, twing, nwing, blade_properties = helpers.make_labbook(
                        chord=chord,
                        span=1,
                        wing_shape_dir=wing_shape_dir,
                    )
                    
                    st.success(f"✅ Wing created with {len(zwing)} points")
                    
                    # Position Aerosense nodes
                    zBaros_list, limit_points, zsens0_baros, zBaros_all, db = helpers.position_aerosense_sensors(
                        zwing=zwing,
                        nwing=nwing,
                        chord=chord,
                        target_real_positions=target_real_positions,
                        target_sides=target_sides,
                        isens0_baros=isens0_baros,
                        nbaros=nbaros,
                        sensor_spacing=sensor_spacing,
                        sensor_thickness=sensor_thickness,
                        limit_before=limit_before,
                        limit_after=limit_after,
                    )
                    
                    # Display statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Aerosense Nodes", len(zsens0_baros))
                    with col2:
                        st.metric("Sensors per Node", nbaros)
                    with col3:
                        st.metric("Total Sensors", len(zBaros_all))
                    
                    # Create visualization
                    st.subheader("Sensor Layout Visualization")
                    
                    # Create color palette
                    colors = sns.color_palette("Set2", len(zBaros_list))
                    
                    # Create traces
                    aerosense_traces = []
                    limit_traces = []
                    
                    for i, zBaros in enumerate(zBaros_list):
                        color_rgb = f"rgb({colors[i][0]*255:.0f},{colors[i][1]*255:.0f},{colors[i][2]*255:.0f})"
                        
                        # Sensor trace
                        aerosense_trace = go.Scatter(
                            x=zBaros.real,
                            y=zBaros.imag,
                            mode="markers",
                            marker=dict(size=8, symbol="diamond", color=color_rgb),
                            name=f"Node {i+1} sensors"
                        )
                        aerosense_traces.append(aerosense_trace)
                        
                        # Limit markers
                        patch_limits = [limit_points[i*2], limit_points[i*2+1]]
                        limit_trace = go.Scatter(
                            x=[p.real for p in patch_limits],
                            y=[p.imag for p in patch_limits],
                            mode="markers",
                            marker=dict(size=6, symbol="cross", color=color_rgb),
                            name=f"Node {i+1} limits"
                        )
                        limit_traces.append(limit_trace)
                    
                    # Create wing trace
                    wing_trace = helpers.make_go_wing(zwing)
                    
                    # Create figure
                    layout = go.Layout(
                        title="Aerosense Sensor Layout on Blade",
                        showlegend=True,
                        xaxis=dict(scaleanchor="y", scaleratio=1, title="Chordwise position"),
                        yaxis=dict(title="Thickness"),
                        height=800
                    )
                    fig = go.Figure(data=[wing_trace] + aerosense_traces + limit_traces, layout=layout)
                    
                    # Display figure
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download button for the image
                    img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
                    st.download_button(
                        label="📥 Download Layout Image",
                        data=img_bytes,
                        file_name="aerosense_layout.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error processing blade geometry: {str(e)}")
                    st.exception(e)
                finally:
                    # Clean up temporary file
                    if os.path.exists(wing_shape_dir):
                        os.unlink(wing_shape_dir)
    else:
        # Display instructions when no file is uploaded
        st.info("👈 Please upload a blade geometry file to get started.")
        st.markdown("""
        ### Instructions:
        1. Upload a blade geometry file (.dat or .txt) using the sidebar
        2. Configure the wing dimensions (chord length)
        3. Set up Aerosense nodes with their positions and orientations
        4. Click "Generate Layout" to visualize the sensor arrangement
        5. Download the resulting image if needed
        
        ### File Format:
        The blade geometry file should contain airfoil coordinates in a standard format 
        (typically x,y pairs representing the airfoil shape).
        """)


if __name__ == "__main__":
    main()
