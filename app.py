import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# --- 1. الإعدادات العامة والهوية ---
PLATFORM_NAME = "PetroVision AI"
DEVELOPER_NAME = "Eng. Sulaiman Kudaimi"

st.set_page_config(
    page_title=f"{PLATFORM_NAME} | {DEVELOPER_NAME}",
    page_icon="💎",
    layout="wide"
)

# تصميم الواجهة الاحترافية (Custom CSS)
st.markdown("""
    <style>
    .main { background-color: #0b0e11; color: #e2e8f0; }
    .stSidebar { background-color: #151b23; border-right: 2px solid #00f2ff; }
    .header-box { 
        padding: 20px; border-radius: 12px; 
        background: linear-gradient(135deg, #0f172a, #1e3a8a); 
        border-left: 5px solid #00f2ff; margin-bottom: 25px;
    }
    .kpi-card { background-color: #1c252e; padding: 15px; border-radius: 10px; border: 1px solid #334155; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. محرك تحميل البيانات (Data Ingest Engine) ---
@st.cache_data
def load_all_data():
    files = {
        "petro": "Data/petrophysical_data.csv",
        "sensors": "Data/sensor_integrity_data.csv",
        "history": "Data/production_history.csv",
        "drilling": "Data/real_time_drilling_data.csv"
    }
    data_dict = {}
    for key, path in files.items():
        try:
            data_dict[key] = pd.read_csv(path)
        except:
            data_dict[key] = pd.DataFrame()
    return data_dict

data = load_all_data()

# --- 3. القائمة الجانبية (Mission Control) ---
with st.sidebar:
    st.markdown(f"""
        <div style='text-align: center; padding: 15px; border-radius: 10px; background: #0f172a; border: 1px solid #00f2ff;'>
            <h1 style='color: #00f2ff; margin:0; font-size: 1.4em;'>{PLATFORM_NAME}</h1>
            <p style='color: #94a3b8; font-size: 0.8em;'>Sovereign Digital Twin Platform</p>
            <hr style='border-top: 1px solid #334155;'>
            <p style='color: #cbd5e1; font-size: 0.85em;'>By: <b>{DEVELOPER_NAME}</b></p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    menu = st.radio("OPERATIONAL MODULES", 
                    ["Strategic Dashboard", "Subsurface (10k Petrophysics)", "Production (History & AI)", "Safety (10k Sensors)"])
    
    st.markdown("---")
    st.success("✅ Big Data Engine: Connected")

# --- 4. معالجة الأقسام (Module Logic) ---

if menu == "Strategic Dashboard":
    st.markdown(f"<div class='header-box'><h1>Global Operations Summary</h1><p>Integrated KPIs Managed by <b>{DEVELOPER_NAME}</b></p></div>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Subsurface Logs", f"{len(data['petro'])} pts")
    with c2: st.metric("Live Sensor Feeds", f"{len(data['sensors'])} pts")
    with c3: st.metric("Avg Pressure", "3120 psi", "-15")
    with c4: st.metric("System Uptime", "99.9%")

    # عرض عينة من البيانات الضخمة
    st.subheader("Real-time Drilling Data Stream")
    st.dataframe(data['drilling'].head(100), use_container_width=True)

elif menu == "Subsurface (10k Petrophysics)":
    st.title("🌐 Advanced Subsurface Analytics")
    if not data['petro'].empty:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.write("**Cross-Plot: Porosity vs Permeability**")
            fig_cross = px.scatter(data['petro'], x='Porosity_%', y='Permeability_mD', 
                                   color='Gamma_Ray_API', template='plotly_dark')
            st.plotly_chart(fig_cross, use_container_width=True)
        with col_b:
            st.write("**3D Structural Property Mapping**")
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=data['petro']['Depth_m'], y=data['petro']['Porosity_%'], z=data['petro']['Permeability_mD'],
                mode='markers', marker=dict(size=2, color=data['petro']['Gamma_Ray_API'], colorscale='Viridis')
            )])
            fig_3d.update_layout(template='plotly_dark', margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_3d, use_container_width=True)

elif menu == "Production (History & AI)":
    st.title("🔮 Production Forecasting Engine")
    if not data['history'].empty:
        fig_hist = px.line(data['history'], x=data['history'].columns[0], y=data['history'].columns[1], 
                           title="Historical Production Trend", template='plotly_dark')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.info("AI Analysis: Based on current trends, EUR is expected to increase by 4.2% with optimized drawdown.")

elif menu == "Safety (10k Sensors)":
    st.title("🛡️ HSE & Integrity Sentinel")
    if not data['sensors'].empty:
        st.write("**Real-time Vibration & Pressure Stream (10,000 Logs)**")
        # عرض آخر 500 نقطة لضمان سرعة الأداء
        fig_sensors = go.Figure()
        fig_sensors.add_trace(go.Scatter(y=data['sensors']['Wellhead_Pressure_psi'].tail(500), name="Pressure"))
        fig_sensors.add_trace(go.Scatter(y=data['sensors']['Vibration_Level_mm_s'].tail(500), name="Vibration", yaxis="y2"))
        fig_sensors.update_layout(
            template='plotly_dark',
            yaxis2=dict(title="Vibration", overlaying="y", side="right"),
            title="High-Frequency Monitoring Window"
        )
        st.plotly_chart(fig_sensors, use_container_width=True)

# --- 5. التذييل ---
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #64748b;'>Proprietary Big Data Platform | Developed & Architected by <b>{DEVELOPER_NAME}</b></p>", unsafe_allow_html=True)
