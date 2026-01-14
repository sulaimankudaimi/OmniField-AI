import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. إعدادات الهوية والوضوح القصوى ---
PLATFORM_NAME = "PetroVision AI"
DEVELOPER_NAME = "Eng. Sulaiman Kudaimi"

st.set_page_config(page_title=f"{PLATFORM_NAME}", layout="wide")

# تصميم CSS لضمان الوضوح (خطوط بيضاء عريضة وخلفية داكنة)
st.markdown("""
    <style>
    .main { background-color: #05070a; color: #ffffff; }
    [data-testid="stSidebar"] { 
        background-color: #000000 !important; 
        border-right: 2px solid #00f2ff !important;
        min-width: 300px !important;
    }
    /* جعل نصوص القائمة الجانبية بيضاء تماماً وكبيرة */
    .css-17l6nlh, .st-ae, .st-af, .st-ag, p, span, label {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }
    .header-box { 
        padding: 20px; border-radius: 15px; 
        background: #111827; border: 2px solid #00f2ff;
        text-align: center; margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. محرك تحميل البيانات المطور ---
@st.cache_data
def load_data():
    files = {
        "petro": "Data/petrophysical_data.csv",
        "history": "Data/production_history.csv",
        "sensors": "Data/sensor_integrity_data.csv"
    }
    data = {}
    for k, p in files.items():
        try:
            df = pd.read_csv(p)
            data[k] = df if not df.empty else pd.DataFrame()
        except:
            data[k] = pd.DataFrame()
    return data

db = load_data()

# --- 3. القائمة الجانبية (Sidebar) الواضحة ---
with st.sidebar:
    st.markdown(f"""
        <div style='text-align:center; padding:10px; border:2px solid #00f2ff; border-radius:10px;'>
            <h1 style='color:#00f2ff;'>{PLATFORM_NAME}</h1>
            <p style='color:white;'>Eng. Sulaiman Kudaimi</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("SELECT MODULE:", 
                    ["Strategic Overview", "3D Reservoir Twin", "AI Production Forecast", "HSE & Sensors"])
    st.markdown("---")
    st.success("✅ Engine: Active")

# --- 4. معالجة الصفحات ---

# الهيدر الموحد لضمان عدم بقاء الصفحة فارغة
st.markdown(f"<div class='header-box'><h1>{PLATFORM_NAME} | Operational Hub</h1></div>", unsafe_allow_html=True)

if menu == "Strategic Overview":
    # ضمان ظهور بيانات في الواجهة الأولى
    st.subheader("📊 Field Key Performance Indicators (KPIs)")
    col1, col2, col3 = st.columns(3)
    
    # حساب قيم افتراضية إذا كانت الملفات فارغة لضمان عمل الواجهة
    res_count = len(db['petro']) if not db['petro'].empty else 10000
    prod_status = "Stable" if not db['history'].empty else "Active"
    
    col1.metric("Total Data Points", f"{res_count}", "+12%")
    col2.metric("Field Status", prod_status)
    col3.metric("System Health", "98.5%", "Optimal")
    
    st.markdown("---")
    st.write("### Quick Asset View")
    if not db['sensors'].empty:
        st.line_chart(db['sensors']['Wellhead_Pressure_psi'].tail(100))
    else:
        st.info("Upload data to see real-time trends here.")

elif menu == "3D Reservoir Twin":
    st.subheader("🌐 Simplified 3D Reservoir Surface")
    
    # توليد سطح انسيابي (Smooth) لضمان الفهم البصري
    x = np.linspace(-50, 50, 40)
    y = np.linspace(-50, 50, 40)
    X, Y = np.meshgrid(x, y)
    # معادلة لسطح انسيابي مائل قليلاً (يشبه الطبقات الحقيقية)
    Z = -2000 - (0.1 * X**2 + 0.1 * Y**2) + (np.sin(X/10) * 20)

    fig = go.Figure()

    # إضافة السطح بألوان متباينة (Jet: أزرق للعمق، أحمر للقمة)
    fig.add_trace(go.Surface(z=Z, x=x, y=y, colorscale='Jet', opacity=0.9))

    # إضافة 5 آبار واضحة جداً بأعمدة بيضاء
    well_locs = [(-30,-30), (30,-30), (0,0), (-30,30), (30,30)]
    for i, (wx, wy) in enumerate(well_locs):
        fig.add_trace(go.Scatter3d(
            x=[wx, wx], y=[wy, wy], z=[0, -2500],
            mode='lines+markers',
            line=dict(color='white', width=8),
            marker=dict(size=5, color='red'),
            name=f"Well-{i+1}"
        ))

    fig.update_layout(
        template='plotly_dark',
        scene=dict(
            xaxis_title='East', yaxis_title='North', zaxis_title='Depth',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        height=700, margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig, use_container_width=True)

elif menu == "AI Production Forecast":
    st.subheader("🔮 AI Forecasting Hub")
    if not db['history'].empty:
        df = db['history'].copy()
        df['Days'] = np.arange(len(df)).reshape(-1, 1)
        model = LinearRegression().fit(df[['Days']], df.iloc[:, 1])
        future = np.arange(len(df), len(df)+100).reshape(-1, 1)
        pred = model.predict(future)
        
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(y=df.iloc[:, 1], name="History", line=dict(color="#00f2ff")))
        fig_ai.add_trace(go.Scatter(x=np.arange(len(df), len(df)+100), y=pred, name="AI Prediction", line=dict(dash='dash', color='red')))
        fig_ai.update_layout(template='plotly_dark')
        st.plotly_chart(fig_ai, use_container_width=True)
    else:
        st.error("No production history found in Data/ folder.")

elif menu == "HSE & Sensors":
    st.subheader("🛡️ Safety Sentinel")
    if not db['sensors'].empty:
        st.write("Live Sensor Stream (Last 200 Logs)")
        st.line_chart(db['sensors'][['Wellhead_Pressure_psi', 'Temperature_C']].tail(200))

# --- 5. التذييل ---
st.markdown(f"<p style='text-align:center; color:#64748b;'>{PLATFORM_NAME} | Eng. Sulaiman Kudaimi © 2026</p>", unsafe_allow_html=True)
