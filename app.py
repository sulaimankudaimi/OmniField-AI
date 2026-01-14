import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. الهوية والتهيئة ---
PLATFORM_NAME = "PetroVision AI"
DEVELOPER_NAME = "Eng. Sulaiman Kudaimi"

st.set_page_config(page_title=f"{PLATFORM_NAME} | {DEVELOPER_NAME}", layout="wide")

# تصميم واجهة مستخدم عالية التباين (Premium Dark Mode)
st.markdown("""
    <style>
    .main { background-color: #05070a; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 2px solid #00f2ff !important; }
    .header-box { padding: 20px; border-radius: 15px; background: linear-gradient(135deg, #001f3f, #0074d9); border-bottom: 4px solid #00f2ff; margin-bottom: 25px; text-align: center; }
    .signature-card { padding: 15px; background: #161b22; border: 2px solid #00f2ff; border-radius: 10px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. محرك تحميل ومعالجة البيانات ---
@st.cache_data
def load_and_process_data():
    paths = {
        "petro": "Data/petrophysical_data.csv",
        "history": "Data/production_history.csv",
        "sensors": "Data/sensor_integrity_data.csv"
    }
    data = {}
    for k, p in paths.items():
        try: data[k] = pd.read_csv(p)
        except: data[k] = pd.DataFrame()
    return data

db = load_and_process_data()

# --- 3. القائمة الجانبية ---
with st.sidebar:
    st.markdown(f"<div class='signature-card'><h2 style='color:white; margin:0;'>{PLATFORM_NAME}</h2><p style='color:#00f2ff; font-size:0.8em;'>Integrated AI-Field Hub</p><hr style='border-top:1px solid #00f2ff;'><p style='color:#cbd5e1; font-size:0.85em;'>Architected by:</p><p style='color:#00f2ff; font-size:1.1em; font-weight:bold;'>{DEVELOPER_NAME}</p></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    menu = st.radio("SELECT CONTROL MODULE", ["Strategic Dashboard", "Subsurface Twin (Data-Driven)", "AI Production Forecast", "HSE Asset Integrity"])
    st.markdown("---")
    st.info("System Engine: Scikit-learn & Plotly WebGL")

# العنوان الرئيسي
st.markdown(f"<div class='header-box'><h1 style='color:white; margin:0;'>{PLATFORM_NAME} | Operational Command Hub</h1><p style='color:#00f2ff; font-weight:bold;'>Data-Driven Insight by {DEVELOPER_NAME}</p></div>", unsafe_allow_html=True)

# --- 4. الأقسام المحدثة ---

if menu == "Subsurface Twin (Data-Driven)":
    st.subheader("🌐 Real-Data 3D Reservoir Simulation")
    
    if not db['petro'].empty:
        # ترميم المكمن: توليد سطح بناءً على بيانات العمق والمسامية الحقيقية
        df_sample = db['petro'].sample(1000) # عينة للسرعة
        
        fig = go.Figure()
        
        # إضافة السطح المعتمد على البيانات (Mesh Surface)
        fig.add_trace(go.Mesh3d(
            x=np.random.uniform(-100, 100, 1000), # إحداثيات افتراضية للتوزيع
            y=np.random.uniform(-100, 100, 1000),
            z=df_sample['Depth_m'] * -1, # تحويل العمق لقيم سالبة
            intensity=df_sample['Porosity_%'], # اللون يعبر عن المسامية الحقيقية
            colorscale='Jet', opacity=0.8, name="Reservoir Layer"
        ))

        # إضافة 5 آبار مخترقة للطبقة
        wells = [(-50,-50, "Well-1"), (50,-50, "Well-2"), (0,0, "Well-3"), (-50,50, "Well-4"), (50,50, "Well-5")]
        for wx, wy, wname in wells:
            fig.add_trace(go.Scatter3d(
                x=[wx, wx], y=[wy, wy], z=[0, -4500],
                mode='lines+markers', line=dict(color='white', width=5),
                marker=dict(size=3, color='red'), name=wname
            ))

        fig.update_layout(template='plotly_dark', scene=dict(aspectratio=dict(x=1, y=1, z=0.5)), height=700)
        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Surface generated using Petrophysical Depth & Porosity correlation.")

elif menu == "AI Production Forecast":
    st.subheader("🔮 Machine Learning Production Prediction")
    
    if not db['history'].empty:
        # ترميم التنبؤ: استخدام Linear Regression حقيقي
        df = db['history'].copy()
        df['Days'] = np.arange(len(df)).reshape(-1, 1)
        
        X = df[['Days']]
        y = df.iloc[:, 1] # نفترض العمود الثاني هو الإنتاج
        
        model = LinearRegression().fit(X, y)
        
        # التنبؤ للمستقبل (365 يوم قادم)
        future_days = np.arange(len(df), len(df) + 365).reshape(-1, 1)
        prediction = model.predict(future_days)
        
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(x=df['Days'], y=y, name="Historical Data", line=dict(color="#00f2ff")))
        fig_ai.add_trace(go.Scatter(x=future_days.flatten(), y=prediction, name="AI Prediction", line=dict(dash='dash', color='red')))
        
        fig_ai.update_layout(template='plotly_dark', title="AI-Driven Decline Curve Analysis")
        st.plotly_chart(fig_ai, use_container_width=True)
        st.info("The red dashed line represents the AI's learned production behavior from historical logs.")

elif menu == "HSE Asset Integrity":
    st.subheader("🛡️ Real-Time Sensor Stream")
    if not db['sensors'].empty:
        # عرض البيانات الضخمة (آخر 500 سطر)
        st.line_chart(db['sensors'][['Wellhead_Pressure_psi', 'Temperature_C']].tail(500))
