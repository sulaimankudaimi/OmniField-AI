import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. إعدادات الهوية والوضوح القصوى ---
PLATFORM_NAME = "PetroVision AI"
DEVELOPER_NAME = "Eng. Sulaiman Kudaimi"

st.set_page_config(page_title=f"{PLATFORM_NAME}", layout="wide", initial_sidebar_state="expanded")

# تصميم CSS لضمان وضوح العناوين والقائمة الجانبية
st.markdown("""
    <style>
    .main { background-color: #05070a; color: #ffffff; }
    [data-testid="stSidebar"] { 
        background-color: #000000 !important; 
        border-right: 2px solid #00f2ff !important;
    }
    /* وضوح نصوص القائمة الجانبية */
    .st-ae, .st-af, .st-ag, p, span, label {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }
    /* تصميم العناوين الكبيرة والمضيئة */
    h1 {
        color: #00f2ff !important;
        font-weight: 900 !important;
        text-shadow: 2px 2px 15px rgba(0, 242, 255, 0.6);
        text-align: center;
        text-transform: uppercase;
    }
    .header-box { 
        padding: 30px; border-radius: 20px; 
        background: rgba(17, 24, 39, 0.9); border: 2px solid #00f2ff;
        text-align: center; margin-bottom: 30px;
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.2);
    }
    .guide-card {
        padding: 20px; background: #161b22; border-left: 5px solid #00f2ff;
        margin-bottom: 15px; border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. محرك تحميل البيانات الديناميكي ---
@st.cache_data
def get_default_data():
    # توليد بيانات افتراضية في حال عدم وجود ملفات لضمان عمل المنصة فوراً
    time = np.arange(100)
    prod = 5000 * np.exp(-0.02 * time) + np.random.normal(0, 50, 100)
    return pd.DataFrame({'Days': time, 'Production': prod})

def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    try:
        return pd.read_csv("Data/production_history.csv")
    except:
        return get_default_data()

# --- 3. القائمة الجانبية (Sidebar) مع بوابة البيانات ---
with st.sidebar:
    st.markdown(f"""
        <div style='text-align:center; padding:15px; border:2px solid #00f2ff; border-radius:15px;'>
            <h1 style='color:#00f2ff; font-size:1.5em;'>{PLATFORM_NAME}</h1>
            <p style='color:white; font-size:0.9em;'>By Eng. Sulaiman Kudaimi</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    menu = st.radio("MAIN NAVIGATION", 
                    ["📖 Quick Start Guide", "📊 Strategic Dashboard", "🌐 3D Reservoir Twin", "🔮 AI Production Forecast", "🛡️ HSE & Sensors"])
    
    st.markdown("---")
    st.markdown("### 📥 FIELD DATA GATEWAY")
    up_file = st.file_uploader("Upload Field CSV", type=['csv'], help="Upload production logs to update the AI model.")
    
    st.markdown("---")
    st.success("✅ System Status: Operational")

# تحميل البيانات بناءً على الرفع
active_df = load_data(up_file)

# --- 4. معالجة الصفحات والمحتوى ---

# العنوان الرئيسي الموحد
st.markdown(f"""
    <div class='header-box'>
        <h1>{PLATFORM_NAME} | Operational Command Hub</h1>
        <p style='color: #00f2ff; font-weight: bold; font-size: 1.2em;'>Next-Gen Oilfield Management by {DEVELOPER_NAME}</p>
    </div>
""", unsafe_allow_html=True)

if menu == "📖 Quick Start Guide":
    st.header("Welcome to PetroVision AI")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("""
        <div class='guide-card'>
            <h3 style='color:#00f2ff;'>1. Analyze Status</h3>
            <p>Go to <b>Strategic Dashboard</b> to view real-time KPIs and general field health.</p>
        </div>
        <div class='guide-card'>
            <h3 style='color:#00f2ff;'>2. Explore Subsurface</h3>
            <p>Use the <b>3D Twin</b> to visualize reservoir topography and well placements.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_g2:
        st.markdown("""
        <div class='guide-card'>
            <h3 style='color:#00f2ff;'>3. Predict Future</h3>
            <p>The <b>AI Forecast</b> uses machine learning to estimate production for the next 365 days.</p>
        </div>
        <div class='guide-card'>
            <h3 style='color:#00f2ff;'>4. Custom Data</h3>
            <p>Use the <b>Data Gateway</b> in the sidebar to upload your own field's CSV files.</p>
        </div>
        """, unsafe_allow_html=True)
    st.image("https://img.icons8.com/clouds/500/oil-rig.png", width=200)

elif menu == "📊 Strategic Dashboard":
    st.subheader("Field Key Performance Indicators")
    k1, k2, k3 = st.columns(3)
    k1.metric("Current Production", f"{int(active_df.iloc[-1,1])} bpd", "+5%")
    k2.metric("Reservoir Pressure", "3120 psi", "-12")
    k3.metric("System Safety", "100%", "Secure")
    st.line_chart(active_df.iloc[:, 1].tail(50))

elif menu == "🌐 3D Reservoir Twin":
    st.subheader("3D Interactive Field Mapping")
    x, y = np.linspace(-50, 50, 40), np.linspace(-50, 50, 40)
    X, Y = np.meshgrid(x, y)
    Z = -2000 - (0.08 * X**2 + 0.08 * Y**2) + (np.cos(X/10) * 30)
    
    fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y, colorscale='Jet')])
    # إضافة الآبار
    well_locs = [(-25,-25), (25,-25), (0,0), (-25,25), (25,25)]
    for wx, wy in well_locs:
        fig.add_trace(go.Scatter3d(x=[wx, wx], y=[wy, wy], z=[0, -2200], mode='lines', line=dict(color='white', width=7)))
        fig.add_trace(go.Scatter3d(x=[wx], y=[wy], z=[0], mode='markers', marker=dict(color='red', size=6)))
    
    fig.update_layout(template='plotly_dark', height=700, margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)

elif menu == "🔮 AI Production Forecast":
    st.subheader("Machine Learning Decline Curve Analysis")
    df = active_df.copy()
    X = np.arange(len(df)).reshape(-1, 1)
    y = df.iloc[:, 1]
    model = LinearRegression().fit(X, y)
    future = np.arange(len(df), len(df)+365).reshape(-1, 1)
    pred = model.predict(future)
    
    fig_ai = go.Figure()
    fig_ai.add_trace(go.Scatter(y=y, name="History", line=dict(color="#00f2ff")))
    fig_ai.add_trace(go.Scatter(x=np.arange(len(df), len(df)+365), y=pred, name="AI Forecast", line=dict(dash='dash', color='red')))
    fig_ai.update_layout(template='plotly_dark', title="Predicted vs Historical Production")
    st.plotly_chart(fig_ai, use_container_width=True)

elif menu == "🛡️ HSE & Sensors":
    st.subheader("Asset Integrity Sentinel")
    st.warning("Monitoring High-Pressure Critical Zones")
    # محاكاة بيانات حساسات
    sensor_data = np.random.normal(2800, 20, 100)
    st.area_chart(sensor_data)

# --- 5. التذييل ---
st.markdown("---")
st.markdown(f"<p style='text-align:center; color:#64748b;'>{PLATFORM_NAME} | Proprietary Tech by {DEVELOPER_NAME} © 2026</p>", unsafe_allow_html=True)
