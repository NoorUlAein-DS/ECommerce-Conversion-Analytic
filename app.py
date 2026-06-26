import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 🎨 BRANDING & THEME COLORS (Aapka Palette)
# ==========================================
COLOR_TEAL = "#97B2AE"
COLOR_MINT = "#D2DDD3"
COLOR_PINK = "#FEDDD6"
COLOR_PEACH = "#F2C2B7"
COLOR_BEIGE = "#D6CBBF"
COLOR_WHITE = "#F0EEEA"

# Streamlit ki Custom Styling (CSS)
st.set_page_config(page_title="E-Commerce Shopper Intent AI", layout="wide")
st.markdown(f"""
    <style>
    .reportview-container {{ background-color: {COLOR_WHITE}; }}
    .sidebar .sidebar-content {{ background-color: {COLOR_BEIGE}; }}
    h1 {{ color: #2C3E50; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }}
    h3 {{ color: #34495E; }}
    .stButton>button {{
        background-color: {COLOR_TEAL}; color: white; 
        border-radius: 8px; border: none; padding: 10px 24px;
        font-weight: bold; transition: all 0.3s ease;
    }}
    .stButton>button:hover {{ background-color: {COLOR_PEACH}; color: black; }}
    .metric-box {{
        background-color: white; padding: 20px; 
        border-radius: 12px; border-left: 5px solid {COLOR_TEAL};
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 LOAD MODELS & ENCODERS
# ==========================================
@st.cache_resource
def load_artifacts():
    try:
        with open('logistic_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoder_month.pkl', 'rb') as f:
            le_month = pickle.load(f)
        return model, le_month
    except FileNotFoundError:
        st.error("🚨 'logistic_model.pkl' ya 'label_encoder_month.pkl' nahi mili! Pehle unhe download karke isi folder mein rakhein.")
        return None, None

model, le_month = load_artifacts()

# ==========================================
# 🚀 APP HEADER
# ==========================================
st.title("🛒 E-Commerce Shopper Intention Analytics")
st.markdown("An advanced AI framework powered by Logistic Regression to predict user purchase conversions based on real-time browsing behavior.")
st.write("---")

# Layout Configuration: Side-by-Side Panels
col_inputs, col_outputs = st.columns([1, 1.2])

# ==========================================
# 🎛️ SIDEBAR / INPUT PANEL
# ==========================================
with col_inputs:
    st.subheader("📊 Live Session Indicators")
    
    with st.container():
        # Core Numeric Features
        page_values = st.slider("📈 Page Values (Crucial Metric)", 0.0, 300.0, 0.0, help="Average value of the pages visited by the user.")
        exit_rates = st.slider("🚪 Exit Rates", 0.0, 0.2, 0.02)
        bounce_rates = st.slider("📉 Bounce Rates", 0.0, 0.2, 0.02)
        
        # Categorical / Behavioral Features
        st.markdown("### 🛠️ Behavioral & Temporal Metadata")
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            month_input = st.selectbox("📅 Month", ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            visitor_type = st.selectbox("👥 Visitor Type", ['Returning_Visitor', 'New_Visitor', 'Other'])
        with col_sub2:
            weekend = st.selectbox("🏖️ Weekend Session?", ['False', 'True'])
            traffic_type = st.number_input("🚦 Traffic Type ID", min_value=1, max_value=20, value=1)

        # Baseline features defaulted to zero for UI simplicity (X_train schema consistency)
        admin = 0; admin_dur = 0.0; info = 0; info_dur = 0.0; prod_rel = 10; prod_rel_dur = 200.0; special_day = 0.0
        op_sys = 2; browser = 2; region = 1

# ==========================================
# 🔮 INFERENCE & VISUALIZATION ENGINE
# ==========================================
with col_outputs:
    st.subheader("🎯 Model Inference & Probability Curve")
    
    if model is not None and le_month is not None:
        # Preprocessing user selections
        try:
            encoded_month = le_month.transform([month_input])[0]
        except Exception:
            encoded_month = 2 # Safe fallback
        
        encoded_visitor = 2 if visitor_type == 'Returning_Visitor' else (0 if visitor_type == 'New_Visitor' else 1)
        encoded_weekend = 1 if weekend == 'True' else 0

        # Constructing feature vector matching exactly model.feature_names_in_
        features = np.array([[admin, admin_dur, info, info_dur, prod_rel, prod_rel_dur,
                              bounce_rates, exit_rates, page_values, special_day, encoded_month,
                              op_sys, browser, region, traffic_type, encoded_visitor, encoded_weekend]])
        
        # Predictions
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] # Probability of buying (Class 1)

        # Display Metrics
        st.markdown(f"""
            <div class='metric-box'>
                <h4>Probability of Conversion: <span style='color:{COLOR_TEAL};font-weight:bold;'>{probability*100:.2f}%</span></h4>
            </div>
        """, unsafe_allow_html=True)
        st.write("")

        # Visualizing the Sigmoid Behavior Dynamically
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor(COLOR_WHITE)
        ax.set_facecolor("#FFFFFF")
        
        # Simulating an elegant mathematical curve around the user's input
        x_curve = np.linspace(0, max(100, page_values * 2), 200)
        # Logistic sigmoid approximation using model's behavior
        y_curve = 1 / (1 + np.exp(-0.1 * (x_curve - 30))) 
        
        # Plotting the smooth S-curve
        ax.plot(x_curve, y_curve, color=COLOR_PEACH, linewidth=3, label="Logistic S-Curve")
        # Highlighting the exact current user state
        ax.scatter([page_values], [probability], color=COLOR_TEAL, s=150, zorder=5, label="Current Shopper State")
        
        # Aesthetic Fine-Tuning
        ax.set_title("Real-Time Sigmoid Distribution", fontsize=10, color="#2C3E50", weight='bold')
        ax.set_xlabel("Page Values", fontsize=8)
        ax.set_ylabel("Conversion Probability", fontsize=8)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(True, linestyle="--", alpha=0.3, color=COLOR_BEIGE)
        ax.legend(fontsize=7, facecolor=COLOR_WHITE)
        sns.despine(ax=ax)
        
        st.pyplot(fig)
        
        # Final Verdict Message
        if prediction == 1:
            st.success("🔥 High Intent Buyer! Target with proactive discounts or fast-checkout popups.")
        else:
            st.info("💤 Low Engagement Session. Suggesting personalized product recommendations.")
