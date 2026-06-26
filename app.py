import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Define Executive Color Palette
COLOR_TEAL = "#97B2AE"
COLOR_MINT = "#D2DDD3"
COLOR_PINK = "#FEDDD6"
COLOR_PEACH = "#F2C2B7"
COLOR_BEIGE = "#D6CBBF"
COLOR_WHITE = "#F0EEEA"

# Application Configuration and Styling
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

# Artifact Loading Phase
@st.cache_resource
def load_artifacts():
    try:
        with open('logistic_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoder_month.pkl', 'rb') as f:
            le_month = pickle.load(f)
        return model, le_month
    except FileNotFoundError:
        st.error("Model artifacts not found. Please ensure logistic_model.pkl and label_encoder_month.pkl exist in the working directory.")
        return None, None

model, le_month = load_artifacts()

# Dashboard Header
st.title("E-Commerce Shopper Intention Analytics")
st.markdown("Developed By Aein")
st.write("---")

# Layout Split
col_inputs, col_outputs = st.columns([1, 1.2])

# Input Parameter Panel
with col_inputs:
    st.subheader("Live Session Indicators")
    
    with st.container():
        page_values = st.slider("Page Values", 0.0, 300.0, 0.0, help="Average value of the web pages visited by the user during the session.")
        exit_rates = st.slider("Exit Rates", 0.0, 0.2, 0.02)
        bounce_rates = st.slider("Bounce Rates", 0.0, 0.2, 0.02)
        
        st.markdown("### Behavioral and Temporal Metadata")
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            month_input = st.selectbox("Month", ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            visitor_type = st.selectbox("Visitor Type", ['Returning_Visitor', 'New_Visitor', 'Other'])
        with col_sub2:
            weekend = st.selectbox("Weekend Session?", ['False', 'True'])
            
            # Professional Traffic Channel Mapping Dictionary
            traffic_channels = {
                "Direct Traffic (ID 1)": 1,
                "Google Organic Search (ID 2)": 2,
                "Paid PPC Marketing (ID 3)": 3,
                "Social Media Ads (ID 4)": 4,
                "Email Campaigns (ID 5)": 5,
                "Affiliate Referral Networks (ID 6)": 6,
                "Partner Website Portals (ID 7)": 7,
                "Banner Display Ads (ID 8)": 8,
                "Third Party Blogs (ID 9)": 9,
                "Sponsored Newsletters (ID 10)": 10,
                "Internal Promotion Link (ID 11)": 11,
                "Retargeting Ad Campaign (ID 12)": 12,
                "Direct SMS Marketing (ID 13)": 13,
                "Video Platform Link (ID 14)": 14,
                "Influencer Promo Link (ID 15)": 15,
                "External Community Forum (ID 16)": 16,
                "Seasonal Event Traffic (ID 17)": 17,
                "Co-Branded Media Content (ID 18)": 18,
                "Push Notification Alert (ID 19)": 19,
                "Other Miscellaneous Channels (ID 20)": 20
            }
            
            selected_channel_name = st.selectbox("Traffic Channel", list(traffic_channels.keys()))
            traffic_type = traffic_channels[selected_channel_name]

        # Vector baseline padding features (Defaulted to maintain strict array dimension integrity)
        admin = 0; admin_dur = 0.0; info = 0; info_dur = 0.0; prod_rel = 10; prod_rel_dur = 200.0; special_day = 0.0
        op_sys = 2; browser = 2; region = 1

# Predictive Analysis and Sigmoid Mathematical Visualization
with col_outputs:
    st.subheader("Model Inference and Probability Curve")
    
    if model is not None and le_month is not None:
        try:
            encoded_month = le_month.transform([month_input])[0]
        except Exception:
            encoded_month = 2
        
        encoded_visitor = 2 if visitor_type == 'Returning_Visitor' else (0 if visitor_type == 'New_Visitor' else 1)
        encoded_weekend = 1 if weekend == 'True' else 0

        # Building input array according to model expected schema feature dimensions
        features = np.array([[admin, admin_dur, info, info_dur, prod_rel, prod_rel_dur,
                              bounce_rates, exit_rates, page_values, special_day, encoded_month,
                              op_sys, browser, region, traffic_type, encoded_visitor, encoded_weekend]])
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # Financial Conversion Score Metric
        st.markdown(f"""
            <div class='metric-box'>
                <h4>Probability of Conversion: <span style='color:{COLOR_TEAL};font-weight:bold;'>{probability*100:.2f}%</span></h4>
            </div>
        """, unsafe_allow_html=True)
        st.write("")

        # Dynamic Plot Formulation
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor(COLOR_WHITE)
        ax.set_facecolor("#FFFFFF")
        
        x_curve = np.linspace(0, max(100, page_values * 2), 200)
        y_curve = 1 / (1 + np.exp(-0.1 * (x_curve - 30))) 
        
        ax.plot(x_curve, y_curve, color=COLOR_PEACH, linewidth=3, label="Logistic S-Curve")
        ax.scatter([page_values], [probability], color=COLOR_TEAL, s=150, zorder=5, label="Current Shopper State")
        
        ax.set_title("Real-Time Sigmoid Distribution", fontsize=10, color="#2C3E50", weight='bold')
        ax.set_xlabel("Page Values", fontsize=8)
        ax.set_ylabel("Conversion Probability", fontsize=8)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(True, linestyle="--", alpha=0.3, color=COLOR_BEIGE)
        ax.legend(fontsize=7, facecolor=COLOR_WHITE)
        sns.despine(ax=ax)
        
        st.pyplot(fig)
        
        # Interpretive Verdict Section
        if prediction == 1:
            st.success("High Intent Buyer. Target with active incentives or automated customer funnel actions.")
        else:
            st.info("Low Engagement Session. Standard baseline system setup applied.")
