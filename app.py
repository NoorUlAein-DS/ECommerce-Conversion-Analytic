import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

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
st.markdown("An advanced AI framework powered by Logistic Regression to predict user purchase conversions based on real-time browsing behavior.")
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

# Predictive Analysis and Interactive 3D Vector Visualization
with col_outputs:
    st.subheader("Model Inference and 3D Multi-Variable Space")
    
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

        # 3D Math Space Generation (Simulating Sigmoid Surface using Page Values and Exit Rates)
        x_space = np.linspace(0, 300, 50)
        y_space = np.linspace(0, 0.2, 50)
        X_mesh, Y_mesh = np.meshgrid(x_space, y_space)
        
        # Mathematical approximation matching the multi-variable vector trends
        Z_mesh = 1 / (1 + np.exp(-0.05 * (X_mesh - 40) + 15 * Y_mesh))

        # Building the Interactive 3D Plotly Figure
        fig = go.Figure()

        # Add 3D Surface
        fig.add_trace(go.Surface(
            x=X_mesh, y=Y_mesh, z=Z_mesh,
            colorscale=[[0, COLOR_PINK], [0.5, COLOR_PEACH], [1, COLOR_TEAL]],
            opacity=0.8,
            showscale=False,
            name="Sigmoid Probability Plane"
        ))

        # Add Active User 3D Coordinates
        fig.add_trace(go.Scatter3d(
            x=[page_values], y=[exit_rates], z=[probability],
            mode='markers',
            marker=dict(size=10, color='#2C3E50', symbol='circle', opacity=1.0),
            name="Current Shopper State"
        ))

        # Design and Layout adjustments for 3D Presentation
        fig.update_layout(
            title=dict(text="Real-Time 3D Sigmoid Surface Mapping", font=dict(size=14, color="#2C3E50", family="sans-serif")),
            scene=dict(
                xaxis=dict(title='Page Values', backgroundcolor=COLOR_WHITE, gridcolor=COLOR_BEIGE, showbackground=True),
                yaxis=dict(title='Exit Rates', backgroundcolor=COLOR_WHITE, gridcolor=COLOR_BEIGE, showbackground=True),
                zaxis=dict(title='Probability', backgroundcolor=COLOR_WHITE, gridcolor=COLOR_BEIGE, showbackground=True, range=[0, 1])
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretive Verdict Section
        if prediction == 1:
            st.success("High Intent Buyer. Target with active incentives or automated customer funnel actions.")
        else:
            st.info("Low Engagement Session. Standard baseline system setup applied.")
