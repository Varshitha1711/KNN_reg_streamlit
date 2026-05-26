import streamlit as st
import numpy as np
import joblib

# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ======================
# LOAD MODEL
# ======================

model = joblib.load(
    "models/knn_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

# ======================
# CSS
# ======================

st.markdown("""
<style>

.main {
    background-color:#f8fafc;
}

.banner {
    background:linear-gradient(
    135deg,
    #2563eb,
    #7c3aed
    );

    padding:25px;
    border-radius:15px;
}

.title {
    color:white;
    text-align:center;
}

.pred-box{
    background:skyblue;
    border:2px solid #10b981;
    border-radius:15px;
    padding:25px;
}

</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================

st.markdown("""
<div class='banner'>
<h1 class='title'>
🏠 California House Price Predictor
</h1>
</div>
""", unsafe_allow_html=True)

st.write("Predict house prices using KNN Regression")

# ======================
# SIDEBAR
# ======================

st.sidebar.header("Property Details")

MedInc = st.sidebar.slider(
    "Median Income",
    0.0,
    15.0,
    3.0
)

HouseAge = st.sidebar.slider(
    "House Age",
    1,
    60,
    20
)

AveRooms = st.sidebar.slider(
    "Average Rooms",
    1.0,
    20.0,
    5.0
)

AveBedrms = st.sidebar.slider(
    "Average Bedrooms",
    0.5,
    5.0,
    1.0
)

Population = st.sidebar.slider(
    "Population",
    1,
    40000,
    1000
)

AveOccup = st.sidebar.slider(
    "Average Occupancy",
    1.0,
    10.0,
    3.0
)

Latitude = st.sidebar.slider(
    "Latitude",
    32.0,
    42.0,
    34.0
)

Longitude = st.sidebar.slider(
    "Longitude",
    -125.0,
    -114.0,
    -118.0
)

# ======================
# PREVIEW
# ======================

c1,c2,c3,c4 = st.columns(4)

c1.metric("Income", MedInc)
c2.metric("Age", HouseAge)
c3.metric("Rooms", AveRooms)
c4.metric("Population", Population)

# ======================
# PREDICT
# ======================

if st.button(
    "🔮 Predict House Price",
    use_container_width=True
):

    sample = np.array([[
        MedInc,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Latitude,
        Longitude
    ]])

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(
        sample_scaled
    )[0]

    actual_price = prediction * 100000

    st.markdown(
        f"""
        <div class='pred-box'>
        <h2>Estimated House Value</h2>
        <h1>${actual_price:,.0f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    