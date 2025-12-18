import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score,root_mean_squared_error,mean_absolute_error

#Page Config #
st.set_page_config("Linear Regression", layout="centered")

# Load css #
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
load_css("styles.css")

# Title #
st.markdown("""
<div class="card">
        <h1>Linear Regression</h1>
        <p>Predict <b> Tip Amount </b> from <b> Total Bill </b> using Linear Regression</p>
</div>
""", unsafe_allow_html=True)

# Load Data #
@st.cache_data
def load_data():
    data = sns.load_dataset("tips")
    return data
df=load_data()

# Dataset preview #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# preparing the data #
X = df[["total_bill"]]
y = df["tip"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the model #
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

#Metrics #
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 2)


# Visualization #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip Amount")
fig, ax = plt.subplots()
ax.scatter(df[["total_bill"]], df["tip"],alpha=0.6)
ax.plot(df[["total_bill"]], model.predict(scaler.transform(X)), color='red')
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip Amount")
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# Performance Metrics #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance Metrics")
c1,c2=st.columns(2)
c1.metric("MAE",f"{mae:.2f}")
c2.metric("RMSE",f"{rmse:.2f}")
c3,c4=st.columns(2)
c3.metric("R2_Score",f"{r2:.3f}")
c4.metric("Adjusted R²",f"{adj_r2:.3f}")
st.markdown('</div>', unsafe_allow_html=True)

# m & c #
st.markdown(f"""
            <div class="card">
            <h3>Model Interception</h3>
            <p><b>Slope (m):</b> {model.coef_[0]:.3f}</p>
            <p><b>Intercept (c):</b> {model.intercept_:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

# Prediction #
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Predict tip amount")
bill=st.slider("Total Bill :",float(df["total_bill"].min()),float(df["total_bill"].max()),30.0)
tip=model.predict(scaler.transform([[bill]]))[0]
st.markdown(f'<div class="prediction-box">Predicted Tip: {tip:.2f}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)



