import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Water Potability Prediction", layout="wide")

st.title("Water Potability Prediction")
st.write("""
    This app predicts whether water is potable based on chemical properties.
    Adjust the parameters on the sidebar and click Predict.
""")

st.sidebar.header("Input Parameters")

def user_input_features():
    ph = st.sidebar.slider("pH", 0.0, 14.0, 7.0)
    hardness = st.sidebar.slider("Hardness", 0, 500, 150)
    solids = st.sidebar.slider("Total Dissolved Solids", 0, 2000, 500)
    chloramines = st.sidebar.slider("Chloramines", 0.0, 10.0, 2.0)
    sulfate = st.sidebar.slider("Sulfate", 0, 500, 150)
    conductivity = st.sidebar.slider("Conductivity", 0, 1000, 300)
    organic_carbon = st.sidebar.slider("Organic Carbon", 0.0, 20.0, 5.0)
    trihalomethanes = st.sidebar.slider("Trihalomethanes", 0.0, 100.0, 20.0)
    turbidity = st.sidebar.slider("Turbidity", 0.0, 10.0, 1.0)

    data = {
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

@st.cache_data
def load_data():
    data = pd.read_csv("water_potability.csv")
    data.fillna(data.mean(), inplace=True)
    return data

data = load_data()

feature_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

X = data[feature_cols]
y = data['Potability']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

if st.sidebar.button("Predict"):
    input_scaled = scaler.transform(df[feature_cols])
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("Water is **Potable** :sunglasses:")
    else:
        st.error("Water is **Not Potable** :warning:")

    st.subheader("Prediction Probability")
    st.write(f"Potable: {prediction_proba[0][1]:.2f}")
    st.write(f"Not Potable: {prediction_proba[0][0]:.2f}")

    y_pred = model.predict(X_test)
    st.subheader("Model Evaluation")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("pH Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(data['ph'], kde=True, ax=ax2, color='blue')
    st.pyplot(fig2)

else:
    st.write("Adjust parameters in the sidebar and click **Predict** to see results.")
