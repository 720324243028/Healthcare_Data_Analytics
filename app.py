import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Page configuration
st.set_page_config(
    page_title="Healthcare Data Analytics",
    page_icon="🏥",
    layout="wide"
)

# Project title
st.title("Healthcare Data Analytics and Disease Risk Prediction")
st.write(
    "Analyze patient health data, explore patterns, "
    "and estimate diabetes risk using machine learning."
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("healthcare_disease_risk_dataset.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found. Place the CSV file in the project folder."
    )
    st.stop()

# Data cleaning
df = df.drop_duplicates()
df = df.dropna()

# Convert categorical values into consistent format
df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
df["Family_History"] = (
    df["Family_History"].astype(str).str.strip().str.title()
)
df["Diabetes"] = df["Diabetes"].astype(str).str.strip().str.title()

# Dashboard summary
st.subheader("Healthcare Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Patients", len(df))

with col2:
    st.metric(
        "Diabetes Cases",
        int((df["Diabetes"] == "Yes").sum())
    )

with col3:
    st.metric(
        "Non-Diabetes Cases",
        int((df["Diabetes"] == "No").sum())
    )

# Dataset preview
st.subheader("Patient Dataset")
st.dataframe(df, use_container_width=True)

# Data analysis and visualization
st.subheader("Diabetes Distribution")

fig1, ax1 = plt.subplots()

sns.countplot(
    data=df,
    x="Diabetes",
    hue="Diabetes",
    legend=False,
    ax=ax1
)

ax1.set_title("Diabetes Cases Distribution")
ax1.set_xlabel("Diabetes Status")
ax1.set_ylabel("Number of Patients")

st.pyplot(fig1)
plt.close(fig1)

# Age analysis
st.subheader("Age Distribution")

fig2, ax2 = plt.subplots()

sns.histplot(
    data=df,
    x="Age",
    bins=8,
    kde=True,
    ax=ax2
)

ax2.set_title("Patient Age Distribution")

st.pyplot(fig2)
plt.close(fig2)

# Glucose analysis
st.subheader("Glucose Level Analysis")

fig3, ax3 = plt.subplots()

sns.boxplot(
    data=df,
    x="Diabetes",
    y="Glucose_Level",
    hue="Diabetes",
    legend=False,
    ax=ax3
)

ax3.set_title("Glucose Levels by Diabetes Status")

st.pyplot(fig3)
plt.close(fig3)

# BMI analysis
st.subheader("BMI Analysis")

fig4, ax4 = plt.subplots()

sns.scatterplot(
    data=df,
    x="Age",
    y="BMI",
    hue="Diabetes",
    ax=ax4
)

ax4.set_title("Age vs BMI")

st.pyplot(fig4)
plt.close(fig4)

# Machine learning model
st.subheader("Diabetes Risk Prediction")

st.warning(
    "Educational demonstration only. "
    "This small synthetic dataset is not suitable for medical diagnosis."
)

features = [
    "Age",
    "Gender",
    "Glucose_Level",
    "Blood_Pressure",
    "BMI",
    "Insulin_Level",
    "Family_History"
]

X = df[features]
y = df["Diabetes"].map({"No": 0, "Yes": 1})

if y.isna().any() or y.nunique() != 2:
    st.error("The dataset must contain both Yes and No diabetes labels.")
    st.stop()

categorical_features = ["Gender", "Family_History"]

numeric_features = [
    "Age",
    "Glucose_Level",
    "Blood_Pressure",
    "BMI",
    "Insulin_Level"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        ("numeric", "passthrough", numeric_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# Split the data for a basic demonstration
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

st.metric("Test Set Accuracy", f"{accuracy * 100:.2f}%")

st.caption(
    "This accuracy is calculated on a small synthetic dataset "
    "and does not indicate real-world medical performance."
)

# Prediction form
st.subheader("Enter Patient Details")

with st.form("prediction_form"):

    age = st.slider("Age", 18, 100, 40)

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=0.0,
        max_value=500.0,
        value=110.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=300.0,
        value=75.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=25.0
    )

    insulin = st.number_input(
        "Insulin Level",
        min_value=0.0,
        max_value=1000.0,
        value=90.0
    )

    family_history = st.selectbox(
        "Family History of Diabetes",
        ["Yes", "No"]
    )

    submitted = st.form_submit_button("Estimate Diabetes Risk")

if submitted:

    patient = pd.DataFrame(
        [[
            age,
            gender,
            glucose,
            blood_pressure,
            bmi,
            insulin,
            family_history
        ]],
        columns=features
    )

    risk_probability = model.predict_proba(patient)[0][1]

    st.write(
        f"Model-estimated probability: "
        f"{risk_probability * 100:.2f}%"
    )

    st.info(
        "This is an experimental model estimate, not a medical diagnosis. "
        "Consult a qualified healthcare professional for medical advice."
    )

# Download cleaned dataset
st.subheader("Download Dataset")

csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Cleaned CSV",
    data=csv_data,
    file_name="cleaned_healthcare_dataset.csv",
    mime="text/csv"
)

st.success("Healthcare analytics dashboard is ready.")