import streamlit as st
import pandas as pd
import joblib

# =====================
# Load Model
# =====================
model = joblib.load("pancreatic_rf.pkl")

# =====================
# Page Title
# =====================
st.title("🧬 AI-Based Prediction of Chemotherapy Response in Pancreatic Cancer")

st.info(
    "This tool is for educational and research purposes only and should not be used for clinical decision-making."
)

st.write("""
### Project Information

**Model:** Random Forest Classifier

**Accuracy (5-Fold Cross Validation):** 80.4%

**Number of Genes Used:** 19

This AI model predicts whether a pancreatic cancer patient is likely to respond to Gemcitabine treatment based on gene expression data.
""")

# =====================
# Gene List
# =====================
genes = [
    "AC010226.4",
    "AC139452.2",
    "ATM",
    "BRCA1",
    "BRCA2",
    "CDKN2A",
    "EGFR",
    "KRAS",
    "LINC00346",
    "PALB2",
    "PES1P2",
    "PPEF2",
    "RP11-179A10.1",
    "RP11-248J23.5",
    "RP11-297K8.2",
    "RP11-474C8.8",
    "SMAD4",
    "TP53",
    "Z83001.1"
]

st.write("### Genes Used")
st.write(", ".join(genes))

# =====================
# Input Section
# =====================
st.write("### Enter Gene Expression Values")

data = {}

for gene in genes:
    data[gene] = st.number_input(
        gene,
        value=0.0,
        format="%.2f"
    )

# =====================
# Prediction
# =====================
if st.button("Predict Response"):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    st.write("## Prediction Result")

    if prediction == 1:

        confidence = probability[1] * 100

        st.success(
            f"Response Predicted ({confidence:.1f}% confidence)"
        )

        st.progress(float(probability[1]))

    else:

        confidence = probability[0] * 100

        st.error(
            f"Non-Response Predicted ({confidence:.1f}% confidence)"
        )

        st.progress(float(probability[0]))