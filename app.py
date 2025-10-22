# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
from geopy.distance import geodesic

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection System")
st.write("Choose an algorithm, enter transaction details, and click **Check for Fraud**.")

# ----- Helpers -----
def haversine_km(lat1, lon1, lat2, lon2):
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).km
    except Exception:
        return 0.0

def safe_encode(value, encoder):
    """Encode value using a fitted LabelEncoder. If unseen, return -1."""
    try:
        # encoder has attribute classes_, and transform accepts list
        if value in list(encoder.classes_):
            return int(encoder.transform([value])[0])
        else:
            return -1
    except Exception:
        # fallback if encoder is not fitted or missing
        return -1

def hash_cc(cc_str, mod=10**2):
    try:
        return int(hash(cc_str) % mod)
    except Exception:
        return 0

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except Exception:
        return None

def get_probability(model, X):
    """Try predict_proba, else decision_function -> sigmoid, else None."""
    try:
        proba = model.predict_proba(X)
        # probability of positive class (1)
        if proba.shape[1] == 2:
            return float(proba[0, 1])
        # in rare multiclass, try index -1
        return float(proba[0, -1])
    except Exception:
        # try decision_function
        try:
            df = model.decision_function(X)
            # df might be array or scalar
            if np.ndim(df) == 0:
                return float(sigmoid(float(df)))
            elif np.ndim(df) == 1:
                return float(sigmoid(float(df[0])))
            else:
                return None
        except Exception:
            return None

# ----- Load model list -----
# Expected model filenames: "<Name>_fraud_model.jb", e.g. "LightGBM_fraud_model.jb"
available_models = {
    "LightGBM": "LightGBM_fraud_model.jb",
    "Logistic Regression": "LogisticRegression_fraud_model.jb",
    "Random Forest": "RandomForest_fraud_model.jb",
    "SVM": "SVM_fraud_model.jb",
    "XGBoost": "XGBoost_fraud_model.jb"
}

# Show only files that actually exist in the directory
existing_models = {k: v for k, v in available_models.items() if (st._is_running_with_streamlit and None) or True}
# We'll lazily check file existence when loading — show the full list to user
algorithm = st.selectbox("Select Algorithm", list(available_models.keys()))

# Load encoder(s)
encoders = {}
encoder_loaded = False
try:
    encoders = joblib.load("label_encoder.jb")
    encoder_loaded = True
except FileNotFoundError:
    st.warning("label_encoder.jb not found in the app folder. Categorical encoding will use fallback (-1).")
except Exception as e:
    st.warning(f"Could not load label_encoder.jb: {e}. Using fallback encodings (-1).")

# Attempt to load selected model
model = None
model_file = available_models.get(algorithm)
if model_file:
    try:
        model = joblib.load(model_file)
    except FileNotFoundError:
        st.error(f"Model file '{model_file}' not found in app directory. Please train & save model as '{model_file}'.")
    except Exception as e:
        st.error(f"Failed to load model '{model_file}': {e}")

# ----- Input fields -----
st.subheader("Transaction Details")
with st.form(key="tx_form"):
    merchant = st.text_input("Merchant Name")
    category = st.text_input("Category")
    amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
    lat = st.number_input("Latitude", format="%.6f")
    long = st.number_input("Longitude", format="%.6f")
    merch_lat = st.number_input("Merchant Latitude", format="%.6f")
    merch_long = st.number_input("Merchant Longitude", format="%.6f")
    hour = st.slider("Transaction Hour", 0, 23, 12)
    day = st.slider("Transaction Day", 1, 31, 15)
    month = st.slider("Transaction Month", 1, 12, 6)
    gender = st.selectbox("Gender", ["Male", "Female"])
    cc_num = st.text_input("Credit Card Number")
    submit = st.form_submit_button("Check for Fraud")

# ----- On submit -----
if submit:
    # basic validation
    if not merchant or not category or not cc_num:
        st.error("Please fill Merchant Name, Category, and Credit Card Number (required).")
    elif model is None:
        st.error("Selected model is not available. Please ensure its .jb file exists in the app folder.")
    else:
        # Prepare input row in same feature order used in training
        distance = haversine_km(lat, long, merch_lat, merch_long)

        # encode categorical features safely
        merchant_enc = safe_encode(merchant, encoders.get("merchant")) if encoder_loaded else -1
        category_enc = safe_encode(category, encoders.get("category")) if encoder_loaded else -1
        gender_enc = safe_encode(gender, encoders.get("gender")) if encoder_loaded else -1

        cc_hashed = hash_cc(cc_num, mod=10**2)

        # Feature order must match the training notebook:
        # ['merchant','category','amt','cc_num','hour','day','month','gender','distance']
        input_df = pd.DataFrame([[
            merchant_enc,
            category_enc,
            amt,
            cc_hashed,
            int(hour),
            int(day),
            int(month),
            gender_enc,
            float(distance)
        ]], columns=['merchant','category','amt','cc_num','hour','day','month','gender','distance'])

        # show input preview
        st.write("**Processed input features:**")
        st.dataframe(input_df)

        # Make prediction
        try:
            pred = model.predict(input_df)[0]
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            pred = None

        prob = get_probability(model, input_df)

        if pred is None:
            st.warning("Could not get a prediction from the selected model.")
        else:
            if pred == 1:
                if prob is not None:
                    st.error(f"⚠️ Fraudulent Transaction detected. (Probability: {prob:.2%})")
                else:
                    st.error("⚠️ Fraudulent Transaction detected.")
            else:
                if prob is not None:
                    st.success(f"✅ Transaction appears Legitimate. (Probability: {prob:.2%})")
                else:
                    st.success("✅ Transaction appears Legitimate.")

            # Optionally show probability gauge / details
            if prob is not None:
                st.progress(min(max(prob, 0.0), 1.0))
                st.caption("Probability shown by the model (closer to 1.00 → higher fraud risk).")

# ----- Footer / tips -----
st.markdown("---")
st.write("**Notes & tips:**")
st.write("""
- Models must be saved in the app folder with names like `LightGBM_fraud_model.jb`, `RandomForest_fraud_model.jb`, etc.
- `label_encoder.jb` should be a dict containing fitted LabelEncoders for the categorical columns: `{'merchant': encoder, 'category': encoder, 'gender': encoder}`.
- If you trained SVM, make sure it was fitted with `probability=True` if you want `predict_proba`. Otherwise the app will try `decision_function` and use a sigmoid approximation.
- For reproducible results, ensure the feature order and preprocessing in training matches this app.
""")

