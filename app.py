import streamlit as st
import multi_disease_model as mdm

# Set page config
st.set_page_config(page_title="Multi Disease Prediction System", layout="centered")

# Sidebar title
st.sidebar.title("Multiple Disease Prediction System")

# Sidebar options
app_mode = st.sidebar.radio(
    "Select Disease to Predict",
    ['Diabetes Prediction', 'Heart Disease Prediction', 'Kidney Disease Prediction', 'Liver Disease Prediction']
)

# Load models once
if 'models_loaded' not in st.session_state:
    mdm.load_and_train_all()
    st.session_state.models_loaded = True

# Function to render forms
def render_prediction_form(disease):
    st.subheader(f"{disease.capitalize()} Prediction using ML")

    features_info = mdm.get_feature_info(disease)
    inputs = []

    with st.form(f"{disease}_form"):
        for feature, min_val, max_val in features_info:
            default_val = (min_val + max_val) / 2
            val = st.number_input(
                label=feature,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=(max_val - min_val) / 100
            )
            inputs.append(val)

        submitted = st.form_submit_button("Predict")

    if submitted:
        result = mdm.predict_disease(disease, inputs)
        if result == 1:
            st.success(f"🟢 Positive for {disease.capitalize()}")
        else:
            st.error(f"🔴 Negative for {disease.capitalize()}")

# Main area
if app_mode == 'Diabetes Prediction':
    render_prediction_form('diabetes')

elif app_mode == 'Heart Disease Prediction':
    render_prediction_form('heart')

elif app_mode == 'Kidney Disease Prediction':
    render_prediction_form('kidney')

elif app_mode == 'Liver Disease Prediction':
    render_prediction_form('liver')
