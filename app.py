import streamlit as st
import joblib
import pandas as pd


# Page config must be first command
st.set_page_config(
    page_title="❤️ Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        production_model = joblib.load('models/uci_heart_disease_model.pkl')
        return production_model['model'], production_model['metadata']['threshold']
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


model, optimal_threshold = load_model()


# Function to process input and make predictions
def predict_heart_disease(user_input):
    try:
        # Feature engineering
        user_input['hr_age_ratio'] = user_input['thalach'] / (user_input['age'] + 1e-5)
        user_input['bp_oldpeak'] = user_input['trestbps'] * (user_input['oldpeak'] + 1)
        user_input['risk_score'] = (user_input['age'] / 50 + user_input['chol'] / 200 + user_input['trestbps'] / 140)

        # Make prediction
        probabilities = model.predict_proba(user_input)[:, 1]
        predictions = (probabilities >= optimal_threshold).astype(int)

        # Create results DataFrame
        results = pd.DataFrame({
            'Prediction': predictions,
            'Diagnosis': ['Heart Disease' if p == 1 else 'Healthy' for p in predictions],
            'Probability': probabilities,
        })

        # Combine with input features for display
        display_data = pd.concat([user_input[['age', 'sex', 'cp', 'trestbps', 'chol']], results], axis=1)

        return results, display_data

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None


# Main app interface
st.title("❤️ Heart Disease Prediction")

# Create tabs
tab1, tab2 ,tab3= st.tabs(["Single Prediction", "Batch Prediction","Data & Model Info"])

with tab1:
    st.header("Single Patient Prediction")

    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Patient Information")
            age = st.slider("Age", 18, 100, 50)
            sex = st.radio("Sex", ["Male (1)", "Female (0)"], index=0)
            cp = st.selectbox("Chest Pain Type",
                              ["Typical angina (1)", "Atypical angina (2)",
                               "Non-anginal pain (3)", "Asymptomatic (4)"])
            trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 120)
            chol = st.slider("Serum Cholesterol (mg/dl)", 150, 350, 200)

        with col2:
            st.subheader("Clinical Measurements")
            fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes (1)", "No (0)"], index=1)
            restecg = st.selectbox("Resting ECG Results",
                                   ["Normal (0)", "ST-T wave abnormality (1)",
                                    "Left ventricular hypertrophy (2)"])
            thalach = st.slider("Maximum Heart Rate Achieved (bpm)", 60, 200, 150)
            exang = st.radio("Exercise Induced Angina", ["Yes (1)", "No (0)"], index=1)
            oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment",
                                 ["Upsloping (1)", "Flat (2)", "Downsloping (3)"])
            ca = st.slider("Number of Major Vessels", 0, 4, 0)
            thal = st.selectbox("Thalassemia",
                                ["Normal (3)", "Fixed defect (6)", "Reversible defect (7)"])

        submitted = st.form_submit_button("Predict Heart Disease Risk")

    if submitted:
        # Preprocess inputs
        user_input = pd.DataFrame({
            'age': [age],
            'sex': [1 if sex.startswith("Male") else 0],
            'cp': [int(cp.split("(")[1].strip(")"))],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [1 if fbs.startswith("Yes") else 0],
            'restecg': [int(restecg.split("(")[1].strip(")"))],
            'thalach': [thalach],
            'exang': [1 if exang.startswith("Yes") else 0],
            'oldpeak': [oldpeak],
            'slope': [int(slope.split("(")[1].strip(")"))],
            'ca': [ca],
            'thal': [int(thal.split("(")[1].strip(")"))],
        })

        # Get predictions
        results, display_data = predict_heart_disease(user_input)

        if results is not None:
            st.subheader("Prediction Results")

            # Display the formatted results
            st.markdown(f"""
            ### Heart Disease Prediction Results
            **Using threshold:** {optimal_threshold:.3f}
            """)

            # Show detailed results in expandable section
            with st.expander("View Detailed Results"):
                st.dataframe(display_data)

            # Show risk assessment
            probability = results['Probability'].iloc[0]
            prediction = results['Diagnosis'].iloc[0]

            if probability > 0.7:
                risk_level = "High"
                recommendation = "Immediate consultation with cardiologist recommended"
                color = "red"
            elif probability > 0.4:
                risk_level = "Medium"
                recommendation = "Further tests recommended"
                color = "orange"
            else:
                risk_level = "Low"
                recommendation = "No immediate concerns, maintain regular checkups"
                color = "green"

            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction", prediction)
            with col2:
                st.metric("Probability", f"{probability * 100:.2f}%")
            with col3:
                st.metric("Risk Level", risk_level)

            # Show recommendation
            st.markdown(f"""
            <div style='background-color:#f0f2f6; padding:10px; border-radius:5px;'>
                <h4 style='color:{color};'>Recommendation: {recommendation}</h4>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.header("Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV file with patient data", type=["csv"])

    if uploaded_file is not None:
        try:
            test_data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")

            # Check for required columns
            required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

            missing_cols = [col for col in required_cols if col not in test_data.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Get predictions
                results, display_data = predict_heart_disease(test_data)

                if results is not None:
                    st.subheader("Prediction Results")

                    # Show summary statistics
                    st.markdown(f"""
                    ### Batch Prediction Results
                    **Using threshold:** {optimal_threshold:.3f}
                    """)

                    # Combine results with original data
                    full_results = test_data.copy()
                    full_results['Probability'] = results['Probability']
                    full_results['Prediction'] = results['Prediction']
                    full_results['Diagnosis'] = results['Diagnosis']

                    # Show results in expandable section
                    with st.expander("View All Predictions"):
                        st.dataframe(full_results)

                    # Show statistics
                    st.subheader("Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Patients", len(full_results))
                    with col2:
                        st.metric("Heart Disease Cases", full_results['Prediction'].sum())
                    with col3:
                        st.metric("Healthy Cases", len(full_results) - full_results['Prediction'].sum())

                    # Add download button
                    csv = full_results.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "heart_disease_predictions.csv",
                        "text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")

sample_data = pd.DataFrame({
    'age': [52, 63, 45, 67, 58],
    'sex': [1, 1, 0, 0, 1],
    'cp': [3, 4, 2, 3, 4],
    'trestbps': [125, 145, 130, 120, 136],
    'chol': [212, 233, 204, 228, 319],
    'fbs': [0, 1, 0, 0, 0],
    'restecg': [0, 1, 0, 1, 0],
    'thalach': [168, 150, 172, 129, 152],
    'exang': [0, 0, 0, 1, 0],
    'oldpeak': [1.0, 2.3, 1.4, 2.6, 0.0],
    'slope': [2, 3, 1, 2, 1],
    'ca': [2, 0, 0, 1, 0],
    'thal': [3, 3, 3, 7, 3]
})

with tab3:
    st.header("Data & Model Information")

    st.subheader("Dataset Information")
    st.markdown("""
    The model was trained on the UCI Heart Disease Dataset containing the following features:
    - **Demographic**: Age, Sex
    - **Clinical**: Blood Pressure, Cholesterol, etc.
    - **Electrocardiographic**: Resting ECG, Exercise ST segment, etc.
    """)

    st.subheader("Sample Data")
    st.dataframe(sample_data)

    st.subheader("Model Performance")
    st.markdown("""
    - **Accuracy**: 85.2% (on test set)
    - **Precision**: 83.1%
    - **Recall**: 87.5%
    - **F1-score**: 85.2%
    
    **📈 Additional Metrics:**
    - **ROC AUC:** `0.909`
    - **Sensitivity (Recall):** `0.95` _(for Heart Disease)_
    - **Specificity:** `0.76` _(for Healthy)_
    - **Balanced Accuracy:** `0.855`
    - **False Positive Rate (FPR):** `0.24`
    - **False Negative Rate (FNR):** `0.05`
    - **Precision (Heart Disease):** `0.80`
    - **Precision (Healthy):** `0.95`
    - **F1 Score (Overall):** `0.85`
    - **Support Size:** `46` patients
    """)

    st.subheader("Risk Interpretation Guide")
    st.markdown("""
    - **High Risk (>70%)**: Strong recommendation for cardiologist consultation
    - **Medium Risk (40-70%)**: Suggest additional tests
    - **Low Risk (<40%)**: Likely healthy, maintain regular checkups
    """)

    st.subheader("Terms of Use")
    st.markdown("""
    This tool is for informational purposes only and should not replace 
    professional medical advice. Always consult a healthcare provider 
    for medical diagnosis and treatment.
    """)

# Sidebar with info
with st.sidebar:
    st.title("❤️ Heart Disease Prediction")
    st.markdown("""
    ## 🧠 Model & System Info
    This application predicts the likelihood of heart disease based on clinical features using a machine learning model.
    - **Developed by Musabbir KM**
    - **Model Name:** Heart-Guard
    - **Version:** 1.1
    
    ### Model Information
    - **Algorithm**: Random Forest Classifier
    - **Dataset**: UCI Heart Disease Dataset
    - **Optimal Threshold**: {:.3f}
    - **Version**: 1.1

    ### How It Works
    1. Enter patient details
    2. Click 'Predict' button
    3. View prediction results
    """.format(optimal_threshold))

    st.markdown("---")
    st.markdown("""
    ### Feature Descriptions
    - **Age**: Patient's age in years
    - **Sex**: Gender (1 = Male, 0 = Female)
    - **CP**: Chest pain type (1-4)
    - **Trestbps**: Resting blood pressure (mmHg)
    - **Chol**: Serum cholesterol (mg/dl)
    - **FBS**: Fasting blood sugar > 120 mg/dl
    - **Restecg**: Resting ECG results
    - **Thalach**: Maximum heart rate achieved
    - **Exang**: Exercise induced angina
    - **Oldpeak**: ST depression induced by exercise
    - **Slope**: Slope of peak exercise ST segment
    - **CA**: Number of major vessels colored by fluoroscopy
    - **Thal**: Thalassemia (3,6,7)
    """)
