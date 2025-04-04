import pandas as pd
import joblib

def test_heart_disease_model(test_data):
    """
    Function to test the trained heart disease prediction model.
    Loads the model and makes predictions on a sample dataset.
    """
    try:
        # model loading
        production_model = joblib.load('models/uci_heart_disease_model.pkl')
        model = production_model['model']
        optimal_threshold = production_model['metadata']['threshold']



        # engineered features to match training data
        test_data['hr_age_ratio'] = test_data['thalach'] / (test_data['age'] + 1e-5)
        test_data['bp_oldpeak'] = test_data['trestbps'] * (test_data['oldpeak'] + 1)
        test_data['risk_score'] = (test_data['age']/50 + test_data['chol']/200 + test_data['trestbps']/140)

        # Make predictions
        probabilities = model.predict_proba(test_data)[:, 1]
        predictions = (probabilities >= optimal_threshold).astype(int)

        # results DataFrame
        results = pd.DataFrame({
            'Prediction': predictions,
            'Diagnosis': ['Heart Disease' if p == 1 else 'Healthy' for p in predictions],
            'Probability': probabilities,
        })

        # data for display
        display_data = pd.concat([test_data[['age', 'sex', 'cp', 'trestbps', 'chol']], results], axis=1)

        print("=== Heart Disease Prediction Results ===")
        print(f"Using threshold: {optimal_threshold:.3f}\n")
        print(display_data.to_string(index=False))

        return results

    except Exception as e:
        print(f"Error testing model: {str(e)}")
        return None


# pf =pd.read_csv('dataset/test_data.csv')
# test_results = test_heart_disease_model(pf)