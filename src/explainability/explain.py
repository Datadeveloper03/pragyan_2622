import shap
import pickle
import pandas as pd
import os

def process_triage_and_shap(patient_features, selected_features):
    model_path = 'models/risk_model.pkl'
    encoder_path = 'models/label_encoder.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)

    # Base values to prevent crashes if features are unselected
    base_defaults = {
        'age': 45.0, 'heart_rate': 80.0, 'systolic_blood_pressure': 120.0,
        'oxygen_saturation': 98.0, 'body_temperature': 37.0, 'pain_level': 5,
        'chronic_disease_count': 0, 'previous_er_visits': 0, 'arrival_mode': 'walk_in'
    }

    final_inputs = base_defaults.copy()
    for sf in selected_features:
        if sf in patient_features:
            final_inputs[sf] = patient_features[sf]

    df = pd.DataFrame([final_inputs])
    cols = list(base_defaults.keys())
    df = df[cols]
    df['arrival_mode'] = encoder.transform(df['arrival_mode'])

    prediction = int(model.predict(df)[0])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    # Multi-class fix: Isolate the SHAP values for the predicted class
    current_shap = shap_values[prediction][0] if isinstance(shap_values, list) else shap_values[0, :, prediction]

    factors = []
    for i, feature in enumerate(cols):
        if feature in selected_features:
            val = current_shap[i]
            # CLINICAL FIX: Rewording the direction based on multi-class contribution
            direction = f"pushed toward Level {prediction}" if val > 0 else f"pulled away from Level {prediction}"
            
            factors.append({
                "feature": feature,
                "value": final_inputs[feature],
                "direction": direction,
                "shap_value": round(float(val), 3)
            })

    top_factors = sorted(factors, key=lambda x: abs(x['shap_value']), reverse=True)[:3]
    return {
        "triage_level": prediction,
        "top_factors": top_factors
    }