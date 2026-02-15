import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def train_triage_model():
    # 1. Resolve project paths and load data
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "synthetic_medical_triage.csv"
    df = pd.read_csv(data_path)
    
    # 2. Preprocessing
    # Encode 'arrival_mode' (walk_in, ambulance, etc.) to numbers
    le = LabelEncoder()
    df['arrival_mode'] = le.fit_transform(df['arrival_mode'])
    
    # Define Features (X) and Target (y)
    X = df.drop('triage_level', axis=1)
    y = df['triage_level']
    
    # 3. Split data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Random Forest Model
    print("Training the Triage Engine...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Save the model and the encoder into project-level models directory
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / 'risk_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open(models_dir / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print(f"Model saved to {models_dir / 'risk_model.pkl'}")

if __name__ == "__main__":
    train_triage_model()