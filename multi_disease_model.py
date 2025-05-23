import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Paths
datasets = {
    "diabetes": "diabetes.csv",
    "heart":    "heart.csv",
    "kidney":   "kidney_disease.csv",
    "liver":    "/content/Indian Liver Patient Dataset (ILPD).csv"
}

models = {}
scalers = {}

# 2. Preprocessing
def preprocess_data(name, df):
    df = df.copy()
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    if name == "kidney":
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.lower()
        if 'classification' in df.columns:
            df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
        df = df.select_dtypes(include=[np.number])

    elif name == "liver":
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        if 'Dataset' in df.columns:
            df['Dataset'] = df['Dataset'].replace({1: 1, 2: 0})

    return df

# 3. Target detection
def get_target_column(name, df):
    candidates = {
        "diabetes": [df.columns[-1]],
        "heart":    [df.columns[-1]],
        "kidney":   ["classification", "class", "target"],
        "liver":    ["Dataset", "class", "target"]
    }
    for c in candidates[name]:
        if c in df.columns:
            return c
    return df.columns[-1]

# 4. Training
def train_model(name, df):
    target = get_target_column(name, df)
    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name.capitalize()} accuracy: {acc:.2f}")

    models[name] = model
    scalers[name] = scaler

def load_and_train_all():
    for name, path in datasets.items():
        print(f"\nLoading and training {name}...")
        df = pd.read_csv(path)
        df_clean = preprocess_data(name, df)
        train_model(name, df_clean)

# 5. Prediction
def predict_disease(name, input_list):
    model = models[name]
    scaler = scalers[name]
    X_scaled = scaler.transform([input_list])
    return model.predict(X_scaled)[0]

# 6. Show sample input and range
def show_sample_inputs():
    print("\n=== Sample Input Format & Ranges ===")
    for name, path in datasets.items():
        df = pd.read_csv(path)
        df_clean = preprocess_data(name, df)
        target = get_target_column(name, df_clean)
        X = df_clean.drop(columns=[target])

        print(f"\n{name.capitalize()} features:")
        for col in X.columns:
            sample_val = X[col].iloc[0]
            col_min = X[col].min()
            col_max = X[col].max()
            print(f"  {col:<20} Example: {sample_val:<8.2f} | Range: [{col_min:.1f}, {col_max:.1f}]")
    print("="*50)

# 7. CLI interface
def run_interface():
    show_sample_inputs()
    print("\nAvailable diseases:", ", ".join(datasets.keys()))
    disease = input("Select a disease to predict: ").strip().lower()
    if disease not in datasets:
        print("Invalid disease name."); return

    df = pd.read_csv(datasets[disease])
    df_clean = preprocess_data(disease, df)
    target = get_target_column(disease, df_clean)
    X = df_clean.drop(columns=[target])
    feature_names = X.columns

    print("\nEnter your values in this order:")
    input_list = []
    for col in feature_names:
        while True:
            try:
                val = float(input(f"  {col}: "))
                input_list.append(val)
                break
            except ValueError:
                print("  Please enter a numeric value.")

    result = predict_disease(disease, input_list)
    print(f"\n✅ Prediction for {disease.capitalize()}: {'Positive' if result == 1 else 'Negative'}")

# Main
if __name__ == "__main__":
    load_and_train_all()
    run_interface()
