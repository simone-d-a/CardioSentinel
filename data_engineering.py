import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# === CONFIGURAZIONE ===
RAW_DATA_FILE = 'heart_cleveland.csv'
COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

def run_data_engineering():
    print("--- INIZIO DATA ENGINEERING ---")

    # 1. CARICAMENTO
    print(f"Caricamento {RAW_DATA_FILE}...")
    df = pd.read_csv(RAW_DATA_FILE, names=COLUMN_NAMES, na_values='?')
    print(f"Dimensioni iniziali: {df.shape}")

    # 2. PULIZIA (Missing Values)
    df = df.dropna()
    print(f"Dimensioni dopo rimozione NaN: {df.shape}")

    # 3. TRASFORMAZIONE TARGET
    # Convertiamo 1,2,3,4 in 1 (Malato) e 0 resta 0 (Sano)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    print("Target binarizzato (0=Sano, 1=Malato).")

    # Lasciamo i numeri originali per avere grafici più puliti.

    # 4. DEFINIZIONE FEATURE E TARGET
    X = df.drop(columns=['target'])
    y = df['target']

    # 5. SCALING (Normalizzazione tra 0 e 1)
    print("Normalizzazione feature (MinMaxScaler)...")
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 6. SPLIT TRAIN/TEST (Stratificato)
    print("Divisione Train/Test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. SALVATAGGIO FILE FINALI
    print("Salvataggio file processati...")
    X_train.to_csv('X_train_final.csv', index=False)
    X_test.to_csv('X_test_final.csv', index=False)
    y_train.to_csv('y_train_final.csv', index=False)
    y_test.to_csv('y_test_final.csv', index=False)

    print("\n--- DATA ENGINEERING COMPLETATO ---")

if __name__ == "__main__":
    run_data_engineering()