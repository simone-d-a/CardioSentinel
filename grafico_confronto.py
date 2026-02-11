import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# === CONFIGURAZIONE ===
IMG_FILE = "tesi_confronto_metriche.png"

def run_comparison_plot():
    print("--- GENERAZIONE GRAFICO CONFRONTO ---")

    # 1. CARICAMENTO DATI
    try:
        X_train = pd.read_csv('X_train_final.csv')
        X_test = pd.read_csv('X_test_final.csv')
        y_train = pd.read_csv('y_train_final.csv').values.ravel()
        y_test = pd.read_csv('y_test_final.csv').values.ravel()
    except FileNotFoundError:
        print("ERRORE: Esegui prima 'data_engineering.py'!")
        return

    # 2. DEFINIZIONE MODELLI
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # 3. CALCOLO METRICHE
    data = []
    
    print("Calcolo metriche in corso...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Salviamo tutte le metriche in un formato comodo per il grafico
        data.append({"Modello": name, "Metrica": "Accuracy", "Valore": accuracy_score(y_test, y_pred)})
        data.append({"Modello": name, "Metrica": "Recall", "Valore": recall_score(y_test, y_pred)})
        data.append({"Modello": name, "Metrica": "Precision", "Valore": precision_score(y_test, y_pred)})
        data.append({"Modello": name, "Metrica": "F1-Score", "Valore": f1_score(y_test, y_pred)})

    # Creiamo un DataFrame per Seaborn
    df_metrics = pd.DataFrame(data)

    # 4. CREAZIONE GRAFICO A BARRE
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Disegniamo le barre
    chart = sns.barplot(x="Metrica", y="Valore", hue="Modello", data=df_metrics, palette="viridis")
    
    # Aggiungiamo i numeri sopra le barre (così non devi leggerli a occhio)
    for container in chart.containers:
        chart.bar_label(container, fmt='%.2f', padding=3)

    plt.ylim(0, 1.1) # Lasciamo spazio in alto per i numeri
    plt.title("Confronto Prestazioni Modelli", fontsize=16)
    plt.ylabel("Punteggio (0-1)", fontsize=12)
    plt.xlabel("")
    plt.legend(title="Modello", loc='lower right')
    
    plt.tight_layout()
    plt.savefig(IMG_FILE, dpi=300)
    print(f"\nGrafico salvato con successo: {IMG_FILE}")
    plt.show()

if __name__ == "__main__":
    run_comparison_plot()